"""Real temporary file/HTTP mocks check bounded parallel original acquisition."""

from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from vesper import spark_artifacts as assets
from vesper.resources import GiB, MiB, MemoryBudget


class Workspace:
    def __init__(self, root):
        self.base = Path(root) / "artifacts/dgx-spark"
        self.epoch = 0
        self.check, self.check_runtime_module = Mock(), Mock()

    def artifact_path(self, name):
        return self.base / name

    def context(self):
        return {"platform": "dgx-spark", "profile": "native", "model": assets.MODEL,
                "revision": assets.REVISION, "epoch": self.epoch}

    def stamp_result(self, payload, *, initial_context=None):
        if initial_context is not None and initial_context != self.context():
            raise ValueError("Source context changed")
        return {**payload, "workspace_context": self.context()}


class Response:
    def __init__(self, source, name, start):
        self.source, self.name, self.start = source, name, start
        self.raw = source.payloads[name]
        self.headers = {"Content-Length": str(len(self.raw) - start), "Content-Encoding": "identity",
                        "Content-Range": f"bytes {start}-{len(self.raw)-1}/{len(self.raw)}"}
        self.status, self.calls = 206, 0
        self.stream = io.BytesIO(self.raw[start:])

    def __enter__(self):
        with self.source.lock:
            if self.name in self.source.active:
                raise AssertionError("Two workers own the same shard")
            self.source.active.add(self.name)
            self.source.peak_active = max(self.source.peak_active, len(self.source.active))
            self.source.starts.append((self.name, self.start))
        if self.source.barrier is not None and self.name in self.source.first_wave:
            self.source.barrier.wait(timeout=5)
        return self

    def read(self, size):
        self.calls += 1
        if self.name == self.source.fail_name and self.calls == 2:
            raise OSError("Injected interrupted shard")
        time.sleep(.001)
        return self.stream.read(size)

    def __exit__(self, *_):
        self.stream.close()
        with self.source.lock:
            self.source.active.remove(self.name)
            self.source.closed += 1


class Source:
    def __init__(self, count=6, workers=None):
        self.payloads = {f"model-{number:05d}-of-00048.safetensors": bytes([number]) * (2 * MiB + 17)
                         for number in range(1, count + 1)}
        self.inventory = [{"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                          for name, raw in self.payloads.items()]
        self.lock, self.active, self.starts = threading.Lock(), set(), []
        self.closed, self.peak_active, self.fail_name = 0, 0, None
        self.first_wave = set(list(self.payloads)[:workers]) if workers else set()
        self.barrier = threading.Barrier(workers) if workers else None

    def open(self, request, **kwargs):
        name = request.full_url.rsplit("/", 1)[1]
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        return Response(self, name, start)


class ParallelAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Windows CI may return an 8.3 TEMP alias. Match the real Workspace's
        # canonical root before the unchanged namespace guard compares paths.
        self.root = Path(self.temp.name).resolve()
        self.workspace = Workspace(self.root)
        self.metadata, self.originals = [self.workspace.base / name for name in ("metadata", "originals")]
        self.metadata.mkdir(parents=True)
        self.originals.mkdir()
        (self.metadata / "tree.json").write_bytes(b"CPU authenticated-inventory seam")
        available = patch("vesper.resources.physical_memory", return_value={"available": 128 * GiB})
        available.start(); self.addCleanup(available.stop)
        disk = patch.object(assets.shutil, "disk_usage", return_value=Mock(free=2**40))
        disk.start(); self.addCleanup(disk.stop)

    def group(self, source, workers=4, checkpoint=lambda: None, progress=None):
        return assets._download_group(self.originals, source.inventory, 2 * MiB, workers=workers,
            opener=source.open, checkpoint=checkpoint, progress=progress)

    def public(self, source, workers=4, progress=None, priority_shards=()):
        with patch.object(assets, "shard_inventory", return_value=source.inventory):
            return assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                chunk_bytes=2 * MiB, execute=True, workspace=self.workspace, opener=source.open,
                progress=progress, workers=workers, priority_shards=priority_shards)

    def test_four_independent_streams_publish_whole_sha_verified_originals(self):
        source = Source(workers=4)
        count = self.group(source)
        self.assertEqual(count, sum(len(raw) for raw in source.payloads.values()))
        self.assertEqual(source.peak_active, 4)
        self.assertEqual(source.active, set())
        self.assertEqual(source.closed, 6)
        self.assertEqual(len(source.starts), len(set(name for name, _ in source.starts)))
        for name, raw in source.payloads.items():
            self.assertEqual((self.originals / name).read_bytes(), raw)
            self.assertFalse((self.originals / (name + ".part")).exists())

    def test_progress_callbacks_are_serial_and_global_counts_are_monotone(self):
        source, calls, active, peak = Source(workers=4), [], 0, 0
        def progress(row):
            nonlocal active, peak
            active += 1; peak = max(active, peak)
            time.sleep(.001)
            calls.append(row)
            active -= 1
        self.group(source, progress=progress)
        self.assertEqual(peak, 1)
        self.assertEqual([row["completed_shards"] for row in calls], sorted(row["completed_shards"] for row in calls))
        self.assertTrue(all(1 <= row["active_shard_tasks"] <= 4 for row in calls))
        self.assertTrue(all(row["workers"] == 4 for row in calls))
        for name in source.payloads:
            offsets = [row["acknowledged_bytes"] for row in calls if row["shard"] == name]
            self.assertEqual(offsets, sorted(offsets))

    def test_failure_stops_submissions_and_joins_other_active_worker_before_raising(self):
        source = Source(count=8)
        barrier, other_closed, stopped_seen = threading.Barrier(2), threading.Event(), threading.Event()
        started = []
        def transfer(root, row, chunk_bytes, *, cancelled, **kwargs):
            started.append(row["file"])
            barrier.wait(timeout=5)
            if row["file"] == source.inventory[0]["file"]:
                raise ValueError("Primary original transfer failure")
            deadline = time.monotonic() + 5
            while not cancelled():
                if time.monotonic() > deadline:
                    raise AssertionError("Active worker did not receive cooperative cancellation")
                time.sleep(.001)
            stopped_seen.set()
            time.sleep(.01)
            other_closed.set()
            raise assets.AcquisitionCancelled("Secondary cancellation")
        with patch.object(assets, "_download_one", side_effect=transfer):
            with self.assertRaisesRegex(ValueError, "Primary original"):
                self.group(source, workers=2)
        self.assertTrue(stopped_seen.is_set())
        self.assertTrue(other_closed.is_set())
        self.assertEqual(set(started), {row["file"] for row in source.inventory[:2]})
        self.assertFalse(any(thread.name.startswith("vesper-original") for thread in threading.enumerate()))

    def test_interrupted_parallel_prefix_resumes_exactly_and_other_workers_are_closed(self):
        source = Source(count=4, workers=2)
        source.fail_name = source.inventory[0]["file"]
        with self.assertRaisesRegex(OSError, "interrupted shard"):
            self.group(source, workers=2)
        self.assertEqual(source.active, set())
        part = self.originals / (source.fail_name + ".part")
        self.assertEqual(part.stat().st_size, 2 * MiB)
        source.fail_name, source.barrier = None, None
        source.starts.clear()
        self.group(source, workers=4)
        self.assertIn((source.inventory[0]["file"], 2 * MiB), source.starts)
        for name, raw in source.payloads.items():
            self.assertEqual((self.originals / name).read_bytes(), raw)

    def test_publication_transition_is_not_observed_as_another_partial_owner(self):
        source, publishing = Source(count=4, workers=4), threading.Event()
        original = assets._publish
        def publish(part, final):
            publishing.set()
            try:
                time.sleep(.003)
                return original(part, final)
            finally:
                publishing.clear()
        def checkpoint():
            self.assertFalse(publishing.is_set(), "Ownership scan overlapped a temporary POSIX hard-link publication")
        with patch.object(assets, "_publish", side_effect=publish):
            self.group(source, checkpoint=checkpoint)

    def test_public_job_scales_admission_and_releases_namespace_after_every_worker(self):
        source, budgets, held = Source(count=4, workers=4), [], []
        def budget(*args, **kwargs):
            value = MemoryBudget(*args, **kwargs)
            budgets.append(value)
            return value
        @contextmanager
        def lock(root):
            held.append(True)
            try:
                yield
            finally:
                self.assertEqual(source.active, set())
                held.pop()
        with patch.object(assets, "MemoryBudget", side_effect=budget), patch.object(assets, "_download_lock", side_effect=lock):
            result = self.public(source)
        self.assertTrue(result["all_worker_tasks_joined"])
        self.assertEqual(result["workers"], 4)
        self.assertEqual(result["transfer_memory_allowance_bytes"], 256 * MiB)
        self.assertEqual(budgets[0].peak, 256 * MiB)
        self.assertEqual(budgets[0].reserved, 0)
        self.assertEqual(held, [])

    def test_source_context_change_cancels_all_workers_and_never_publishes_completion(self):
        source = Source(count=4, workers=2)
        def progress(row):
            self.workspace.epoch = 1
        with self.assertRaisesRegex(ValueError, "Source context changed"):
            self.public(source, workers=2, progress=progress)
        self.assertEqual(source.active, set())
        self.assertFalse((self.originals / "download-manifest.json").exists())

    def test_parallel_memory_admission_refuses_before_any_stream_starts(self):
        source = Source(count=4)
        with patch("vesper.resources.physical_memory", return_value={"available": 32 * GiB + 128 * MiB}):
            with self.assertRaises(MemoryError):
                self.public(source)
        self.assertEqual(source.starts, [])
        self.assertFalse((self.originals / "download-manifest.json").exists())

    def test_parallel_disk_pressure_stops_all_streams_without_completion_metadata(self):
        source, free = Source(count=4, workers=2), 2**40
        def progress(row):
            nonlocal free
            free = 64 * GiB + 32 * MiB
        with patch.object(assets.shutil, "disk_usage", side_effect=lambda root: Mock(free=free)):
            with self.assertRaisesRegex(OSError, "SSD reserve"):
                self.public(source, workers=2, progress=progress)
        self.assertEqual(source.active, set())
        self.assertFalse((self.originals / "download-manifest.json").exists())

    def test_invalid_worker_counts_and_duplicate_shards_cannot_start_network(self):
        source = Source(count=2)
        for workers in (0, 3, 5, True, 2.0):
            with patch.object(assets, "_scope") as scope:
                with self.assertRaises(ValueError):
                    self.public(source, workers=workers)
                scope.assert_not_called()
        source.inventory.append(source.inventory[0])
        with patch.object(assets, "_download_one") as transfer:
            with self.assertRaisesRegex(ValueError, "only one acquisition task"):
                self.group(source)
            transfer.assert_not_called()

    def test_priority_changes_only_transfer_order_without_omitting_original_shards(self):
        source = Source(count=6)
        original = [row["file"] for row in source.inventory]
        priority = [original[5], original[3], original[1]]
        result = self.public(source, workers=1, priority_shards=priority)
        wanted = priority + [name for name in original if name not in priority]
        self.assertEqual([name for name, _ in source.starts], wanted)
        self.assertEqual(result["priority_shards"], priority)
        self.assertEqual(result["transfer_order"], wanted)
        self.assertEqual(result["shards"], 6)
        self.assertEqual([row["file"] for row in source.inventory], original)
        self.assertEqual(result["original_bytes"], sum(row["bytes"] for row in source.inventory))
        self.assertEqual({p.name for p in self.originals.glob("*.safetensors")}, set(original))

    def test_priority_prefix_fills_first_parallel_wave_and_every_shard_is_required(self):
        source = Source(count=6)
        names = [row["file"] for row in source.inventory]
        priority, seen, barrier = [names[5], names[2]], [], threading.Barrier(2)
        def transfer(root, row, chunk_bytes, **kwargs):
            seen.append(row["file"])
            if row["file"] in priority:
                barrier.wait(timeout=5)
            return row["bytes"]
        with patch.object(assets, "_download_one", side_effect=transfer):
            result = self.public(source, workers=2, priority_shards=priority)
        self.assertEqual(set(seen[:2]), set(priority))
        self.assertEqual(len(seen), len(names))
        self.assertEqual(set(seen), set(names))
        self.assertEqual(result["shards"], len(names))

    def test_invalid_priority_cannot_start_network_or_create_partials(self):
        source = Source(count=2)
        registered = source.inventory[0]["file"]
        for priority in ((registered, registered), ("../../outside",), ("model-00043-of-00048.safetensors",),
                         (True,), registered, {registered}):
            with self.subTest(priority=priority):
                with patch.object(assets, "_download_one") as transfer:
                    with self.assertRaises(ValueError):
                        self.public(source, priority_shards=priority)
                    transfer.assert_not_called()
                self.assertEqual(list(self.originals.iterdir()), [])

    def test_default_single_worker_retains_one_stream_and_64mib_admission(self):
        source = Source(count=2)
        callback_threads = []
        with patch.object(assets, "shard_inventory", return_value=source.inventory):
            plan = assets.plan_originals(self.metadata, self.originals, quota_bytes=GiB, workspace=self.workspace)
            result = assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                chunk_bytes=2 * MiB, execute=True, workspace=self.workspace, opener=source.open,
                progress=lambda row: callback_threads.append(threading.current_thread()))
        self.assertEqual(plan["workers"], 1)
        self.assertEqual(plan["transfer_memory_allowance_bytes"], 64 * MiB)
        self.assertEqual(result["workers"], 1)
        self.assertEqual(source.peak_active, 1)
        self.assertTrue(callback_threads)
        self.assertTrue(all(thread is threading.current_thread() for thread in callback_threads))


if __name__ == "__main__":
    unittest.main()
