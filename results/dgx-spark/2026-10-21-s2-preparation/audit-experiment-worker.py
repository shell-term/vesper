"""Private finite experiment: readback versus GPU-SHA with identical S2 setup.

This external wrapper does not edit Vesper source or adopt an audit mode. Run
only after the existing GPU worker releases its lock. The parent owns one
300-second child and up to 40 seconds for cleanup, with no automatic retry.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

EXPECTED_HEAD = "c0439df8cf6961461f052a21be2d3e715cadb16f"
CLEAN_DIGEST = hashlib.sha256(bytes(24)).hexdigest()
AUDIT_MODES = ("readback", "gpu-sha256")
EXPECTED_ARRAYS = 324
TIMEOUT_SECONDS = 300
CLEANUP_STEPS = ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 10)) if os.name != "nt" else ()
MAX_REPORT_BYTES = 4 * 1024**2
MAX_WRAPPER_BYTES = 32 * 1024
SCHEMA = "vesper.external-spark-audit-comparison.v1"
IDENTITY_KEYS = ("platform", "profile", "repository", "model", "revision", "git_head",
                 "dirty_diff_sha256", "registry_sha256", "binding_id", "run_id")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded_read(path, limit):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Expected a bounded regular evidence file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("Evidence grew beyond its bound")
    return raw


def write_exclusive(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("Private evidence exceeded its bound")
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def guard(workspace, run_id, *, initial=None, wrapper_sha=None):
    from vesper import spark_s2_probe as probe
    now = probe._guard(workspace, run_id)
    if now["git_head"] != EXPECTED_HEAD or now["dirty_diff_sha256"] != CLEAN_DIGEST:
        raise RuntimeError("Only the declared clean c043 source may run")
    if initial is not None and any(now[key] != initial[key] for key in IDENTITY_KEYS):
        raise RuntimeError("Workspace/source identity changed")
    if wrapper_sha is not None and sha(bounded_read(Path(__file__), MAX_WRAPPER_BYTES)) != wrapper_sha:
        raise RuntimeError("Frozen external wrapper changed")
    return now


def make_plan(workspace, run_id, initial, wrapper_raw):
    from vesper import spark_s2_probe as probe
    return {"schema": SCHEMA, "source_context": initial, "source_head": EXPECTED_HEAD,
        "run_id": run_id, "external_wrapper": {"sha256": sha(wrapper_raw), "bytes": len(wrapper_raw)},
        "audit_mode_order": list(AUDIT_MODES), "preparation_mode_per_audit": "s2",
        "seed": probe.SEED, "layers_per_mode": list(probe.LAYERS),
        "input_arrays": probe._hash_arrays(probe._inputs(), "input"),
        "expected_compared_arrays": EXPECTED_ARRAYS,
        "component_operations": ["hc.ffn.mixes", "hc.ffn.pre", "projection.wq_a",
                                 "projection.grouped_wo_a", "moe.native_six_routes"],
        "mode_lifetimes": "Fresh assets, authenticated catalog, backends and empty scoped compiler cache per audit mode",
        "instrumentation": "External explicit LayerOperatorPool adapter replaces only ConnectedMoE.config.cache.integrity_mode before entry; actual snapshots are checked",
        "existing_probe_used": "spark_s2_probe._mode('s2', ...); experimental report has its own schema and namespace",
        "probe_module_sha256": sha(bounded_read(Path(probe.__file__), 128 * 1024)),
        "timeout_seconds": TIMEOUT_SECONDS, "timeout_enforced_by": "parent worker child process",
        "cleanup_grace_seconds": 40, "cleanup_deadline_scope": "One absolute deadline shared across all cleanup attempts",
        "gpu_lock_artifact": "s1/gpu-execution.lock", "gpu_lock_inherited_by_child_until_exit": True,
        "physical_available_floor_bytes": 32 * 1024**3,
        "memory_policy": "Existing Spark S2 component budgets, diagnostic admission and shared unified-memory policy; no capacity increase",
        "comparison": "All 324 diagnostic dtype/shape/byte/SHA records equal and layer0 revisit exact in both modes",
        "timing_limitations": ["Fixed readback then gpu-sha256 order; OS/driver/storage state is not reset",
            "Per-mode initialization, first compilation, hashes, diagnostic downloads and cleanup are included",
            "This does not measure full-model throughput or establish superiority of an audit mode"],
        "no_eos_generation": True, "model_generations": 0, "model_calls": 0,
        "no_quality_claim": True, "image_functionality_quality_qualified": False,
        "automatic_retry": False, "scheduled_task": False, "source_modifications": False,
        "downloads": 0, "existing_service_changes": 0, "audit_mode_adopted": False}


class AuditedMoE:
    """Observe the actual opened config, counters and cleanup without arithmetic changes."""
    def __init__(self, owner, audit_mode, rows, failures=None):
        if audit_mode not in AUDIT_MODES or owner.stack is not None or owner.active:
            raise ValueError("Audit configuration may change only before owner entry")
        owner.config = replace(owner.config, cache=replace(owner.config.cache, integrity_mode=audit_mode))
        self.owner, self.audit_mode, self.rows = owner, audit_mode, rows
        self.failures = failures if failures is not None else []
        self.row = {"layer": owner.layer, "requested_integrity_mode": audit_mode, "cleanup_established": False}
    def remember(self, error, stage):
        identity = {"failure_type": type(error).__name__, "stage": stage,
            "cancelled": isinstance(error, KeyboardInterrupt)}
        from vesper.memory import RuntimeCancelled
        identity["cancelled"] |= isinstance(error, RuntimeCancelled)
        if not self.failures:
            self.failures.append({**identity, "traceback": "".join(traceback.format_exception(error))[-60000:]})
            self.row["first_observed_failure"] = identity
        else:
            self.row.setdefault("secondary_observation_failures", [])
            if len(self.row["secondary_observation_failures"]) < 8:
                self.row["secondary_observation_failures"].append(identity)
    def __getattr__(self, name):
        return getattr(self.owner, name)
    def __enter__(self):
        try:
            self.owner.__enter__()
            self.rows.append(self.row)
            self.row["snapshot_after_open"] = self.owner.snapshot()
            self._verify_mode(self.row["snapshot_after_open"])
            return self
        except (Exception, KeyboardInterrupt) as error:
            self.remember(error, "moe.entry")
            raise
    def _verify_mode(self, snapshot):
        if snapshot["cache_config"]["integrity_mode"] != self.audit_mode or snapshot["cache"]["integrity_mode"] != self.audit_mode:
            raise RuntimeError("The opened audit mode differs from the declared mode")
    def forward(self, *args, **kwargs):
        try:
            return self.owner.forward(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as error:
            self.remember(error, "moe.forward")
            raise
        finally:
            try:
                self.row["snapshot_before_close"] = self.owner.snapshot()
                self._verify_mode(self.row["snapshot_before_close"])
            except (Exception, KeyboardInterrupt) as error:
                self.remember(error, "moe.snapshot")
                raise
    def __exit__(self, *error):
        try:
            return self.owner.__exit__(*error)
        except (Exception, KeyboardInterrupt) as failure:
            self.remember(failure, "moe.cleanup")
            raise
        finally:
            self.row["cleanup"] = dict(getattr(self.owner, "cleanup", {}))
            self.row["cleanup_established"] = bool(self.row["cleanup"] and all(value == 0 for value in self.row["cleanup"].values()))


@contextmanager
def audit_pool(probe, audit_mode, rows, failures=None):
    native_pool = probe.LayerOperatorPool
    class ExperimentalAuditPool(native_pool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_factory = self.moe_factory
            self.moe_factory = lambda layer: AuditedMoE(original_factory(layer), audit_mode, rows, failures)
        def __exit__(self, *error):
            from vesper.memory import current_policy
            try:
                policy = current_policy()
                if error[0] is None and policy is not None:
                    policy.checkpoint()
            finally:
                super().__exit__(*error)
    with patch.object(probe, "LayerOperatorPool", ExperimentalAuditPool):
        yield


def terminal_checks(mode):
    after = mode.get("memory_after_close", {})
    minimum = after.get("minimum_observed_available_bytes")
    return {"existing_owned_cleanup": mode.get("owned_resources_closed") is True,
        "runtime_closed": mode.get("memory_runtime_closed") is True,
        "zero_charges": after.get("charged_host_bytes") == 0 and after.get("pinned_bytes") == 0
            and isinstance(after.get("device_used_bytes"), dict) and not any(after["device_used_bytes"].values()),
        "zero_owners_and_operations": after.get("live_leases") == 0 and after.get("live_reclaimers") == 0
            and after.get("inflight_operations") == 0,
        "monitor_stopped_and_healthy": after.get("monitor_thread_stopped") is True
            and "monitor_error" in after and after["monitor_error"] is None,
        "physical_floor_preserved": type(minimum) is int
            and minimum >= after.get("available_floor_bytes", 32 * 1024**3) >= 32 * 1024**3}


def audit_counters(rows, audit_mode):
    if len(rows) != 3 or [row["layer"] for row in rows] != [0, 1, 0]:
        raise ValueError("Expected the three predeclared MoE visits")
    totals = {}
    for row in rows:
        snapshot = row["snapshot_before_close"]
        if (snapshot["cache_config"]["integrity_mode"] != audit_mode
                or snapshot["cache"]["integrity_mode"] != audit_mode or not row["cleanup_established"]):
            raise RuntimeError("Observed mode or owned MoE cleanup differs")
        counters = snapshot["cache"]["counters"]
        if counters["requests_completed"] != 6 or counters["integrity_failures"]:
            raise ValueError("Each original MoE visit must complete six audited routes")
        for key, value in counters.items():
            if type(value) is not int or value < 0:
                raise ValueError("Invalid measured cache counter")
            totals[key] = totals.get(key, 0) + value
    readback = totals["primary_audit_readback_bytes"] + totals["helper_audit_readback_bytes"]
    payload = totals["primary_gpu_audit_payload_bytes"] + totals["helper_gpu_audit_payload_bytes"]
    digest = totals["primary_gpu_audit_digest_d2h_bytes"] + totals["helper_gpu_audit_digest_d2h_bytes"]
    if audit_mode == "readback" and not (readback > 0 and payload == 0 and digest == 0):
        raise ValueError("Readback mode did not perform the expected actual audit transfers")
    if audit_mode == "gpu-sha256" and not (readback == 0 and payload > digest > 0):
        raise ValueError("GPU-SHA mode did not perform the expected actual digest audit")
    return {"cache_counter_totals": totals, "actual_full_audit_readback_bytes": readback,
        "actual_gpu_hashed_payload_bytes": payload, "actual_gpu_digest_d2h_bytes": digest,
        "scope": "Summed measured cache counters from three distinct original MoE owner lifetimes; not whole-model I/O"}


def compare_modes(modes):
    if len(modes) != 2 or [m["requested_integrity_mode"] for m in modes] != list(AUDIT_MODES):
        raise ValueError("Both declared audit modes must finish exactly once")
    first, second = [mode["array_hashes"] for mode in modes]
    mismatches = sorted(name for name in set(first) | set(second) if first.get(name) != second.get(name))
    if len(first) != EXPECTED_ARRAYS or len(second) != EXPECTED_ARRAYS or mismatches:
        raise ValueError("Expected all 324 exact diagnostic array records to match")
    if not all(mode["revisited_layer_array_bits_exact"] for mode in modes):
        raise ValueError("A layer0 revisit changed diagnostic array bits")
    return {"array_bits_exact": True, "compared_array_count": len(first), "mismatched_arrays": []}


def collect(workspace, plan, plan_sha, job, *, lock_fd=None):
    from vesper import spark_s2_probe as probe
    from vesper.memory import RuntimeCancelled
    initial = plan["source_context"]
    run_id = plan["run_id"]
    report = {"schema": SCHEMA, "status": "failed", "external_wrapper": plan["external_wrapper"],
        "source_head": EXPECTED_HEAD, "plan": plan, "plan_sha256": plan_sha,
        "standard_probe_default_report": False, "preparation_mode": "s2", "modes": [],
        "array_bits_exact": False, "owned_resources_closed": False, "model_generations": 0,
        "eos_qualified": False, "full_model_quality_qualified": False, "image_quality_qualified": False,
        "general_performance_claim": False, "audit_mode_adopted": False, "automatic_retry": False}
    paths = {name: workspace.artifact_path(value) for name, value in (("metadata", "s1/import-source"),
        ("originals", "s1/originals"), ("integrity", "s1/original-integrity"), ("fixtures", "s1/moe-original"))}
    started = time.perf_counter()
    stage = "platform"
    failures = []
    try:
        if probe.platform.system() != "Linux" or probe.platform.machine().lower() not in ("aarch64", "arm64"):
            raise probe.Unsupported("Only Linux ARM64 Spark may perform this experiment")
        report["physical_memory_before"] = probe.host_memory()
        probe.require_capacity(report["physical_memory_before"], probe.DIAGNOSTIC_ALLOWANCE)
        report["hardware"] = probe.Driver().inventory()
        inputs = probe._inputs()
        if probe._hash_arrays(inputs, "input") != plan["input_arrays"]:
            raise ValueError("Fixed input hashes differ from the frozen plan")
        for audit_mode in AUDIT_MODES:
            stage = audit_mode
            if lock_fd is not None:
                validate_inherited_lock(workspace, lock_fd)
            guard(workspace, run_id, initial=initial, wrapper_sha=plan["external_wrapper"]["sha256"])
            if sha(bounded_read(job / "plan.json", MAX_REPORT_BYTES)) != plan_sha:
                raise RuntimeError("Frozen experiment plan changed")
            rows = []
            count_before = len(report["modes"])
            try:
                with audit_pool(probe, audit_mode, rows, failures):
                    probe._mode("s2", workspace, paths, inputs, report["modes"])
            finally:
                if len(report["modes"]) > count_before:
                    mode = report["modes"][-1]
                    mode["preparation_mode"] = mode.pop("mode")
                    mode["requested_integrity_mode"] = audit_mode
                    mode["actual_moe_audit_observations"] = rows
            mode["terminal_checks"] = terminal_checks(mode)
            if not all(mode["terminal_checks"].values()):
                raise RuntimeError("Terminal memory, monitoring or ownership checks failed")
            mode["actual_audit_counters"] = audit_counters(rows, audit_mode)
        report.update(compare_modes(report["modes"]))
        report["status"] = "equivalent_experimental_audit_components"
    except (Exception, KeyboardInterrupt) as error:
        first = failures[0] if failures else {"failure_type": type(error).__name__, "stage": stage,
            "cancelled": isinstance(error, (KeyboardInterrupt, RuntimeCancelled)),
            "traceback": "".join(traceback.format_exception(error))[-60000:]}
        report.update(status="cancelled" if first["cancelled"] else "failed",
            failure_type=first["failure_type"], failure_stage=stage, first_failure_observer_stage=first["stage"])
        raw = first
        failure_raw = write_exclusive(job / "failure.json", raw)
        report["private_failure"] = {"artifact": f"s2/audit/{run_id}/failure.json", "sha256": sha(failure_raw)}
    finally:
        gc.collect()
        report["owned_resources_closed"] = bool(report["modes"] and all(
            all(terminal_checks(mode).values()) for mode in report["modes"]))
        report["wall_seconds"] = time.perf_counter() - started
    if len(json.dumps(report).encode()) > MAX_REPORT_BYTES:
        raise ValueError("Experimental report exceeded its bound")
    guard(workspace, run_id, initial=initial, wrapper_sha=plan["external_wrapper"]["sha256"])
    if lock_fd is not None:
        validate_inherited_lock(workspace, lock_fd)
    return report


class ChildCleanup:
    """One monotonic deadline, shared by timeout, unwinding and final cleanup."""
    def __init__(self):
        self.child = None
        self.started = self.deadline = None
        self.sent = set()
    def stop(self):
        child = self.child
        if child is None or child.poll() is not None:
            return
        if self.deadline is None:
            self.started = time.monotonic()
            self.deadline = self.started + sum(grace for _, grace in CLEANUP_STEPS)
        phase_end = self.started
        for sig, grace in CLEANUP_STEPS:
            phase_end += grace
            if child.poll() is not None:
                return
            remaining = min(phase_end, self.deadline) - time.monotonic()
            if remaining <= 0:
                continue
            if sig not in self.sent:
                try: os.killpg(child.pid, sig)
                except ProcessLookupError: pass
                self.sent.add(sig)
            try: child.wait(timeout=remaining)
            except subprocess.TimeoutExpired: pass
        if child.poll() is None:
            # An interrupted earlier attempt must not skip the final signal,
            # but a repeated attempt receives no fresh waiting allowance.
            sig = CLEANUP_STEPS[-1][0]
            if sig not in self.sent:
                try: os.killpg(child.pid, sig)
                except ProcessLookupError: pass
                self.sent.add(sig)
            raise RuntimeError("Own experiment child did not exit within its one cleanup deadline")


def validate_inherited_lock(workspace, descriptor):
    """Require the inherited writable descriptor and its already-held lock.

    Linux flock ownership follows the shared open-file description. The child
    deliberately keeps this descriptor until process exit, including if the
    supervising parent is terminated or fails to observe child termination.
    """
    import fcntl
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("Expected the explicitly inherited GPU-lock descriptor")
    path = workspace.artifact_path("s1/gpu-execution.lock")
    opened, named = os.fstat(descriptor), path.stat()
    if (path.is_symlink() or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDWR):
        raise RuntimeError("Inherited GPU-lock identity or descriptor configuration differs")
    with path.open("a+b") as independent:
        try:
            fcntl.flock(independent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(independent.fileno(), fcntl.LOCK_UN)
            raise RuntimeError("Inherited GPU-lock description was not already locked")
    # This succeeds only for our inherited open-file description; another
    # process holding an unrelated lock cannot make the validation pass.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "vesper-spark")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-sha", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.workspace.resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    from vesper.workspace import Workspace
    workspace = Workspace.open(root)
    initial = guard(workspace, args.run_id)
    result_relative = f"s2/audit/{args.run_id}.json"
    if workspace.result_path(result_relative).exists():
        raise FileExistsError("Refusing to overwrite earlier audit evidence")
    job = workspace.artifact_path(f"s2/audit/{args.run_id}", create_parent=True)
    if args.child:
        validate_inherited_lock(workspace, args.lock_fd)
        plan_raw = bounded_read(job / "plan.json", MAX_REPORT_BYTES)
        if sha(plan_raw) != args.plan_sha:
            raise RuntimeError("Frozen child plan differs")
        plan = json.loads(plan_raw)
        if plan["run_id"] != args.run_id:
            raise RuntimeError("Frozen plan run identity differs")
        guard(workspace, args.run_id, initial=plan["source_context"], wrapper_sha=plan["external_wrapper"]["sha256"])
        report = collect(workspace, plan, args.plan_sha, job, lock_fd=args.lock_fd)
        path = workspace.write_result(result_relative, report, run_id=args.run_id, initial_context=plan["source_context"])
        print(json.dumps({"status": report["status"], "result": path.relative_to(root).as_posix(),
            "array_bits_exact": report["array_bits_exact"], "owned_resources_closed": report["owned_resources_closed"]}))
        return 0 if report["status"] == "equivalent_experimental_audit_components" else 2
    if os.name == "nt":
        raise RuntimeError("The finite parent worker requires Linux process groups and advisory GPU lock")
    import fcntl
    if args.lock_fd is not None:
        raise ValueError("Only the child may receive an inherited GPU-lock descriptor")
    job.mkdir(exist_ok=False)
    wrapper_raw = bounded_read(Path(__file__), MAX_WRAPPER_BYTES)
    with (job / "worker.py").open("xb") as stream:
        stream.write(wrapper_raw)
    plan = make_plan(workspace, args.run_id, initial, wrapper_raw)
    plan_sha = sha(write_exclusive(job / "plan.json", plan))
    state = {"status": "running", "run_id": args.run_id, "source_head": EXPECTED_HEAD,
        "external_wrapper_sha256": sha(wrapper_raw), "plan_sha256": plan_sha, "started_unix": time.time(),
        "timeout_seconds": TIMEOUT_SECONDS, "automatic_retry": False, "model_generations": 0}
    child = None
    cleanup = ChildCleanup()
    def save():
        temporary = job / "state.tmp"
        write_exclusive(temporary, state)
        os.replace(temporary, job / "state.json")
    try:
        save()
        with ExitStack() as owned:
            lock = owned.enter_context(workspace.artifact_path("s1/gpu-execution.lock").open("a+b"))
            # Registered after the lock owner: the child is always stopped
            # before this process relinquishes the GPU-execution lock.
            owned.callback(cleanup.stop)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard(workspace, args.run_id, initial=initial, wrapper_sha=sha(wrapper_raw))
            env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
            env.pop("PYTHONHOME", None)
            with (job / "probe.log").open("xb") as log:
                child = subprocess.Popen([sys.executable, "-B", str(job / "worker.py"), "--workspace", str(root),
                    "--run-id", args.run_id, "--child", "--plan-sha", plan_sha, "--lock-fd", str(lock.fileno())],
                    cwd=root, env=env, pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                cleanup.child = child
                state["child_pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    state["timed_out"] = True
                    cleanup.stop()
                    code = child.returncode
                state.update(exit_code=code, owned_child_exited=child.poll() is not None,
                    status="finished" if code == 0 and not state.get("timed_out") else "failed")
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            failure_type=type(error).__name__, private_reason=str(error)[:4096])
    finally:
        try:
            if child is not None and child.poll() is None:
                cleanup.stop()
        except Exception as error:
            state.update(status="failed", child_cleanup_failure=type(error).__name__)
        state.update(finished_unix=time.time(), owned_child_exited=child is None or child.poll() is not None)
        save()
    if workspace.result_path(result_relative).exists():
        identity = workspace.check_result(result_relative)
        state["result"] = {"relative": result_relative, "sha256": identity["result_sha256"]}
        save()
    print(json.dumps(state))
    return 0 if state["status"] == "finished" else 2


def interrupted(*_):
    raise KeyboardInterrupt("Own finite audit comparison cancelled")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
