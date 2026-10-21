"""Private finite current-source revalidation: fresh gates and two chat partials.

The adjacent frozen plan supplies source_head, worker_sha256, created_unix,
deadline_unix and overall_timeout_seconds=900. No production source changes,
checkpoint download, EOS run, practical-quality run or automatic retry occurs.
This distinct loop iteration validates the newly committed native source;
historical source-bound records and earlier failed attempts remain untouched.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import stat
import subprocess
import sys
import time
import traceback

JOB = "s2-short-loop-validation-r01"
JOB_NAMESPACE = "s2/short-loop-validation-r01"
CLEAN_DIGEST = hashlib.sha256(bytes(24)).hexdigest()
LIMIT = 4 * 1024**2
OVERALL_SECONDS = 900
CHILD_SECONDS = 300
CLEANUP_STEPS = ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 10)) if os.name != "nt" else ()
IDENTITY_KEYS = ("platform", "profile", "repository", "model", "revision", "git_head",
                 "dirty_diff_sha256", "registry_sha256", "binding_id", "run_id")
PROMPT_IDS = [0, 128803, 3085, 344, 223, 20, 940, 223, 20, 33, 9361, 418, 270, 1167, 1353, 16, 128804, 128822]
PINNED_INTERPRETER_TARGET = Path("/usr/bin/python3.12")
SYSTEM_INTERPRETER_ALIAS = Path("/usr/bin/python3")


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path, limit=LIMIT):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Missing or excessive regular private evidence")
    with path.open("rb") as stream: raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit: raise ValueError("Evidence grew beyond its bound")
    return raw


def exclusive(path, value, limit=LIMIT):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if len(raw) > limit: raise ValueError("Private evidence bound exceeded")
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return raw


def validate_plan(plan, worker_raw, now):
    import re
    if (not isinstance(plan, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("source_head")))
            or plan.get("worker_sha256") != sha(worker_raw) or len(worker_raw) > 40 * 1024
            or plan.get("overall_timeout_seconds") != OVERALL_SECONDS
            or not isinstance(plan.get("interpreters"), dict) or set(plan["interpreters"]) != {"native", "reference"}
            or any(not isinstance(value, dict) or value.get("target") != str(PINNED_INTERPRETER_TARGET)
                or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("binary_sha256")))
                or value.get("runtime", {}).get("version_info", [])[:2] != [3, 12]
                or value.get("runtime", {}).get("implementation") != "cpython" for value in plan["interpreters"].values())
            or type(plan.get("created_unix")) not in (int, float)
            or type(plan.get("deadline_unix")) not in (int, float)
            or not 0 < plan["deadline_unix"] - plan["created_unix"] <= OVERALL_SECONDS
            or not plan["created_unix"] <= now < plan["deadline_unix"]):
        raise ValueError("Frozen source, worker or finite plan differs")
    return min(OVERALL_SECONDS, plan["deadline_unix"] - now)


def _runtime_version_identity(target_stat):
    if (sys.platform != "linux" or Path(sys._base_executable).resolve(strict=True) != PINNED_INTERPRETER_TARGET
            or (os.stat("/proc/self/exe").st_dev, os.stat("/proc/self/exe").st_ino) != (target_stat.st_dev, target_stat.st_ino)
            or tuple(sys.version_info[:2]) != (3, 12) or sys.implementation.name != "cpython"):
        raise RuntimeError("Tool version must come from the running pinned interpreter inode")
    return {"version_info": list(sys.version_info), "version": sys.version, "implementation": sys.implementation.name}


def _binary_identity(path):
    from vesper.workspace import _no_links
    _no_links(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_mode & 0o022 or os.access(path, os.W_OK)
                or not before.st_mode & 0o111 or not 0 < before.st_size <= 128 * 1024**2):
            raise RuntimeError("Pinned interpreter must be bounded, root owned and not group/other writable")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while raw := stream.read(1024**2): digest.update(raw)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_mode")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise RuntimeError("Pinned interpreter changed during read-only identity collection")
        named = path.stat()
        if any(getattr(after, key) != getattr(named, key) for key in fields):
            raise RuntimeError("Pinned interpreter path changed during identity collection")
        return {"binary_sha256": digest.hexdigest(), "binary_bytes": after.st_size,
            "binary_mode": stat.S_IMODE(after.st_mode), "binary_owner_uid": after.st_uid,
            "runtime": _runtime_version_identity(after)}
    finally: os.close(descriptor)


def _link_version(info):
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        "st_uid", "st_mode", "st_nlink"))


def _protected_system_alias(path, info, raw_link):
    from vesper.workspace import _no_links
    _no_links(path.parent)
    parent = path.parent.stat()
    if (path != SYSTEM_INTERPRETER_ALIAS or info.st_uid != 0
            or raw_link != "python3.12" or path.parent / raw_link != PINNED_INTERPRETER_TARGET
            or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0 or parent.st_mode & 0o022
            or os.access(path.parent, os.W_OK)):
        raise ValueError("System interpreter alias or its protected parent differs")


def interpreter_identity(workspace, name):
    """Explicit executable-link policy; never relax evidence-path validation."""
    if name not in ("native", "reference"): raise ValueError("Unknown fixed interpreter")
    directory = workspace.artifact_path("venvs/" + name + "/bin")
    if not directory.is_dir(): raise ValueError("Missing guarded real interpreter directory")
    configuration = workspace.artifact_path("venvs/" + name + "/pyvenv.cfg")
    config_sha = sha(read(configuration, 64 * 1024))
    candidate = directory / "python"
    current, links, seen = candidate, [], set()
    for _ in range(5):
        if current == PINNED_INTERPRETER_TARGET: break
        if (current.parent != directory and current != SYSTEM_INTERPRETER_ALIAS) or current in seen:
            raise ValueError("Interpreter link leaves the fixed venv/bin or forms a cycle")
        seen.add(current)
        info = current.lstat()
        if not stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Expected an intentional single-link interpreter symlink")
        raw_link = os.readlink(current)
        if not raw_link or len(os.fsencode(raw_link)) > 256:
            raise ValueError("Interpreter symlink text exceeds its bound")
        if current == SYSTEM_INTERPRETER_ALIAS:
            _protected_system_alias(current, info, raw_link)
        links.append({"scope": "system_alias" if current == SYSTEM_INTERPRETER_ALIAS else "venv_bin",
            "name": current.name, "target": raw_link, "version": list(_link_version(info))})
        target = Path(raw_link)
        current = Path(os.path.abspath(target if target.is_absolute() else current.parent / target))
    else: raise ValueError("Interpreter symlink chain exceeds its bound")
    if current != PINNED_INTERPRETER_TARGET or not links:
        raise ValueError("Interpreter does not resolve to the one pinned target")
    binary = _binary_identity(current)
    # Symlink0777 is normal. Recheck its identity/text and protected parents.
    workspace.artifact_path("venvs/" + name + "/bin")
    for entry in links:
        link = SYSTEM_INTERPRETER_ALIAS if entry["scope"] == "system_alias" else directory / entry["name"]
        info = link.lstat()
        if list(_link_version(info)) != entry["version"] or os.readlink(link) != entry["target"]:
            raise ValueError("Interpreter link changed during identity collection")
        if entry["scope"] == "system_alias": _protected_system_alias(link, info, entry["target"])
    if sha(read(workspace.artifact_path("venvs/" + name + "/pyvenv.cfg"), 64 * 1024)) != config_sha:
        raise ValueError("Venv configuration changed during identity collection")
    return {"venv": name, "target": str(current), "link_chain": links,
        "pyvenv_cfg_sha256": config_sha, **binary}


def checked_interpreter(workspace, name, expected):
    actual = interpreter_identity(workspace, name)
    if actual != expected: raise RuntimeError("Frozen interpreter link/binary/configuration/tool version changed")
    return workspace.artifact_path("venvs/" + name + "/bin") / "python"


def text_args(run_id, mode):
    return ["generate", "--run-id", run_id, "--bundle", "s1/attention-bundle-r02",
        "--attention-reference-result", "s1/attention/reference-s2-loop-r01.json",
        "--attention-receipt-result", "s1/attention/native-validation-s2-loop-r01.json",
        "--max-new-tokens", "64", "--temperature", "0", "--seed", "2026",
        "--max-model-calls", "1", "--reuse-profile", mode]


def phases():
    result = [{"name": "attention-reference-loop-r01", "interpreter": "reference", "module": None, "args": [],
        "result": "s1/attention/reference-s2-loop-r01.json", "status": "pass", "expected_exit": 0},
        {"name": "attention-native-loop-r01", "interpreter": "native", "module": "vesper.spark_attention_validation",
        "args": ["native", "--bundle", "s1/attention-bundle-r02", "--reference-result",
            "s1/attention/reference-s2-loop-r01.json", "--run-id", "s2-loop-r01"],
        "result": "s1/attention/native-validation-s2-loop-r01.json", "status": "pass", "expected_exit": 0}]
    for mode in ("native", "reference", "compare"):
        args = [mode, "--run-id", "s2-head-" + mode + "-loop-r01"]
        if mode == "compare":
            args += ["--native-result", "s1/head/s2-head-native-loop-r01.json",
                     "--reference-result", "s1/head/s2-head-reference-loop-r01.json"]
        result.append({"name": "head-" + mode + "-loop-r01", "interpreter": "reference" if mode == "reference" else "native",
            "module": "vesper.spark_head_probe", "args": args, "result": "s1/head/s2-head-" + mode + "-loop-r01.json",
            "status": "pass" if mode == "compare" else mode + "_execution_pass", "expected_exit": 0})
    for mode in ("s2", "baseline"):
        run_id = "s2-call-" + mode + "-loop-r01"
        result.append({"name": run_id, "interpreter": "native", "module": "vesper.spark_text_probe",
            "args": text_args(run_id, mode), "result": "s1/text/" + run_id + ".json",
            "status": "partial_model_call_budget", "expected_exit": 2, "preparation_mode": mode})
    return result


class ChildCleanup:
    """One absolute deadline per child, also capped by the overall deadline."""
    def __init__(self, overall_deadline):
        self.child = None; self.overall_deadline = overall_deadline
        self.started = self.deadline = None; self.sent = set()
    def stop(self):
        if self.child is None or self.child.poll() is not None: return
        if self.deadline is None:
            self.started = time.monotonic()
            self.deadline = min(self.started + 40, self.overall_deadline)
        phase_end = self.started
        for sig, grace in CLEANUP_STEPS:
            phase_end += grace
            if self.child.poll() is not None: return
            remaining = min(phase_end, self.deadline) - time.monotonic()
            if remaining <= 0: continue
            if sig not in self.sent:
                try: os.killpg(self.child.pid, sig)
                except ProcessLookupError: pass
                self.sent.add(sig)
            try: self.child.wait(timeout=remaining)
            except subprocess.TimeoutExpired: pass
        if self.child.poll() is None:
            sig = CLEANUP_STEPS[-1][0]
            if sig not in self.sent:
                try: os.killpg(self.child.pid, sig)
                except ProcessLookupError: pass
                self.sent.add(sig)
            raise RuntimeError("Own phase child exceeded its one cleanup deadline")


def inherited_lock(workspace, descriptor):
    import fcntl
    if type(descriptor) is not int or descriptor < 3: raise ValueError("Expected inherited GPU lock FD")
    path = workspace.artifact_path("s1/gpu-execution.lock")
    opened, named = os.fstat(descriptor), path.stat()
    if (path.is_symlink() or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDWR):
        raise RuntimeError("Inherited GPU lock identity/configuration differs")
    with path.open("a+b") as independent:
        try: fcntl.flock(independent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(independent.fileno(), fcntl.LOCK_UN)
            raise RuntimeError("GPU lock was not already held")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def clean_partial(value, mode, source_head):
    if (value["workspace_context"]["git_head"] != source_head
            or value["workspace_context"]["dirty_diff_sha256"] != CLEAN_DIGEST
            or value["status"] != "partial_model_call_budget" or value["preparation_reuse_profile"] != mode
            or len(value["steps"]) != 1 or value["completion"]["calls"] != 1
            or value["completion"]["prompt_ids"] != PROMPT_IDS or value["completion"]["completion_ids"]
            or value["completion"]["selected_ids_including_eos"]):
        raise ValueError("Expected one current-source fixed-input partial model call")
    after = value["memory_after_runtime_close"]
    minimum = after["minimum_observed_available_bytes"]
    if (not value["owned_resources_closed"] or not value["memory_runtime_closed"]
            or after["charged_host_bytes"] or after["pinned_bytes"] or after["live_leases"]
            or after["live_reclaimers"] or after["inflight_operations"] or any(after["device_used_bytes"].values())
            or not after["monitor_thread_stopped"] or after["monitor_error"] is not None
            or type(minimum) is not int or minimum < after["available_floor_bytes"] or minimum < 32 * 1024**3):
        raise RuntimeError("Partial call cleanup, monitoring or physical floor failed")
    step = value["steps"][0]
    if (set(step["layers"]) != {str(layer) for layer in range(40)} or set(step["engrams"]) != {"1", "14"}
            or step["logits_fp32"]["shape"] != [1, 129280]):
        raise ValueError("Partial does not cover all40 layers, both Engrams and the full head")
    return {key: step[key] for key in ("layers", "engrams", "logits_fp32", "output_bf16")}


def attention_reference(workspace):
    from vesper.spark_attention_reference import SparkAttentionMemoryReference
    from vesper.checkpoint import REFERENCE_HASHES
    from unittest.mock import patch
    initial = workspace.context("s2-attention-reference-loop-r01")
    target = workspace.artifact_path("s2/attention-reference-loop-r01")
    target.mkdir(exist_ok=False)
    for name in ("kernel.py", "LICENSE"):
        source = workspace.artifact_path("s1/attention-reference-r03") / name
        raw = read(source, 2 * 1024**2)
        if sha(raw) != REFERENCE_HASHES[name]: raise ValueError("Pinned local reference differs")
        with (target / name).open("xb") as stream: stream.write(raw)
    with patch("vesper.checkpoint.read_small", side_effect=RuntimeError("Only local pinned references")):
        with SparkAttentionMemoryReference(target, workspace=workspace) as reference: value = reference.qualify()
    workspace.write_result("s1/attention/reference-s2-loop-r01.json", value,
        run_id="s2-attention-reference-loop-r01", initial_context=initial)
    return 0 if value["status"] == "pass" else 2


class Worker:
    def __init__(self, workspace, here, plan_raw, worker_raw):
        self.ws, self.here, self.plan_raw = workspace, here, plan_raw
        self.plan = json.loads(plan_raw); remaining = validate_plan(self.plan, worker_raw, time.time())
        self.plan_sha, self.worker_sha = sha(plan_raw), sha(worker_raw)
        self.overall_deadline = time.monotonic() + remaining
        self.initial = workspace.context(JOB)
        self.state = {"schema": "vesper.private-native-short-loop-worker.v1", "status": "running",
            "stage": "preflight", "source_head": self.plan["source_head"], "worker_sha256": self.worker_sha,
            "plan_sha256": self.plan_sha, "started_unix": time.time(), "phases": {}, "results": {},
            "mode_order": ["s2", "baseline"], "maximum_model_calls": 2, "automatic_retry": False,
            "iteration_purpose": "Fresh current-source native validation after bounded source improvements; prior calibration evidence is preserved at its original source",
            "eos_qualified": False, "practical_quality_qualified": False, "overall_timeout_seconds": 900}
        self.active_cleanup = None
    def save(self):
        temporary = self.here / "state.tmp"
        exclusive(temporary, self.state, 128 * 1024)
        os.replace(temporary, self.here / "state.json")
    def fail(self, error, stage):
        if "failure_type" not in self.state:
            self.state.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                failure_type=type(error).__name__, failure_stage=stage)
            try:
                raw = exclusive(self.here / "failure.json", {"failure_type": type(error).__name__, "stage": stage,
                    "traceback": "".join(traceback.format_exception(error))[-12000:]}, 64 * 1024)
                self.state["private_failure"] = {"saved": True, "sha256": sha(raw)}
            except (Exception, KeyboardInterrupt): self.state["private_failure"] = {"saved": False}
        else:
            rows = self.state.setdefault("secondary_failures", [])
            if len(rows) < 8: rows.append({"failure_type": type(error).__name__, "stage": stage})
    def guard(self, *, check_deadline=True):
        from vesper.checkpoint import MODEL, REVISION
        now = self.ws.context(JOB)
        if (tuple(now[key] for key in ("platform", "profile", "model", "revision")) != ("dgx-spark", "native", MODEL, REVISION)
                or any(now[key] != self.initial[key] for key in IDENTITY_KEYS)
                or now["git_head"] != self.plan["source_head"] or now["dirty_diff_sha256"] != CLEAN_DIGEST):
            raise RuntimeError("Declared clean runtime source identity changed")
        if sha(read(self.here / "worker.py", 40 * 1024)) != self.worker_sha or sha(read(self.here / "plan.json")) != self.plan_sha:
            raise RuntimeError("Frozen worker or plan changed")
        if check_deadline and time.monotonic() >= self.overall_deadline: raise TimeoutError("Overall finite deadline expired")
    def result(self, phase):
        self.guard()
        identity = self.ws.check_result(phase["result"])
        raw = read(self.ws.result_path(phase["result"]))
        if sha(raw) != identity["result_sha256"]: raise ValueError("Result hash differs")
        value = json.loads(raw)
        if value["status"] != phase["status"]: raise ValueError("Required fresh gate or partial did not pass")
        self.state["results"][phase["result"]] = {"sha256": identity["result_sha256"], "status": value["status"]}
        self.save()
        return value
    def execute(self, phase, lock_fd):
        started = time.monotonic(); name = phase["name"]
        self.state["stage"] = name
        row = self.state["phases"][name] = {"status": "preparing", "launched": False, "launch_state": "not_launched",
            "owned_child_exited": True, "timeout_seconds": None,
            "expected_exit": phase["expected_exit"], "module": phase["module"], "args": phase["args"]}
        cleanup = self.active_cleanup = ChildCleanup(self.overall_deadline)
        env = {**os.environ, "PYTHONPATH": str(self.ws.root / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        env.pop("PYTHONHOME", None)
        try:
            self.save()
            self.guard(); remaining = self.overall_deadline - time.monotonic() - 40
            if remaining <= 0: raise TimeoutError("No overall cleanup allowance remains")
            timeout = min(CHILD_SECONDS, remaining); row["timeout_seconds"] = timeout
            expected = self.plan["interpreters"][phase["interpreter"]]
            interpreter = checked_interpreter(self.ws, phase["interpreter"], expected)
            row["interpreter_identity_sha256"] = sha(json.dumps(expected, sort_keys=True).encode())
            with (self.here / (name + ".log")).open("xb") as log:
                cleanup.child = subprocess.Popen([str(interpreter), "-B", str(self.here / "worker.py"),
                    "--workspace", str(self.ws.root), "--phase", name, "--plan-sha", self.plan_sha,
                    "--lock-fd", str(lock_fd), "--child-timeout", str(timeout)], cwd=self.ws.root, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,),
                    start_new_session=True, close_fds=True)
                row.update(status="running", launched=True, launch_state="launched")
                self.state["active_child_pid"] = cleanup.child.pid; self.save()
                code = cleanup.child.wait(timeout=timeout)
                row["exit_code"] = code
                if code != phase["expected_exit"]: raise RuntimeError("Unexpected phase exit: " + name)
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error, name)
            row["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            try: cleanup.stop()
            except (Exception, KeyboardInterrupt) as error:
                self.fail(error, name + "/cleanup")
                raise
            finally:
                row["wall_seconds"] = time.monotonic() - started
                row["owned_child_exited"] = cleanup.child is None or cleanup.child.poll() is not None
                self.state.pop("active_child_pid", None); self.save()
        self.active_cleanup = None
        row["status"] = "finished"; self.save(); self.guard()
    def run(self):
        import fcntl
        self.guard()
        if self.here != self.ws.artifact_path(JOB_NAMESPACE): raise RuntimeError("Wrong own job namespace")
        fixed = {"schema": "vesper.native-short-loop-r01-plan.v1", "workspace_context": self.initial,
            "source_head": self.plan["source_head"], "external_worker_sha256": self.worker_sha, "input_plan_sha256": self.plan_sha,
            "phases": phases(), "overall_timeout_seconds": 900, "per_child_maximum_seconds": 300,
            "single_absolute_cleanup_seconds_per_child": 40, "gpu_lock_inherited_until_child_exit": True,
            "interpreter_identity_policy": "Explicit bounded final-link chain to pinned root-owned /usr/bin/python3.12, binary/configuration/tool-version comparison before spawn and in child",
            "current_source_revalidation": True, "comparison_scope": "Semantic equality of the two new native partial modes only; historical cross-source comparison is a separate derived audit",
            "physical_available_floor_bytes": 32 * 1024**3, "model_calls": 2, "mode_order": ["s2", "baseline"],
            "prompt_ids": PROMPT_IDS, "temperature": 0, "seed": 2026, "max_new_tokens_setting": 64,
            "max_model_calls_per_process": 1, "downloads": 0, "automatic_retry": False,
            "timing_limitations": "Fresh processes; fixed S2 then baseline order; OS/driver state uncontrolled; no throughput claim",
            "eos_run": False, "practical_quality_run": False, "existing_service_changes": 0}
        exclusive(self.here / "execution-plan.json", fixed)
        with ExitStack() as owned:
            lock = owned.enter_context(self.ws.artifact_path("s1/gpu-execution.lock").open("a+b"))
            owned.callback(lambda: self.active_cleanup.stop() if self.active_cleanup is not None else None)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reports = []
            for phase in phases():
                self.execute(phase, lock.fileno())
                value = self.result(phase)
                if "preparation_mode" in phase:
                    semantic = clean_partial(value, phase["preparation_mode"], self.plan["source_head"])
                    reports.append((semantic, {"mode": phase["preparation_mode"], "result": phase["result"],
                        "total_seconds": value["wall_seconds_total"],
                        "model_call_seconds": value["steps"][0]["model_call_wall_seconds_including_native_diagnostics"],
                        "minimum_available_bytes": value["memory_after_runtime_close"]["minimum_observed_available_bytes"],
                        "peak_charged_host_bytes": value["memory_after_runtime_close"]["charged_host_peak_bytes"]}))
            if len(reports) != 2 or reports[0][0] != reports[1][0]: raise ValueError("Two native partial semantic records differ")
            self.state.update(one_call_semantic_arrays_exact=True, all40_layers_both_engrams_full_head_compared=True,
                single_call_timings=[item[1] for item in reports])
        self.guard(); self.state.update(status="finished", stage="finished")
    def finish(self):
        try:
            if self.active_cleanup is not None: self.active_cleanup.stop()
        except (Exception, KeyboardInterrupt) as error: self.fail(error, "final_cleanup")
        self.state["finished_unix"] = time.time(); self.save()
        self.guard(check_deadline=False)
        result = {**self.state, "full_model_eos_qualified": False, "full_model_quality_qualified": False,
            "general_performance_claim": False, "old_results_restamped": False, "new_model_calls_maximum": 2,
            "owned_children_exited": all(row.get("owned_child_exited") is True for row in self.state["phases"].values())}
        return self.ws.write_result("s2/calibration/" + JOB + ".json", result,
            run_id=JOB, initial_context=self.initial)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "vesper-spark")
    parser.add_argument("--phase", help=argparse.SUPPRESS)
    parser.add_argument("--plan-sha", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-timeout", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.workspace.resolve(strict=True); here = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from vesper.workspace import Workspace
    workspace = Workspace.open(root)
    worker_raw, plan_raw = read(here / "worker.py", 40 * 1024), read(here / "plan.json")
    if Path(__file__).resolve() != here / "worker.py": raise RuntimeError("Expected the frozen private worker copy")
    worker = Worker(workspace, here, plan_raw, worker_raw)
    if args.phase is not None:
        if sha(plan_raw) != args.plan_sha or not args.child_timeout or not 0 < args.child_timeout <= 300:
            raise ValueError("Child frozen plan or time budget differs")
        worker.guard(); inherited_lock(workspace, args.lock_fd)
        candidates = [phase for phase in phases() if phase["name"] == args.phase]
        if len(candidates) != 1: raise ValueError("Unknown fixed phase")
        phase = candidates[0]
        checked_interpreter(workspace, phase["interpreter"], worker.plan["interpreters"][phase["interpreter"]])
        # Child retains the inherited lock through all unwinding and process
        # exit, and has its own alarm if the parent is externally terminated.
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, min(args.child_timeout, worker.overall_deadline - time.monotonic()))
        try:
            if phase["module"] is None: return attention_reference(workspace)
            sys.argv = [phase["module"], *phase["args"]]
            runpy.run_module(phase["module"], run_name="__main__")
            return 0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    if os.name == "nt" or args.lock_fd is not None: raise ValueError("Parent requires Linux GPU-lock ownership")
    with (here / "claim.json").open("x") as stream: json.dump({"worker_sha256": worker.worker_sha, "plan_sha256": worker.plan_sha}, stream)
    worker.save()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, worker.overall_deadline - time.monotonic()))
    try:
        try: worker.run()
        except (Exception, KeyboardInterrupt) as error: worker.fail(error, worker.state["stage"])
        finally: path = worker.finish()
    finally: signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": worker.state["status"], "source_head": worker.plan["source_head"],
        "result": path.relative_to(root).as_posix(), "model_calls_maximum": 2}))
    return 0 if worker.state["status"] == "finished" else 2


def interrupted(*_): raise KeyboardInterrupt("Own finite native calibration cancelled")


def expired(*_): raise TimeoutError("Own overall 900-second native calibration deadline expired")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
