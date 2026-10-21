"""Real temporary Git checkouts and mocked GitHub; no live GPU or API writes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vesper.workspace import BINDING, CONTEXT, Workspace, WorkspaceError, main


REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def git(root, *args):
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=", "-C", str(root), *args],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout.decode().strip()


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "spark"
        self.root.mkdir()
        # Windows TEMP may use an 8.3 alias; compare the same real path as the guard.
        self.root = self.root.resolve(strict=True)
        git(self.root, "init", "--initial-branch=feature/dgx-spark-isolation")
        git(self.root, "config", "user.email", "test@example.invalid")
        git(self.root, "config", "user.name", "Test")
        git(self.root, "remote", "add", "origin", "https://github.com/shell-term/vesper.git")
        (self.root / ".gitignore").write_text(f"{BINDING}\nartifacts/\nresults/dgx-spark/\n")
        (self.root / "source.py").write_text("VALUE = 1\n")
        profile = {"native": {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "revision": REVISION}}
        self.registry = {"schema_version": 1, "repository": "shell-term/vesper", "platforms": {
            "dgx-spark": {"label": "platform:dgx-spark", "issues": [13, 14], "milestones": [5, 6, 7, 8],
                          "artifact_root": "artifacts/dgx-spark", "result_root": "results/dgx-spark",
                          "branch_prefix": "feature/dgx-spark-", "profiles": profile,
                          "shared_read_dependencies": [{"issue": 10, "platform": "pc-heterogeneous",
                                                       "label": "platform:pc-heterogeneous", "milestone": 4}]},
            "pc-heterogeneous": {"label": "platform:pc-heterogeneous", "issues": [1, 10, 12],
                                 "milestones": [1, 2, 3, 4, None], "artifact_root": "artifacts/pc-heterogeneous",
                                 "result_root": "results/pc-heterogeneous", "branches": ["main"], "profiles": profile}}}
        (self.root / "platforms").mkdir()
        self.registry_file = self.root / "platforms/registry.json"
        self.registry_file.write_text(json.dumps(self.registry))
        git(self.root, "add", ".")
        git(self.root, "commit", "-m", "fixture")

    def bind(self):
        return Workspace.bind(self.root, platform="dgx-spark")

    def artifact_scope_fixture(self):
        directory = self.root / "src/vesper"
        directory.mkdir(parents=True)
        modules = []
        for name in ("spark_artifacts.py", "workspace.py"):
            path = directory / name
            path.write_text("VALUE = 1\n")
            modules.append((path, "src/vesper/" + name))
        paths = tuple(self.root / "artifacts/dgx-spark" / name for name in ("metadata", "originals", "integrity"))
        identity = ("dgx-spark", "native", "deepseek-ai/DeepSeek-V4.1-Flash-1017", REVISION)
        return modules, paths, identity

    def test_artifact_batch_runs_one_fresh_check_without_unused_git_context(self):
        from vesper import spark_artifacts, workspace as workspace_module
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        with patch.object(workspace, "check", wraps=workspace.check) as check, \
                patch.object(workspace, "context", side_effect=AssertionError("No unused source context")), \
                patch.object(workspace_module, "_git", wraps=workspace_module._git) as git_calls:
            for _ in range(2):
                workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
            self.assertEqual(check.call_count, 2)
            self.assertEqual(git_calls.call_count, 10)
        with patch.object(spark_artifacts, "__file__", str(modules[0][0])), \
                patch.object(workspace_module, "__file__", str(modules[1][0])), \
                patch.object(workspace, "check", wraps=workspace.check) as check, \
                patch.object(workspace, "context", side_effect=AssertionError("No unused source context")):
            self.assertIs(spark_artifacts._scope(workspace, *paths), workspace)
            check.assert_called_once_with()

    def test_artifact_batch_refuses_binding_branch_origin_registry_changes_and_allows_recovery(self):
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        def validate():
            workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
        validate()
        binding = self.root / BINDING
        original_binding = binding.read_bytes()
        changed = json.loads(original_binding); changed["binding_id"] = "0" * 32
        binding.write_text(json.dumps(changed))
        with self.assertRaises(WorkspaceError): validate()
        binding.write_bytes(original_binding); validate()
        git(self.root, "switch", "-c", "feature/dgx-spark-batch-other")
        with self.assertRaises(WorkspaceError): validate()
        git(self.root, "switch", "feature/dgx-spark-isolation"); validate()
        git(self.root, "remote", "set-url", "origin", "https://github.com/someone/other.git")
        with self.assertRaises(WorkspaceError): validate()
        git(self.root, "remote", "set-url", "origin", "https://github.com/shell-term/vesper.git"); validate()
        original_registry = self.registry_file.read_bytes()
        changed = deepcopy(self.registry); changed["platforms"]["dgx-spark"]["profiles"]["native"]["revision"] = "f" * 40
        self.registry_file.write_text(json.dumps(changed))
        with self.assertRaises(WorkspaceError): validate()
        self.registry_file.write_bytes(original_registry); validate()

    def test_artifact_batch_pins_identity_and_retains_module_namespace_path_rules(self):
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        def validate(test_paths=paths, test_modules=modules, expected=identity):
            return workspace.validate_artifact_scope(test_paths, runtime_modules=test_modules, expected_identity=expected)
        for index, value in enumerate(("pc-heterogeneous", "other", "other/model", "f" * 40)):
            wrong = list(identity); wrong[index] = value
            with self.subTest(identity=index), self.assertRaises(WorkspaceError): validate(expected=tuple(wrong))
        outside = Path(self.temp.name) / "foreign.py"; outside.write_text("VALUE = 1\n")
        for entry in ((outside, modules[0][1]), (modules[0][0], "../outside.py"),
                (modules[0][0], "src/vesper/other.py")):
            with self.subTest(module=entry), self.assertRaises(WorkspaceError): validate(test_modules=[entry])
        for path in (self.root / "artifacts/pc-heterogeneous/input", self.root / "outside",
                paths[0] / "../escape", self.root / "artifacts/dgx-spark/a:stream"):
            with self.subTest(path=path), self.assertRaises(WorkspaceError): validate(test_paths=[path])
        validate()

    def test_artifact_batch_rejects_symlink_reparse_paths_and_modules_without_link_privilege(self):
        from types import SimpleNamespace
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        original = Path.lstat
        for forbidden in (paths[0], modules[0][0]):
            for mode, flags in ((0o120777, 0), (0o40755, 0x400)):
                def attributes(path, *args, **kwargs):
                    if path == forbidden:
                        return SimpleNamespace(st_mode=mode, st_file_attributes=flags)
                    return original(path, *args, **kwargs)
                with self.subTest(forbidden=forbidden, mode=mode), patch.object(Path, "lstat", attributes):
                    with self.assertRaises(WorkspaceError):
                        workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
        workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)

    def test_artifact_batch_does_not_replace_entry_or_result_source_context_checks(self):
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        initial = workspace.context("batch-source-run")
        workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
        (self.root / "source.py").write_text("VALUE = 9\n")
        # The batch checks namespace/model identity, not HEAD/dirty state. The
        # unchanged publication gate must still reject the changed source.
        workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
        with self.assertRaises(WorkspaceError):
            workspace.write_result("batch-source.json", {"status": "pass"}, initial_context=initial)
        self.assertFalse((self.root / "results/dgx-spark/batch-source.json").exists())

    def test_artifact_batch_cancellation_is_not_cached_and_retry_checks_again(self):
        from vesper.memory import RuntimeCancelled
        workspace = self.bind()
        modules, paths, identity = self.artifact_scope_fixture()
        for error in (KeyboardInterrupt("stop"), RuntimeCancelled("stop")):
            with self.subTest(error=type(error).__name__), patch.object(workspace, "check", side_effect=error):
                with self.assertRaises(type(error)):
                    workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
        with patch.object(workspace, "check", wraps=workspace.check) as check:
            workspace.validate_artifact_scope(paths, runtime_modules=modules, expected_identity=identity)
            check.assert_called_once_with()

    def issue(self, number=13, label="platform:dgx-spark", milestone=5, body="current"):
        return {"number": number, "labels": [{"name": label}], "milestone": None if milestone is None else {"number": milestone},
                "pull_request": None, "body": body, "title": "Scoped progress", "state": "open",
                "html_url": f"https://github.com/shell-term/vesper/issues/{number}"}

    def test_binding_is_required_and_cannot_be_overwritten(self):
        with self.assertRaises(WorkspaceError): Workspace.open(self.root)
        workspace = self.bind()
        self.assertEqual(workspace.check()["platform"], "dgx-spark")
        with self.assertRaises(WorkspaceError): self.bind()

    def test_binding_must_be_ignored_and_registry_tracked(self):
        (self.root / ".gitignore").write_text("artifacts/\n")
        with self.assertRaises(WorkspaceError): self.bind()
        (self.root / ".gitignore").write_text(f"{BINDING}\n")
        git(self.root, "rm", "--cached", "platforms/registry.json")
        with self.assertRaises(WorkspaceError): self.bind()

    def test_wrong_origin_refused_before_binding(self):
        git(self.root, "remote", "set-url", "origin", "https://github.com/someone/other.git")
        with self.assertRaises(WorkspaceError): self.bind()

    def test_branch_and_origin_changes_refused(self):
        workspace = self.bind()
        git(self.root, "switch", "-c", "feature/dgx-spark-other")
        with self.assertRaises(WorkspaceError): workspace.check()
        git(self.root, "switch", "feature/dgx-spark-isolation")
        git(self.root, "remote", "set-url", "origin", "git@github.com:shell-term/vesper.git")
        with self.assertRaises(WorkspaceError): workspace.check()

    def test_cross_platform_bind_is_refused_even_in_same_repository(self):
        with self.assertRaises(WorkspaceError): Workspace.bind(self.root, platform="pc-heterogeneous")
        git(self.root, "switch", "-c", "main")
        with self.assertRaises(WorkspaceError): self.bind()
        pc = Workspace.bind(self.root, platform="pc-heterogeneous")
        self.assertEqual(pc.platform, "pc-heterogeneous")
        with patch.object(pc, "_gh", side_effect=[self.issue(12, "platform:pc-heterogeneous", None)] * 2):
            self.assertEqual(pc.read_progress(12)["number"], 12)

    def test_copied_checkout_binding_is_refused(self):
        self.bind()
        other = Path(self.temp.name) / "copied"
        shutil.copytree(self.root, other)
        with self.assertRaises(WorkspaceError): Workspace.open(other)

    def test_recreated_checkout_at_same_path_is_refused(self):
        self.bind()
        old = self.root.with_name("old")
        self.root.rename(old)
        shutil.copytree(old, self.root)
        with self.assertRaises(WorkspaceError): Workspace.open(self.root)

    def test_subdirectory_and_git_environment_override_refused(self):
        self.bind()
        with self.assertRaises(WorkspaceError): Workspace.open(self.root / "platforms")
        with patch.dict(os.environ, {"GIT_DIR": str(self.root / ".git")}):
            with self.assertRaises(WorkspaceError): Workspace.open(self.root)

    def test_registry_changes_refuse_bound_instance(self):
        workspace = self.bind()
        self.registry["platforms"]["dgx-spark"]["issues"].append(15)
        self.registry_file.write_text(json.dumps(self.registry))
        with self.assertRaises(WorkspaceError): workspace.check()

    def test_registry_refuses_ambiguous_issue_milestone_or_label_ownership(self):
        original = deepcopy(self.registry)
        for field, value in (("issues", [10, 13]), ("milestones", [4, 5]), ("label", "platform:pc-heterogeneous")):
            registry = deepcopy(original)
            registry["platforms"]["dgx-spark"][field] = value
            self.registry_file.write_text(json.dumps(registry))
            with self.subTest(field=field):
                with self.assertRaises(WorkspaceError): self.bind()

    def test_registry_allows_explicit_unassigned_milestone_for_both_platforms(self):
        self.registry["platforms"]["dgx-spark"]["milestones"].append(None)
        self.registry_file.write_text(json.dumps(self.registry))
        workspace = self.bind()
        with patch.object(workspace, "_gh", side_effect=[self.issue(milestone=None)] * 2):
            self.assertEqual(workspace.read_progress(13)["number"], 13)

    def test_profile_model_revision_is_bound(self):
        workspace = self.bind()
        result = workspace.stamp_result({"status": "pass"}, "first")
        self.assertEqual(result[CONTEXT]["revision"], REVISION)
        self.assertEqual(result[CONTEXT]["profile"], "native")
        for key in ("root", "gitdir", "origin", "root_identity", "gitdir_identity"):
            self.assertNotIn(key, result[CONTEXT])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_own_evidence_only_and_exclusive_write(self):
        workspace = self.bind()
        for path in ("../pc/file.json", "/tmp/out.json", "C:/out.json", "a\\..\\out.json", "a:stream", ".//out.json",
                     "results/pc-heterogeneous/result.json", "artifacts/pc-heterogeneous/result.json"):
            with self.subTest(path=path):
                with self.assertRaises(WorkspaceError): workspace.result_path(path)
        output = workspace.write_result("s0/result.json", {"status": "pass"}, "run-01")
        self.assertEqual(output, self.root / "results/dgx-spark/s0/result.json")
        self.assertEqual(workspace.check_result("results/dgx-spark/s0/result.json")["run_id"], "run-01")
        with self.assertRaises(WorkspaceError): workspace.write_result("s0/result.json", {"status": "pass"})

    def test_result_rejects_foreign_platform_model_and_binding(self):
        workspace = self.bind()
        path = workspace.write_result("result.json", {"status": "pass"}, "result-run")
        original = json.loads(path.read_text())
        for key, value in (("platform", "pc-heterogeneous"), ("profile", "other"), ("revision", "f" * 40),
                           ("binding_id", "0" * 32), ("git_head", "f" * 40)):
            changed = deepcopy(original); changed[CONTEXT][key] = value
            path.write_text(json.dumps(changed))
            with self.assertRaises(WorkspaceError): workspace.check_result("result.json")
        path.write_text(json.dumps(original))
        self.assertEqual(workspace.check_result("result.json")["status"], "pass")

    def test_dirty_source_changes_invalidate_results(self):
        workspace = self.bind()
        workspace.write_result("result.json", {"status": "pass"}, "dirty-run")
        (self.root / "source.py").write_text("VALUE = 2\n")
        with self.assertRaises(WorkspaceError): workspace.check_result("result.json")
        (self.root / "source.py").write_text("VALUE = 1\n")
        (self.root / "new_source.py").write_text("NEW = 1\n")
        with self.assertRaises(WorkspaceError): workspace.check_result("result.json")

    def test_dirty_run_is_valid_until_its_actual_source_changes(self):
        workspace = self.bind()
        (self.root / "source.py").write_text("VALUE = 2\n")
        workspace.write_result("result.json", {"status": "pass"}, "dirty-run")
        self.assertEqual(workspace.check_result("result.json")["status"], "pass")
        git(self.root, "add", "source.py")
        with self.assertRaises(WorkspaceError): workspace.check_result("result.json")

    def test_run_beginning_and_end_must_match_when_context_supplied(self):
        workspace = self.bind()
        initial = workspace.context("run-bound")
        workspace.write_result("stable.json", {"status": "pass"}, initial_context=initial)
        with self.assertRaises(WorkspaceError): workspace.write_result("wrong-run.json", {}, "different", initial_context=initial)
        (self.root / "source.py").write_text("VALUE = 4\n")
        with self.assertRaises(WorkspaceError): workspace.write_result("changed.json", {}, initial_context=initial)
        self.assertFalse((self.root / "results/dgx-spark/changed.json").exists())

    def test_reparse_point_detection_without_symlink_privilege(self):
        from types import SimpleNamespace
        from vesper.workspace import _no_links
        target = self.root / "artifacts/link/result.json"
        original = Path.lstat
        def attributes(path, *args, **kwargs):
            if path == self.root / "artifacts/link":
                return SimpleNamespace(st_mode=0o40755, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", attributes):
            with self.assertRaises(WorkspaceError): _no_links(target)

    def test_symlink_result_parent_and_binding_are_refused(self):
        workspace = self.bind()
        outside = Path(self.temp.name) / "outside"; outside.mkdir()
        base = workspace.result_path("s0/x.json", create_parent=True).parent
        link = base / "linked"
        try: link.symlink_to(outside, target_is_directory=True)
        except OSError: self.skipTest("Symlink creation is not permitted")
        with self.assertRaises(WorkspaceError): workspace.write_result("s0/linked/x.json", {})
        raw = (self.root / BINDING).read_bytes()
        (self.root / BINDING).unlink()
        marker = outside / "binding.json"; marker.write_bytes(raw)
        (self.root / BINDING).symlink_to(marker)
        with self.assertRaises(WorkspaceError): Workspace.open(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows junction check")
    def test_windows_junction_escape_is_refused(self):
        workspace = self.bind()
        outside = Path(self.temp.name) / "outside"; outside.mkdir()
        base = workspace.result_path("s0/x.json", create_parent=True).parent
        junction = base / "junction"
        # Setup only: no deletion commands cross shells. TemporaryDirectory owns both paths.
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode: self.skipTest("Cannot create a junction")
        self.addCleanup(lambda: junction.rmdir() if junction.exists() else None)
        with self.assertRaises(WorkspaceError): workspace.write_result("s0/junction/x.json", {})

    def test_progress_denies_wrong_issue_before_network(self):
        workspace = self.bind()
        with patch.object(workspace, "_gh") as api:
            with self.assertRaises(WorkspaceError): workspace.read_progress(10)
            api.assert_not_called()

    def test_progress_checks_metadata_before_reading_body(self):
        workspace = self.bind()
        for wrong in (self.issue(label="platform:pc-heterogeneous"), self.issue(milestone=4), self.issue(milestone=None)):
            with patch.object(workspace, "_gh", return_value=wrong) as api:
                with self.assertRaises(WorkspaceError): workspace.read_progress(13)
                self.assertEqual(api.call_count, 1)
                self.assertTrue(api.call_args.kwargs["metadata"])

    def test_progress_rechecks_second_response_metadata(self):
        workspace = self.bind()
        with patch.object(workspace, "_gh", side_effect=[self.issue(), self.issue(label="other", body="secret")]):
            with self.assertRaises(WorkspaceError): workspace.read_progress(13)

    def test_progress_refuses_ambiguous_live_platform_labels_before_body(self):
        workspace = self.bind()
        ambiguous = self.issue()
        ambiguous["labels"].append({"name": "platform:pc-heterogeneous"})
        with patch.object(workspace, "_gh", return_value=ambiguous) as api:
            with self.assertRaises(WorkspaceError): workspace.read_progress(13)
            self.assertEqual(api.call_count, 1)
            self.assertTrue(api.call_args.kwargs["metadata"])
        with patch.object(workspace, "_gh", side_effect=[self.issue(), ambiguous]):
            with self.assertRaises(WorkspaceError): workspace.read_progress(13)

    def test_github_host_is_pinned_despite_environment_default(self):
        workspace = self.bind()
        with patch.dict(os.environ, {"GH_HOST": "another-host.invalid"}), patch("vesper.workspace._process", return_value=b"{}") as process:
            workspace._gh("repos/shell-term/vesper/issues/13", metadata=True)
            self.assertEqual(process.call_args.args[0][:5], ["gh", "api", "--hostname", "github.com", "repos/shell-term/vesper/issues/13"])

    def test_shared_dependency_is_read_only(self):
        workspace = self.bind()
        issue = self.issue(10, "platform:pc-heterogeneous", 4)
        with patch.object(workspace, "_gh", side_effect=[issue, issue]):
            self.assertTrue(workspace.read_progress(10, shared=True)["shared_read_only"])
        with patch.object(workspace, "_gh") as api:
            with self.assertRaises(WorkspaceError): workspace.prepare_update(10, "body.md")
            api.assert_not_called()

    def prepare_packet(self, workspace):
        body = workspace.artifact_path("draft/body.md", create_parent=True)
        body.write_text("Scoped Spark update.\n", newline="\n")
        with patch.object(workspace, "_gh", side_effect=[self.issue(), self.issue()]):
            packet = workspace.prepare_update(13, "draft/body.md", run_id="update-01")
        return packet.relative_to(workspace.root).as_posix()

    def test_prepared_update_rechecks_hash_source_and_remote_body(self):
        workspace = self.bind()
        packet = self.prepare_packet(workspace)
        with patch.object(workspace, "_gh", side_effect=[self.issue(), self.issue()]):
            self.assertEqual(workspace.preflight_update(packet)["issue"], 13)
        with patch.object(workspace, "_gh", side_effect=[self.issue(body="changed"), self.issue(body="changed")]):
            with self.assertRaises(WorkspaceError): workspace.preflight_update(packet)
        (self.root / "source.py").write_text("VALUE = 3\n")
        with patch.object(workspace, "_gh") as api:
            with self.assertRaises(WorkspaceError): workspace.preflight_update(packet)
            api.assert_not_called()

    def test_modified_update_body_and_other_issue_never_write(self):
        workspace = self.bind()
        packet_name = self.prepare_packet(workspace)
        path = workspace.artifact_path(packet_name); packet = json.loads(path.read_text())
        for mutation in ({"body": "replaced"}, {"issue": 10}):
            changed = {**packet, **mutation}; path.write_text(json.dumps(changed))
            with patch.object(workspace, "_gh") as api:
                with self.assertRaises(WorkspaceError): workspace.commit_update(packet_name)
                api.assert_not_called()

    def test_commit_helper_only_sends_prepared_body_to_owned_issue(self):
        workspace = self.bind(); packet = self.prepare_packet(workspace)
        with patch.object(workspace, "_gh", side_effect=[self.issue(), self.issue(), self.issue(body="Scoped Spark update.\n")]) as api:
            result = workspace.commit_update(packet)
            self.assertEqual(result["issue"], 13)
            self.assertEqual(api.call_args.args[0], "repos/shell-term/vesper/issues/13")
            self.assertEqual(api.call_args.kwargs["body"], {"body": "Scoped Spark update.\n"})

    def test_commit_refuses_ambiguous_metadata_before_patch_and_checks_response(self):
        workspace = self.bind(); packet = self.prepare_packet(workspace)
        ambiguous = self.issue()
        ambiguous["labels"].append({"name": "platform:pc-heterogeneous"})
        for replies in ([ambiguous], [self.issue(), ambiguous]):
            with patch.object(workspace, "_gh", side_effect=replies) as api:
                with self.assertRaises(WorkspaceError): workspace.commit_update(packet)
                self.assertFalse(any(call.kwargs.get("body") is not None for call in api.call_args_list))
        with patch.object(workspace, "_gh", side_effect=[self.issue(), self.issue(), ambiguous]) as api:
            with self.assertRaises(WorkspaceError): workspace.commit_update(packet)
            self.assertEqual(sum(call.kwargs.get("body") is not None for call in api.call_args_list), 1)

    def test_invalid_run_identity_and_prestamped_results_refused(self):
        workspace = self.bind()
        with self.assertRaises(WorkspaceError): workspace.stamp_result({}, "../elsewhere")
        with self.assertRaises(WorkspaceError): workspace.stamp_result({CONTEXT: {}})

    def test_cli_returns_machine_readable_refusal_without_body(self):
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        with redirect_stdout(output): code = main(["--root", str(self.root), "check"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "refused")

    def test_runtime_module_must_be_from_the_bound_checkout(self):
        workspace = self.bind()
        module = self.root / "src/vesper/workspace.py"
        module.parent.mkdir(parents=True)
        module.write_text("# fixture module identity\n")
        workspace.check_runtime_module(module, "src/vesper/workspace.py")
        for candidate, relative in ((Path(__file__), "src/vesper/workspace.py"),
                                    (module, "src/vesper/../outside.py"), (module, "other/module.py")):
            with self.assertRaises(WorkspaceError): workspace.check_runtime_module(candidate, relative)

    def test_cli_refuses_another_imported_clone_even_when_root_is_bound(self):
        from contextlib import redirect_stdout
        from io import StringIO
        self.bind()
        module = self.root / "src/vesper/workspace.py"
        module.parent.mkdir(parents=True)
        module.write_text("# fixture module identity\n")
        output = StringIO()
        with redirect_stdout(output): code = main(["--root", str(self.root), "check"])
        self.assertEqual(code, 2)
        self.assertIn("another checkout", json.loads(output.getvalue())["error"])
        output = StringIO()
        with patch("vesper.workspace.__file__", str(module)), redirect_stdout(output):
            code = main(["--root", str(self.root), "check"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["platform"], "dgx-spark")


if __name__ == "__main__":
    unittest.main()
