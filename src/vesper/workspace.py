"""Fail-closed helpers for an explicitly bound platform checkout.

These checks protect these helpers, not arbitrary shell commands or another
process. The ignored binding is machine-local; result contexts contain no local
paths. Reading another platform requires an explicit shared dependency, and
updates can target only the bound platform's registered issues.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import uuid


BINDING = ".vesper-workspace.json"
REGISTRY = "platforms/registry.json"
MAX_JSON = 4 * 1024 * 1024
MAX_BODY = 64 * 1024
CONTEXT = "workspace_context"


class WorkspaceError(ValueError):
    """The requested operation does not match this checkout's binding."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _no_links(path):
    """Reject symlinks and Windows reparse points, including parent junctions."""
    path = Path(os.path.abspath(path))
    for node in (*reversed(path.parents), path):
        try:
            info = node.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise WorkspaceError("Symlinks and reparse points are not workspace paths")
    return path


def _read(path, limit=MAX_JSON):
    path = _no_links(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise WorkspaceError("Missing or excessive workspace file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise WorkspaceError("Workspace file grew beyond its bound")
    return data


def _identity(path):
    info = _no_links(path).stat()
    return {"device": info.st_dev, "inode": info.st_ino}


def _runtime_module(root, module_file, relative):
    parts = str(relative).split("/")
    if (len(parts) < 3 or parts[:2] != ["src", "vesper"] or any(part in ("", ".", "..") for part in parts)
            or "\\" in str(relative) or ":" in str(relative) or not parts[-1].endswith(".py")):
        raise WorkspaceError("Expected a module below this checkout's Vesper source")
    try:
        root = _no_links(root).resolve(strict=True)
        expected = _no_links(root.joinpath(*parts)).resolve(strict=True)
        actual = _no_links(module_file).resolve(strict=True)
    except OSError as error:
        raise WorkspaceError("The expected runtime module is missing") from error
    if not expected.is_file() or actual != expected:
        raise WorkspaceError("The imported runtime module belongs to another checkout or installation")


def _process(argv, *, cwd, data=None):
    try:
        result = subprocess.run(argv, cwd=cwd, input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkspaceError("Workspace command could not complete") from error
    if result.returncode:
        # Do not expose git/gh diagnostics that might contain local paths or tokens.
        raise WorkspaceError("Workspace command refused the operation")
    if len(result.stdout) > 64 * 1024 * 1024:
        raise WorkspaceError("Excessive workspace command output")
    return result.stdout


def _git(root, *args):
    forbidden = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE")
    if any(os.environ.get(key) for key in forbidden) or any(key.startswith("GIT_CONFIG") for key in os.environ):
        raise WorkspaceError("Git environment overrides are not allowed")
    return _process(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *args], cwd=root)


def _origin_repository(origin):
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", origin)
    if not match:
        raise WorkspaceError("Only an explicit GitHub origin is supported")
    return match[1]


def _checkout(root):
    root = _no_links(root).resolve(strict=True)
    _no_links(root / ".git")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve(strict=True)
    if root != top:
        raise WorkspaceError("Use the checkout root, not a parent or subdirectory")
    gitdir = Path(_git(root, "rev-parse", "--absolute-git-dir").decode().strip())
    gitdir = _no_links(gitdir).resolve(strict=True)
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD").decode().strip()
    origin = _git(root, "remote", "get-url", "origin").decode().strip()
    return {"root": str(root), "gitdir": str(gitdir), "root_identity": _identity(root),
            "gitdir_identity": _identity(gitdir), "branch": branch, "origin": origin,
            "repository": _origin_repository(origin)}


def _positive_ids(values, *, allow_none=False):
    return (isinstance(values, list) and all((allow_none and n is None) or (type(n) is int and n > 0) for n in values)
            and len(values) == len(set(values)))


def _branch_allowed(branch, platform):
    if "branches" in platform and branch not in platform["branches"]:
        raise WorkspaceError("Branch does not belong to the selected platform")
    if "branch_prefix" in platform and not branch.startswith(platform["branch_prefix"]):
        raise WorkspaceError("Branch does not belong to the selected platform")


def _registry(root):
    _git(root, "ls-files", "--error-unmatch", "--", REGISTRY)
    raw = _read(root / REGISTRY)
    try:
        registry = json.loads(raw)
        if registry["schema_version"] != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", registry["repository"]):
            raise WorkspaceError("Invalid platform registry identity")
        owned_issues, owned_milestones, owned_labels = set(), set(), set()
        for name, platform in registry["platforms"].items():
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
                raise WorkspaceError("Invalid platform name")
            if platform["artifact_root"] != f"artifacts/{name}" or platform["result_root"] != f"results/{name}":
                raise WorkspaceError("Platform evidence roots must be isolated")
            if not _positive_ids(platform["issues"]) or not _positive_ids(platform["milestones"], allow_none=True):
                raise WorkspaceError("Invalid progress allowlist")
            if "branches" in platform and (not isinstance(platform["branches"], list) or not platform["branches"]
                    or any(not isinstance(branch, str) or not branch for branch in platform["branches"])):
                raise WorkspaceError("Invalid platform branches")
            if "branch_prefix" in platform and (not isinstance(platform["branch_prefix"], str) or not platform["branch_prefix"]):
                raise WorkspaceError("Invalid platform branch prefix")
            if not isinstance(platform["label"], str) or not platform["label"]:
                raise WorkspaceError("Missing platform label")
            milestones = {number for number in platform["milestones"] if number is not None}
            if (owned_issues.intersection(platform["issues"]) or owned_milestones.intersection(milestones)
                    or platform["label"] in owned_labels):
                raise WorkspaceError("Platform issue, milestone and label ownership must be unique")
            owned_issues.update(platform["issues"])
            owned_milestones.update(milestones)
            owned_labels.add(platform["label"])
            if not platform["profiles"]:
                raise WorkspaceError("Missing platform profiles")
            for key, profile in platform["profiles"].items():
                if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) or not isinstance(profile["model"], str)
                        or not profile["model"] or not re.fullmatch(r"[0-9a-f]{40,64}", profile["revision"])):
                    raise WorkspaceError("Invalid pinned model profile")
            for dependency in platform.get("shared_read_dependencies", []):
                if (type(dependency["issue"]) is not int or dependency["issue"] <= 0
                        or type(dependency["milestone"]) is not int or dependency["milestone"] <= 0
                        or dependency["platform"] == name or not dependency["label"]):
                    raise WorkspaceError("Invalid shared read dependency")
        return registry, _sha(raw)
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise WorkspaceError("Malformed platform registry") from error


def _exclusive_json(path, payload):
    path = _no_links(path)
    data = _json_bytes(payload)
    if len(data) > MAX_JSON:
        raise WorkspaceError("Excessive result or update packet")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise WorkspaceError("Evidence destination must be new and regular") from error
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class Workspace:
    def __init__(self, root, binding, registry, registry_hash):
        self.root, self.binding, self.registry = root, binding, registry
        self.registry_hash = registry_hash
        self.platform = binding["platform"]
        self.profile = binding["profile"]
        self.config = registry["platforms"][self.platform]

    @classmethod
    def bind(cls, root=None, *, platform, profile="native"):
        root = _no_links(Path.cwd() if root is None else root).resolve(strict=True)
        checkout = _checkout(root)
        registry, registry_hash = _registry(root)
        if checkout["repository"] != registry["repository"]:
            raise WorkspaceError("Origin does not match the registered repository")
        if platform not in registry["platforms"] or profile not in registry["platforms"][platform]["profiles"]:
            raise WorkspaceError("Unknown platform or model profile")
        _branch_allowed(checkout["branch"], registry["platforms"][platform])
        _git(root, "check-ignore", "--", BINDING)
        binding = {"schema_version": 1, **checkout, "platform": platform, "profile": profile,
                   "registry_sha256": registry_hash, "binding_id": uuid.uuid4().hex}
        _exclusive_json(root / BINDING, binding)
        return cls.open(root)

    @classmethod
    def open(cls, root=None):
        root = _no_links(Path.cwd() if root is None else root).resolve(strict=True)
        try:
            binding = json.loads(_read(root / BINDING))
            registry, registry_hash = _registry(root)
            workspace = cls(root, binding, registry, registry_hash)
            workspace.check()
            return workspace
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise WorkspaceError("Missing or malformed workspace binding") from error

    def check(self):
        try:
            stored = json.loads(_read(self.root / BINDING))
            if stored != self.binding or stored["schema_version"] != 1:
                raise WorkspaceError("Workspace binding changed")
            actual = _checkout(self.root)
            if any(stored[key] != value for key, value in actual.items()):
                raise WorkspaceError("Checkout root, Git directory, branch or origin changed")
            registry, digest = _registry(self.root)
            if (registry != self.registry or digest != stored["registry_sha256"]
                    or actual["repository"] != registry["repository"]
                    or self.profile not in registry["platforms"][self.platform]["profiles"]):
                raise WorkspaceError("The bound platform registry changed")
            if not re.fullmatch(r"[0-9a-f]{32}", stored["binding_id"]):
                raise WorkspaceError("Invalid binding identity")
            _branch_allowed(actual["branch"], registry["platforms"][self.platform])
            return {"status": "pass", "platform": self.platform, "profile": self.profile,
                    "repository": registry["repository"], "branch": actual["branch"]}
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise WorkspaceError("Malformed workspace binding") from error

    def _path(self, relative, kind, create_parent=False):
        self.check()
        return self._checked_path(relative, kind, create_parent)

    def _checked_path(self, relative, kind, create_parent=False):
        """Path rules after a check in this same synchronous helper call."""
        value = str(relative)
        if "\\" in value or ":" in value or "\x00" in value:
            raise WorkspaceError("Evidence paths must use plain relative POSIX components")
        parts = value.split("/")
        if not value or any(part in ("", ".", "..") for part in parts) or PurePosixPath(value).is_absolute():
            raise WorkspaceError("Evidence path escapes its platform root")
        base = PurePosixPath(self.config[f"{kind}_root"])
        if parts[0] in ("artifacts", "results"):
            if tuple(parts[:2]) != base.parts:
                raise WorkspaceError("Another platform's evidence is not an allowed target")
            parts = parts[2:]
        if not parts:
            raise WorkspaceError("Expected an evidence file below the platform root")
        target = self.root.joinpath(*base.parts, *parts)
        _no_links(target)
        if not target.resolve(strict=False).is_relative_to(self.root / Path(*base.parts)):
            raise WorkspaceError("Evidence path leaves the platform root")
        if create_parent:
            parent = self.root
            for part in (*base.parts, *parts[:-1]):
                parent = parent / part
                _no_links(parent)
                parent.mkdir(exist_ok=True)
                if not parent.is_dir():
                    raise WorkspaceError("Evidence parent is not a directory")
        return target

    def artifact_path(self, relative, create_parent=False):
        return self._path(relative, "artifact", create_parent)

    def validate_artifact_scope(self, paths, *, runtime_modules, expected_identity):
        """Validate one bounded artifact/module batch after one fresh check.

        No authorization or source snapshot is retained or returned. The model
        identity comes from the binding/registry validated by this call; it
        needs no unused HEAD/worktree diff. Public context() and result/update
        publication keep their full source-context checks unchanged. Every
        later batch starts with a new full binding/checkout/registry check.
        """
        self.check()
        if (not isinstance(paths, (tuple, list)) or len(paths) > 16
                or not isinstance(runtime_modules, (tuple, list)) or not 1 <= len(runtime_modules) <= 16
                or any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in runtime_modules)
                or not isinstance(expected_identity, tuple) or len(expected_identity) != 4
                or any(not isinstance(value, str) or not value for value in expected_identity)):
            raise WorkspaceError("Invalid bounded artifact/module validation batch")
        profile = self.config["profiles"][self.profile]
        if (self.platform, self.profile, profile["model"], profile["revision"]) != expected_identity:
            raise WorkspaceError("Artifact scope belongs to another platform, profile or pinned model")
        for module_file, relative in runtime_modules:
            _runtime_module(self.root, module_file, relative)
        base = self._checked_path("namespace-check", "artifact").parent.resolve()
        for value in paths:
            path = Path(value).absolute()
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError as error:
                raise WorkspaceError("Artifact paths must stay in the bound platform namespace") from error
            if self._checked_path(relative, "artifact").absolute() != path:
                raise WorkspaceError("Artifact path differs from its bound namespace")

    def check_runtime_module(self, module_file, relative):
        self.check()
        _runtime_module(self.root, module_file, relative)

    def result_path(self, relative, create_parent=False):
        return self._path(relative, "result", create_parent)

    def _git_state(self):
        exclusions = [f":(exclude){self.config[key]}/**" for key in ("artifact_root", "result_root")]
        paths = ["--", ".", f":(exclude){BINDING}", *exclusions]
        head = _git(self.root, "rev-parse", "HEAD").decode().strip()
        digest = hashlib.sha256()
        for data in (_git(self.root, "diff", "--binary", "--no-ext-diff", "--no-textconv", *paths),
                     _git(self.root, "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", *paths),
                     _git(self.root, "status", "--porcelain=v1", "-z", "--untracked-files=all", *paths)):
            digest.update(len(data).to_bytes(8, "big")); digest.update(data)
        names = _git(self.root, "ls-files", "--others", "--exclude-standard", "-z", *paths).split(b"\0")
        if len(names) > 4097:
            raise WorkspaceError("Too many untracked source files to bind a result")
        total = 0
        for name in sorted(n for n in names if n):
            path = self.root / os.fsdecode(name)
            raw = _read(path, 16 * 1024 * 1024)
            total += len(raw)
            if total > 64 * 1024 * 1024:
                raise WorkspaceError("Untracked source inputs exceed the evidence bound")
            digest.update(len(name).to_bytes(8, "big")); digest.update(name); digest.update(hashlib.sha256(raw).digest())
        return head, digest.hexdigest()

    def context(self, run_id=None):
        self.check()
        run_id = uuid.uuid4().hex if run_id is None else run_id
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id):
            raise WorkspaceError("Invalid run identity")
        head, dirty = self._git_state()
        profile = self.config["profiles"][self.profile]
        return {"schema_version": 1, "binding_id": self.binding["binding_id"], "platform": self.platform,
                "profile": self.profile, "repository": self.registry["repository"], "model": profile["model"],
                "revision": profile["revision"], "git_head": head, "dirty_diff_sha256": dirty,
                "registry_sha256": self.registry_hash, "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat()}

    def stamp_result(self, payload, run_id=None, *, initial_context=None):
        if not isinstance(payload, dict) or CONTEXT in payload:
            raise WorkspaceError("Expected a new result without an existing workspace context")
        if initial_context is not None:
            self._verify_context(initial_context)
            if run_id is not None and run_id != initial_context["run_id"]:
                raise WorkspaceError("Run identity changed during execution")
            context = deepcopy(initial_context)
        else:
            context = self.context(run_id)
        return {**deepcopy(payload), CONTEXT: context}

    def write_result(self, relative_path, payload, run_id=None, *, initial_context=None):
        value = self.stamp_result(payload, run_id, initial_context=initial_context)
        path = self.result_path(relative_path, create_parent=True)
        _exclusive_json(path, value)
        return path

    def _verify_context(self, context):
        if not isinstance(context, dict):
            raise WorkspaceError("Missing result workspace context")
        current = self.context(context.get("run_id"))
        if set(context) != set(current) or any(context[key] != value for key, value in current.items() if key != "created_at"):
            raise WorkspaceError("Result platform, model, checkout or source revision differs")
        if not isinstance(context["created_at"], str):
            raise WorkspaceError("Missing result creation time")

    def check_result(self, relative_path):
        path = self.result_path(relative_path)
        try:
            value = json.loads(_read(path))
            self._verify_context(value.get(CONTEXT))
        except (AttributeError, json.JSONDecodeError) as error:
            raise WorkspaceError("Malformed platform result") from error
        return {"status": "pass", "platform": self.platform, "profile": self.profile,
                "run_id": value[CONTEXT]["run_id"], "result_sha256": _sha(_read(path))}

    def _gh(self, endpoint, *, metadata=False, body=None):
        args = ["gh", "api", "--hostname", "github.com", endpoint]
        if metadata:
            args += ["--jq", "{number,labels:[.labels[].name],milestone:(.milestone.number // null),pull_request:.pull_request}"]
        if body is not None:
            args += ["--method", "PATCH", "--input", "-"]
        raw = _process(args, cwd=self.root, data=None if body is None else _json_bytes(body))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise WorkspaceError("Malformed GitHub metadata") from error

    def _issue_scope(self, number, shared=False):
        self.check()
        if type(number) is not int or number <= 0:
            raise WorkspaceError("Expected a registered issue number")
        if shared:
            matches = [d for d in self.config.get("shared_read_dependencies", []) if d["issue"] == number]
            if len(matches) != 1:
                raise WorkspaceError("Shared issue is not an explicit dependency")
            return matches[0]["label"], [matches[0]["milestone"]]
        if number not in self.config["issues"]:
            raise WorkspaceError("Issue belongs to another progress scope")
        return self.config["label"], self.config["milestones"]

    def _issue_matches(self, issue, number, label, milestones):
        labels = [row if isinstance(row, str) else row.get("name") for row in issue.get("labels", [])]
        platform_labels = {entry["label"] for entry in self.registry["platforms"].values()}
        milestone = issue.get("milestone")
        milestone = milestone.get("number") if isinstance(milestone, dict) else milestone
        if (issue.get("number") != number or issue.get("pull_request") is not None
                or labels.count(label) != 1 or set(labels).intersection(platform_labels) != {label}
                or milestone not in milestones):
            raise WorkspaceError("GitHub issue label or milestone differs from the registry")

    def read_progress(self, number, *, shared=False):
        label, milestones = self._issue_scope(number, shared)
        endpoint = f"repos/{self.registry['repository']}/issues/{number}"
        meta = self._gh(endpoint, metadata=True)
        self._issue_matches(meta, number, label, milestones)
        self.check()
        issue = self._gh(endpoint)
        self._issue_matches(issue, number, label, milestones)
        self.check()
        return {"number": number, "title": issue.get("title"), "body": issue.get("body"),
                "state": issue.get("state"), "url": issue.get("html_url"), "shared_read_only": shared}

    def prepare_update(self, number, body_path, *, run_id=None):
        self._issue_scope(number)  # Shared dependencies can never be update targets.
        body = _read(self.artifact_path(body_path), MAX_BODY).decode("utf-8")
        original = self.read_progress(number)
        packet = {"kind": "vesper_platform_issue_update", "issue": number, "body": body,
                  "body_sha256": _sha(body.encode("utf-8")),
                  "previous_body_sha256": _sha((original["body"] or "").encode("utf-8")),
                  CONTEXT: self.context(run_id)}
        path = self.artifact_path(f"updates/{packet[CONTEXT]['run_id']}.json", create_parent=True)
        _exclusive_json(path, packet)
        return path

    def preflight_update(self, relative_packet):
        packet = json.loads(_read(self.artifact_path(relative_packet)))
        if (packet.get("kind") != "vesper_platform_issue_update" or not isinstance(packet.get("body"), str)
                or len(packet["body"].encode("utf-8")) > MAX_BODY
                or _sha(packet["body"].encode("utf-8")) != packet.get("body_sha256")):
            raise WorkspaceError("Update body differs from its prepared packet")
        self._verify_context(packet.get(CONTEXT))
        issue = self.read_progress(packet.get("issue"))
        if _sha((issue["body"] or "").encode("utf-8")) != packet.get("previous_body_sha256"):
            raise WorkspaceError("Issue body changed since update preparation")
        return packet

    def commit_update(self, relative_packet):
        packet = self.preflight_update(relative_packet)
        self._verify_context(packet[CONTEXT])
        response = self._gh(f"repos/{self.registry['repository']}/issues/{packet['issue']}", body={"body": packet["body"]})
        self._issue_matches(response, packet["issue"], self.config["label"], self.config["milestones"])
        return {"status": "updated", "platform": self.platform, "issue": packet["issue"], "url": response.get("html_url")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind"); bind.add_argument("--platform", required=True); bind.add_argument("--profile", default="native")
    sub.add_parser("check")
    progress = sub.add_parser("read-progress")
    target = progress.add_mutually_exclusive_group(); target.add_argument("--issue", type=int); target.add_argument("--shared-issue", type=int)
    check = sub.add_parser("check-result"); check.add_argument("--result", required=True)
    prepare = sub.add_parser("prepare-update"); prepare.add_argument("--issue", type=int, required=True)
    prepare.add_argument("--body-file", required=True); prepare.add_argument("--run-id")
    for command in ("preflight-update", "commit-update"):
        entry = sub.add_parser(command); entry.add_argument("--packet", required=True)
    args = parser.parse_args(argv)
    try:
        _runtime_module(Path.cwd() if args.root is None else args.root, __file__, "src/vesper/workspace.py")
        if args.command == "bind":
            result = Workspace.bind(args.root, platform=args.platform, profile=args.profile).check()
        else:
            workspace = Workspace.open(args.root)
            if args.command == "check": result = workspace.check()
            elif args.command == "check-result": result = workspace.check_result(args.result)
            elif args.command == "read-progress":
                numbers = [args.shared_issue or args.issue] if args.shared_issue or args.issue else workspace.config["issues"]
                result = [workspace.read_progress(n, shared=args.shared_issue is not None) for n in numbers]
            elif args.command == "prepare-update":
                path = workspace.prepare_update(args.issue, args.body_file, run_id=args.run_id)
                result = {"status": "prepared", "packet": path.relative_to(workspace.root).as_posix()}
            elif args.command == "preflight-update":
                packet = workspace.preflight_update(args.packet)
                result = {"status": "pass", "issue": packet["issue"], "body_sha256": packet["body_sha256"]}
            else: result = workspace.commit_update(args.packet)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (WorkspaceError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
