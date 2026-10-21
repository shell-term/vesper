"""Offline GPQA development protocol; never invokes a model, judge or full run.

Question text, answers and completions belong in private Spark artifacts. The
manifest is content-bound, not an assertion that the author's official setup
or a practical-quality gate has been reproduced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .checkpoint import MODEL, REVISION

SCHEMA = "spark-gpqa-development-v1"
DATASET = "idavidrein/gpqa"
SPLIT = "gpqa_diamond"
DOMAINS = ("biology", "chemistry", "physics")
PHASES = ("startup", "input", "thinking", "answer", "image", "tool", "grading", "cleanup", "other")
STATUSES = ("completed", "timeout", "truncated", "cancelled", "error", "unsupported", "not_run")
PROMPT_PREFIX = "Answer the following multiple-choice question. End the final answer with exactly one line: Answer: (A), Answer: (B), Answer: (C), or Answer: (D)."
PROPOSED_SETTINGS = {"thinking": True, "reasoning_effort": 100, "temperature": 0.0,
                     "top_p": 1.0, "seed": 2026, "tools": [], "repetitions": 1,
                     "batch_size": 1, "concurrency": 1, "answer_parser": "final-line-v1"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _hex(value, length, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError(f"Expected pinned {name}")
    return value


def _number(value, name, *, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite {name}")
    if value < 0 or (positive and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid {name}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {name}")
    return value.strip()


def read_dataset(raw, *, revision, expected_sha256, domain_column="High-level domain"):
    """Validate a local author CSV and normalize only evaluation-relevant fields.

    File bytes and canonical per-question bytes are both pinned. No network or
    dataset download occurs here. The complete Diamond input must have 198 rows.
    """
    _hex(revision, 40, "dataset revision")
    _hex(expected_sha256, 64, "dataset file SHA256")
    if not isinstance(raw, bytes) or len(raw) > 16 * 1024**2:
        raise ValueError("Expected bounded local CSV bytes")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Dataset file SHA256 mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) != 198:
        raise ValueError("Expected all 198 GPQA Diamond questions")
    normalized, seen, source_ids = [], set(), set()
    for row in rows:
        question = _text(row.get("Question"), "question")
        domain = _text(row.get(domain_column), "high-level domain").lower()
        if domain not in DOMAINS:
            raise ValueError("Unexpected GPQA high-level domain")
        choices = [_text(row.get("Correct Answer"), "correct answer")]
        choices.extend(_text(row.get(f"Incorrect Answer {i}"), "incorrect answer") for i in (1, 2, 3))
        if choices[0] in choices[1:]:
            raise ValueError("Correct answer is duplicated among incorrect choices")
        question_sha = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if question_sha in seen:
            raise ValueError("Duplicate question")
        seen.add(question_sha)
        source_id = _text(row.get("Record ID"), "author Record ID")
        if source_id in source_ids:
            raise ValueError("Duplicate author Record ID")
        source_ids.add(source_id)
        item = {"question": question, "domain": domain, "choices": choices,
                "source_record_id": source_id}
        item["id"] = digest(item)
        item["question_sha256"] = question_sha
        normalized.append(item)
    if {r["domain"] for r in normalized} != set(DOMAINS):
        raise ValueError("Expected all three GPQA domains")
    pin = {"dataset": DATASET, "revision": revision, "split": SPLIT,
           "file_sha256": expected_sha256, "canonical_sha256": digest(sorted(normalized, key=lambda r: r["id"])),
           "count": len(normalized), "domain_column": domain_column}
    return pin, normalized


def _allocation(groups, total):
    """At least one per nonempty domain, then proportional largest remainders."""
    sizes = {k: len(v) for k, v in groups.items() if v}
    if isinstance(total, bool) or not isinstance(total, int) or not len(sizes) <= total <= sum(sizes.values()):
        raise ValueError("Subset size must cover each domain and fit the remaining questions")
    quotas = {k: 1 for k in sizes}
    remaining = total - len(sizes)
    capacities = {k: v - 1 for k, v in sizes.items()}
    capacity = sum(capacities.values())
    if remaining:
        floors = {k: remaining * v // capacity for k, v in capacities.items()}
        for k, value in floors.items():
            quotas[k] += value
        left = remaining - sum(floors.values())
        order = sorted(capacities, key=lambda k: (-(remaining * capacities[k] % capacity), k))
        for k in order[:left]:
            quotas[k] += 1
    return quotas


def request(item, *, seed=2026):
    """Deterministic choice order independent of CSV row order; no answer leak."""
    order = sorted(range(4), key=lambda i: digest([seed, item["id"], "choice", i]))
    choices = [item["choices"][i] for i in order]
    prompt = PROMPT_PREFIX + "\n\n" + item["question"] + "\n\n" + "\n".join(
        f"({label}) {choice}" for label, choice in zip("ABCD", choices))
    return {"id": item["id"], "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "choice_order": order, "correct_label": "ABCD"[order.index(0)], "domain": item["domain"],
            "source_record_id": item["source_record_id"]}


def prepare(pin, rows, *, development=12, held_out=6, seed=2026):
    """Freeze membership without evaluating correctness or exposing prompts."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Expected integer selection seed")
    if pin.get("count") != 198 or len(rows) != 198 or pin.get("split") != SPLIT:
        raise ValueError("Expected validated complete Diamond input")
    if pin.get("canonical_sha256") != digest(sorted(rows, key=lambda r: r["id"])):
        raise ValueError("Normalized dataset changed")
    groups = {d: sorted([r for r in rows if r["domain"] == d],
                        key=lambda r: digest([seed, "membership", r["id"]])) for d in DOMAINS}
    dev_quota = _allocation(groups, development)
    dev = [r for d in DOMAINS for r in groups[d][:dev_quota[d]]]
    rest = {d: groups[d][dev_quota[d]:] for d in DOMAINS}
    held_quota = _allocation(rest, held_out)
    hold = [r for d in DOMAINS for r in rest[d][:held_quota[d]]]
    entries = []
    for r in rows:
        prepared = request(r, seed=seed)
        entries.append({k: v for k, v in prepared.items() if k != "prompt"})
    value = {"schema": SCHEMA, "dataset": pin, "selection_seed": seed,
             "selection_method": "per-domain-minimum-one-then-proportional-largest-remainder-v1",
             "development": [r["id"] for r in dev], "held_out": [r["id"] for r in hold],
             "population": sorted(entries, key=lambda r: r["id"]),
             "full_run_authorized": False, "quality_tolerance_adopted": False}
    value["manifest_sha256"] = digest(value)
    return value


def _manifest(value):
    check = {k: v for k, v in value.items() if k != "manifest_sha256"}
    if value.get("schema") != SCHEMA or value.get("manifest_sha256") != digest(check):
        raise ValueError("Manifest changed or has an unknown schema")
    population = value.get("population", [])
    ids = [r["id"] for r in population]
    dev, hold = value.get("development", []), value.get("held_out", [])
    if len(ids) != 198 or len(set(ids)) != 198 or len(set(dev + hold)) != len(dev + hold) or not set(dev + hold) <= set(ids):
        raise ValueError("Invalid or overlapping subset membership")
    return {r["id"]: r for r in population}


def plan(manifest, *, subset, identity, budget, settings=None):
    """Freeze a finite development run. This creates no execution capability."""
    _manifest(manifest)
    if subset not in ("development", "held_out"):
        raise ValueError("Only small development/held-out plans can be prepared")
    if identity.get("model") != MODEL or identity.get("model_revision") != REVISION:
        raise ValueError("Expected the pinned original model identity")
    _hex(identity.get("code_commit"), 40, "runtime source commit")
    dirty = identity.get("dirty_diff_sha256")
    if dirty is not None:
        _hex(dirty, 64, "dirty source digest")
    for name in ("profile", "runtime", "hardware", "toolchain", "diagnostic_mode"):
        _text(identity.get(name), name)
    if identity["profile"] != "native":
        raise ValueError("No approximate quality profile or tolerance has been adopted")
    conditions = json.loads(json.dumps(PROPOSED_SETTINGS if settings is None else settings))
    if conditions != PROPOSED_SETTINGS:
        raise ValueError("This version implements only the declared primary development protocol")
    required = {"context_tokens", "max_generated_tokens", "max_model_calls", "per_question_seconds", "total_seconds"}
    if set(budget) != required:
        raise ValueError("Declare every input/generation/call/time budget explicitly")
    for key, value in budget.items():
        _number(value, key, positive=True, integer=key.endswith("tokens") or key == "max_model_calls")
    if budget["max_generated_tokens"] > budget["context_tokens"]:
        raise ValueError("Generation budget exceeds total context budget")
    if budget["context_tokens"] > 1152:
        raise ValueError("Current bounded Spark text context is at most 1152 tokens; longer reasoning is unqualified")
    value = {"schema": SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
             "dataset": manifest["dataset"], "subset": subset, "question_ids": manifest[subset],
             "identity": dict(identity), "settings": conditions, "budget": dict(budget),
             "protocol_status": "concrete_development_proposal_not_official_reproduction",
             "quality_tolerance_adopted": False, "full_run_authorized": False}
    value["plan_sha256"] = digest(value)
    return value


def _plan(value):
    check = {k: v for k, v in value.items() if k != "plan_sha256"}
    if value.get("schema") != SCHEMA or value.get("plan_sha256") != digest(check):
        raise ValueError("Run plan changed")


def parse_answer(final_text):
    """A single terminal answer line; never mine a letter from reasoning text."""
    if not isinstance(final_text, str):
        raise ValueError("Expected final-channel text")
    markers = re.findall(r"(?m)^Answer: \(([ABCD])\)\s*$", final_text.strip())
    terminal = re.search(r"(?:^|\n)Answer: \(([ABCD])\)$", final_text.strip())
    if len(markers) != 1 or terminal is None or re.search(r"</?think>|<\||<｜|｜DSML｜", final_text):
        return None
    return terminal.group(1)


def grade(manifest, run_plan, records):
    """Preserve non-completions and denominator; reject mismatched provenance."""
    entries = _manifest(manifest)
    _plan(run_plan)
    if run_plan["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Manifest/run mismatch")
    selected = run_plan["question_ids"]
    if run_plan.get("subset") not in ("development", "held_out") or selected != manifest[run_plan["subset"]]:
        raise ValueError("Run membership differs from the frozen selected subset")
    incoming = {}
    for record in records:
        qid = record.get("question_id")
        if qid not in selected or qid in incoming:
            raise ValueError("Unknown or duplicate question result")
        if record.get("plan_sha256") != run_plan["plan_sha256"]:
            raise ValueError("Result source/settings/plan mismatch")
        incoming[qid] = record
    output = []
    totals = Counter()
    phases = Counter()
    wall_total = 0.0
    missing_phases = set()
    for qid in selected:
        if qid not in incoming:
            missing_phases.update(PHASES)
            output.append({"question_id": qid, "domain": entries[qid]["domain"], "status": "not_run",
                           "correct": None, "score": None, "reason": "missing result"})
            continue
        raw = incoming[qid]
        status = raw.get("status")
        if status not in STATUSES:
            raise ValueError("Unknown completion status")
        wall = _number(raw.get("wall_seconds"), "question wall time")
        wall_total += wall
        tokens = {}
        for key in ("input_tokens", "generated_tokens", "thinking_tokens", "answer_tokens", "special_tokens", "model_calls"):
            tokens[key] = _number(raw.get(key), key, integer=True)
            totals[key] += tokens[key]
        if tokens["generated_tokens"] != tokens["thinking_tokens"] + tokens["answer_tokens"] + tokens["special_tokens"]:
            raise ValueError("Generated token accounting must include thinking, answer and special tokens")
        if status == "completed" and raw.get("eos_complete") is True:
            if (tokens["input_tokens"] < 1 or tokens["generated_tokens"] < 1 or tokens["special_tokens"] < 1
                    or tokens["model_calls"] != tokens["input_tokens"] + tokens["generated_tokens"] - 1):
                raise ValueError("Normal EOS completion requires positive tokens and consistent native model-call accounting")
        measured = raw.get("phase_seconds", {})
        if not isinstance(measured, Mapping) or not set(measured) <= set(PHASES):
            raise ValueError("Unknown phase accounting")
        for key, seconds in measured.items():
            phases[key] += _number(seconds, key)
        if sum(measured.values()) > wall + 0.01:
            raise ValueError("Phase times must be exclusive within wall time")
        missing_phases.update(set(PHASES) - set(measured))
        budget = run_plan["budget"]
        violations = []
        if tokens["input_tokens"] + max(0, tokens["generated_tokens"] - 1) > budget["context_tokens"]:
            violations.append("context")
        if tokens["generated_tokens"] > budget["max_generated_tokens"]:
            violations.append("generation")
        if tokens["model_calls"] > budget["max_model_calls"]:
            violations.append("model_calls")
        if wall > budget["per_question_seconds"]:
            violations.append("per_question_time")
        final = raw.get("final_text", "")
        if not isinstance(final, str):
            raise ValueError("Expected final-channel text")
        predicted = parse_answer(final) if status == "completed" and raw.get("eos_complete") is True and not violations else None
        if status == "completed" and (raw.get("eos_complete") is not True or violations):
            effective = "invalid_completion"
        elif status == "completed" and predicted is None:
            effective = "invalid_answer"
        else:
            effective = status
        correct = predicted == entries[qid]["correct_label"] if effective == "completed" else None
        output.append({"question_id": qid, "domain": entries[qid]["domain"], "status": effective,
                       "reported_status": status, "correct": correct, "score": None if correct is None else int(correct),
                       "predicted_label": predicted, "budget_violations": violations, "wall_seconds": wall,
                       "tokens": tokens, "phase_seconds": dict(measured), "raw_record_sha256": digest(raw)})
    count = len(selected)
    graded = [r for r in output if r["score"] is not None]
    correct_count = sum(r["score"] for r in graded)
    total_over = wall_total > run_plan["budget"]["total_seconds"]
    protocol_valid = not total_over and not any(r.get("budget_violations") for r in output)
    value = {"schema": SCHEMA, "run_plan": run_plan, "records": output,
             "summary": {"planned_questions": count, "graded_questions": len(graded),
                         "ungraded_questions": count - len(graded), "correct": correct_count,
                         "completed_accuracy": correct_count / len(graded) if graded else None,
                         "correct_over_planned": correct_count / count,
                         "completion_coverage": len(graded) / count,
                         "full_subset_score": correct_count / count if len(graded) == count and protocol_valid else None,
                         "status_counts": dict(Counter(r["status"] for r in output)),
                         "tokens": dict(totals), "wall_seconds": wall_total,
                         "phase_seconds": dict(phases), "unmeasured_phases": sorted(missing_phases),
                         "total_budget_exceeded": total_over, "protocol_valid": protocol_valid,
                         "score_scope": "completed_accuracy is conditional; correct_over_planned is coverage-aware earned credit, not ungraded model ability"},
             "scope": "descriptive small native regression screen; no broad quality or official-score pass",
             "full_run_authorized": False}
    value["report_sha256"] = digest(value)
    return value


def _report(value):
    check = {k: v for k, v in value.items() if k != "report_sha256"}
    if value.get("schema") != SCHEMA or value.get("report_sha256") != digest(check):
        raise ValueError("Quality report changed")
    _plan(value["run_plan"])


def compare(baseline, candidate):
    """Paired task changes, with strict matched conditions; never auto-adopts."""
    _report(baseline)
    _report(candidate)
    left, right = baseline["run_plan"], candidate["run_plan"]
    for key in ("manifest_sha256", "dataset", "subset", "question_ids", "settings", "budget"):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched comparison {key}")
    for key in ("model", "model_revision", "profile", "runtime", "hardware", "toolchain", "diagnostic_mode"):
        if left["identity"][key] != right["identity"][key]:
            raise ValueError(f"Unmatched runtime condition {key}")
    pairs = []
    for a, b in zip(baseline["records"], candidate["records"]):
        if a["question_id"] != b["question_id"]:
            raise ValueError("Unmatched paired record ordering")
        pairs.append({"question_id": a["question_id"], "baseline_status": a["status"], "candidate_status": b["status"],
                      "baseline_correct": a["correct"], "candidate_correct": b["correct"]})
    return {"schema": SCHEMA, "baseline_report_sha256": baseline["report_sha256"],
            "candidate_report_sha256": candidate["report_sha256"], "pairs": pairs,
            "correct_to_incorrect": [p["question_id"] for p in pairs if p["baseline_correct"] is True and p["candidate_correct"] is False],
            "incorrect_to_correct": [p["question_id"] for p in pairs if p["baseline_correct"] is False and p["candidate_correct"] is True],
            "completion_loss": [p["question_id"] for p in pairs if p["baseline_correct"] is not None and p["candidate_correct"] is None],
            "completion_gain": [p["question_id"] for p in pairs if p["baseline_correct"] is None and p["candidate_correct"] is not None],
            "full_subset_score_delta": (candidate["summary"]["full_subset_score"] - baseline["summary"]["full_subset_score"]
                                        if candidate["summary"]["full_subset_score"] is not None and baseline["summary"]["full_subset_score"] is not None else None),
            "protocol_valid": baseline["summary"]["protocol_valid"] and candidate["summary"]["protocol_valid"],
            "decision": "review_required_no_quality_tolerance_adopted", "full_run_authorized": False}


def estimate(manifest, reports, *, repetitions, fixed_overhead_seconds, excluded_costs):
    """Stratified measured-wall projection, with honest censoring and gaps.

    Min/max are empirical scenarios, not a confidence interval or guaranteed
    bounds. Each report is projected separately; baseline and candidate costs
    are then added, rather than silently timing only the faster configuration.
    """
    entries = _manifest(manifest)
    _number(repetitions, "full-run repetitions", positive=True, integer=True)
    _number(fixed_overhead_seconds, "additional fixed overhead")
    if not reports or not isinstance(excluded_costs, list) or not all(isinstance(x, str) and x.strip() for x in excluded_costs):
        raise ValueError("Declare measured reports and excluded costs explicitly")
    populations = Counter(r["domain"] for r in entries.values())
    projections = []
    seen = set()
    for report in reports:
        _report(report)
        if report["report_sha256"] in seen:
            raise ValueError("Duplicate measured report")
        seen.add(report["report_sha256"])
        if report["run_plan"]["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Estimate dataset mismatch")
        groups = defaultdict(list)
        censored = []
        for record in report["records"]:
            qid = record["question_id"]
            if qid not in entries or record["domain"] != entries[qid]["domain"]:
                raise ValueError("Estimate question identity mismatch")
            if "wall_seconds" in record:
                groups[record["domain"]].append(record["wall_seconds"])
            if record["status"] not in ("completed", "invalid_answer"):
                censored.append(qid)
        absent = sorted(set(populations) - set(groups))
        scenarios = {}
        for name, func in (("observed_min", min), ("stratified_mean", lambda a: sum(a) / len(a)), ("observed_max", max)):
            scenarios[name] = None if absent else repetitions * sum(populations[d] * func(groups[d]) for d in populations)
        # Timed-out/length-stopped work is right-censored, not evidence of a
        # complete-answer distribution. Keep only a partial-cost scenario.
        protocol_invalid = not report["summary"]["protocol_valid"]
        if censored or protocol_invalid:
            scenarios["stratified_mean"] = scenarios["observed_max"] = None
        projections.append({"report_sha256": report["report_sha256"], "identity": report["run_plan"]["identity"],
                            "budget": report["run_plan"]["budget"], "settings": report["run_plan"]["settings"],
                            "observed_questions": len(report["records"]), "samples_per_domain": {d: len(groups[d]) for d in populations},
                            "missing_domains": absent, "censored_question_ids": censored, "seconds": scenarios,
                            "protocol_invalid": protocol_invalid,
                            "unmeasured_phase_breakdown": report["summary"]["unmeasured_phases"]})
    totals = {}
    for name in ("observed_min", "stratified_mean", "observed_max"):
        values = [p["seconds"][name] for p in projections]
        totals[name] = None if any(v is None for v in values) else sum(values) + fixed_overhead_seconds
    return {"schema": SCHEMA, "dataset": manifest["dataset"], "full_coverage_questions_per_repetition": len(entries),
            "repetitions": repetitions, "configurations": projections, "additional_fixed_overhead_seconds": fixed_overhead_seconds,
            "total_seconds": totals, "excluded_costs": excluded_costs,
            "uncertainty": "Observed min/max scenarios are not confidence limits. Small samples miss long tails; source, cache, load, context and stopping changes invalidate transfer. Censored records prevent a finite complete-answer mean/upper estimate.",
            "required_user_decision": "Present this estimate, gaps and uncertainty; obtain an explicit decision before any all-question execution.",
            "full_run_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="private local CSV -> frozen subset and finite plan")
    p.add_argument("--dataset", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--domain-column", default="High-level domain")
    p.add_argument("--conditions", required=True, help="private JSON with identity and budget")
    p.add_argument("--development", type=int, default=12)
    p.add_argument("--held-out", type=int, default=6)
    p.add_argument("--subset", choices=("development", "held_out"), default="development")
    p = sub.add_parser("grade")
    p.add_argument("--prepared", required=True)
    p.add_argument("--records", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p = sub.add_parser("estimate")
    p.add_argument("--prepared", required=True)
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--repetitions", type=int, required=True)
    p.add_argument("--fixed-overhead-seconds", type=float, required=True)
    p.add_argument("--excluded-cost", action="append", default=[])
    for p in sub.choices.values():
        p.add_argument("--output", required=True, help="new Spark artifact-relative JSON")
    args = parser.parse_args(argv)
    from .workspace import Workspace
    ws = Workspace.open(args.root)
    ws.check()
    if ws.context().get("platform") != "dgx-spark":
        raise ValueError("Spark workspace required")
    def load(relative):
        path = ws.artifact_path(relative)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError("Offline evaluation input exceeds 32 MiB")
        return json.loads(path.read_text(encoding="utf-8"))
    if args.command == "prepare":
        path = ws.artifact_path(args.dataset)
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("Dataset exceeds bounded CSV limit")
        pin, rows = read_dataset(path.read_bytes(), revision=args.revision, expected_sha256=args.sha256, domain_column=args.domain_column)
        manifest = prepare(pin, rows, development=args.development, held_out=args.held_out)
        conditions = load(args.conditions)
        run_plan = plan(manifest, subset=args.subset, identity=conditions["identity"], budget=conditions["budget"])
        # Requests carry no correct labels; private manifest holds the key.
        by_id = {row["id"]: row for row in rows}
        requests = [{k: v for k, v in request(by_id[qid]).items() if k not in ("correct_label", "choice_order")}
                    for qid in manifest[args.subset]]
        result = {"manifest": manifest, "run_plan": run_plan, "requests": requests}
    elif args.command == "grade":
        prepared = load(args.prepared)
        result = grade(prepared["manifest"], prepared["run_plan"], load(args.records))
    elif args.command == "compare":
        result = compare(load(args.baseline), load(args.candidate))
    else:
        prepared = load(args.prepared)
        result = estimate(prepared["manifest"], [load(r) for r in args.reports], repetitions=args.repetitions,
                          fixed_overhead_seconds=args.fixed_overhead_seconds, excluded_costs=args.excluded_cost)
    # Recheck source/binding before exclusive publication. These are private
    # derived protocol artifacts, never source-stamped GPU qualification.
    ws.check()
    path = ws.artifact_path(args.output, create_parent=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "written", "command": args.command, "full_run_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
