---
name: Implementation or validation task
about: Define a scoped outcome and the evidence needed to accept it.
title: ''
labels: ''
assignees: ''
---

## Outcome

What observable behavior or validation gate should this work establish?

## Target and candidate

Name the platform, milestone and numerical profile. A workspace profile such as
`native` is not a precision profile. For PC work, follow the current
[quality/speed policy](https://github.com/shell-term/vesper/blob/main/docs/quality-speed.md);
other platforms retain their own gates.

## Acceptance

- [ ] Define observable success for the claim: exact comparator checks, actual task quality, runtime/resource correctness or performance, as applicable.
- [ ] State resource limits and test scope.
- [ ] Link complete evidence and relevant CI before closing.

Choose the smallest relevant verification plan. Detailed intermediate tracing
is diagnostic or deliberate qualification work, not the default for every PC
change. Keep optional mechanisms conditional; an idea is not a mandatory gate
unless selected. Documentation-only work does not require model execution.

## Source and dependencies

Pin reference artifacts where applicable. Link valid cached comparator evidence;
state whether a new original execution is needed. Distinguish a dependency from
an optional investigation. Keep unfinished named tasks open rather than silently
shortening their prompts/budgets.

## Evidence and limitations

Add reproduction commands, results and known failures. Keep specification,
official-reference parity, full-model quality and performance claims separate.
Declare approximation and compare actual output quality, speed and memory; do
not call a failed exact check passed. Preserve historical scope and execution
contexts. Do not include checkpoints, private paths, credentials or private user
prompts.
