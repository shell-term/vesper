# Vesper development rules

- Before platform-specific work, select `platforms/README.md` and the explicit PC/Spark entry. Use separate checkouts and local bindings; run `python -m vesper.workspace check` before guarded progress, results or Spark probes. Do not silently rebind a mismatch or infer current progress from an unfiltered repository-wide search. Migrate a legacy PC checkout only in its own active task.
- Shared model code does not transfer qualification. Use the registry allowlist for progress reads/update packets and retain hardware/profile/source identity in results. Helper checks are workflow checks, not OS-level read isolation.

- Preserve the released model's weights, scales, activation quantization, KV rounding, routing and sampling semantics. A faster approximation is not a successful optimization for this project.
- The model's trained router correction bias is required. Additional cache-dependent routing bias, expert pruning and lossy requantization are prohibited.
- Report format checks, specification checks, official-kernel parity, and full-model quality as separate results. Passing one does not prove the others.
- Keep host memory available for other work. The provisional full-engine cap is 96 GiB, with a 32 GiB system-available floor. Small probes have their own much smaller limits. Track pinned buffers and scratch as well as caches. A reservation counter is not proof of an OS-level memory limit.
- Do not stop other workloads, change the page file, reset GPUs, or download the full checkpoint as part of a small probe.
- Treat design proposals as hypotheses. Keep measured results and assumed performance ceilings distinct. Explain the evidence when revising a conclusion.
- Do not commit model weights, downloaded reference sources, local paths, credentials, GPU UUIDs, or machine names. Preserve third-party licenses for adapted code.
- Use English for public source documentation; maintain a short Japanese entry point. Keep status and unresolved validation gates honest.
- Track development in the repository's Issues and Milestones; see docs/roadmap.md. Link the active issue in commits and update its progress with scoped evidence. Use status:needs-validation for implemented work with an open reference gate. Add verified and close only when the issue's stated acceptance checks pass; do not close a milestone for a component-only pass.
