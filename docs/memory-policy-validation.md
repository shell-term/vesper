# Whole-runtime memory policy

The six original [Issue #9](https://github.com/shell-term/vesper/issues/9)
acceptance checks pass within the resource scope below. The engine keeps the
default **96 GiB owned-host ceiling and 32 GiB system-available floor**.
The [sealed audit](../results/2026-10-19-memory-policy/acceptance.json) binds
complete measurements and original numerical goldens. Full forty-layer
inference, logits/sampling and text generation remain Issue #10 work.

## One admission authority

Start `RuntimeMemory` before opening runtime components. Its default
`MemoryPolicy` applies to every `MemoryBudget` and CUDA `Device` in that process,
including worker threads. Component ceilings remain additional restrictions;
each component cannot independently spend the entire 96 GiB allowance.
The API currently permits lowering the host ceiling or raising the physical
reserve. Raising the 96 GiB ceiling requires a code/policy change and evidence.

```python
from vesper.memory import RuntimeMemory

with RuntimeMemory() as memory:
    # Open existing components here, then perform complete runtime calls.
    # Close their resources before leaving the policy context.
    pass
```

| Resource | Accounting and lifetime |
| --- | --- |
| Expert/Engram cache copies | Typed cache/Engram charges; eligible, unborrowed entries can be evicted without changing source bytes or routes |
| Weights and activations | Explicit owned buffers, or conservative bounded component admissions covering existing native operators, fixtures and intermediate arrays |
| Window/compressed KV, index keys, candidates and history | Component state/snapshot admissions; resource-scale buffers use exact full-backbone shapes; logical state is never pressure-evicted |
| Pinned transfer buffers | Typed host charges plus a shared 1 GiB pinned ceiling; release follows a successful native free |
| Metadata and integrity tables | Typed store/index/row metadata; a 64 MiB control admission, at most 16,384 live allocation records and 512 reclaimers |
| Scratch and diagnostics | Typed direct-I/O/comparison scratch, explicit probe buffers and bounded native-component envelopes; diagnostic arrays are covered while runtime-owned |
| GPU payloads | Shared counters across all contexts/operators on each ordinal, in addition to local component caps and observed free VRAM |

The default combined payload caps are 20 GiB on ordinal 0 and 10 GiB on ordinal
1, with a 2 GiB observed free-VRAM floor. Ordinals must be checked against the
actual machine; the recorded run uses the 4090 for computation and the 3080 Ti
for exact byte storage/audits. Existing numerical kernels and their registered
binaries are unchanged.

The host total combines explicit payload bytes with **conservative admissions**.
Existing component envelopes cover weights, returned intermediate diagnostics,
KV/snapshot copies and temporary arrays; overlapping envelopes may overestimate
actual payloads. The report preserves this distinction in `accounting_kinds`
and `explicit_host_bytes`. Categories describe ownership, not an exhaustive
breakdown of every Python object. Exported caller-owned results must have their
own bounded retention policy when integrated into a service.

Before allocation, admission checks the combined cap, pinned cap and observed
physical reserve. Eligible cache callbacks try their own locks without waiting
and skip live borrowers. If releasing cache charges does not restore **observed
physical** room, work is refused. Periodic observation also attempts reclamation
while idle; it fails closed if observation fails. Large owned-buffer initialization
checks physical headroom in at most 64 MiB touch increments.

State publication and cancellation share the governor lock. Public stateful
calls remain registered as in flight through completion/recovery; cancellation
cannot be cleared halfway through a call. Exact snapshot recovery after cancel
uses the same caps and reserve, then explicit resume permits retry. If physical
pressure prevents recovery, existing pending-snapshot contracts block new calls
until recovery succeeds. Cleanup can release allocations after cancellation.
Closing with live leases, GPU payloads or callbacks is refused. Escaped NumPy
views of `OwnedBuffer` retain their charge until the base allocation dies.

## Measurements and exclusions

This is a cooperative runtime policy, **not a process RSS/commit limit or an OS
quota**. Driver/context/module allocations, compiler libraries and general
Python/NumPy allocator overhead have separate process observations. Unmanaged
allocations in another library/process do not become governed by this counter.
Observation is periodic and at admission/publication boundaries; another
application can change available RAM between observations. The policy refuses
new work when it cannot restore the reserve and never stops other workloads.

Windows working-set residency and private commit are recorded separately using
`GetProcessMemoryInfo`; private commit is not resident RAM. See
[Microsoft's metric definitions](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex).
Linux reads its own `smaps_rollup` residency/private-resident metrics when
available, with an explicitly approximate `status` fallback; it does not label
those fields private commit. See the
[kernel's proc documentation](https://docs.kernel.org/filesystems/proc.html).
Unavailable metrics remain null/absent rather than zero.

Driver initialization, context creation, native compilation and module load
have named before/after process snapshots. Deltas include allocator effects and
other simultaneous activity in the process; they are **not exact attribution**
to a component. The optional original Torch/TileLang reference is measured in a
separate short-lived process, including its own GPU VMM accounting. Torch's
allocator figures exclude direct-driver/VMM allocations, which are reported
separately. That reference is not required by the native runtime.

## Recorded live gates

The [complete scale report](../results/2026-10-19-memory-policy/scale.json)
derives its plan from the pinned original config and all 96,085 tensor
descriptors. It visits all **40 × 384 = 15,360 backbone expert identities**,
six original packed tensors and **18,800,640 bytes per expert**. MTP experts are
excluded explicitly. It actually touches **288,777,830,400 synthetic expert
bytes over time**, with at most a 6 GiB expert cache and an 8 GiB host ceiling.
This is a bounded resource workload; the payload contents are synthetic and
are not original-weight arithmetic or SSD throughput measurements.

Batch one, context 131,072 retains all original window/source/index/pooling/
candidate shapes: **424,714,240 bytes per state copy**, plus an equally sized
rollback copy. The run also touches the original-sized 1,323,827,200-byte
embedding/head footprint, per-layer working weights, activations/scratch,
integrity tables, metadata and two actual expert-sized CUDA pinned buffers.
KV bytes remain unchanged through pressure, partial reads, cancellation,
allocation refusal and retry. A separate owned 512 MiB companion keeps working
through these checks and exits normally; existing workloads are not controlled.

| Measured gate | Result |
| --- | --- |
| Full-geometry bounded resource pass | Peak total charge 8,369,594,496 bytes; explicit payload peak 7,570,652,800 bytes; 244 global reclaim events |
| Actual parent-process observations in that pass | Sampled resident peak 7,850,340,352 bytes; private-commit peak 8,837,902,336 bytes; separate from charged bytes |
| Sampled physical reserve during that pass | Minimum 128,392,814,592 bytes, approximately 119.58 GiB |
| Original Engram under the same governor and retained full-geometry KV | All 110 original steps / 6,820 typed/shaped boundaries match; six extra warm/retry steps / 372 boundaries match; native environment has no Torch/TileLang/Transformers/SymPy installed or imported |
| Native failure additions | Cancel after layer 1, cancel after layer 14, backend GPU allocation failure, host allocation failure and pinned allocation failure; no leaked allocations or committed partial history |
| Governed complete 20..25 chain | All 40 steps / 240 block calls / 59,600 boundaries match original goldens, with ten original failures/retries and snapshots; combined charge peak 10,510,468,032 bytes under an explicit 16 GiB probe cap |
| Complete-chain GPU payload peaks | 1,660,849,784 primary bytes and 226,488,960 helper bytes, combined across operators/contexts |
| Default 96 GiB / 32 GiB boundary check | Symbolic admission reaches 96 GiB, then another byte is refused; injected 32 GiB availability refuses another byte; no 96 GiB payload allocation |
| Optional separate original reference | All 110 steps / 6,820 boundaries still match; sampled resident peak 2,298,871,808 bytes and private commit 8,582,000,640 bytes; original-address VMM physical/virtual allocations are released |
| Final cleanup | All owned host/GPU charges, live leases/reclaimers and in-flight operations reach zero; monitor threads stop |

Restorable/unrestorable low-available-memory conditions are injected while real
available memory is observed separately. The probe does not intentionally fill
the machine down to 32 GiB. The default 96 GiB cap was tested as an admission
boundary; physical allocation was tested with the explicit smaller caps above.
No full checkpoint was downloaded, no GPU reset occurred, and page-file settings
were unchanged. This does not establish whole-model quality, numerical execution
of all forty layers, text generation or token speed.

## Reproduce

The CPU acceptance audit reads complete published evidence and original
goldens, checks file hashes against the pinned manifest and retains all six
original acceptance criteria. Altered/missing requirements, geometry, results,
state, framework independence, memory metrics and cleanup are rejected by tests.
It does not silently accept a newly generated result as qualified evidence.

```console
python examples/qualify_memory.py --output artifacts/memory-acceptance.json
python -m unittest discover -s tests -v
```

For a new GPU resource run, first prepare the original metadata/Engram fixtures
as described in the [artifact report](engram-artifacts-validation.md). These
commands use the independent native environment and do not fetch model weights:

```console
python examples/probe_memory_admission.py --output artifacts/default-boundaries.json
python examples/probe_memory_policy.py --metadata artifacts/import-source --engram-fixtures artifacts/engram-fixture --output artifacts/model-memory.json
python examples/recheck_memory_components.py --component block_chain --memory-output artifacts/chain-memory.json -- --reference-report results/2026-10-19-refresh-block-chain/chain.json --topology-fixtures artifacts/topology-fixture --block-root artifacts --attention-bundle artifacts/native-attention-v1 --output artifacts/governed-chain.json
```

The chain needs the already authenticated fixtures/binaries from the
[six-block report](refreshed-block-chain-validation.md). Smaller
`--experts-per-layer` values are smoke checks and cannot satisfy the
representative gate. The default resource workload is synthetic host allocation;
it does not need the full original checkpoint or another SSD.

Run `examples/profile_reference_memory.py` **in the separate pinned reference
environment** with `--metadata`, `--fixtures`, `--reference`, `--output` and
`--original-output`. On Windows, `--reference-nvrtc` preloads the compatible
reference DLL, as in the Engram report; native compilation keeps its registered
12.4 compiler. No raw comparison/source bytes are rewritten for qualification.

Complete reports: [scale](../results/2026-10-19-memory-policy/scale.json),
[chain](../results/2026-10-19-memory-policy/chain.json),
[chain memory](../results/2026-10-19-memory-policy/chain-memory.json),
[default boundaries](../results/2026-10-19-memory-policy/default-boundaries.json),
[separate reference memory](../results/2026-10-19-memory-policy/reference-memory.json),
[fresh original comparison](../results/2026-10-19-memory-policy/reference-original.json),
[manifest](../results/2026-10-19-memory-policy/manifest.json) and
[acceptance](../results/2026-10-19-memory-policy/acceptance.json).
