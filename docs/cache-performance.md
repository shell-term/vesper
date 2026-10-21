# Profiling and GPU integrity audits — 2026-10-18

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

GPU SHA-256 audits reduce the time of a fixed **warm synthetic expert trace** by
a median factor of **5.02** across six alternating pairs (range **4.93–5.13**).
Both modes match the unchanged official expert in all thirteen checked stages
of 232 calls each. This is not a model-token rate: routing, layers and generation
remain unimplemented, and the timed trace contains no SSD misses.

## Measured bottleneck

The earlier cache implementation copied each uploaded expert back to the CPU
and hashed it, in addition to hashing intermediate host copies. Profiling
separates direct-file reads, staging/RAM copies, CPU hashing, CUDA transfers,
GPU hashing, model kernels and diagnostic output copies. Sections do not overlap
in these serialized measurements; small Python/control overhead is unassigned.

In the six-pair benchmark, the readback mode spent 10,329.05 ms of 13,664.84 ms
of timed calls on CPU SHA-256, or **75.59%**. This evidence makes verification
placement a better immediate target than asynchronous storage. It does not show
that SSD latency will be unimportant for the full model.

## Move verification to the consumer

The optional `gpu-sha256` mode uses an independent implementation of SHA-256 from
[NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final). It changes
verification placement while preserving all original weight bytes and model
arithmetic:

1. On a source read, CPU whole-expert and six original tensor hashes must pass.
2. Only then derive an immutable SHA-256 table for consecutive 4 KiB chunks.
   The tables are lazy, counted in the host budget and released on close.
3. Every GPU weight upload hashes all 4,590 chunks on the destination GPU.
4. Return 146,880 digest bytes and compare every byte with the source table.
5. Publish the cache slot only after a match. Model computation cannot use a
   failed or partly uploaded slot.

The readback for an upload audit is **128 times smaller**, from 18,800,640 bytes
to 146,880 bytes. Ordinary weight promotion/demotion transfers still occur; this
is not a 128-fold reduction of all PCIe traffic. SHA-256 is a cryptographic
integrity check, not a mathematical proof against arbitrary hash collisions or
subsequent hardware faults.

RAM and other intermediate host copies defer their checks until GPU admission.
A corrupt RAM copy is therefore allowed to occupy RAM temporarily, but the
next upload must be rejected before GPU-slot publication or model arithmetic.
The request fails visibly; no substitute expert, rerouting or tolerance is used.
The original readback mode remains available and is the default baseline.

The 3080 Ti performs integer hashing of stored bytes only. It still performs no
model arithmetic, and its experimental BF16 expert path remains unqualified.
There is no asynchronous prefetch or transfer/compute overlap in this change.

## Exact-output and failure checks

Each mode passes the existing 232-call matrix for layer 0 experts 0, 1 and 2:
four input seeds, two scales, four route conditions, source-tier changes,
pressure refusal/recovery and forced reloads with zero RAM. Every call compares
all thirteen stages, including FP32 SwiGLU intermediates and final BF16 values.
Together, the two modes compare **2,375,680 final output values**, with zero
mismatches. This is repeated coverage of the same conditions under two policies,
not twice as many independent input conditions.

The model CUDA cubin SHA-256 is unchanged from the previous experiment:
`e4d5330f1f0f9893463370ba2361f4002b06f7fc496b94490fdbc8ef5ce1ccbf`.
The pinned official source and TileLang NVRTC handle-isolation workaround are
unchanged. Numerical gates still do not establish full-layer/model fidelity.

GPU hash validation compares 48 payload cases per GPU with Python `hashlib`:
random, zero and all-one bytes across sixteen lengths, including SHA padding
and 4 KiB boundaries and one complete expert. A known `abc` digest is checked
separately. Both GPUs pass all 96 payload cases and reject all ten deliberate
single-bit faults at offsets 0, 63, 4095, 4096 and the final expert byte. Restored
bytes must pass again. These are engineering checks, not FIPS certification.

CPU tests also cover corrupt RAM/uploads, publication refusal, source-table
allocation/cap failures and cleanup, immutable anchors and overlapping audit
buffers that could otherwise overwrite their own expected digests.

## Paired timing

The machine is the same Windows / Core i7-12700KF / RTX 4090 / RTX 3080 Ti setup
as the [initial report](first-probe.md#transfers-and-storage). The timing interpreter
uses CPython 3.13.2, NumPy 2.5.4 and `_hashlib.HASH` backed by OpenSSL 3.0.15.
The separate official-comparison environment and its versions are recorded in
each parity result. Vesper's own kernels still use CUDA 12.4 NVRTC.

Each of six trials runs both modes in alternating order. Each run first executes
one seven-call warmup, then times eight repetitions of
`0, 1, 0, 2, 2, 1, 0`: **56 timed calls per run, 336 per mode**. Both modes have
one compute-GPU slot, one helper-GPU slot and one RAM expert. All three identities
are retained after warmup. No timed call reads the store. Other workloads remain
running; the pair range describes observed variation, not a confidence interval.

| Source of requested expert | Readback median | GPU-audit median |
| --- | ---: | ---: |
| Compute GPU | 0.657 ms | 0.617 ms |
| Helper GPU | 43.094 ms | 7.625 ms |
| RAM | 65.507 ms | 13.162 ms |

The paired ratio uses the total call time of each entire trace, not ratios of
these source medians. Total timed call wall times are 13,664.84 ms for readback
and 2,720.33 ms for GPU audits. All 672 timed calls and 84 warmup calls match
baseline output fingerprints covering every returned array. Official-expert
comparisons are separate from this timing harness.

Call timing includes transfers, integrity checks, model kernels and diagnostic
copies. It excludes runtime setup, warmup, output-fingerprint comparisons and
the official oracle. Setup medians were 330.89 ms for readback and 1,341.79 ms
for GPU audits, including additional NVRTC compilation. Cold source reads also
pay for initial CPU digest preparation. These costs are excluded from the warm
ratio and may matter for short runs or low-locality workloads. An SSD-miss or
full-model speedup cannot be inferred.

## Buffer accounting

| Explicit state | Readback | GPU audits |
| --- | ---: | ---: |
| Host peak | 76,251,136 bytes | 76,985,536 bytes |
| RTX 4090 peak | 18,903,528 bytes | 19,050,408 bytes |
| RTX 3080 Ti peak | 18,800,640 bytes | 18,947,520 bytes |

The extra **734,400 host bytes (0.70 MiB)** are three original digest tables,
one portable pinned digest readback and one comparison scratch array. Each GPU
adds one 146,880-byte output buffer. Host-buffer cap remains 96 MiB and the
system-available floor remains 32 GiB; RAM cache capacity is unchanged. All
tracked buffers are released on exit. Python, driver/compiler objects and
returned diagnostics remain outside the ledger; this is not an OS memory cap.

## Reproduction and next work

Use the fixtures/store and optional reference setup from the
[cache validation report](cache-validation.md). To repeat hash validation and
paired timing without installing the optional official oracle:

```sh
vesper validate-integrity --devices 0 1 --output artifacts/integrity.json
vesper bench-cache --store artifacts/native-store --trials 6 --repeats 8 --output artifacts/cache-benchmark.json
```

Repeat the README's `validate-cache` command with `--profile` for the readback
baseline, then with `--integrity gpu-sha256 --profile` for the new mode. The full
[JSON records](../results/2026-10-18-integrity/README.md) include every stage check,
timed call, traffic counter, resource peak and cleanup result.

After this change, ordinary weight transfers and RAM copies dominate the warm
trace. Overlap is a future experiment, but realistic scheduling first needs the
unchanged router, shared expert and contribution aggregation. The synthetic
three-expert trace does not justify a routing predictor or a production capacity.
