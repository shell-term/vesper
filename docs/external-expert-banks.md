# Bounded PC expert banks on a separate SSD

The declared reasoning preparation reached the system SSD's **64 GiB free-space
reserve** after 69 complete single-query model calls: the unchanged 63-token
prompt and seven sampled continuation IDs. No EOS, completed preparation plan,
independent reference capture or quality score was produced by that attempt.
It is a capacity failure, not a completed reasoning evaluation.

The PC can explicitly place its immutable expert banks on a separate SSD.
Published indexes keep their original relative names and exact bytes. A private
mapping resolves those logical names to owned physical directories, with model,
revision, platform, profile and marker hashes checked before use. Physical paths
stay in ignored local configuration; execution receipts expose only identities
and digests. This does not transfer model qualification to other hardware.

## What changes

The two registered logical banks are `pc-heterogeneous/expert-bank-v1` and
`pc-heterogeneous/expert-bank-v2`. Migration retains the completed v1 objects as
inputs and selects a fresh v2 for subsequent preparation imports. Each bank keeps
its **64 GiB packed-payload cap**. There is no automatic bank rollover. The
selected SSD must admit the copied files, the entire fresh 64 GiB bank, and its
64 GiB free-space reserve before copying starts. Each import still checks free
space, uses at most four expert owners/eight transfers, and preserves every
original weight/scale byte. Host-memory caps and the 32 GiB available-RAM floor
are unchanged.

Run files and expert payloads have separate admissions. When the active bank is
on another filesystem, the preparation run admits **1 GiB** on its own volume:
the bounded embedding/row caches, aligned frozen rows, two copies of the forty
index manifests, capped plan/report metadata, scratch and progress allowance
sum to less than that. Co-located run/bank storage retains the earlier 8 GiB
preparation admission. Each independent capture separately admits its sealed
actual call count times the 4 MiB per-call maximum, plus 64 MiB for its report
and 4 MiB for its context. The original framework capture additionally admits
**8 GiB of startup headroom**: its framework/compiler/driver activity is not
bounded by the record census. A reference attempt was interrupted during its
first token when system-volume free space fell below 64 GiB after admission.
The small explicitly directed build cache did not explain that decrease; the
responsible system allocation has not been established.

Preparation and both capture runners now check the **64 GiB available-disk
floor** at every progress checkpoint, failing the owned run before publishing
that checkpoint if space has fallen. Plans/reports and row metadata also refuse
their bounds before publication. These are cooperative admissions/checkpoints,
not OS quotas or protection against another application's intervening writes.
The task and output budget do not change when storage moves.

The [source resolver](../src/vesper/source_roots.py) rejects unregistered roots,
foreign markers, overlapping roots, changed configuration, traversal and linked
mounts. A process lease prevents cooperating PC captures and migration from
running concurrently. The original/native comparison remains offline; missing
original experts cannot trigger a network fallback.

## Copy, verify and activate

The [migration tool](../src/vesper/bank_migration.py) first inventories only the
closed, single-expert objects in the PC's v1 bank. It never moves unrelated
files or the legacy shared fixtures. The saved private plan binds every relative
object, original manifest hash, payload hash and byte count.

Execution copies into a dedicated owned directory, independently reads and
verifies every target file, then activates the logical mapping. The default
retains every original source file and verifies both copies again. Source
retirement requires the explicit `--retire-verified-source` option; only then
does the tool remove each exact verified original file. It never recursively
deletes a source tree or replaces a different destination file.
An interruption after activation can resume without recopying data;
the existing indexes remain readable throughout that resumed phase. Unexpected
source contents, altered copies or a different ownership journal refuse.

Returned receipts describe verified file relocation. They are not GPU execution,
answer scoring or a model-quality gate. Their source context identifies the
actual migration; later model captures retain their own execution contexts and
bind the selected source-root identity.

The [file-bound PC evidence](../results/pc-heterogeneous/2026-10-20-expert-bank-ssd/summary.json)
retains the failed preparation inspection and the actual independent file-copy
receipt. The copy verifies **3,375 objects / 63,452,160,000 packed payload bytes**
and **63,467,411,542 total file bytes**, with all **6,750 original source files
retained** and **zero source bytes deleted or weight bytes converted**. Its
execution context records source `461ba64`; the summary is explicitly metadata
publication. The earlier failed GPU preparation has no completed execution
receipt and is not retrospectively stamped as a successful run. Reasoning
quality remains open until a fresh complete original/native comparison passes.

The [later retirement evidence](../results/pc-heterogeneous/2026-10-20-expert-bank-retirement/summary.json)
records a separately authorized cleanup at source `cd0407c`. It rechecks the
same 3,375 copied objects and immediately verifies both copies before removing
each original source file: **6,750 files / 63,467,411,542 bytes** retired, zero
original source files left, zero new copies and zero weight bytes converted.
The mounted v1/v2 census remains **3,375 / 552 objects**, and logical root/index
identities are unchanged. The earlier retained-copy receipt remains intact at
its original execution context. This cleanup is CPU file evidence; the
interrupted first-token reference attempt remains unscored and must restart.

## Reproduction

Select the [PC workspace](../platforms/pc-heterogeneous/README.md), stop its own
capture normally, and pass `python -m vesper.workspace check`. Choose a dedicated
new directory on a suitable SSD. Existing nonempty, unowned directories refuse.
The plan contains private absolute paths and belongs under ignored PC artifacts.

```powershell
$ssdDirectory = Read-Host "Absolute new directory on the SSD"
python -m vesper.bank_migration plan --ssd-directory $ssdDirectory --plan artifacts/pc-heterogeneous/management/bank-migration-plan.json
python -m vesper.bank_migration execute --retain-source --plan artifacts/pc-heterogeneous/management/bank-migration-plan.json --receipt artifacts/pc-heterogeneous/management/bank-migration-receipt.json
```

Review the plan before executing. Source retention leaves the original volume's
disk use unchanged. Retire verified source copies only when that cleanup is
explicitly intended and authorized. To resume an interrupted execution, use the
same plan and a fresh receipt filename. Do not replace the plan or manually
change active roots during a capture. Failed preparation can reuse only verified
input caches and completed immutable objects; start model/hash/RNG state afresh.
Keep the original task, reasoning effort and output budget intact.

CPU tests cover unchanged schema-1 index bytes and actual full-sized direct
reads after relocation, corruption rejection, partial-copy recovery, partial
retirement recovery, refusal before mutation on insufficient capacity, and
capture/migration exclusion. Windows privilege-dependent link checks are
reported separately; CPU tests alone do not establish model-level fidelity.
