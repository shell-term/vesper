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

The [source resolver](../src/vesper/source_roots.py) rejects unregistered roots,
foreign markers, overlapping roots, changed configuration, traversal and linked
mounts. A process lease prevents cooperating PC captures and migration from
running concurrently. The original/native comparison remains offline; missing
original experts cannot trigger a network fallback.

## Copy, verify, activate, retire

The [migration tool](../src/vesper/bank_migration.py) first inventories only the
closed, single-expert objects in the PC's v1 bank. It never moves unrelated
files or the legacy shared fixtures. The saved private plan binds every relative
object, original manifest hash, payload hash and byte count.

Execution copies into a dedicated owned directory, independently reads and
verifies every target file, then activates the logical mapping. Only after that
does it verify both copies again and remove each exact original file. It never
recursively deletes a source tree or replaces a different destination file.
An interruption after activation can resume retirement without recopying data;
the existing indexes remain readable throughout that resumed phase. Unexpected
source contents, altered copies or a different ownership journal refuse.

Returned receipts describe verified file relocation. They are not GPU execution,
answer scoring or a model-quality gate. Their source context identifies the
actual migration; later model captures retain their own execution contexts and
bind the selected source-root identity.

## Reproduction

Select the [PC workspace](../platforms/pc-heterogeneous/README.md), stop its own
capture normally, and pass `python -m vesper.workspace check`. Choose a dedicated
new directory on a suitable SSD. Existing nonempty, unowned directories refuse.
The plan contains private absolute paths and belongs under ignored PC artifacts.

```powershell
$ssdDirectory = Read-Host "Absolute new directory on the SSD"
python -m vesper.bank_migration plan --ssd-directory $ssdDirectory --plan artifacts/pc-heterogeneous/management/bank-migration-plan.json
python -m vesper.bank_migration execute --plan artifacts/pc-heterogeneous/management/bank-migration-plan.json --receipt artifacts/pc-heterogeneous/management/bank-migration-receipt.json
```

Review the plan before executing. To resume an interrupted execution, use the
same plan and a fresh receipt filename. Do not replace the plan or manually
change active roots during a capture. Failed preparation can reuse only verified
input caches and completed immutable objects; start model/hash/RNG state afresh.
Keep the original task, reasoning effort and output budget intact.

CPU tests cover unchanged schema-1 index bytes and actual full-sized direct
reads after relocation, corruption rejection, partial-copy recovery, partial
retirement recovery, refusal before mutation on insufficient capacity, and
capture/migration exclusion. Windows privilege-dependent link checks are
reported separately; CPU tests alone do not establish model-level fidelity.
