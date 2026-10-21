"""Bounded, explicitly unqualified Spark composition of shared native operators.

Only weight/operator residency changes here. NativeBackbone retains all forty
blocks, both Engrams, transactional KV/history/sampling, and the complete head.
One HC and one projection operator may be resident at a time. Original expert
routes are resolved offline over all 384 experts; no route replay or fetch is
used during forward. Component execution does not establish model quality.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time

import numpy as np

from .attention_topology import BACKBONE_COMPRESSED_LAYERS, INDEX_SOURCES, KV_SOURCES
from .backbone import LAYERS, NativeBackbone
from .compression import NativeCompression
from .checkpoint import MODEL, REVISION
from .connected import ConnectedConfig, ConnectedMoE
from .engram import NativeEngram
from .execution_target import spark_kernel_cache
from .hc import NativeHC
from .indexer import NativeIndexer
from .memory import MemoryPolicy, current_policy
from .ngram import NgramState
from .normalization import NativeNormRoPE
from .original_engram_rows import OriginalShardRows
from .projections import NativeProjections
from .resources import GiB, MiB
from .sampling import NativeSampler
from .shared_attention import SharedAttentionFrame
from .spark_artifacts import OriginalShardExpertStore
from .spark_attention import SparkAttentionCandidate
from .spark_output import SparkOutputCandidate
from .tiered import CacheConfig
from .tokenizer import PinnedTokenizer, compressed_token_map
from .window_attention import NativeWindowAttention

TARGET = "spark-sm121-candidate"

# Short S1 composition: legacy component admissions (including control,
# per-step diagnostics and transient catalog parsing) need about 5.3 GiB.
# Keep 6 GiB for those conservative, possibly overlapping admissions, 1 GiB
# for leased original assets, and the existing 5 GiB device ceiling. These
# are one shared physical-pool ledger, never RAM plus additional VRAM.
SPARK_COMPONENT_ALLOWANCE = 6 * GiB
SPARK_ASSET_CAP = GiB
SPARK_DEVICE_CAP = 5 * GiB
SPARK_SHARED_CAP = SPARK_COMPONENT_ALLOWANCE + SPARK_ASSET_CAP + SPARK_DEVICE_CAP


def _layer(layer):
    if type(layer) is not int or layer not in LAYERS:
        raise ValueError("Expected one original backbone layer")
    return layer


@dataclass(frozen=True)
class LayerProxy:
    """Immutable operator identity; opened means the owning lazy pool is open.

    It does not claim every GPU operator has been materialized. The shared
    composition checks identity/readiness without causing eager weight loads.
    Calls acquire an actual opened operator while the pool lock is held.
    """
    pool: object
    kind: str
    layer: int

    @property
    def opened(self):
        return self.pool.opened and not self.pool.poisoned

    @property
    def ordinal(self):
        return self.pool.ordinal

    def __getattr__(self, name):
        allowed = {"hc": {"mixes", "pre", "post"},
                   "projection": {"fp8", "grouped", "round_window"}, "moe": {"forward"}}
        if name not in allowed.get(self.kind, set()):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.pool.call(self.kind, self.layer, name, *args, **kwargs)


class LayerOperatorPool:
    """Serialize layer switches; close old ownership before creating new work."""
    def __init__(self, assets, ordinal=0, *, hc_factory=None, projection_factory=None, moe_factory=None, progress=None,
                 backend_pool=None):
        self.assets, self.ordinal = assets, ordinal
        self.factories = {"hc": hc_factory or NativeHC, "projection": projection_factory or NativeProjections}
        self.backend_pool = backend_pool
        self.native_factories = {"hc": hc_factory is None, "projection": projection_factory is None}
        self.moe_factory = moe_factory or self._moe
        self.lock, self.opened, self.poisoned = threading.RLock(), False, False
        self.initialized, self.active = False, {}
        self.moe_owner = None
        self.moe_entered = False
        self.counts = {"hc_opened": 0, "projection_opened": 0, "moe_opened": 0, "calls": 0}
        self.peak_resident = {"hc": 0, "projection": 0}
        self.progress = progress
        self.timings = {kind:{"open_calls":0,"open_seconds":0.,"compute_seconds":0.} for kind in ("hc","projection","moe")}

    def _notify(self, kind, layer, phase):
        if self.progress is not None:
            self.progress({"kind":kind,"layer":layer,"phase":phase})

    def __enter__(self):
        with self.lock:
            if self.opened or self.initialized:
                raise RuntimeError("Layer pools open once")
            self.assets.workspace.check()
            self.initialized, self.opened = True, True
            return self

    def proxy(self, kind, layer):
        _layer(layer)
        if kind not in ("hc", "projection", "moe"):
            raise ValueError("Unknown streamed operator kind")
        return LayerProxy(self, kind, layer)

    def _ready(self):
        if not self.opened or self.poisoned:
            raise RuntimeError("The layer pool is closed or requires cleanup")
        self.assets.workspace.check()

    def _close_kind(self, kind):
        owned = self.active.get(kind)
        if owned is not None:
            try:
                owned[2].close()
            except BaseException:
                self.poisoned = True
                raise
            del self.active[kind]

    def _operator(self, kind, layer):
        current = self.active.get(kind)
        if current is not None and current[0] == layer:
            return current[1]
        self._close_kind(kind)
        owner = ExitStack()
        started = time.perf_counter()
        self._notify(kind, layer, "operator_open")
        try:
            with self.assets.upload_arrays(layer) as arrays:
                reuse = ({"backend_pool": self.backend_pool}
                    if self.backend_pool is not None and self.native_factories[kind] else {})
                operator = owner.enter_context(self.factories[kind](arrays, self.ordinal, layer=layer,
                    profile="backbone_probe", execution_target=TARGET, **reuse))
            if not operator.opened or operator.layer != layer or operator.ordinal != self.ordinal:
                raise ValueError("Streamed operator identity differs")
        except BaseException:
            owner.close()
            raise
        self.active[kind] = layer, operator, owner
        self.timings[kind]["open_calls"] += 1
        self.timings[kind]["open_seconds"] += time.perf_counter() - started
        self.counts[kind + "_opened"] += 1
        self.peak_resident[kind] = 1
        return operator

    def _moe(self, layer):
        gate, router, shared = self.assets.moe_paths(layer)
        cache = CacheConfig(host_cap_bytes=128 * MiB, ram_cache_bytes=0,
            primary_device=self.ordinal, primary_slots=1, helper_device=None, helper_slots=0,
            device_cap_bytes=64 * MiB, metadata_allowance_bytes=MiB, integrity_mode="readback",
            max_store_experts=48)
        # max_store_experts applies only to the legacy repacked NativeStore.
        # The explicit source factory validates all 384 original experts instead.
        def source_factory(_, budget):
            catalog = getattr(self.assets, "catalog", None)
            reuse = {} if catalog is None else {"catalog": catalog}
            return OriginalShardExpertStore(self.assets.metadata, self.assets.originals,
                self.assets.integrity, budget, layer=layer, workspace=self.assets.workspace, **reuse)
        return ConnectedMoE(self.assets.originals, gate, router, shared,
            ConnectedConfig(cache=cache), layer=layer, profile="backbone_probe", source_factory=source_factory)

    def _close_moe(self, error=None):
        if self.moe_owner is not None:
            try:
                if self.moe_entered:
                    self.moe_owner.__exit__(type(error) if error is not None else None, error,
                                            error.__traceback__ if error is not None else None)
                else:
                    # ConnectedMoE normally closes its own stack on failed
                    # entry. If that cleanup itself failed, retry only that
                    # stack: its normal __exit__ diagnostics require fields
                    # which a failed constructor may never have created.
                    owner_stack = getattr(self.moe_owner, "stack", None)
                    if owner_stack is not None:
                        owner_stack.close()
                        self.moe_owner.stack = None
            except BaseException:
                self.poisoned = True
                raise
            self.moe_owner = None
            self.moe_entered = False

    def call(self, kind, layer, method, *args, **kwargs):
        _layer(layer)
        with self.lock:
            self._ready()
            if kind == "moe":
                if method != "forward":
                    raise ValueError("Unsupported streamed MoE operation")
                # No cache-dependent gate changes or retained stale accumulator.
                self.moe_owner = self.moe_factory(layer)
                self.moe_entered = False
                started = time.perf_counter()
                self._notify(kind, layer, "operator_open")
                try:
                    operator = self.moe_owner.__enter__()
                    self.moe_entered = True
                    self.timings[kind]["open_calls"] += 1
                    self.timings[kind]["open_seconds"] += time.perf_counter() - started
                except BaseException:
                    if getattr(self.moe_owner, "stack", None) is not None:
                        self.poisoned = True
                    else:
                        self.moe_owner = None
                    raise
                error = None
                started = time.perf_counter()
                self._notify(kind, layer, "operator_compute")
                try:
                    if operator.layer != layer:
                        raise ValueError("Streamed MoE identity differs")
                    self.counts["moe_opened"] += 1
                    result = operator.forward(*args, **kwargs)
                except BaseException as failure:
                    error = failure
                    raise
                finally:
                    self._close_moe(error)
                    self.timings[kind]["compute_seconds"] += time.perf_counter() - started
                self._notify(kind, layer, "layer_complete")
            else:
                if kind not in self.factories or method not in (
                        {"mixes", "pre", "post"} if kind == "hc" else {"fp8", "grouped", "round_window"}):
                    raise ValueError("Unsupported streamed weight operation")
                operator = self._operator(kind, layer)
                started = time.perf_counter()
                self._notify(kind, layer, "operator_compute")
                try:
                    result = getattr(operator, method)(*args, **kwargs)
                finally:
                    self.timings[kind]["compute_seconds"] += time.perf_counter() - started
            self.counts["calls"] += 1
            return result

    def stats(self):
        return {**self.counts, "active_layers": {kind: value[0] for kind, value in self.active.items()},
                "maximum_resident_operators": dict(self.peak_resident),
                "moe_cleanup_owner_retained": self.moe_owner is not None,
                "opened": self.opened, "poisoned": self.poisoned,
                "expert_ram_cache_bytes": 0, "helper_device": None,
                "host_wall_timings": {kind:dict(values) for kind,values in self.timings.items()},
                "arithmetic_qualification": "not_run", "network_in_forward": False}

    def __exit__(self, *_):
        with self.lock:
            errors = []
            try:
                self._close_moe()
            except BaseException as error:
                errors.append(error)
            for kind in list(self.active):
                try:
                    self._close_kind(kind)
                except BaseException as error:
                    errors.append(error)
            self.opened = False
            if errors:
                raise RuntimeError("Streamed operator cleanup failed") from errors[0]


class EmbeddingProvider:
    """Admit any vocabulary ID, then validate that one verified original row.

    This is an explicit lazy provider, not a fixed dictionary with invented
    items. NativeBackbone's fixed-dictionary validation is bypassed only by the
    Spark subclass, which validates the actual row on every access.
    """
    def __init__(self, read):
        if not callable(read):
            raise ValueError("Expected a verified offline embedding row reader")
        self.read = read

    def __contains__(self, token):
        return type(token) is int and 0 <= token < 129280

    def __getitem__(self, token):
        if token not in self:
            raise KeyError("Token is outside the original vocabulary")
        value = self.read(token)
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint16 or value.shape != (5120,)
                or not value.flags.c_contiguous or np.any((value & 0x7f80) == 0x7f80)):
            raise ValueError("Expected one finite contiguous original BF16 embedding row")
        return value


class SparkBackboneCandidate(NativeBackbone):
    def __init__(self, *args, embedding_reader, pool, **kwargs):
        # The shared constructor still validates all forty operators and state
        # owners. Only its explicit prepared-embedding dictionary is replaced.
        super().__init__(*args, **kwargs)
        self.embeddings = EmbeddingProvider(embedding_reader)
        self.operator_pool = pool

    def stats(self):
        return {"execution_target": TARGET, "qualification": "unqualified_candidate",
                "all_backbone_layers": list(LAYERS), "engram_layers": [1, 14],
                "original_kernel_parity_qualified": False, "full_model_quality_qualified": False,
                "state": self.transaction.stats(), "layer_pool": self.operator_pool.stats(),
                "engram_operators": {str(layer):op.stats() for layer,op in self.engrams.items()},
                "head_operator": self.head.stats(), "sampler_operator": self.transaction.sampler.stats(),
                "compiled_kernel_cache": (self.compilation_cache.stats() if hasattr(getattr(self,"compilation_cache",None),"stats") else None),
                "preparation_backends": (self.preparation_backends.stats() if getattr(self,"preparation_backends",None) is not None else None),
                "network_in_forward": False, "trajectory_replay": False}

    def __exit__(self, *error):
        try:
            return super().__exit__(*error)
        finally:
            # The row may carry an assets memory lease. Drop our ownership
            # before the caller closes assets and then the shared governor.
            self.norm_weight = None


def spark_memory_policy():
    """Enter RuntimeMemory with this policy before opening SparkModelAssets."""
    return MemoryPolicy(host_cap_bytes=SPARK_SHARED_CAP, pinned_cap_bytes=128 * MiB,
                        device_caps={0:SPARK_DEVICE_CAP}, unified_device_memory=True)


def open_spark_backbone(stack, assets, attention_bundle, *, temperature=.7, seed=2026, progress=None,
                        reuse_profile="baseline"):
    """Compose a bounded candidate in a caller-owned ExitStack.

    Assets must already be open and locally verified. This function performs no
    checkpoint download. It deliberately does not use the eight-call selected
    fixture TextGeneration limit; callers choose an explicit short functional
    request and retain EOS/quality as separate S1/S2 acceptance gates.
    """
    assets.workspace.check()
    if reuse_profile not in ("baseline", "s2"):
        raise ValueError("Expected an explicit baseline or s2 preparation profile")
    context = assets.workspace.context()
    if tuple(context[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("Spark composition requires its explicitly bound native workspace")
    assets.workspace.check_runtime_module(__file__, "src/vesper/spark_backbone_runtime.py")
    policy = current_policy()
    if policy is None:
        raise ValueError("Open the unified runtime memory policy before opening model assets")
    elif not policy.config.unified_device_memory:
        raise ValueError("Spark operators require one shared physical host/device memory ceiling")
    compilation = stack.enter_context(spark_kernel_cache())
    backends = None
    if reuse_profile == "s2":
        from .spark_operator_backends import SparkOperatorBackends
        backends = stack.enter_context(SparkOperatorBackends(assets.workspace))
    pool = stack.enter_context(LayerOperatorPool(assets, progress=progress, backend_pool=backends))
    hc = {layer: pool.proxy("hc", layer) for layer in LAYERS}
    projections = {layer: pool.proxy("projection", layer) for layer in LAYERS}
    moe = {layer: pool.proxy("moe", layer) for layer in LAYERS}
    norm = stack.enter_context(NativeNormRoPE(execution_target=TARGET))
    sparse = stack.enter_context(SparkAttentionCandidate(attention_bundle, profile="long_context"))
    compression = stack.enter_context(NativeCompression(assets.compression_arrays(), layers=KV_SOURCES, execution_target=TARGET))
    indexer = stack.enter_context(NativeIndexer(assets.indexer_arrays(), layers=INDEX_SOURCES, execution_target=TARGET))
    windows = {layer: stack.enter_context(NativeWindowAttention(assets.arrays, projections[layer], norm, sparse,
        profile="backbone_probe", execution_target=TARGET)) for layer in (0, 1)}
    sharing = stack.enter_context(SharedAttentionFrame(assets.arrays,
        {layer: projections[layer] for layer in BACKBONE_COMPRESSED_LAYERS}, norm, compression, indexer, sparse,
        BACKBONE_COMPRESSED_LAYERS, binding_profile="backbone_probe"))
    engrams = {}
    for layer in (1, 14):
        rows = stack.enter_context(OriginalShardRows(assets.metadata, assets.originals, assets.integrity, layer))
        engrams[layer] = stack.enter_context(NativeEngram(assets.engram_controls(layer), rows, execution_target=TARGET))
    mapping, tokenizer_identity = compressed_token_map(PinnedTokenizer(assets.metadata))
    sampler = stack.enter_context(NativeSampler(seed, execution_target=TARGET))
    head = stack.enter_context(SparkOutputCandidate(assets.head_rows, block_rows=512))
    engine = stack.enter_context(SparkBackboneCandidate(assets.arrays, {}, hc, norm, windows,
        sharing, moe, engrams, NgramState(mapping), sampler, head, assets.norm_weight(),
        temperature=temperature, embedding_reader=assets.embedding, pool=pool))
    engine.tokenizer_identity = tokenizer_identity
    engine.memory_policy = policy
    engine.compilation_cache = compilation
    engine.preparation_backends = backends
    return engine
