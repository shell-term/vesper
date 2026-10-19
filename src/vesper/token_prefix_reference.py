"""Independent original four-block prefix and diagnostic head, never generation."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .attention_memory_reference import AttentionMemoryReference
from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .block_reference import BlockOracle
from .block_pair_reference import BlockPairOracle
from .connected_reference import OfficialConnected
from .engram_reference import EngramOracle
from .engram_reference_storage import VirtualTables
from .hc_reference import HCOracle
from .model_artifacts import read_checked
from .normalization_reference import ReferenceMath
from .output_reference import OutputOracle
from .projection_reference import ProjectionOracle
from .sampling_reference import original_sample
from .token_prefix import LAYERS, TOKENS
from .token_prefix_runtime import moe_files
from .topology_reference import SharingOracle
from .token_prefix_validation import capture, identity, goldens, scope
from .topology_validation import state_digest
from .token_prefix_runtime import load_arrays
from .output_assets import load as load_head
from .memory import RuntimeMemory, MemoryPolicy
from .resources import MemoryBudget, GiB


def original_hash(source, *, max_seq_len=4):
    import torch
    import transformers
    source = Path(source)
    read_checked(source, "inference/engram.py")
    name = "vesper_original_token_prefix_hash"
    spec = importlib.util.spec_from_file_location(name, source / "engram.py")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    if type(max_seq_len) is not int or not 4 <= max_seq_len <= 1152: raise ValueError("Expected a bounded original hash history")
    args = SimpleNamespace(**json.loads(read_checked(source, "inference/config.json")), max_batch_size=1, max_seq_len=max_seq_len)
    with torch.device("cpu"):
        layout = module.EngramLayout.from_args(args)
        state = module.NgramHashState(args, layout, tokenizer)
    return state


class TokenPrefixOracle:
    def __init__(self, stack, root, arrays, embeddings, head_arrays, ordinal=0):
        import torch
        self.torch, self.ordinal, self.embeddings = torch, ordinal, embeddings
        root = Path(root); directory = root / "reference"
        self.hash = original_hash(root / "import-source")
        self.memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile="long_context"))
        self.qualification = self.memory.qualify()
        if self.qualification["status"] != "pass":
            raise RuntimeError("Token prefix requires a qualified original arithmetic Attention adapter")
        self.hc = {l: HCOracle(directory, arrays, ordinal, layer=l, profile="token_prefix") for l in LAYERS}
        self.p = {l: ProjectionOracle(directory, arrays, ordinal, layer=l) for l in (0, 1)}
        self.moe = {l: OfficialConnected(*moe_files(root, l), root / f"token-prefix-experts/layer{l}", directory,
            ordinal, layer=l, profile="token_prefix") for l in LAYERS}
        self.blocks = {l: BlockOracle(arrays, self.hc[l], self.p[l], self.memory, self.moe[l]) for l in (0, 1)}
        for block in self.blocks.values(): block.reset()
        self.sharing = SharingOracle(directory, arrays, (2, 3), self.memory, ordinal)
        stack.callback(self.sharing.close)
        self.norm = ReferenceMath(directory, {n: arrays[n] for l in (2, 3)
            for n in (f"layers.{l}.attn_norm.weight", f"layers.{l}.ffn_norm.weight")}, ordinal)
        self.pair = BlockPairOracle(arrays, self.hc, self.sharing, self.norm, self.moe); self.pair.reset()
        virtual = stack.enter_context(VirtualTables(ordinal))
        self.engram = EngramOracle(directory, root / "import-source", root / "engram-fixture", virtual, ordinal)
        self.head = OutputOracle(directory, head_arrays, ordinal)
        self.position = 0
        torch.cuda.manual_seed(2026)
        self.last_before_engram = self.last_result = None

    def step(self, ids):
        torch, pos = self.torch, self.position
        tokens = torch.from_numpy(ids.copy()).to("cpu")
        hashes = self.hash(tokens, pos).numpy().copy()
        compressed = self.hash.cache[:1, pos:pos + 1].numpy().copy()
        hidden = np.repeat(self.embeddings[int(ids[0, 0])].reshape(1, 1, 1, 5120), 4, axis=2)
        pre = _numpy(self.p[0].scope["make_identity_pre_mix"](self.hc[0].tensor(hidden), 4))
        result = {"embedding_bf16": hidden.copy(), "initial_pre_fp32": pre.copy(),
            "hash_ids_i64": hashes, "compressed_i64": compressed}
        # Read actual original hash inputs from its retained compressed history.
        pad = int(self.hash.pad_id)
        result["lookbacks_i64"] = np.array([[[int(self.hash.cache[0, pos - shift]) if pos >= shift else pad
            for shift in range(4)]]], np.int64)
        carry = [hidden, pre]
        for layer in (0, 1):
            if layer == 1:
                self.last_before_engram = carry[0].copy()
                result["engram1"] = self.engram.step(1, carry[0], np.ascontiguousarray(hashes[:, :, 0, :]))
                carry[0] = result["engram1"]["output_bf16"]
            initial = {"input_hidden_bf16": carry[0].copy(), "incoming_pre_fp32": carry[1].copy()}
            block = self.blocks[layer].step(*carry, False, pos)
            result[f"layer{layer}"] = {**initial, **block}
            carry = [block["output_bf16"], block["next_pre_mix_fp32"]]
        pair, state = self.pair.step(*carry, False, pos)
        result.update({f"layer{l}": pair[f"layer{l}"] for l in (2, 3)})
        result["sharing_attention"] = pair["attention"]
        result["output_bf16"], result["next_pre_mix_fp32"] = pair["output_bf16"], pair["next_pre_mix_fp32"]
        result["probe_head"] = self.head.evaluate(result["output_bf16"], result["next_pre_mix_fp32"])
        sample, before, after = original_sample(torch, self.head.sample, result["probe_head"]["logits_fp32"], .7)
        result["probe_sampling"] = {**sample, "offset_before": before, "offset_after": after}
        self.position += 1; self.last_result = result
        return result, {"window0": _numpy(self.blocks[0].original_attention.window_kv_cache),
            "window1": _numpy(self.blocks[1].original_attention.window_kv_cache), "sharing": state,
            "history": self.hash.cache[:1, max(0, self.position - 3):self.position].numpy().copy(),
            "position": self.position, "rng_offset": result["probe_sampling"]["offset_after"]}

    def negatives(self):
        result = self.last_result
        x = result["layer1"]["attn_norm_output_bf16"]
        wrong = self.p[0].fp8("wq_a", x)["output_bf16"]
        stale = self.hc[2].pre(result["layer2"]["input_hidden_bf16"], result["layer0"]["next_pre_mix_fp32"])["output_bf16"]
        controls = [check_arrays(self.last_before_engram, result["engram1"]["output_bf16"], "omit_Engram_before_block1"),
            check_arrays(wrong, result["layer1"]["attention_wq_a_output_bf16"], "use_layer0_weights_for_window1"),
            check_arrays(stale, result["layer2"]["attn_pre_output_bf16"], "reuse_stale_pre_mix_at_block2")]
        if any(c["status"] != "fail" or not c.get("mismatches") for c in controls):
            raise RuntimeError("Token-prefix negative control failed to distinguish altered semantics")
        return controls


def capture_original(root, progress=None):
    import torch
    root = Path(root)
    plan = json.loads((root / "token-prefix-plan.json").read_bytes())
    if (plan["model"], plan["revision"], plan["layers"], plan["tokens"], plan["status"]) != (
            scope()["model"], scope()["revision"], list(LAYERS), list(TOKENS), "pass"):
        raise ValueError("Unpinned token-prefix preparation plan")
    previous = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=24 * GiB)) as policy:
            with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(0), torch.device("cuda:0"):
                budget = MemoryBudget(cap=8 * GiB, owner="token_prefix_original_fixture_and_framework")
                budget.acquire(budget.cap); stack.callback(budget.release, budget.cap)
                arrays, embeddings, fixtures = load_arrays(root)
                head_arrays, output_fixture = load_head(root / "output-fixture")
                stack.callback(arrays.clear); stack.callback(head_arrays.clear)
                oracle = TokenPrefixOracle(stack, root, arrays, embeddings, head_arrays)
                if (oracle.engram.manifests["1"]["controls"] != plan["additional_identities"]["Engram1"]
                        or oracle.engram.manifests["1"]["rows"] != plan["additional_identities"]["rows1"]):
                    raise ValueError("Independent original Engram fixture differs from preparation")
                from .model_artifacts import digest
                if digest(oracle.hash.token_map.numpy().astype(np.int64).tobytes()) != plan["additional_identities"]["tokenizer"]["compressed_map_sha256"]:
                    raise ValueError("Independent original compressed vocabulary differs")
                cases = []
                for pos, token in enumerate(TOKENS):
                    result, state = oracle.step(np.array([[token]], np.int64))
                    cases.append({"meta": {"position": pos, "token_id": token}, "state_sha256": state_digest(state),
                        "boundaries": capture(result, state),
                        "ordered_routes": {str(l): {"indices": result[f"layer{l}"]["moe_gate.indices"].tolist(),
                            "weight_bits": result[f"layer{l}"]["moe_gate.weights"].view(np.uint32).tolist()} for l in LAYERS}})
                    if progress: progress({"original_prefix_position": pos, "boundaries": len(cases[-1]["boundaries"]), "status": "pass"})
                negatives = oracle.negatives()
                report = {"schema_version": 1, "probe": "original_connected_token_prefix_goldens", "status": "pass", "scope": scope(),
                    "fixtures": fixtures, "additional_identities": plan["additional_identities"], "output_fixture": output_fixture,
                    "expert_plan": plan["experts"], "cases": cases, "negative_controls": negatives,
                    "reference_methods": {"HC": {str(l): o.methods for l, o in oracle.hc.items()},
                        "window_projection": {str(l): o.methods for l, o in oracle.p.items()}, "sharing": oracle.sharing.methods,
                        "Engram": oracle.engram.methods, "output": oracle.head.methods},
                    "original_controls": {"blocks0_1": {str(l): b.controls for l, b in oracle.blocks.items()},
                        "blocks2_3": oracle.pair.block_controls, "attention2_3": oracle.pair.attention_controls,
                        "Engram": oracle.engram.controls}, "original_attention_adapter": oracle.qualification,
                    "original_grouped_conversion": {str(l): (oracle.p[l] if l < 2 else oracle.sharing.p[l]).converted_sha256 for l in LAYERS},
                    "weight_payloads_fetched_in_reference": 0}
                report["identity_sha256"], report["complete_goldens_sha256"] = identity(report), goldens(report)
                # Drop GPU framework owners before the memory policy closes.
                for moe in oracle.moe.values(): moe.experts.clear()
                del result, state, moe, oracle
            import gc
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
            report["memory_before_policy_close"] = policy.snapshot()
        report["memory_after_policy_close"] = policy.snapshot()
        return report
    finally:
        torch.set_default_dtype(previous)
