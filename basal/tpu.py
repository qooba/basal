"""Google TPU backend
"""
import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .engine import EagerBackend, GraphBackend
from .prompt import PREFILL

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def load_safetensors(path):
    """Memory-map a .safetensors file into numpy arrays without torch (the safetensors numpy API cannot read BF16)."""
    from ml_dtypes import bfloat16
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", mm[:8].tobytes())[0]
    header = json.loads(mm[8: 8 + n].tobytes())
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = bfloat16 if meta["dtype"] == "BF16" else _ST_DTYPES.get(meta["dtype"])
        if dt is None:
            raise ValueError(f"{path}: unsupported safetensors dtype {meta['dtype']} ({name})")
        a, b = meta["data_offsets"]
        out[name] = mm[8 + n + a: 8 + n + b].view(dt).reshape(meta["shape"])
    return out


def model_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if cfg.get("model_type") != "llama":
        raise ValueError(f"TPU backend supports Llama-architecture checkpoints, got {cfg.get('model_type')!r}")
    rope = cfg.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or cfg.get("rope_scaling"):
        raise ValueError(f"TPU backend supports default RoPE only, got {rope or cfg.get('rope_scaling')}")
    nh = cfg["num_attention_heads"]
    return dict(layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"], ffn=cfg["intermediate_size"], heads=nh,
                kv_heads=cfg.get("num_key_value_heads", nh), head_dim=cfg.get("head_dim") or cfg["hidden_size"] // nh,
                eps=cfg.get("rms_norm_eps", 1e-6), theta=float(rope.get("rope_theta", cfg.get("rope_theta", 10000.0))))


def stack_params(w, c, dtype="bfloat16"):
    """HF-named weights -> host pytree with the decoder layers stacked on axis 0, q / k / v fused, linear weights
    [in, out]. Biases are always present (zeros when the checkpoint has none), so the compiled program does not depend
    on the checkpoint."""
    from ml_dtypes import bfloat16
    dt = {"bfloat16": bfloat16, "float32": np.float32}[dtype]
    L, H, F = c["layers"], c["hidden"], c["ffn"]
    Q, KV = c["heads"] * c["head_dim"], c["kv_heads"] * c["head_dim"]
    shapes = dict(ln1=(H,), wqkv=(H, Q + 2 * KV), bqkv=(Q + 2 * KV,), wo=(Q, H), bo=(H,), ln2=(H,),
                  wg=(H, F), bg=(F,), wu=(H, F), bu=(F,), wd=(F, H), bd=(H,))
    lay = {k: np.zeros((L, *s), dt) for k, s in shapes.items()}

    def get(name):
        return np.asarray(w[name]).astype(dt)

    def bias(name, out, sl=slice(None)):
        if name in w:
            out[sl] = get(name)

    def fill(i):
        p = f"model.layers.{i}"
        lay["ln1"][i], lay["ln2"][i] = get(f"{p}.input_layernorm.weight"), get(f"{p}.post_attention_layernorm.weight")
        for name, sl in (("q", slice(0, Q)), ("k", slice(Q, Q + KV)), ("v", slice(Q + KV, None))):
            lay["wqkv"][i, :, sl] = get(f"{p}.self_attn.{name}_proj.weight").T
            bias(f"{p}.self_attn.{name}_proj.bias", lay["bqkv"][i], sl)
        lay["wo"][i] = get(f"{p}.self_attn.o_proj.weight").T
        bias(f"{p}.self_attn.o_proj.bias", lay["bo"][i])
        for name, wk, bk in (("gate", "wg", "bg"), ("up", "wu", "bu"), ("down", "wd", "bd")):
            lay[wk][i] = get(f"{p}.mlp.{name}_proj.weight").T
            bias(f"{p}.mlp.{name}_proj.bias", lay[bk][i])

    with ThreadPoolExecutor(min(16, os.cpu_count() or 1)) as ex:  # numpy copies release the GIL: ~5x faster load
        list(ex.map(fill, range(L)))
    embed = get("model.embed_tokens.weight")
    head = get("lm_head.weight") if "lm_head.weight" in w else embed  # tied embeddings
    return dict(embed=embed, norm=get("model.norm.weight"), lm_head=head, layers=lay)


def load_params(model_dir, dtype="bfloat16"):
    """Checkpoint directory -> (config, device-resident parameter pytree)."""
    import jax
    c = model_config(model_dir)
    w = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        w.update(load_safetensors(f))
    return c, jax.device_put(stack_params(w, c, dtype))


def forward(params, ids, pos, seg, readout, lids, *, c):
    """ids / pos / seg (b, L) int32, readout (b, R) int32 positions to read, lids (S,) int32 output-head rows ->
    fp32 logits (b, R, S) of the next token at the readout positions."""
    import jax
    import jax.numpy as jnp

    from . import tpu_kernels as K
    b, L = ids.shape
    nh, nkv, hd, H = c["heads"], c["kv_heads"], c["head_dim"], c["hidden"]
    Q, KV = nh * hd, nkv * hd
    cos, sin = K.rope_cache(L, hd, c["theta"])  # positions never exceed the row length
    scale = 1.0 / math.sqrt(hd)

    def layer(h, lp):
        x = K.rms_norm(h.reshape(b * L, H), lp["ln1"], c["eps"])
        qkv = K.linear(x, lp["wqkv"], lp["bqkv"])
        q = K.rope(qkv[:, :Q].reshape(b, L, nh, hd), pos, cos, sin)
        k = K.rope(qkv[:, Q: Q + KV].reshape(b, L, nkv, hd), pos, cos, sin)
        v = qkv[:, Q + KV:].reshape(b, L, nkv, hd)
        o = K.attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), seg, scale)
        h = h + K.linear(o.transpose(0, 2, 1, 3).reshape(b * L, Q), lp["wo"], lp["bo"]).reshape(b, L, H)
        x = K.rms_norm(h.reshape(b * L, H), lp["ln2"], c["eps"])
        f = K.swiglu(x, lp["wg"], lp["bg"], lp["wu"], lp["bu"])
        return h + K.linear(f, lp["wd"], lp["bd"]).reshape(b, L, H), None

    h, _ = jax.lax.scan(layer, params["embed"][ids], params["layers"])
    hr = K.rms_norm(jnp.take_along_axis(h, readout[:, :, None], axis=1), params["norm"], c["eps"])
    return jnp.einsum("brh,sh->brs", hr.astype(jnp.float32), params["lm_head"][lids].astype(jnp.float32),
                      precision=jax.lax.Precision.HIGHEST)


def tpu_available():
    """True when JAX is installed and sees a TPU."""
    import importlib.util
    if importlib.util.find_spec("jax") is None:
        return False
    import jax
    try:
        return bool(jax.devices("tpu"))
    except RuntimeError:
        return False


def _softmax(x):
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


class TPUBackend(GraphBackend):
    """Google TPU through JAX / XLA (plain-JAX kernels, bf16 or fp32). GraphBackend's packing, buckets and token-budget
    batching; each (batch, length, readouts) shape compiles once (all bucket shapes at start-up with warm=True).
    Prompts longer than the largest bucket get a length rounded up to a multiple of 512 (compiled on first use)."""

    LETTER_SLOTS = 16  # output-head rows read per position; padded to a fixed size so new letter ids do not recompile

    def __init__(self, model_dir, dtype="bfloat16", shared=True, warm=True):
        import jax
        self.jax = jax
        cache = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(Path.home() / ".cache" / "basal" / "jax")
        jax.config.update("jax_compilation_cache_dir", cache)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.prefill = PREFILL
        self.shared = shared
        self.R = 2 if shared else 1  # readout positions per row: the option orders of one question
        self.cfg, self.params = load_params(model_dir, dtype)
        self.device = jax.devices()[0]
        c = self.cfg
        self._fwd = jax.jit(lambda p, ids, pos, seg, ro, li: forward(p, ids, pos, seg, ro, li, c=c))
        self.slots, self._lids = {}, None
        if warm:
            for L in self.LENS:
                for b in self.BATCHES:
                    if b * L <= self.TOKEN_BUDGET or b == 1:
                        self._forward_rows([([0] * L, list(range(L)), [1] * L, [L - 1])] * b, b, L)

    def _bucket(self, n, xs):
        for x in xs:
            if n <= x:
                return x
        return -(-n // 512) * 512  # beyond the largest bucket: one extra compiled length per 512 tokens

    def _letters(self, ids_list):
        """Column of each letter id in the logits; the device array of output-head rows changes only for new ids."""
        new = [i for ids in ids_list for i in ids if i not in self.slots]
        if new or self._lids is None:
            for i in new:
                self.slots.setdefault(i, len(self.slots))
            n = -(-max(len(self.slots), 1) // self.LETTER_SLOTS) * self.LETTER_SLOTS
            lids = np.zeros(n, np.int32)
            for i, s in self.slots.items():
                lids[s] = i
            self._lids = self.jax.device_put(lids, self.device)
        return self.slots

    def _forward_rows(self, packs, b, L):
        """Packed rows (ids, positions, segment ids, readout positions) padded to (b, L) -> fp32 logits
        (len(packs), R, LETTER_SLOTS) on the host."""
        if self._lids is None:
            self._letters([])
        R = max(self.R, max(len(pk[3]) for pk in packs))
        ids = np.full((b, L), self.tok.pad_token_id, np.int32)
        pos = np.tile(np.arange(L, dtype=np.int32), (b, 1))
        seg = np.full((b, L), -1, np.int32)
        ro = np.zeros((b, R), np.int32)
        for r, (t, pp, sg, last) in enumerate(packs):
            ids[r, : len(t)], pos[r, : len(t)], seg[r, : len(t)], ro[r, : len(last)] = t, pp, sg, last
        return np.asarray(self._fwd(self.params, ids, pos, seg, ro, self._lids))[: len(packs)]

    def run(self, prompts, ids_list, policy=None):
        return [r[0] for r in self._run_groups([([p], [x]) for p, x in zip(prompts, ids_list)])]

    def run_shared(self, groups, policy=None):
        if not self.shared:  # one row per option order
            return EagerBackend.run_shared(self, groups)
        return self._run_groups(groups)

    def _run_groups(self, groups):
        packs = [self._pack([self.tok(p, add_special_tokens=False).input_ids for p in prompts]) for prompts, _ in groups]
        slots = self._letters([x for _, ids_list in groups for x in ids_list])
        res = [None] * len(packs)
        for idx in self._chunks([len(pk[0]) for pk in packs]):
            L = self._bucket(max(len(packs[k][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            lg = self._forward_rows([packs[k] for k in idx], b, L).astype(np.float64)
            for r, k in enumerate(idx):
                res[k] = [_softmax(lg[r, o, [slots[i] for i in ids]]) for o, ids in enumerate(groups[k][1])]
        return res

    def memory_gb(self):
        stats = self.device.memory_stats() or {}
        return stats.get("peak_bytes_in_use", float("nan")) / 2**30

    def device_name(self):
        return self.device.device_kind
