"""Plain-JAX kernels of the TPU backend
"""
import math
import jax
import jax.numpy as jnp


def dot_precision(dtype):
    return jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None


def rms_norm(x, w, eps=1e-6):
    """x (..., H), w (H,): normalised in fp32, returned in x.dtype."""
    x32 = x.astype(jnp.float32)
    x32 = x32 / jnp.sqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * w).astype(x.dtype)


def linear(x, w, b=None):
    """x (M, K) @ w (K, N) + b (N,), fp32 accumulation."""
    out = jnp.dot(x, w, preferred_element_type=jnp.float32, precision=dot_precision(x.dtype))
    if b is not None:
        out = out + b.astype(jnp.float32)
    return out.astype(x.dtype)


def swiglu(x, wg, bg, wu, bu):
    """silu(x @ wg + bg) * (x @ wu + bu); biases are added before the activation (Llama mlp_bias)."""
    prec = dot_precision(x.dtype)
    g = jnp.dot(x, wg, preferred_element_type=jnp.float32, precision=prec) + bg.astype(jnp.float32)
    u = jnp.dot(x, wu, preferred_element_type=jnp.float32, precision=prec) + bu.astype(jnp.float32)
    return (g * jax.nn.sigmoid(g) * u).astype(x.dtype)


def rope_cache(n, head_dim, theta):
    """cos / sin tables (n, head_dim // 2), fp32."""
    inv = jnp.exp(jnp.arange(head_dim // 2, dtype=jnp.float32) * -(math.log(theta) * 2.0 / head_dim))
    ang = jnp.arange(n, dtype=jnp.float32)[:, None] * inv[None]
    return jnp.cos(ang), jnp.sin(ang)


def rope(x, pos, cos, sin):
    """x (b, L, heads, d), pos (b, L): HF "rotate half" RoPE at explicit positions."""
    half = x.shape[-1] // 2
    c, s = cos[pos][:, :, None], sin[pos][:, :, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1).astype(x.dtype)


def tree_mask(end):
    """end (b, L): for every token, the end (exclusive) of the subtree of its block in the depth-first packed prefix
    trie (padding: its own index + 1) -> bool (b, L, L). allowed(i, j) = j <= i and i < end[j]: in depth-first order
    the queries inside the subtree of key j's block are exactly those of its own block and its descendants, so every
    token sees its ancestor blocks and itself only, as in a separate forward of its own prompt. Padding sits after
    every real token, so causality hides it from every row that is read out."""
    L = end.shape[1]
    i = jnp.arange(L, dtype=end.dtype)
    return (i[None, None, :] <= i[None, :, None]) & (i[None, :, None] < end[:, None, :])


def attention(q, k, v, end, scale):
    """q (b, nh, L, d), k / v (b, nkv, L, d) with GQA, end (b, L) (see tree_mask) -> (b, nh, L, d). The query heads
    of one KV head are grouped instead of repeating K / V, so K / V are never copied per query head."""
    b, nh, L, d = q.shape
    nkv = k.shape[1]
    prec = dot_precision(q.dtype)
    qg = q.reshape(b, nkv, nh // nkv, L, d)
    s = jnp.einsum("bkgqd,bkld->bkgql", qg, k, precision=prec, preferred_element_type=jnp.float32) * scale
    s = jnp.where(tree_mask(end)[:, None, None], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("bkgql,bkld->bkgqd", p.astype(v.dtype), v, precision=prec)
    return o.reshape(b, nh, L, d).astype(q.dtype)
