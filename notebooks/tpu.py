# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # basal-1.0 on Google TPU
#
# Typed decisions (`choice` / `noul` / `score`) with the `tpu` mode of basal: JAX / XLA on plain-JAX kernels, shared
# prefix for the two option orders, one compiled executable per (batch, length) bucket.
#
# **Before running**: `Runtime` -> `Change runtime type` -> select the `v5e-1` (or `v6e-1`) **TPU** hardware accelerator.
#
# The notebook installs the repository, starts the model through the same `Server` class as `basal-serve`, answers the
# bundled examples and benchmarks latency, throughput and agreement with the fp32 reference.

# %% [markdown]
# ## 1. Clone the repository

# %%
import os

REPO_URL = "https://github.com/qooba/basal.git"
BRANCH = "main"
REPO_DIR = "/content/basal"

if not os.path.isdir(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# %% [markdown]
# ## 2. Install dependencies
#
# The Colab TPU image already has CPU torch and transformers (used only for the tokenizer and chat template), so basal is
# installed without pulling a CUDA torch. Its preinstalled jax ships a `libtpu` that cannot run code from its own
# `jaxlib`, so the validated pair `jax[tpu]==0.11.1` is force-installed and **the runtime restarts once** (the session
# "crashes" on purpose). After the restart, continue with the next cell; this cell is a no-op when run again.

# %%
import importlib.metadata as md
import importlib.util

JAX_VERSION = "0.11.1"

!pip install -q -e . --no-deps
!pip install -q "transformers==5.17.0" "huggingface_hub>=0.34" "safetensors>=0.6" starlette uvicorn httpx pandas matplotlib
if importlib.util.find_spec("torch") is None:
    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu

if md.version("jax") != JAX_VERSION:
    !pip install -q -U "jax[tpu]=={JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    os.kill(os.getpid(), 9)  # restart the runtime so the new jax / libtpu are loaded

# %%
%cd /content/basal
import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # the basal models are public; skip the Colab secrets lookup

import jax

print(jax.__version__, jax.devices())

# %% [markdown]
# ## 3. Load the model
#
# `basal-1.0-1.5B` (3.2 GB of bf16 weights) or `basal-1.0-4.5B` (9.6 GB, fits one 16 GB v5e chip). Like `basal-serve`,
# the server compiles every bucket shape before answering: about 2 minutes for the 1.5B the first time. Compiled
# executables are cached in `~/.cache/basal/jax`, so loading again in the same runtime is much faster.

# %%
import asyncio
import os
import time

os.chdir("/content/basal")  # the repo dir must not shadow the installed package (a restart resets the cwd)
from basal.server import Server, parser

MODEL = "Remek/basal-1.0-1.5B"  # @param ["Remek/basal-1.0-1.5B", "Remek/basal-1.0-4.5B"]

t0 = time.perf_counter()
srv = Server(parser().parse_args(["--model", MODEL, "--mode", "tpu"]))
worker = asyncio.get_running_loop().create_task(srv.worker())  # the batching loop of basal-serve
be = srv.backend
print(f"{MODEL} ready in {time.perf_counter() - t0:.0f}s on {be.device_name()}, "
      f"{be.memory_gb():.1f} GB HBM peak, calibration: {srv.temps}")

# %% [markdown]
# ## 4. Ask a question
#
# `srv.decide` takes the body of a `POST /v1/systemone` request and returns the response of the HTTP server.

# %%
import json

body = {
    "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
    "questions": {
        "dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
                 "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej",
                              "loans": "Kredyty"}},
        "no_access": {"type": "noul", "instructions": "Czy klient nie ma dostępu do konta?"},
        "urgency": {"type": "score", "instructions": "Jak pilne jest to zgłoszenie?",
                    "criteria": ["niska", "średnia", "wysoka", "krytyczna"]},
    },
}
await srv.decide(body)  # first call: warms up the Python path
print(json.dumps(await srv.decide(body), ensure_ascii=False, indent=1))

# %% [markdown]
# ## 5. The bundled examples
#
# `basal/examples/*.jsonl`: Polish and English routing, policy rules, deadlines, scores and full multi-question
# requests. Simple items carry a `gold` answer. The model is weak at arithmetic and exact thresholds, so a few
# confident mistakes are expected (see *What to expect on these files* in the README).

# %%
from pathlib import Path

import pandas as pd

from basal.run import summarise, to_request


async def answer(item):
    return summarise(item, await srv.decide(to_request(item)))


rows = []
for name in ["choice", "noul", "score"]:
    items = [json.loads(l) for l in Path(f"basal/examples/{name}.jsonl").read_text().splitlines() if l.strip()]
    for it, r in zip(items, await asyncio.gather(*(answer(it) for it in items))):
        rows.append({"file": name, "id": it["id"], "question": it["question"][:70], "answer": r["option"][:45],
                     "confidence": round(r["confidence"], 3), "gold": it["options"][it["gold"]][:45],
                     "correct": r["correct"]})
df = pd.DataFrame(rows)
display(df.groupby("file")["correct"].agg(["sum", "count", "mean"]).rename(columns={"sum": "correct", "mean": "accuracy"}))
pd.set_option("display.max_colwidth", 80)
df

# %%
# Full requests: several typed questions about one JSON state (fan-out, web agent, deadline, loan triage, guard)
for line in Path("basal/examples/complex.jsonl").read_text().splitlines():
    item = json.loads(line)
    resp = await srv.decide(to_request(item))
    print(f"--- {item['id']}  ({resp['usage']['latency_ms']} ms)")
    for qname, a in resp["answers"].items():
        top = a.get("choice") or ("yes" if a["type"] == "noul" and a["noul"] >= 0.5 else
                                 "no" if a["type"] == "noul" else f"score {a['score']:.2f}")
        print(f"  {qname:<22} {a['type']:<7} {top:<22} confidence {a['confidence']:.3f}")

# %% [markdown]
# ## 6. Benchmark: latency and throughput
#
# The measurement of `basal-bench`, run in this process on the loaded backend (a TPU can be opened by one process only,
# so `!basal-bench` in a cell would fail while the model is loaded here):
#
# - `lat2_ms`: median latency of one decision with **both** option orders at batch size 1 (what the server does);
# - `lat1_ms`: the same with one option order;
# - `dec_s`: two-order decisions per second, 16 questions (32 option-order passes) per forward.

# %%
from basal.bench import DEFAULT_QUESTIONS, groups_for, load_questions, measure

qs = load_questions(DEFAULT_QUESTIONS, 500)
groups = groups_for(be.tok, qs)
dec_tpu, r = measure(be, groups, len(groups) - 5)
top_tpu = [max(range(len(d)), key=d.__getitem__) for d in dec_tpu]
r["acc"] = sum(t == g[0]["gold"] for t, g in zip(top_tpu, groups)) / len(groups)
bench = pd.DataFrame([{"model": MODEL, "device": be.device_name(), "items": len(groups),
                       **{k: round(v, 3) for k, v in r.items()}}])
bench

# %% [markdown]
# For orientation, the README's CUDA numbers (`fast`, bf16). They were measured on a private 500-item sample with longer
# prompts (mean about 360 tokens), while this notebook uses the 44 bundled examples (shorter), so they are not
# like-for-like; pass your own JSONL file to `load_questions` for numbers that describe your workload.
#
# | GPU | 4.5B, 2 orders | 4.5B dec/s | 1.5B, 2 orders | 1.5B dec/s |
# |---|---|---|---|---|
# | B300 | 8.8 ms | 109 | 4.7 ms | 250 |
# | H100 | 12.5 ms | 63 | 6.2 ms | 157 |
# | RTX 5090 | 27.3 ms | 24 | 12.7 ms | 67 |
# | TPU v5e-1 (`tpu`, bundled examples) | 17.9 ms | 49 | 6.6 ms | 145 |

# %% [markdown]
# ## 7. Benchmark: prompt length and batch size
#
# Device time of one forward per bucket shape (all were compiled at start-up). A decision is a prefill, so even one
# 256-token prompt sends 256 rows through every matmul: the TPU is close to compute-bound at batch 1, and the length
# bucket matters much more than the batch size. Above about 1,500 tokens latency grows faster than linearly: the
# plain-JAX attention materialises the full length x length score matrix.

# %%
import numpy as np


def forward_ms(b, L, reps=10):
    rows = [([0] * L, list(range(L)), [1] * L, [L - 1])] * b
    be._forward_rows(rows, b, L)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        be._forward_rows(rows, b, L)  # returns host arrays: includes the device sync
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1000


by_len = pd.DataFrame([{"length": L, "ms": forward_ms(1, L)} for L in be.LENS])
by_len["tokens_per_s"] = (by_len["length"] / by_len["ms"] * 1000).round(0)
L0 = 256
by_batch = pd.DataFrame([{"batch": b, "ms": forward_ms(b, L0)} for b in be.BATCHES if b * L0 <= be.TOKEN_BUDGET])
by_batch["rows_per_s"] = (by_batch["batch"] / by_batch["ms"] * 1000).round(1)
display(by_len.round(2), by_batch.round(2))

# %%
import matplotlib.pyplot as plt

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # headroom for the value label


fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(by_len["length"], by_len["ms"], color=BLUE, linewidth=2, marker="o", markersize=7)
style(a1, f"Forward latency vs prompt length (batch 1, {MODEL.split('/')[-1]})", "bucket length (tokens)", "ms")
last = by_len.iloc[-1]
a1.annotate(f"{last['ms']:.0f} ms", (last["length"], last["ms"]), textcoords="offset points", xytext=(-10, 8),
            ha="right", color=INK)
a2.plot(by_batch["batch"], by_batch["rows_per_s"], color=BLUE, linewidth=2, marker="o", markersize=7)
style(a2, f"Rows per second vs batch size (length {L0})", "batch size", "rows / s")
a2.set_xscale("log", base=2)
a2.set_xticks(by_batch["batch"], [str(b) for b in by_batch["batch"]])
best = by_batch.loc[by_batch["rows_per_s"].idxmax()]
a2.annotate(f"{best['rows_per_s']:.0f} rows/s", (best["batch"], best["rows_per_s"]), textcoords="offset points",
            xytext=(0, 8), ha="center", color=INK)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Agreement with the fp32 reference (optional)
#
# Runs the plain transformers model in fp32 on the VM's CPU and compares the top answer of every decision with the TPU.
# Slow: about 1 minute for the 1.5B, 4 minutes for the 4.5B (19 GB of host RAM).

# %%
RUN_FP32_REFERENCE = False  # @param {type: "boolean"}

if RUN_FP32_REFERENCE:
    import gc

    from basal.bench import decisions
    from basal.engine import EagerBackend, resolve

    ref = EagerBackend(resolve(MODEL), "float32", device="cpu")
    res = []
    for i in range(0, len(groups), 8):
        res += ref.run_shared([(g[2], g[3]) for g in groups[i: i + 8]])
    dec_ref = decisions(groups, res)
    top_ref = [max(range(len(d)), key=d.__getitem__) for d in dec_ref]
    agree = sum(a == b for a, b in zip(top_tpu, top_ref)) / len(groups)
    max_dp = max(abs(p - q) for a, b in zip(dec_tpu, dec_ref) for p, q in zip(a, b))
    print(f"agreement with fp32: {agree:.3f}   max |dp|: {max_dp:.4f}   "
          f"fp32 accuracy: {sum(t == g[0]['gold'] for t, g in zip(top_ref, groups)) / len(groups):.3f}")
    del ref
    gc.collect()

# %% [markdown]
# ## 9. Serve over HTTP
#
# To serve the model instead, stop this notebook's model first (one process per TPU) and run the server, e.g. in a
# terminal of the runtime:
#
# ```bash
# basal-serve --model Remek/basal-1.0-1.5B --port 8000     # --mode tpu is the default on a TPU VM
# basal-loadtest --url http://127.0.0.1:8000/v1/systemone
# ```
