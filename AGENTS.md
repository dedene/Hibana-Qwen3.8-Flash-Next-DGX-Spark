# AGENTS.md

Serve Qwen3.8-Flash-Next (125B MoE, RadixArk NVFP4 checkpoint, 135 GB, plus Hibana's calibrated 3-bit copies, 51 GB)
on one DGX Spark with [Hibana](https://github.com/sharar-project/hibana) via Docker Compose.

## Layout

- `compose.yaml`: `download` (one-shot, Hibana's own pinned fetch) -> `hibana` (the server on :8888). Both use the
  image built from `Dockerfile`.
- `tools/download.sh`: runs Hibana's `preflight.ensure_snapshots` + `ensure_w3` for the `fast` profile, so every profile
  can start without a new download. `exit 0` last.
- `tools/serve.sh`: every engine flag; maps this recipe's settings to `hibana serve flashnext` flags and `HIBANA_*`.
- `tools/check.py`: stdlib-only functional checks against a running server.
- Knobs are `${VAR:-default}` in compose and serve.sh; `.env.example` documents each one.

## Spark

`ssh zenjoy@10.0.0.158`, repo at `~/Development/Hibana-Qwen3.8-Flash-Next-DGX-Spark`. Checkpoint in the shared
`~/.cache/huggingface/hub`, 3-bit copies, SSD tier, logs and kernel builds in `~/.cache/hibana-flashnext`. GB10,
sm_121, 128 GB unified memory, driver 580, CUDA 13. :8888 is the daily model for Hermes: stop it for a bring-up or bench
and always put it back.

## Verify

```bash
docker compose config --quiet
shellcheck tools/*.sh
python3 -m py_compile tools/*.py
docker compose up -d --wait      # on the Spark, with :8888 free
tools/check.py                   # all ok
```

## Pins

- Hibana `d474b74d5a892b2cd1cc8fc50cb17946e2a539bb` (tag v0.1.0, 2026-11-13). Its `uv.lock` pins the Python stack
  (torch 2.13.0+cu130, flashinfer 0.6.17, triton 3.7.1, b12x 1.5.0).
- Base image `nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d`,
  uv `ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0`.
- Image tag `hibana-flashnext:d474b74d-r1`: bump `-rN` with every Dockerfile change.
- Weights, pinned in Hibana's source (`hibana/model/flashnext/config.py`), not here:
  `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594`,
  `yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad@5c4c62a62853caebea8cedda7fbed952efa1effe`,
  DFlash2 `incoai/Qwen3.8-Flash-Next-DFlash2@a22a1744e85c17364dcc54a6a420af7135d4c7f6`. A new Hibana commit can move them.

## Model facts

- Qwen3.8-Flash-Next: 48 layers (36 Gated DeltaNet, 12 attention with a sparse indexer), 512 experts, top 10, MTP head,
  262K context. Text only here (the vision tower is not loaded).
- Not the checkpoint the TensorFold recipe serves (`nvidia/Qwen3.8-Flash-Next-FP4-Local`): a different NVFP4 quantization
  of the same model.
- Hibana loads the trunk as FP8 in every profile. `exact` keeps NVFP4 experts and exact speculative acceptance; `fast`
  (Hibana's default) serves 3-bit experts and trunk copies and relaxed sampled acceptance (`accept_soft` 0.5).
- Thinking is on unless a request sends `chat_template_kwargs.enable_thinking: false` (hard default in
  `hibana/api/validation.py`, no server switch). Reasoning comes back in `reasoning_content`.
- Any `model` id is accepted; `/v1/models` lists the served checkpoint. No API key check.
- Omitted `temperature` is 1.0 with top_p 0.95, top_k 20 (the generation config).

## Findings

- 2026-11-16, bring-up: image builds in ~12 min on the Spark; weights load in 30 s (68.9 GiB, `exact`); cold start with
  kernel builds ~4 min. `tools/check.py` all ok, needle at 39K and 161K tokens found. `/v1/models` lists
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` (use that as MODEL in spark-bench-kit), any id answers.
- 2026-11-16, download: Hibana's `w3_checkpoint fetch` writes to `$HIBANA_RESULTS/w3-checkpoints` with `local_dir`,
  not into the HF hub cache, so a hub prefetch of the yemara repo is wasted space.
- 2026-11-16, speed (spark-bench-kit `ONLY=speed`, thinking on, one window; runs `tune-hibana-flashnext-{exact,fast}-think`,
  `tune-qwen38-tf-think`): weighted decode TensorFold 49.1, `exact` 47.2, `fast` 105.3 tok/s; 4 streams 97.3 / 96.6 /
  214.3; prefill at 8K 2,333 / 3,454 / 3,174. `exact` has a slow first token on short prompts (0.8-1.2 s against 0.3-0.6).
  Table in the README.

## Constraints

- Stock Hibana at a pinned commit, no patches. Hibana runs on the host with uv upstream; this recipe runs the same
  locked environment in Docker so it follows the other recipes.
- Lossless by default (`PROFILE=exact`). `fast` and `throughput` only as named, measured options.
- Never `hf cache prune` on the Spark (it deletes every revision-pinned snapshot). Write our own scripts.
