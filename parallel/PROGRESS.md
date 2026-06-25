# Multi-Pi Parallelization — Progress / Resume

**Goal:** cut single-image inference LATENCY via 2-way tensor parallelism (TP) of
the main denoiser, over the direct wired GbE link between two Pi 5s.

## ▶ RESUME HERE
TP attention verified on the REAL denoiser (all 60 modules, MQA-aware). NEXT =
distributed bootstrap: load denoiser per rank, shard via shard_denoiser_attention_,
run the FULL pipeline split across the 2 Pis, check 2-Pi output == single-Pi (1
step), then measure real per-step latency.

## Setup facts (don't re-derive)
- rank 0 = this Pi, eth0 `192.168.100.1`; rank 1 = peer, eth0 `192.168.100.2`
- gloo pinned to eth0 via `GLOO_SOCKET_IFNAME=eth0`
- Link measured: 117 MB/s raw TCP, gloo all_gather 138 MB/s, RTT ~283us
- Stage 0 verdict: **~1.83x feasible, PASS**
- Both Pis: same venv (torch 2.12.1+cpu); checkpoints present
- Single-Pi baseline (fp32 + garment caching): ~119 s/step, 8 steps

## TP scheme (Megatron-style, per main-denoiser attention)
- column-parallel `to_q/to_k/to_v` (each rank keeps its head slice; no comm)
- row-parallel `to_out[0]` (partial output per rank) -> **one all_reduce** sums them
- bias kept on rank 0 only (so all_reduce counts it once)
- per attention: exactly 1 all_reduce of [B, S, out_dim]

## TODO (delete each as done)
- [ ] Distributed bootstrap for the full pipeline (load denoiser per rank, shard, run)
- [ ] Equivalence: 2-Pi TP output == single-Pi output (1 step, atol)
- [ ] Measure real 2-Pi per-step latency (target ~1.8x vs 119s)
- [ ] Stage 1b: channel-parallel conv (resnet) + all-gather at block boundaries
- [ ] Stage 2: compute/comm overlap, fuse small all_reduces
- [ ] (later) rope path in TPAttnProcessor (not needed yet; GQA/MQA already done)
- [ ] (later) 4-Pi: 2-way TP x 2-way data-parallel via router

## Done (terse log)
- Stage 0 complete: link + gloo benched, PASS (1.83x).
  scripts: `parallel/link_bench.py`, `parallel/tp_stage0_commbench.py`
- Env reproducible: `setup_env.sh` + `requirements.txt` (peer cloned + set up).
- Stage 1 MVP core: TP attention sharding (head-parallel q/k/v + row-parallel
  to_out + all_reduce) verified == full attention, diff ~1e-7 (self & cross).
  `parallel/tp_attention.py`, test `parallel/tp_test_attention.py`.
- Model is MQA (kv_heads=1, query heads 4/8/14, head_dim 64). TP updated to
  shard query heads + replicate/shard kv (shard_attention_, tp_supported,
  TPAttnProcessor now GQA/MQA-aware). Verified on REAL denoiser: all 60
  Attention modules TP == full, max|diff| 2.2e-5. test `parallel/tp_test_denoiser.py`.
