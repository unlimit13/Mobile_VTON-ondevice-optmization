# Multi-Pi Parallelization — Progress / Resume

**Goal:** cut single-image inference LATENCY via 2-way tensor parallelism (TP) of
the main denoiser, over the direct wired GbE link between two Pi 5s.

## ▶ RESUME HERE
Stage 1 MVP: TP attention math is verified. NEXT = load the real denoiser, walk &
shard ALL its Attention modules, set TPAttnProcessor, run the pipeline split
across the 2 Pis (distributed bootstrap), then check 2-Pi output == single-Pi.

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
- [ ] Walk + shard ALL Attention modules in a loaded denoiser
- [ ] Distributed bootstrap for the full pipeline (load denoiser per rank, shard, run)
- [ ] Equivalence: 2-Pi TP output == single-Pi output (1 step, atol)
- [ ] Measure real 2-Pi per-step latency (target ~1.8x vs 119s)
- [ ] Stage 1b: channel-parallel conv (resnet) + all-gather at block boundaries
- [ ] Stage 2: compute/comm overlap, fuse small all_reduces
- [ ] (later) GQA/MQA + rope path in TPAttnProcessor (MVP asserts MHA, no rope)
- [ ] (later) 4-Pi: 2-way TP x 2-way data-parallel via router

## Done (terse log)
- Stage 0 complete: link + gloo benched, PASS (1.83x).
  scripts: `parallel/link_bench.py`, `parallel/tp_stage0_commbench.py`
- Env reproducible: `setup_env.sh` + `requirements.txt` (peer cloned + set up).
- Stage 1 MVP core: TP attention sharding (head-parallel q/k/v + row-parallel
  to_out + all_reduce) verified == full attention, diff ~1e-7 (self & cross).
  `parallel/tp_attention.py`, test `parallel/tp_test_attention.py`.
