#!/usr/bin/env bash
# Low-memory variant of run.sh: bf16 + 512x384 + tiled VAE + in-process dataloader.
# Keeps the full pipeline inside the Pi's 8GB RAM so it does not thrash on swap.
"$(dirname "$0")/.venv/bin/python" run_single_lowmem.py --person samples/person0.jpg --cloth samples/cloth1.jpg --desc "The garment is a white sleeveless camisole top with a blue and green tropical floral print, lightweight chiffon, V-neckline with thin straps" --steps 8 --guidance 2.5
