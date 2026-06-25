"""Build a 1-pair VITON-HD-style dataset from a person image + a garment image
and run Mobile-VTON inference on CPU (single image try-on).

Usage:
    python run_single.py --person path/to/person.jpg --cloth path/to/cloth.jpg \
        --desc "a red long-sleeve cotton t-shirt" \
        --steps 28 --guidance 2.5

Output is written to ./output/single/<person>_<cloth>
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="path to person image")
    ap.add_argument("--cloth", required=True, help="path to garment image")
    ap.add_argument("--desc", required=True, help="text description of the garment")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "single_data"))
    ap.add_argument("--output_dir", default=os.path.join(HERE, "output", "single"))
    ap.add_argument("--checkpoint_path", default=os.path.join(HERE, "checkpoint"))
    args = ap.parse_args()

    p_name = os.path.basename(args.person)
    c_name = os.path.basename(args.cloth)

    test = os.path.join(args.data_dir, "test")
    os.makedirs(os.path.join(test, "image"), exist_ok=True)
    os.makedirs(os.path.join(test, "cloth"), exist_ok=True)
    shutil.copy(args.person, os.path.join(test, "image", p_name))
    shutil.copy(args.cloth, os.path.join(test, "cloth", c_name))

    with open(os.path.join(test, "image_descriptions.txt"), "w") as f:
        f.write(f"{c_name}: {args.desc}\n")
    # unpaired order so person != cloth is allowed: "<person> <cloth>"
    with open(os.path.join(args.data_dir, "test_pairs.txt"), "w") as f:
        f.write(f"{p_name} {c_name}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(HERE, "inference.py"),
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--order", "unpaired",
        "--height", str(args.height),
        "--width", str(args.width),
        "--test_batch_size", "1",
        "--num_inference_steps", str(args.steps),
        "--guidance_scale", str(args.guidance),
        "--mixed_precision", "no",
        "--checkpoint_path", args.checkpoint_path,
    ]
    print("Running:", " ".join(cmd))
    # Force CPU + single worker (Pi has no GPU)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(os.cpu_count() or 4))
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
