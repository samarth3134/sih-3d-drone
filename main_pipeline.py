# main_pipeline.py
# Samarth — Team Lead + Integration
# Entry point for the full SIH 26158 pipeline.
# Run: python main_pipeline.py --video data/input_video/drone.mp4

import yaml
import argparse
import time

# ── Load config ───────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Single-Pass Drone Video to 3D Model Pipeline")
parser.add_argument("--video", type=str, help="Path to input drone video file")
args = parser.parse_args()

if args.video:
    config["paths"]["input_video"] = args.video

# ── Stage 1: Preprocessing + Pose Estimation (Alissa) ─────────
print("\n[1/4] Running preprocessing and pose estimation (Alissa)...")
t = time.time()
from modules.preprocessing.run import run as preprocess
preprocess(config)
print(f"      Done in {time.time() - t:.1f}s")
print(f"      Output: {config['colmap']['sparse_geo_dir']}")

# ── Stage 2: Dynamic Object Masking (Yathansh) ────────────────
print("\n[2/4] Running dynamic object masking (Yathansh)...")
t = time.time()
from modules.dynamic_masking.run import run as mask
mask(config)
print(f"      Done in {time.time() - t:.1f}s")
print(f"      Output: {config['paths']['masked_frames_dir']}")

# ── Stage 3: Depth Estimation (Ashwika) ───────────────────────
print("\n[3/4] Running depth estimation (Ashwika)...")
t = time.time()
from modules.depth_estimation.run import run as depth
depth(config)
print(f"      Done in {time.time() - t:.1f}s")
print(f"      Output: {config['paths']['depth_maps_dir']}")

# ── Stage 4: 3D Reconstruction + Georeferencing (Srujan) ──────
print("\n[4/4] Running 3D reconstruction (Srujan)...")
t = time.time()
from modules.reconstruction.run import run as reconstruct
reconstruct(config)
print(f"      Done in {time.time() - t:.1f}s")
print(f"      Output: {config['gsplat']['result_dir']}")

# ── Done ───────────────────────────────────────────────────────
print("\nPipeline complete.")
print(f"Point cloud : {config['paths']['output_dir']}splat.ply")
print(f"Open splat.ply in SuperSplat: https://supersplat.playcanvas.com")
