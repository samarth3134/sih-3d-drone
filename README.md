# SIH 3D Drone Reconstruction

Monocular 3D scene reconstruction from drone footage.

## Pipeline

1. **Preprocessing** — Frame extraction & normalization
2. **Depth Estimation** — Monocular depth inference (DPT/MiDaS)
3. **Dynamic Masking** — Moving object segmentation via optical flow
4. **Reconstruction** — 3D point cloud / mesh generation (TSDF)
5. **Visualization** — Interactive 3D rendering & export

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Place your drone footage in `data/input_video/` and run:

```bash
python main_pipeline.py
```

Output is saved to `data/output/`.

## Project Structure

| Directory | Owner | Description |
|-----------|-------|-------------|
| `modules/preprocessing/` | Yathansh | Frame extraction & cleanup |
| `modules/depth_estimation/` | Srujan | Monocular depth models |
| `modules/dynamic_masking/` | Alissa | Moving object segmentation |
| `modules/reconstruction/` | Ashwika | 3D reconstruction engine |
| `modules/visualization/` | Dhruvi | 3D rendering & export |
| `main_pipeline.py` | Samarth | Pipeline orchestration |
