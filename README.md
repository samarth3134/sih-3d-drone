# SIH 2025 - Problem ID 26158
# Single-Pass Drone Video to Accurate 3D Model Generation System

Organisation: National Technical Research Organisation (NTRO)
Theme: Robotics and Drones
Category: Software

---

## Team

| Member  | Role                                      | Key Tools                        |
|---------|-------------------------------------------|----------------------------------|
| Samarth | Team Lead + Integration                   | Python, Git, PyYAML              |
| Alissa  | Video Preprocessing + Pose Estimation     | OpenCV, COLMAP, ExifTool         |
| Yathansh| Dynamic Object Detection + Masking        | YOLOv8, SAM2, LaMa               |
| Ashwika | Depth Estimation + Metric Scale           | Depth Anything V2, PyTorch       |
| Srujan  | 3D Reconstruction + Georeferencing        | gsplat, Open3D                   |
| Dhruvi  | Visualization + PPT + Documentation       | SuperSplat, PowerPoint, OBS      |

---

## Problem Statement

Traditional drone-based 3D reconstruction requires multiple flight passes with heavy image overlap and hours of post-processing. In time-critical scenarios like disaster response or military reconnaissance, only a single drone pass is possible.

This system generates a georeferenced, metrically accurate 3D model from a single drone video pass, replacing the need for multi-pass photogrammetry.

---

## Pipeline Overview

Stage 1 - Alissa - Video Preprocessing and Pose Estimation
- Frame extraction, blur filtering, lens undistortion
- COLMAP sparse reconstruction
- GPS model alignment using COLMAP model_aligner
- Output: sparse_geo folder, frames folder, geo_reference.txt

Stage 2 - Yathansh - Dynamic Object Masking
- YOLOv8 detection of cars, people, bikes etc
- SAM2 pixel-level segmentation masks
- LaMa inpainting to fill masked regions
- Output: masked_frames folder

Stage 3 - Ashwika - Depth Estimation
- Depth Anything V2 inference on each frame
- AGL altitude anchoring for metric scale
- Output: .npy depth maps in metres

Stage 4 - Srujan - 3D Reconstruction
- gsplat Gaussian Splatting training on COLMAP output
- Point cloud export and georeferencing
- Output: splat.ply, GeoTIFF

Final output is displayed in SuperSplat web viewer by Dhruvi.

---

## Tech Stack

| Stage            | Tool                  | Purpose                                 |
|------------------|-----------------------|-----------------------------------------|
| Pose Estimation  | COLMAP                | Structure from Motion, camera poses     |
| GPS Alignment    | COLMAP model_aligner  | Georeferencing to real-world coordinates|
| GPS Parsing      | ExifTool              | Extract GPS metadata from drone footage |
| Object Detection | YOLOv8                | Detect dynamic objects per frame        |
| Segmentation     | SAM2                  | Pixel-accurate masks                    |
| Inpainting       | LaMa                  | Fill masked regions cleanly             |
| Depth Estimation | Depth Anything V2     | Per-frame monocular depth inference     |
| Reconstruction   | gsplat                | Gaussian Splatting from COLMAP output   |
| Visualization    | SuperSplat            | Web-based 3D Gaussian Splat viewer      |
| Language         | Python 3.10+          | Core development                        |
| Deep Learning    | PyTorch + CUDA        | Model inference backbone                |

---

## Repository Structure

sih-3d-drone/
    data/
        input_video/        place drone.mp4 here
        frames/             Alissa's extracted frames
        masked_frames/      Yathansh's cleaned frames
        depth_maps/         Ashwika's .npy depth arrays
        colmap_output/
            sparse_geo/
                cameras.txt
                images.txt
                points3D.txt
            images/         copy of frames/
        output/             final .ply and .splat files
    modules/
        preprocessing/      Alissa
        dynamic_masking/    Yathansh
        depth_estimation/   Ashwika
        reconstruction/     Srujan
        visualization/      Dhruvi
    main_pipeline.py        Samarth - integration entry point
    config.yaml             shared config for all modules
    requirements.txt
    README.md

---

## Setup Instructions

Prerequisites:
- Python 3.10 or higher
- CUDA-compatible GPU (required for gsplat and Depth Anything V2)
- COLMAP installed separately from https://colmap.github.io

Step 1 - Clone the repo

    git clone https://github.com/your-repo/sih-3d-drone.git
    cd sih-3d-drone

Step 2 - Install PyTorch with CUDA first

Do this BEFORE running pip install -r requirements.txt.
Check your CUDA version at https://pytorch.org and adjust accordingly.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Step 3 - Install remaining dependencies

    pip install -r requirements.txt

Step 4 - Install SAM2 and LaMa separately

These cannot be installed via requirements.txt.

    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    pip install git+https://github.com/advimman/lama.git

Step 5 - Verify CUDA and gsplat

    python -c "import torch; print('CUDA:', torch.cuda.is_available())"
    python -c "import gsplat; print('gsplat ready')"

If CUDA returns False, gsplat will not work. Contact the team lead immediately.

---

## How to Run

Place your drone video:

    cp /path/to/drone.mp4 data/input_video/drone.mp4

Run the full pipeline:

    python main_pipeline.py --video data/input_video/drone.mp4

Output files:

    data/output/splat.ply       open this in SuperSplat
    data/output/splat.splat     Gaussian Splat file
    data/output/georef.tif      GeoTIFF overlay

To visualize: open https://supersplat.playcanvas.com and load splat.ply.

---

## Data Format Standards

All modules must read paths from config.yaml. No hardcoded paths.

| Data          | Format                          | Produced By | Consumed By     |
|---------------|---------------------------------|-------------|-----------------|
| Clean frames  | .jpg in data/frames/            | Alissa      | Ashwika,Yathansh|
| Camera poses  | cameras.txt, images.txt         | Alissa      | Srujan          |
| GPS reference | geo_reference.txt               | Alissa      | Ashwika, Srujan |
| Masked frames | .jpg in data/masked_frames/     | Yathansh    | Srujan          |
| Depth maps    | .npy shape (H, W) in metres     | Ashwika     | Srujan          |
| Point cloud   | data/output/splat.ply           | Srujan      | Dhruvi          |

---

## Module Entry Point Convention

Every module must expose a run(config) function:

    def run(config):
        input_dir  = config["paths"]["your_input"]
        output_dir = config["paths"]["your_output"]
        # your code here
        return output_dir

main_pipeline.py will call each module's run(config) in sequence.

---

## Key Technical Decisions

- gsplat used instead of Open3D MVS because it accepts COLMAP output directly and produces Gaussian Splat files compatible with SuperSplat
- Postshot and Scaniverse rejected because they are black-box tools that cannot be defended as original work to NTRO judges
- Potree replaced by SuperSplat because it requires no converter tool and works directly with .ply files
- AGL altitude used instead of raw GPS altitude above sea level for accurate metric depth anchoring
- LaMa used instead of OpenCV inpainting because it produces significantly better results on large masked regions
