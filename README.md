# SIH 3D Drone

A modular pipeline for turning drone video into a reconstructed 3D output.

## Repository structure

```text
data/
  input_video/       Source drone videos (not committed)
  frames/            Extracted video frames (not committed)
  masked_frames/     Frames after dynamic-object masking (not committed)
  depth_maps/        Depth-estimation outputs (not committed)
  output/            Reconstruction and visualization outputs (not committed)
modules/
  preprocessing/     Yathansh
  depth_estimation/  Srujan
  dynamic_masking/   Alissa
  reconstruction/    Ashwika
  visualization/     Dhruvi
main_pipeline.py     Samarth — pipeline orchestration
config.yaml          Shared configuration
```

## Getting started

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place source video files in `data/input_video/`.
4. Run `python main_pipeline.py`.

Each module directory is reserved for its assigned owner. Add module-specific documentation and dependencies as the implementation develops.
