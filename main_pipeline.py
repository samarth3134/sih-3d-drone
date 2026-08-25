# main_pipeline.py - Samarth owns this
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Step 1: Preprocessing (Yathansh's module)
from modules.preprocessing.extract_frames import extract_frames
frames = extract_frames(config)

# Step 2: Dynamic Masking (Alissa's module)
from modules.dynamic_masking.mask import mask_dynamic_objects
masked = mask_dynamic_objects(frames, config)

# Step 3: Depth Estimation (Srujan's module)
from modules.depth_estimation.depth import estimate_depth
depths = estimate_depth(frames, config)

# Step 4: 3D Reconstruction (Ashwika's module)
from modules.reconstruction.reconstruct import reconstruct
model = reconstruct(masked, depths, config)

print("Pipeline complete. Output at:", config["paths"]["output_dir"])