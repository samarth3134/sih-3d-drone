"""
modules/preprocessing/run.py
Alissa — Video Preprocessing + Pose Estimation

Called by main_pipeline.py exactly like this:
    from modules.preprocessing.run import run as preprocess
    preprocess(config)

config comes from config.yaml (already loaded as a dict by main_pipeline.py) —
you never build config yourself, you only read from it and write your outputs
to the paths it specifies.
"""
import os
import subprocess

from .extract_frames import extract_frames
from .undistort import build_intrinsics_from_specs, undistort_dir


def run(config):
    video_path = config["paths"]["input_video"]
    frames_dir = config["paths"]["frames_dir"]
    images_dir = config["colmap"]["images_dir"]          # undistorted frames COLMAP reads
    geo_reference_path = config["paths"]["geo_reference"]
    colmap_output_dir = config["colmap"]["output_dir"]
    sparse_dir = config["colmap"]["sparse_dir"]           # e.g. .../sparse/0/
    sparse_parent = os.path.dirname(os.path.normpath(sparse_dir))  # e.g. .../sparse
    sparse_geo_dir = config["colmap"]["sparse_geo_dir"]

    fps = config["video"].get("fps_extract", 2)
    width, height = config["video"].get("resolution", [1920, 1080])
    altitude_agl = config["gps"].get("altitude_ref", 50.0)

    # 1. Extract + blur-filter frames
    os.makedirs(frames_dir, exist_ok=True)
    extract_frames(video_path, frames_dir, target_fps=fps, blur_thresh=100.0)

    # 2. Undistort into the folder COLMAP will actually read from
    os.makedirs(images_dir, exist_ok=True)
    K, dist = build_intrinsics_from_specs(6.3, 4.5, width, height)
    undistort_dir(frames_dir, images_dir, K, dist)

    # 3. geo_reference.txt — one line per image, constant AGL altitude
    #    (this project intentionally uses config['gps']['altitude_ref'] instead
    #    of a real per-frame flight log — no real drone flight is happening)
    os.makedirs(os.path.dirname(geo_reference_path), exist_ok=True)
    frame_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
    with open(geo_reference_path, "w") as f:
        f.write("# image_name lat lon alt_agl -- alt_agl is a constant from config['gps']['altitude_ref']\n")
        for name in frame_files:
            f.write(f"{name} 0.0 0.0 {altitude_agl}\n")

    # 4. COLMAP: feature extraction -> sequential matching -> mapping
    os.makedirs(colmap_output_dir, exist_ok=True)
    db_path = os.path.join(colmap_output_dir, "database.db")

    subprocess.run(["colmap", "feature_extractor",
                     "--database_path", db_path,
                     "--image_path", images_dir,
                     "--ImageReader.camera_model", "OPENCV",
                     "--ImageReader.single_camera", "1"], check=True)

    subprocess.run(["colmap", "sequential_matcher",
                     "--database_path", db_path,
                     "--SequentialMatching.loop_detection", "1"], check=True)

    os.makedirs(sparse_parent, exist_ok=True)
    subprocess.run(["colmap", "mapper",
                     "--database_path", db_path,
                     "--image_path", images_dir,
                     "--output_path", sparse_parent], check=True)

    # 5. Georeference using geo_reference.txt, export TXT model into sparse_geo_dir
    os.makedirs(sparse_geo_dir, exist_ok=True)
    subprocess.run(["colmap", "model_aligner",
                     "--input_path", sparse_dir,
                     "--output_path", sparse_geo_dir,
                     "--ref_images_path", geo_reference_path,
                     "--ref_is_gps", "1",
                     "--alignment_type", "ecef",
                     "--robust_alignment", "1",
                     "--robust_alignment_max_error", "3.0"], check=True)

    subprocess.run(["colmap", "model_converter",
                     "--input_path", sparse_geo_dir,
                     "--output_path", sparse_geo_dir,
                     "--output_type", "TXT"], check=True)

    return sparse_geo_dir
