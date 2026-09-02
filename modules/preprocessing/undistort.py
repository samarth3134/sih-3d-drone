"""
Lens distortion correction.

Two modes:
1. --calib-json: you have real calibration (K + distortion coeffs) -> just undistort.
2. --sensor-width-mm / --focal-length-mm: you don't have a calibration -> approximate K
   from the drone's known sensor size and focal length, assume near-zero distortion.
   This does not need to be perfect: COLMAP re-refines intrinsics during bundle
   adjustment as long as you give it a reasonable starting point and enough frames.

Usage:
    python undistort.py --indir clean_frames --outdir clean_frames_undist \
        --sensor-width-mm 6.3 --focal-length-mm 4.5 --image-width 3840 --image-height 2160
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def build_intrinsics_from_specs(sensor_width_mm, focal_length_mm, image_width, image_height):
    fx = fy = (focal_length_mm / sensor_width_mm) * image_width
    cx, cy = image_width / 2.0, image_height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def load_calibration(path):
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)
    return K, dist


def undistort_dir(indir, outdir, K, dist):
    os.makedirs(outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(indir, "*.jpg")))
    if not paths:
        raise RuntimeError(f"No .jpg frames found in {indir}")

    effective_K = None
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        und = cv2.undistort(img, K, dist, None, newK)
        cv2.imwrite(os.path.join(outdir, os.path.basename(p)), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i == 0:
            effective_K = newK

    np.savez(os.path.join(outdir, "_intrinsics.npz"), K=effective_K, dist=np.zeros(5))
    print(f"Undistorted {len(paths)} frames -> {outdir}")
    print("Saved effective K to _intrinsics.npz — feed fx from here into COLMAP's "
          "--ImageReader.camera_params if you want to seed the mapper with it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calib-json", help="Path to a {\"K\": [[...]], \"dist\": [...]} JSON file")
    ap.add_argument("--sensor-width-mm", type=float)
    ap.add_argument("--focal-length-mm", type=float)
    ap.add_argument("--image-width", type=int)
    ap.add_argument("--image-height", type=int)
    args = ap.parse_args()

    if args.calib_json:
        K, dist = load_calibration(args.calib_json)
    else:
        required = [args.sensor_width_mm, args.focal_length_mm, args.image_width, args.image_height]
        if any(v is None for v in required):
            raise SystemExit("Provide --calib-json, or all of --sensor-width-mm/--focal-length-mm/"
                              "--image-width/--image-height (check the drone's spec sheet).")
        K, dist = build_intrinsics_from_specs(
            args.sensor_width_mm, args.focal_length_mm, args.image_width, args.image_height)

    undistort_dir(args.indir, args.outdir, K, dist)
