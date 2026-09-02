"""
Extract frames from drone video, filter blurry frames using Laplacian variance.

Usage:
    python extract_frames.py --video input.mp4 --outdir clean_frames --fps 4 --blur-thresh 100
"""
import argparse
import os
import cv2
import numpy as np


def laplacian_variance(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def extract_frames(video_path, outdir, target_fps=4.0, blur_thresh=100.0, resize_width=None):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(src_fps / target_fps))

    frame_idx = 0
    saved_idx = 0
    kept, dropped = 0, 0
    blur_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = laplacian_variance(gray)
            blur_scores.append(score)

            if score >= blur_thresh:
                if resize_width:
                    h, w = frame.shape[:2]
                    scale = resize_width / w
                    frame = cv2.resize(frame, (resize_width, int(h * scale)))
                fname = os.path.join(outdir, f"frame_{saved_idx:06d}.jpg")
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_idx += 1
                kept += 1
            else:
                dropped += 1

        frame_idx += 1

    cap.release()

    print(f"Source FPS: {src_fps:.2f}, sampling every {frame_interval} frames (~{target_fps} fps target)")
    print(f"Frames kept: {kept}, dropped as blurry: {dropped}")
    if blur_scores:
        print(f"Blur score range: min={min(blur_scores):.1f} max={max(blur_scores):.1f} "
              f"median={float(np.median(blur_scores)):.1f}")
    print("NOTE: single-pass footage has less redundancy than multi-pass photogrammetry, "
          "so start with a HIGH sampling fps (4-6) and a LOW blur threshold. It's much safer "
          "to keep marginal frames and let COLMAP down-weight them than to starve the mapper "
          "of viewpoints it needs for triangulation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="clean_frames")
    ap.add_argument("--fps", type=float, default=4.0,
                     help="Target sampling rate in frames/sec.")
    ap.add_argument("--blur-thresh", type=float, default=100.0,
                     help="Laplacian variance threshold. Print the score range first, "
                          "then pick a cutoff that only drops the visibly-bad frames.")
    ap.add_argument("--resize-width", type=int, default=None,
                     help="Optional downscale (e.g. 1920) to speed up COLMAP matching.")
    args = ap.parse_args()
    extract_frames(args.video, args.outdir, args.fps, args.blur_thresh, args.resize_width)
