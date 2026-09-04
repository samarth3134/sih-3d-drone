import cv2
import glob
import numpy as np
import os
from pathlib import Path
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def run(config):
    """
    Depth estimation stage for the SIH pipeline.

    Reads frames from the masked-frames directory produced by the
    previous pipeline stage and writes Depth Anything V2 depth maps
    to the configured depth_maps directory.
    """

    # ---------------------------------------------------------
    # 1. Read paths from config.yaml
    # ---------------------------------------------------------
    masked_dir = config["paths"]["masked_frames_dir"]
    frames_dir = config["paths"]["frames_dir"]
    output_dir = config["paths"]["depth_maps_dir"]

    # ---------------------------------------------------------
    # 2. Prefer masked frames.
    #    If they aren't available, fall back to normal frames.
    # ---------------------------------------------------------
    filenames = []

    for extension in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        filenames.extend(glob.glob(os.path.join(masked_dir, extension)))

    if not filenames:
        print("No masked frames found. Using original frames.")

        for extension in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            filenames.extend(glob.glob(os.path.join(frames_dir, extension)))

    filenames.sort()

    if not filenames:
        raise FileNotFoundError(
            f"No input frames found in:\n"
            f"  {masked_dir}\n"
            f"or\n"
            f"  {frames_dir}"
        )

    # ---------------------------------------------------------
    # 3. Create output directory
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 4. Select device
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Depth Anything V2 device: {device}")

    # ---------------------------------------------------------
    # 5. Depth Anything V2 Small (ViT-S)
    # ---------------------------------------------------------
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        }
    }

    depth_anything = DepthAnythingV2(**model_configs["vits"])

    # Checkpoint is expected at:
    # <repo>/checkpoints/depth_anything_v2_vits.pth
    checkpoint_path = Path(config["depth"]["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Depth Anything checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    depth_anything.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu"
        )
    )

    depth_anything = depth_anything.to(device).eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # ---------------------------------------------------------
    # 6. Process every frame
    # ---------------------------------------------------------
    total = len(filenames)

    for index, filename in enumerate(filenames, start=1):

        print(f"Depth estimation {index}/{total}: {filename}")

        image = cv2.imread(filename)

        if image is None:
            print(f"Warning: could not read {filename}")
            continue

        # Generate relative depth
        depth = depth_anything.infer_image(
            image,
            518
        )

        # -----------------------------------------------------
        # Save raw model output for future metric calibration
        # -----------------------------------------------------
        base_name = os.path.splitext(
            os.path.basename(filename)
        )[0]

        np.save(
            os.path.join(output_dir, base_name + ".npy"),
            depth
        )

        # -----------------------------------------------------
        # Create an 8-bit grayscale depth map for the pipeline
        # -----------------------------------------------------
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max > depth_min:
            depth_normalized = (
                (depth - depth_min)
                / (depth_max - depth_min)
                * 255.0
            )
        else:
            depth_normalized = np.zeros_like(depth)

        depth_normalized = depth_normalized.astype(np.uint8)

        output_path = os.path.join(
            output_dir,
            base_name + ".png"
        )

        cv2.imwrite(
            output_path,
            depth_normalized
        )

    print(
        f"Depth estimation complete. "
        f"{total} frames processed."
    )

    print(f"Depth maps saved to: {output_dir}")

    return output_dir