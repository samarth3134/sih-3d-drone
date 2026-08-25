import os
import yaml
from pathlib import Path

from modules.preprocessing import Preprocessor
from modules.depth_estimation import DepthEstimator
from modules.dynamic_masking import DynamicMasker
from modules.reconstruction import Reconstructor
from modules.visualization import Visualizer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    input_video = config["input"]["video_path"]
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preprocessing video frames...")
    preprocessor = Preprocessor(config["preprocessing"])
    frames = preprocessor.extract_frames(input_video)

    print("[2/5] Estimating depth maps...")
    depth_estimator = DepthEstimator(config["depth_estimation"])
    depth_maps = depth_estimator.estimate_depth(frames)

    print("[3/5] Generating dynamic masks...")
    masker = DynamicMasker(config["dynamic_masking"])
    masks, masked_frames = masker.generate_masks(frames)

    print("[4/4] Reconstructing 3D scene...")
    reconstructor = Reconstructor(config["reconstruction"])
    point_cloud = reconstructor.reconstruct(frames, depth_maps, masks)

    print("[5/5] Visualizing results...")
    visualizer = Visualizer(config["visualization"])
    visualizer.render(point_cloud, output_dir)

    print(f"Pipeline complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
