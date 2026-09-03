import os
import glob
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# MODEL PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

SIH_ROOT = os.path.dirname(PROJECT_ROOT)

YOLO_MODEL = os.path.join(
    SIH_ROOT,
    "dynamic_object_removal",
    "yolov8n.pt"
)

SAM2_DIR = os.path.join(
    SIH_ROOT,
    "sam2"
)

SAM2_CHECKPOINT = os.path.join(
    SAM2_DIR,
    "checkpoints",
    "sam2.1_hiera_small.pt"
)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.20

DYNAMIC_CLASSES = {
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}


# ============================================================
# MODELS
# ============================================================

_models = None


def load_models():

    global _models

    if _models is not None:
        return _models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dynamic masking device: {device}")

    if device == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("Loading YOLO...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading SAM2...")
    sam2_model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=device,
    )

    sam_predictor = SAM2ImagePredictor(
        sam2_model
    )

    print("Loading LaMa...")
    lama = SimpleLama()

    print("Dynamic masking models loaded.")

    _models = (
        yolo,
        sam_predictor,
        lama
    )

    return _models


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    output_dir,
    mask_dir,
    yolo,
    sam_predictor,
    lama,
):

    filename = os.path.basename(image_path)

    print(f"Processing: {filename}")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = yolo(
        image,
        conf=CONFIDENCE,
        imgsz=1280,
        verbose=False
    )[0]

    boxes = []

    for box in results.boxes:

        class_id = int(box.cls[0])

        if class_id not in DYNAMIC_CLASSES:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        if x2 > x1 and y2 > y1:
            boxes.append(
                [x1, y1, x2, y2]
            )

    print(
        f"Dynamic objects detected: "
        f"{len(boxes)}"
    )

    # --------------------------------------------------------
    # NO OBJECTS
    # --------------------------------------------------------

    if not boxes:

        output_path = os.path.join(
            output_dir,
            filename
        )

        mask_path = os.path.join(
            mask_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        cv2.imwrite(
            output_path,
            image
        )

        empty_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.imwrite(
            mask_path,
            empty_mask
        )

        return output_path

    # --------------------------------------------------------
    # SAM2 SEGMENTATION
    # --------------------------------------------------------

    sam_predictor.set_image(rgb)

    final_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for box in boxes:

        box_np = np.array(
            box,
            dtype=np.float32
        )

        masks, scores, _ = (
            sam_predictor.predict(
                box=box_np,
                multimask_output=True
            )
        )

        best_index = np.argmax(scores)

        object_mask = (
            masks[best_index]
            .astype(bool)
        )

        final_mask[object_mask] = 255

    # --------------------------------------------------------
    # DILATION
    # --------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    final_mask = cv2.dilate(
        final_mask,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    mask_path = os.path.join(
        mask_dir,
        os.path.splitext(filename)[0] + ".png"
    )

    cv2.imwrite(
        mask_path,
        final_mask
    )

    # --------------------------------------------------------
    # LAMA INPAINTING
    # --------------------------------------------------------

    print("Running LaMa...")

    from PIL import Image

    pil_image = Image.fromarray(rgb)
    pil_mask = Image.fromarray(final_mask)

    result = lama(
        pil_image,
        pil_mask
    )

    result = np.array(result)

    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        filename
    )

    cv2.imwrite(
        output_path,
        result
    )

    print(f"Saved: {output_path}")

    return output_path


# ============================================================
# MAIN TEAM INTERFACE
# ============================================================

def run(config):

    frames_dir = config["paths"]["frames_dir"]
    masked_frames_dir = config["paths"]["masked_frames_dir"]

    # Convert paths relative to the team repository
    if not os.path.isabs(frames_dir):
        frames_dir = os.path.join(
            PROJECT_ROOT,
            frames_dir
        )

    if not os.path.isabs(masked_frames_dir):
        masked_frames_dir = os.path.join(
            PROJECT_ROOT,
            masked_frames_dir
        )

    mask_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "dynamic_masks"
    )

    os.makedirs(
        masked_frames_dir,
        exist_ok=True
    )

    os.makedirs(
        mask_dir,
        exist_ok=True
    )

    images = sorted(
        glob.glob(
            os.path.join(
                frames_dir,
                "*.jpg"
            )
        )
    )

    print(
        f"Dynamic masking found "
        f"{len(images)} frames."
    )

    if not images:
        raise RuntimeError(
            f"No JPG frames found in {frames_dir}"
        )

    yolo, sam_predictor, lama = load_models()

    outputs = []

    for image_path in images:

        output = process_image(
            image_path,
            masked_frames_dir,
            mask_dir,
            yolo,
            sam_predictor,
            lama,
        )

        if output:
            outputs.append(output)

    print(
        f"Dynamic masking complete: "
        f"{len(outputs)} frames."
    )

    return outputs


# ============================================================
# COMPATIBILITY INTERFACE
# ============================================================

def mask_dynamic_objects(frames, config):

    return run(config)
