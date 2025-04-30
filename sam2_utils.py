import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.85,
    min_mask_region_area=100
)

def segment_cells_with_sam(image):
    """
    Segment cells in an image using SAM2.
    
    Args:
        image: numpy array of the image in BGR format (as returned by cv2.imread)
    
    Returns:
        numpy array: Binary mask of the segmented cells
    """
    if image is None:
        raise ValueError("Input image is None")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)
    print(f"[DEBUG] Total masks generated: {len(masks)}")

    if not masks:
        print("[DEBUG] No masks found.")
        return None

    MIN_AREA = 500
    large_masks = [m for m in masks if np.sum(m['segmentation']) > MIN_AREA]

    if large_masks:
        best_mask_info = max(large_masks, key=lambda m: np.sum(m['segmentation']))
    else:
        print("[DEBUG] No large masks found. Using first available mask.")
        best_mask_info = masks[0]

    mask = (best_mask_info['segmentation'] > 0).astype(np.uint8) * 255
    print("[DEBUG] Mask extracted.")
    return mask