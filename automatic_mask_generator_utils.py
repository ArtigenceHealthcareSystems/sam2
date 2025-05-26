import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import base64
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.85,
    min_mask_region_area=150
)

def decode_base64_image(image_base64: str) -> np.ndarray:
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)

def is_fully_inside(inner_mask, outer_mask):
    if inner_mask.shape != outer_mask.shape:
        return False
    return np.all(outer_mask[inner_mask])

def find_center_cell_mask(masks, image_shape, min_area=1500, max_area_ratio=0.9):
    img_center = np.array([image_shape[1] // 2, image_shape[0] // 2])
    img_size = image_shape[0] * image_shape[1]

    filtered = []
    for m in masks:
        mask = m['segmentation']
        area = np.sum(mask)

        if area < min_area or area > img_size * max_area_ratio:
            continue

        if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue

        ys, xs = np.where(mask)
        centroid = np.array([np.mean(xs), np.mean(ys)])
        dist = np.linalg.norm(centroid - img_center)

        filtered.append((m, dist, area))

    filtered.sort(key=lambda x: x[1])

    for i, (m, _, _) in enumerate(filtered):
        mask_i = m['segmentation']
        is_inner = False
        for j, (n, _, _) in enumerate(filtered):
            if i == j:
                continue
            if is_fully_inside(mask_i, n['segmentation']):
                is_inner = True
                break
        if not is_inner:
            return m

    return None

def encode_mask_to_base64(mask):
    pil_img = Image.fromarray(mask)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_mask(image_base64):
    image = decode_base64_image(image_base64)
    image = cv2.resize(image, (512, 512))

    masks = mask_generator.generate(image)
    center_mask_info = find_center_cell_mask(masks, image.shape)

    mask = (center_mask_info['segmentation'] > 0).astype(np.uint8) * 255

    if center_mask_info is None:
        print("No valid mask found for crop")
        return None
    
    mask_base64 = encode_mask_to_base64(mask)

    return mask_base64