import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh = 0.6,
    stability_score_thresh = 0.7,
    min_mask_region_area = 1500
)

def resize_with_padding(img, target_size=512):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    delta_w = target_size - resized.shape[1]
    delta_h = target_size - resized.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    return padded_img, scale, left, top

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

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.equalizeHist(s)
    hsv_enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


def create_mask(image: np.ndarray) -> np.ndarray | None:
    enhanced = enhance_contrast(image)
    denoised = denoise_image(enhanced)
    sharpened = sharpen_image(denoised)

    padded_img, scale, x_off, y_off = resize_with_padding(sharpened, target_size=512)
    masks = mask_generator.generate(padded_img)
    center_mask_info = find_center_cell_mask(masks, padded_img.shape)

    if center_mask_info is None:
        print("No valid mask found for crop")
        return None

    mask = (center_mask_info['segmentation'] > 0).astype(np.uint8) * 255
    h, w = image.shape[:2]
    mask_cropped = mask[y_off:y_off + int(h*scale), x_off:x_off + int(w*scale)]
    mask_final = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_final