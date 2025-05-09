import base64
from io import BytesIO
import io
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2ImagePredictor(
    sam_model = sam_model
)

def decode_base64_image(image_base64: str) -> np.ndarray:
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(image)

def numpy_mask_to_base64(mask):
    if isinstance(mask, np.ndarray):
        mask = (mask * 255).astype(np.uint8)
        img = Image.fromarray(mask)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str
    else:
        raise ValueError("Expected numpy array for mask, got: {}".format(type(mask)))

def get_masks_from_sam2(mask_generator, images, bboxes):
    masks_batch = []
    mask_generator.set_image(images[0])
    
    masks, ious, low_res_masks = mask_generator.predict(
        box=bboxes,
        multimask_output=False,
        return_logits=False,
    )

    print(f"Generated {len(masks)} masks for the image.")

    for mask, bbox in zip(masks, bboxes):
        mask = np.squeeze(mask)

        x1, y1, x2, y2 = bbox

        cropped_mask = mask[y1:y2, x1:x2]

        mask_base64 = numpy_mask_to_base64(cropped_mask)
        masks_batch.append({
            "mask_base64": mask_base64,
            "bbox": bbox.tolist()
        })

    return masks_batch
