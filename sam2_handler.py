import runpod
import traceback
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sam2_utils import segment_cells_with_sam

def decode_base64_to_numpy(image_base64):
    img_data = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_data))
    # Convert PIL Image to numpy array in BGR format
    img_np = np.array(img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:  # If RGBA, convert to RGB
        img_np = img_np[:, :, :3]
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # If RGB, convert to BGR
        img_np = img_np[:, :, ::-1]
    return img_np

def handler(job):
    try:
        image_base64 = job["input"]["image_base64"]
        
        # Convert base64 directly to numpy array
        image_np = decode_base64_to_numpy(image_base64)
        
        # Get mask directly from numpy array
        mask_image = segment_cells_with_sam(image_np)

        if mask_image is None:
            return None

        # Convert numpy array to PIL Image for base64 encoding
        mask_pil = Image.fromarray(mask_image)
        
        buffered = BytesIO()
        mask_pil.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "mask_base64": mask_base64
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred: {traceback.format_exc()}"
        }

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False
})