import runpod
import traceback
import tempfile
import os
from sam2_utils import segment_cells_with_sam
import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(image_base64):
    img_data = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_data))
    return img

def save_temp_image(image, filename):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        filename = f"{os.path.splitext(filename)[0]}.png"
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, filename)
    image.save(temp_image_path)
    return temp_image_path, temp_dir

def handler(job):
    try:
        image_base64 = job["input"]["image_base64"]
        filename = job["input"]["filename"]

        image = decode_base64_image(image_base64)
        temp_image_path, temp_dir = save_temp_image(image, filename)

        mask_image = segment_cells_with_sam(temp_image_path)

        if mask_image is None:
            return None

        if not isinstance(mask_image, Image.Image):
            mask_image = Image.fromarray(mask_image)

        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
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