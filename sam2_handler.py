import runpod
import traceback
import numpy as np
from sam2_utils import *

def handler(job):
    try:
        image_base64 = job["input"]["image_base64"]
        bboxes = job["input"]["bboxes"]
        
        images = [decode_base64_image(image_base64)]
        filtered_bboxes = []
        for bbox in bboxes:
            if len(bbox) == 5 and bbox[4] == "Circular_RBC":
                filtered_bboxes.append(np.array(bbox[:4]))
            else:
                print(f"Skipped BBOX: {bbox}")
        bboxes = filtered_bboxes

        masks_base64 = get_masks_from_sam2(mask_generator, images, bboxes)

        return {"results": masks_base64}

    except Exception as e:
        return {"message": f"An error occurred: {traceback.format_exc()}"}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False
})