import runpod
import traceback
from automatic_mask_generator_utils import *


def handler(job):
    try:
        image_base64 = job["input"]["image_base64"]
        masks_base64 = create_mask(image_base64)

        return {"results": masks_base64}

    except Exception as e:
        return {"message": f"An error occurred: {traceback.format_exc()}"}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False
})