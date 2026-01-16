import base64
from io import BytesIO
import io
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
from azure.storage.blob import BlobServiceClient
import requests
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

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

def download_image_from_azure_blob(account_name: str, container_name: str, blob_name: str, account_key: str = None, connection_string: str = None, sas_token: str = None) -> np.ndarray:
    """
    Download an image from Azure Blob Storage and return it as a numpy array.
    
    Args:
        account_name: Azure storage account name
        container_name: Container name where the blob is stored
        blob_name: Name of the blob (image file)
        account_key: Account key for authentication (optional)
        connection_string: Full connection string (optional)
        sas_token: SAS token for authentication (optional)
    
    Returns:
        np.ndarray: Image as numpy array in RGB format
    """
    try:
        # Initialize BlobServiceClient based on available credentials
        if connection_string:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_key:
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=account_key
            )
        elif sas_token:
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=sas_token
            )
        else:
            # Try with default credential (managed identity, etc.)
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net"
            )
        
        # Get blob client and download the image
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Download blob content
        blob_data = blob_client.download_blob().readall()
        
        # Convert to PIL Image and then to numpy array
        image = Image.open(io.BytesIO(blob_data)).convert("RGB")
        return np.array(image)
        
    except Exception as e:
        raise Exception(f"Failed to download image from Azure Blob Storage: {str(e)}")

def download_image_from_url(url: str) -> np.ndarray:
    """
    Download an image from a URL and return it as a numpy array.
    
    Args:
        url: Direct URL to the image
    
    Returns:
        np.ndarray: Image as numpy array in RGB format
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(image)
        
    except Exception as e:
        raise Exception(f"Failed to download image from URL: {str(e)}")

def generate_yolo_annotations_from_image(image_array: np.ndarray, class_id: int = 0, 
                                       min_area_ratio: float = 0.0001, 
                                       max_area_ratio: float = 0.5, generator_type: str = "single_shot") -> tuple:
    """
    Generate YOLO format annotations and combined mask from an image using SAM2 automatic mask generation.
    
    Args:
        image_array: Input image as numpy array (RGB format)
        class_id: Class ID for YOLO annotations (default: 0)
        min_area_ratio: Minimum mask area as ratio of total image area (default: 0.0001)
        max_area_ratio: Maximum mask area as ratio of total image area (default: 0.5)
        generator_type: Type of generator ("single_shot" or "multi_shot")
    
    Returns:
        Tuple containing:
        - List of YOLO format strings: ["class_id center_x center_y width height", ...]
        - Combined binary mask as numpy array (uint8, 0-255)
    """
    
    if generator_type == "multi_shot":
        # Initialize automatic mask generator for full image segmentation
        automatic_mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            # Optimized settings for cell images
            points_per_side=64,           # Reduced for smaller cell images
            points_per_batch=128,          # Batch size for processing
            pred_iou_thresh=0.7,          # Higher threshold for quality
            stability_score_thresh=0.92,   # Higher stability requirement
            stability_score_offset=0.7,   # Stability offset
            crop_n_layers=1,              # No crops for small cell images
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,    # Minimum area for small cells
            use_m2m=True,                 # Use mask-to-mask refinement
        )
    elif generator_type == "single_shot":
        automatic_mask_generator = SAM2AutomaticMaskGenerator(model=sam_model)
    
    
    try:
        # Ensure image is in RGB format
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be RGB format with shape (H, W, 3)")
        
        print(f"Generating masks for image with shape: {image_array.shape}")
        
        # Generate masks using SAM2 automatic mask generator
        masks = automatic_mask_generator.generate(image_array)
        
        # Get image dimensions
        img_height, img_width = image_array.shape[:2]
        
        if not masks:
            print("No masks generated")
            # Return empty annotations and empty mask
            empty_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            return [], empty_mask
        
        print(f"Generated {len(masks)} masks")
        
        # Calculate total pixels
        total_pixels = img_height * img_width
        
        # Filter masks to remove very large (background) and very small masks
        filtered_masks = filter_masks_by_area(masks, total_pixels, min_area_ratio, max_area_ratio)
        print(f"Filtered to {len(filtered_masks)} masks after area filtering")
        
        # Create combined binary mask
        combined_mask = np.zeros((img_height, img_width), dtype=bool)
        
        # Convert masks to YOLO annotations and combine masks
        yolo_annotations = []
        for mask_data in filtered_masks:
            mask = mask_data['segmentation']
            
            # Add to combined mask using logical OR
            combined_mask = np.logical_or(combined_mask, mask)
            
            # Convert mask to bounding box
            bbox = mask_to_bbox(mask)
            
            # Skip very small bounding boxes
            if bbox[2] < 5 or bbox[3] < 5:
                continue
            
            # Convert to YOLO format
            yolo_line = bbox_to_yolo_format(bbox, img_width, img_height, class_id)
            yolo_annotations.append(yolo_line)
        
        # Convert combined mask to uint8 format (0-255)
        combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        
        print(f"Generated {len(yolo_annotations)} YOLO annotations")
        print(f"Combined mask shape: {combined_mask_uint8.shape}, unique values: {np.unique(combined_mask_uint8)}")
        
        return yolo_annotations, combined_mask_uint8
        
    except Exception as e:
        print(f"Error generating YOLO annotations: {str(e)}")
        # Return empty results in case of error
        img_height, img_width = image_array.shape[:2]
        empty_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        return [], empty_mask

def filter_masks_by_area(masks: list, total_pixels: int, 
                        min_area_ratio: float = 0.0001, 
                        max_area_ratio: float = 0.5) -> list:
    """
    Filter masks based on area ratios to exclude very large background masks and very small noise.
    
    Args:
        masks: List of mask dictionaries from SAM2
        total_pixels: Total pixels in the image
        min_area_ratio: Minimum area ratio (default: 0.0001 = 0.01%)
        max_area_ratio: Maximum area ratio (default: 0.5 = 50%)
    
    Returns:
        Filtered list of mask dictionaries
    """
    filtered_masks = []
    
    for mask_data in masks:
        area = mask_data.get('area', 0)
        area_ratio = area / total_pixels
        
        # Keep masks within the specified area range
        if min_area_ratio <= area_ratio < max_area_ratio:
            filtered_masks.append(mask_data)
    
    # Sort by quality (IoU * stability_score) for better ordering
    filtered_masks.sort(
        key=lambda x: x.get('predicted_iou', 0) * x.get('stability_score', 0), 
        reverse=True
    )
    
    return filtered_masks

def mask_to_bbox(mask: np.ndarray) -> tuple:
    """
    Convert binary mask to bounding box.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Bounding box as (x, y, width, height)
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return (0, 0, 0, 0)
        
        # Get bounding rectangle of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
        
    except Exception as e:
        print(f"Error converting mask to bbox: {str(e)}")
        return (0, 0, 0, 0)

def bbox_to_yolo_format(bbox: tuple, img_width: int, img_height: int, class_id: int = 0) -> str:
    """
    Convert bounding box to YOLO format.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_id: Class ID (default: 0)
        
    Returns:
        YOLO format string: "class_id center_x center_y width height"
    """
    x, y, w, h = bbox
    
    # Convert to YOLO format (normalized coordinates)
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    
    # Ensure values are within [0, 1] range
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))
    
    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"


if __name__ == "__main__":
    #local read image
    image_path = "images/test.png"
    image_array = cv2.imread(image_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    yolo_annotations, combined_mask = generate_yolo_annotations_from_image(
        image_array, 
        class_id=0, 
        min_area_ratio=0.0001, 
        max_area_ratio=0.5, 
        generator_type="single_shot"
    )
    print("YOLO Annotations:")
    print(yolo_annotations)
    print(f"Combined mask shape: {combined_mask.shape}")
    print(f"Mask pixel values: {np.unique(combined_mask)}")
    
    # Optionally save the mask
    cv2.imwrite("output_mask.png", combined_mask)