import runpod
import traceback
import numpy as np
from sam2_utils import generate_yolo_annotations_from_image
from azure.storage.blob import BlobServiceClient
from PIL import Image
import io
import os
import cv2
import dotenv
dotenv.load_dotenv(override=True)

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_CONTAINER_NAME = "wsi"


def download_from_azure_blob(connection_string, container_name, blob_name):
    """
    Download a file from Azure Blob Storage and convert to numpy array
    
    Args:
        connection_string (str): Azure Storage connection string
        container_name (str): Name of the container
        blob_name (str): Name of the blob to download
        
    Returns:
        numpy.ndarray: The downloaded image as a numpy array
    """
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get a reference to the blob
        blob_client = container_client.get_blob_client(blob_name)
        
        # Download the blob
        download_stream = blob_client.download_blob()
        blob_data = download_stream.readall()
        
        # Convert bytes to numpy array using PIL
        image = Image.open(io.BytesIO(blob_data)).convert("RGB")
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        print(f"Error downloading from Azure Blob Storage: {str(e)}")
        raise


def upload_text_to_azure_blob(connection_string, container_name, blob_name, text_content):
    """
    Upload text content to Azure Blob Storage
    
    Args:
        connection_string (str): Azure Storage connection string
        container_name (str): Name of the container
        blob_name (str): Name of the blob to upload
        text_content (str): Text content to upload
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get a reference to the blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Upload the text content
        blob_client.upload_blob(text_content, overwrite=True)
        
        print(f"Successfully uploaded text to {blob_name}")
        return True
        
    except Exception as e:
        print(f"Error uploading text to Azure Blob Storage: {str(e)}")
        return False


def upload_image_to_azure_blob(connection_string, container_name, blob_name, image_array):
    """
    Upload image array to Azure Blob Storage as PNG
    
    Args:
        connection_string (str): Azure Storage connection string
        container_name (str): Name of the container
        blob_name (str): Name of the blob to upload
        image_array (numpy.ndarray): Image array to upload
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get a reference to the blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Convert numpy array to PNG bytes
        if len(image_array.shape) == 2:  # Grayscale
            image_pil = Image.fromarray(image_array, mode='L')
        else:  # RGB
            image_pil = Image.fromarray(image_array)
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Upload the image content
        blob_client.upload_blob(img_buffer.getvalue(), overwrite=True)
        
        print(f"Successfully uploaded image to {blob_name}")
        return True
        
    except Exception as e:
        print(f"Error uploading image to Azure Blob Storage: {str(e)}")
        return False


def handler(job):
    try:
        # Extract input parameters
        image_blob_name = job["input"]["image_blob_name"]
        sample_type = job["input"]["sample_type"]
        patient_id = job["input"]["patient_id"]
        generator_type = job["input"].get("generator_type", "single_shot")
        
        # Extract base name without extension for output files
        base_name = os.path.splitext(image_blob_name)[0]
        
        # Construct blob path
        blob_name = f"{patient_id}_{sample_type}/{image_blob_name}"
        
        print(f"Processing image: {blob_name}")
        print(f"Generator type: {generator_type}")
        
        # Download image from Azure
        image_array = download_from_azure_blob(AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, blob_name)
        print(f"Downloaded image with shape: {image_array.shape}")
        
        # Generate YOLO annotations and mask
        yolo_annotations, combined_mask = generate_yolo_annotations_from_image(
            image_array, 
            class_id=0, 
            min_area_ratio=0.0001, 
            max_area_ratio=0.5, 
            generator_type=generator_type
        )
        
        print(f"Generated {len(yolo_annotations)} YOLO annotations")
        print(f"Generated mask with shape: {combined_mask.shape}")
        
        # Prepare output file names
        annotations_blob_name = f"{patient_id}_{sample_type}/{base_name}.txt"
        mask_blob_name = f"{patient_id}_{sample_type}/{base_name}_mask.png"
        
        # Convert YOLO annotations list to text
        annotations_text = "\n".join(yolo_annotations) if yolo_annotations else ""
        
        # Upload annotations to Azure
        annotations_uploaded = upload_text_to_azure_blob(
            AZURE_STORAGE_CONNECTION_STRING,
            AZURE_STORAGE_CONTAINER_NAME,
            annotations_blob_name,
            annotations_text
        )
        
        # Upload mask to Azure
        mask_uploaded = upload_image_to_azure_blob(
            AZURE_STORAGE_CONNECTION_STRING,
            AZURE_STORAGE_CONTAINER_NAME,
            mask_blob_name,
            combined_mask
        )
        
        # Prepare response
        response = {
            "success": True,
            "input_image": blob_name,
            "total_annotations": len(yolo_annotations),
            "annotations_file": annotations_blob_name if annotations_uploaded else None,
            "mask_file": mask_blob_name if mask_uploaded else None,
            "mask_shape": combined_mask.shape,
            "generator_type": generator_type,
            "upload_status": {
                "annotations_uploaded": annotations_uploaded,
                "mask_uploaded": mask_uploaded
            }
        }
        
        if yolo_annotations:
            response["sample_annotations"] = yolo_annotations[:3]  # Show first 3 annotations as sample
        
        print(f"Processing completed successfully")
        print(f"Annotations uploaded: {annotations_uploaded}")
        print(f"Mask uploaded: {mask_uploaded}")
        
        return response

    except Exception as e:
        error_msg = f"An error occurred: {traceback.format_exc()}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False
})