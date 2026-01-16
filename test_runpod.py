import requests
import time
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def runpod_request(endpoint_id, json_file):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"
    }
    
    with open(json_file, "r") as file:
        data = json.load(file)
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def check_status(job_id, status_endpoint):
    url = f"https://api.runpod.ai/v2/{status_endpoint}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"
    }

    while True:
        response = requests.get(url, headers=headers).json()
        status = response.get("status", "UNKNOWN")
        print(f"Job Status: {status}")

        if status in ["COMPLETED", "FAILED"]:
            return response
        
        time.sleep(5)  # Wait for 5 seconds before checking again

# Example usage
endpoint_id = "1za6jg908909ki"
json_file = "test_input_sam_od.json"

job_response = runpod_request(endpoint_id, json_file)
job_id = job_response.get("id")

if job_id:
    final_response = check_status(job_id, endpoint_id)
    print(final_response)
