import os
import cv2
import numpy as np
from tqdm import tqdm
from automatic_mask_generator_utils import create_mask
from PIL import Image
import base64
import zipfile
import shutil

def norm_to_pixel_bbox(norm_bbox, img_w, img_h):
    cx, cy, bw, bh = norm_bbox
    x_min = int((cx - bw / 2) * img_w)
    y_min = int((cy - bh / 2) * img_h)
    x_max = int((cx + bw / 2) * img_w)
    y_max = int((cy + bh / 2) * img_h)
    return max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)

def process_split(split_path, output_base_dir, max_images=None):
    split_name = os.path.basename(split_path.rstrip("/"))
    cropped_img_dir = os.path.join(output_base_dir, split_name, "cropped_images")
    cropped_mask_dir = os.path.join(output_base_dir, split_name, "cropped_mask_images")
    original_mask_dir = os.path.join(output_base_dir, split_name, "original_mask_images")
    os.makedirs(cropped_img_dir, exist_ok=True)
    os.makedirs(cropped_mask_dir, exist_ok=True)
    os.makedirs(original_mask_dir, exist_ok=True)

    processed, skipped = 0, 0

    image_files = [f for f in os.listdir(split_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc=f"Processing {split_name}"):
        if max_images is not None and processed >= max_images:
            break

        image_path = os.path.join(split_path, filename)
        label_path = os.path.join(split_path, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
            skipped += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue

        H, W = img.shape[:2]
        base_name = os.path.splitext(filename)[0]
        combined_mask = np.zeros((H, W), dtype=np.uint8)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            norm_bbox = list(map(float, parts[1:]))
            x0, y0, x1, y1 = norm_to_pixel_bbox(norm_bbox, W, H)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            mask = create_mask(crop)
            if mask is None:
                continue

            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            elif isinstance(mask, str):
                mask_data = base64.b64decode(mask)
                mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)

            crop_h, crop_w = y1 - y0, x1 - x0
            if mask.shape != (crop_h, crop_w):
                mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

            crop_filename = f"{base_name}_{idx}.png"
            mask_filename = f"{base_name}_{idx}_mask.png"

            cv2.imwrite(os.path.join(cropped_img_dir, crop_filename), crop)
            cv2.imwrite(os.path.join(cropped_mask_dir, mask_filename), mask)

            combined_mask[y0:y1, x0:x1] = np.maximum(combined_mask[y0:y1, x0:x1], mask)

        combined_mask_path = os.path.join(original_mask_dir, f"{base_name}_mask.png")
        cv2.imwrite(combined_mask_path, combined_mask)

        processed += 1

    print(f"Split: {split_name} | Processed: {processed} | Skipped: {skipped}")

def process_all_splits(dataset_root, output_dir, max_images_per_split=None):
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_root, split)
        if os.path.isdir(split_path):
            process_split(split_path, output_dir, max_images=max_images_per_split)

if __name__ == "__main__":
    dataset_root = "combined_dataset_split_rabin_second_all"
    output_dir = "combined_dataset_split_rabin_second_all_masks"
    dataset_zip = dataset_root + ".zip"
    output_zip = output_dir + ".zip"

    if not os.path.exists(dataset_root):
        if os.path.exists(dataset_zip):
            print(f"Extracting {dataset_zip}...")
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                os.makedirs(dataset_root, exist_ok=True)
                zip_ref.extractall(dataset_root)
            print("Extraction complete.")
        else:
            print(f"Dataset zip file not found: {dataset_zip}")
            exit(1)

    process_all_splits(
        dataset_root=dataset_root,
        output_dir=output_dir
    )

    print(f"Zipping output directory to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    print("Zipping complete.")
    shutil.rmtree(output_dir)