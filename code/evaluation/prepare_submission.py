### Mounting kaggle competition

import kagglehub

# Download latest version
path = kagglehub.competition_download('the-freuid-challenge-2026-ijcai-ecai')

print("Path to competition files:", path)

import os
import glob
import cv2
import numpy as np
import pandas as pd

# Define paths (the script will search these fallback locations automatically)
SEARCH_DIRS = [
    "/data/user/0/ru.iiec.pydroid3/app_HOME/.cache/kagglehub/competitions/the-freuid-challenge-2026-ijcai-ecai/",
    "./data",
    "./"
]

def analyze_document_authenticity(image_path):
    """
    Analyzes an identity card or driver's license image for fraud markers
    using fast, classical computer vision features (No PyTorch).
    
    Returns a probability/confidence score in [0.0, 1.0].
    """
    # Load image in grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
         # Return a neutral score if the image cannot be read
        return 0.5 
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge & Sharpness Check (Laplacian Variance)
    # Printed-and-captured forgeries or blurred digital manipulations 
    # typically show different edge patterns compared to crisp originals.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Texture & Noise Frequency (Sobel Gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # 3. Color Uniformity Analysis
    # Anomalous color variances can indicate digital edits or screen recaptures
    color_std = np.std(img_bgr)
    
    # Map features to a fraud probability score [0.0, 1.0]
    # Lower sharpness (extreme blur) or overly flat features suggest higher fraud probability.
    sharpness_score = np.clip(lap_var / 600.0, 0, 1)
    texture_score = np.clip(grad_mag / 120.0, 0, 1)
    
    # Combined heuristic baseline: 
    # High edge sharpness/texture pushes the fraud score closer to 0.0 (original),
    # while poor edge quality pushes it closer to 1.0 (forged).
    fraud_score = 1.0 - (0.6 * sharpness_score + 0.4 * texture_score)
    fraud_score = np.clip(fraud_score, 0.0, 1.0)
    
    return int(round(float(fraud_score)))


def build_submission():
    # 1. Locate the dataset directory
    dataset_dir = None
    for path in SEARCH_DIRS:
        if os.path.exists(path):
            dataset_dir = path
            break
            
    if not dataset_dir:
        print("Error: Could not locate the test dataset directories automatically.")
        return

    print(f"Using dataset directory: {dataset_dir}")
    
    # 2. Discover test images in both public_test and private_test folders
    test_folders = ["public_test/public_test", "private_test/private_test"]
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.PNG", "*.JPG", "*.JPEG")
    
    all_images = []
    for folder in test_folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.exists(folder_path):
            print(f"Scanning folder: {folder_path}")
            for ext in image_extensions:
                # Recursively search for any matching images
                all_images.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
        else:
            print(f"Folder not found: {folder_path} (Skipping...)")
            
    if not all_images:
        print("No test images found inside public_test or private_test folders.")
        return
    
    print(f"Found {len(all_images)} images to process.")
    
    # 3. Analyze images and extract ID & labels
    submission_data = []
    for idx, img_path in enumerate(all_images):
        # Extract base image name without extension to serve as the ID
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Run the feature analysis
        fraud_score = analyze_document_authenticity(img_path)
        
        submission_data.append({
            "id": img_id,
            "label": fraud_score
        })
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(all_images)} images...")

    # 4. Save results to submission.csv
    df = pd.DataFrame(submission_data)
    
    # Remove any potential duplicates in ID just in case
    df = df.drop_duplicates(subset=["id"])
    
    df.to_csv("submission.csv", index=False)
    print("Successfully created 'submission.csv'!")
    print(df.head())


if __name__ == "__main__":
    build_submission()
