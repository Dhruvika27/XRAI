import pandas as pd
import os
import re
from pathlib import Path
from PIL import Image # Need to import Image to test opening, not just existence

# --- Configuration (EDIT THESE PATHS TO MATCH YOUR SYSTEM) ---

# 1. Your training CSV file location
INPUT_CSV_PATH = r"C:\xray\train__new_data.csv"

# 2. The *actual* root folder where your images reside 
IMAGE_ROOT_DIR = r"C:\xray\deid_png" 

# Number of samples to check
SAMPLE_SIZE = 50  

# =========================================================
# Utility Function: Long Path Handling
# =========================================================

def handle_long_path(path_str):
    """
    Prepends the Windows long path prefix '\\\\?\\' if the OS is Windows (nt).
    This allows accessing paths longer than 260 characters.
    """
    if os.name == 'nt' and path_str.startswith(('C:', 'D:')):
        # Ensure path uses backslashes for the prefix to work consistently
        # Note: os.path.exists and Image.open can usually handle forward slashes, 
        # but the prefix is safer with backslashes.
        normalized_path = os.path.normpath(path_str)
        return f"\\\\?\\{normalized_path}"
    return path_str.replace('\\', '/')

# =========================================================
# Image Path Correction Function 
# (Ensures all paths in DataFrame are standardized)
# =========================================================

def correct_image_paths_in_df(df, image_root_dir):
    """
    Robustly corrects and standardizes paths in the DataFrame's 'img_path' column.
    """
    print("\n--- Applying Robust Image Path Correction Logic ---")
    
    root_path = Path(image_root_dir)
    corrected_paths = []
    root_segment_lower = root_path.name.lower() # 'deid_png'

    for i, original_path in enumerate(df['img_path']):
        path_str = original_path.replace('\\', '/')
        new_path = path_str
        
        # Check 1: If already absolute and starts with the root
        if Path(path_str).is_absolute() and path_str.lower().startswith(str(root_path).lower().replace('\\', '/')):
             new_path = os.path.normpath(path_str)
        else:
            # Check 2: Reconstruct path using the root segment
            try:
                path_str_lower = path_str.lower()
                root_index = path_str_lower.rindex(root_segment_lower)
                
                relative_path_part = path_str[root_index:]
                path_to_join = relative_path_part[len(root_segment_lower):].lstrip('/\\')
                
                new_path = os.path.normpath(Path(image_root_dir) / path_to_join)
                
            except ValueError:
                # Fallback: keep the original path segment
                new_path = os.path.normpath(Path(image_root_dir) / Path(path_str).name)

        # Store the standardized path (no long-path prefix yet)
        # We will add the prefix dynamically when checking/opening files
        new_path_str = str(new_path).replace('\\', '/')
        corrected_paths.append(new_path_str)
        
        if i < 5:
            print(f"   Sample {i}: Original: {original_path}")
            print(f"   Sample {i}: Corrected: {new_path_str}")

    df['img_path'] = corrected_paths
    print("✅ Path correction applied.")
    return df

# =========================================================
# Main Check Logic
# =========================================================

print(f"Loading data from: {INPUT_CSV_PATH}")

try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Input CSV not found at {INPUT_CSV_PATH}. Please check the path.")
    exit()

# Apply the correction logic
df = correct_image_paths_in_df(df, IMAGE_ROOT_DIR)

# Select a sample for checking
df_sample = df.head(SAMPLE_SIZE)
total_samples = len(df_sample)

found_count = 0
not_found_list = []

print(f"\n--- Checking existence and ability to OPEN {total_samples} image files ---")

for index, row in df_sample.iterrows():
    img_path = row['img_path']
    long_path = handle_long_path(img_path)
    
    # --- YOUR SPECIFIC CHECK LOGIC IMPLEMENTED HERE ---
    is_found = os.path.exists(long_path)
    is_openable = False

    if is_found:
        try:
            # Attempt to open the image using the long path prefix
            image = Image.open(long_path)
            image.close()
            is_openable = True
            found_count += 1
            if found_count <= 5:
                 print(f"✅ Found and opened: {img_path}")

        except Exception as e:
            # File exists but cannot be opened (e.g., corruption, wrong format)
            not_found_list.append(f"{img_path} (Failed to open: {e})")
            if len(not_found_list) <= 5:
                 print(f"⚠️ Found, but failed to open: {img_path} Error: {e}")
    else:
        # File not found
        not_found_list.append(img_path)

# --- Summary Report ---
print("\n--- Image Path Validation Summary ---")
print(f"Total samples checked: {total_samples}")
print(f"Files found and opened: {found_count}")
print(f"Files NOT found or failed to open: {len(not_found_list)}")

if len(not_found_list) > 0:
    print("\n❌ The following paths failed the check (first 5 examples):")
    for i, path in enumerate(not_found_list):
        if i >= 5:
            break
        print(f"   {path}")
    print("\nACTION REQUIRED: If 'Files found' is 0, manually adjust `IMAGE_ROOT_DIR` until the samples are found.")
else:
    print("\n✅ All sampled files were found and opened! Proceed to the full training script.")