import os
import shutil
import pandas as pd

def filter_and_isolate_cards(csv_path, image_folder_path, output_true_csv, output_false_csv, true_folder="selected_true_cards", false_folder="selected_false_cards"):
    """
    Cross-references images in a folder with a master CSV.
    Splits data based on label threshold (0.5):
    - True cards (< 0.5) go to true_folder and output_true_csv
    - False cards (>= 0.5) go to false_folder and output_false_csv
    """
    # 1. Load the master CSV file
    if not os.path.exists(csv_path):
        print(f"❌ Error: Input CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Verify required tracking columns exist
    required_columns = ['id', 'image_path', 'label', 'is_digital', 'type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The input CSV is missing expected columns: {missing_cols}")
        return
    
    # 2. Verify if the source image folder exists
    if not os.path.exists(image_folder_path):
        print(f"❌ Error: Image folder not found at {image_folder_path}")
        return

    # 3. Map filenames inside the image folder to handle extension variations smoothly
    all_files = os.listdir(image_folder_path)
    file_map = {}
    
    for file_name in all_files:
        if file_name.startswith('.'):
            continue
        # Map both full name ("egypt_dl.jpg") and base name ("egypt_dl") to the actual file
        file_map[file_name] = file_name                      
        file_map[os.path.splitext(file_name)[0]] = file_name  

    # 4. Standardize data types for accurate matching and math evaluation
    df['id_str'] = df['id'].astype(str)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 5. Filter records that actually exist physically in the image folder
    matched_df = df[df['id_str'].isin(file_map.keys())]

    if matched_df.empty:
        print("\n⚠️ Job finished: Zero items found in the folder matching the CSV IDs.")
        return

    # 6. Separate into True (label < 0.5) and False (label >= 0.5) dataframes
    true_df = matched_df[matched_df['label'] < 0.5]
    false_df = matched_df[matched_df['label'] >= 0.5]

    # Helper function to handle the physical copying of files to reduce redundant code
    def copy_card_images(target_df, target_folder):
        if target_df.empty:
            return 0
        os.makedirs(target_folder, exist_ok=True)
        copied_count = 0
        for _, row in target_df.iterrows():
            id_val = row['id_str']
            actual_filename = file_map.get(id_val)
            
            if actual_filename:
                source_file_path = os.path.join(image_folder_path, actual_filename)
                destination_file_path = os.path.join(target_folder, actual_filename)
                
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Failed to copy image {actual_filename}: {e}")
        return copied_count

    # 7. Process True Cards
    print(f"\n⚡ Processing True Cards (label < 0.5)...")
    true_copied = copy_card_images(true_df, true_folder)
    if not true_df.empty:
        true_df[required_columns].to_csv(output_true_csv, index=False)
        print(f"📊 Rows added to '{os.path.basename(output_true_csv)}': {len(true_df)}")
        print(f"🖼️ Images successfully copied to '{true_folder}': {true_copied}")
    else:
        print("ℹ️ No true cards found meeting the criteria.")

    # 8. Process False Cards
    print(f"\n⚡ Processing False Cards (label >= 0.5)...")
    false_copied = copy_card_images(false_df, false_folder)
    if not false_df.empty:
        false_df[required_columns].to_csv(output_false_csv, index=False)
        print(f"📊 Rows added to '{os.path.basename(output_false_csv)}': {len(false_df)}")
        print(f"🖼️ Images successfully copied to '{false_folder}': {false_copied}")
    else:
        print("ℹ️ No false cards found meeting the criteria.")

    print(f"\n✅ All execution operations completed successfully!")
    # --- CONFIGURATION AND PATH SETTINGS ---
if __name__ == "__main__":
    # Input paths
    INPUT_CSV = "/kaggle/input/datasets/holeneckles/the-freuid-challenge-2026-ijcai-ecai/train_labels.csv"
    IMAGE_FOLDER = "/kaggle/input/datasets/holeneckles/the-freuid-challenge-2026-ijcai-ecai/train/train"
    
    # Requested Output CSV paths
    OUTPUT_TRUE_CSV = "/kaggle/working/output_true.csv"
    OUTPUT_FALSE_CSV = "/kaggle/working/output_false.csv"
    
    # Requested Output Folder paths
    TARGET_TRUE_FOLDER = "selected_true_cards"
    TARGET_FALSE_FOLDER = "selected_false_cards"
    
    # Run the complete routine
    filter_and_isolate_cards(
        csv_path=INPUT_CSV, 
        image_folder_path=IMAGE_FOLDER, 
        output_true_csv=OUTPUT_TRUE_CSV, 
        output_false_csv=OUTPUT_FALSE_CSV, 
        true_folder=TARGET_TRUE_FOLDER, 
        false_folder=TARGET_FALSE_FOLDER
    )


