import os
import pandas as pd
import numpy as np
import shutil

def generate_audit_sample(csv_path, src_dir, dest_dir, sample_size=150):
    """
    Draws a stratified random sample of scans across the fill-percentage range.
    Copies those scans to an audit folder for manual review.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Run generate_ml_dataset.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Dataset is empty.")
        return
        
    # We want to stratify based on geometric label (Empty, Partial, Full)
    # to ensure we get a good mix of all edge cases.
    
    # If the dataset is smaller than the requested sample size, just take everything
    if len(df) <= sample_size:
        sampled_df = df
    else:
        # Calculate how many samples we need per class to get a balanced audit
        classes = df['geometric_label'].unique()
        n_classes = len(classes)
        samples_per_class = sample_size // n_classes
        
        sampled_dfs = []
        for c in classes:
            class_subset = df[df['geometric_label'] == c]
            
            # If a class has fewer samples than requested, take them all
            if len(class_subset) <= samples_per_class:
                sampled_dfs.append(class_subset)
            else:
                # Random sample
                sampled_dfs.append(class_subset.sample(n=samples_per_class, random_state=42))
                
        sampled_df = pd.concat(sampled_dfs)
        
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Drawing audit sample of {len(sampled_df)} trucks...")
    
    # Generate an audit review sheet
    review_sheet_path = os.path.join(dest_dir, "audit_review_sheet.csv")
    
    # We will output a simplified CSV for the human to review
    # The human is expected to add a column 'human_verified_label'
    review_df = sampled_df[['filename', 'geometric_volume_m3', 'normalized_fill_pct', 'geometric_label']].copy()
    review_df['human_verified_label'] = ""
    review_df['human_notes'] = ""
    
    review_df.to_csv(review_sheet_path, index=False)
    
    # Copy the files so the user can inspect them visually
    for idx, row in sampled_df.iterrows():
        src_file = os.path.join(src_dir, row['filename'])
        if os.path.exists(src_file):
            dest_file = os.path.join(dest_dir, row['filename'])
            shutil.copy(src_file, dest_file)
            
    print(f"Audit sampling complete!")
    print(f"Audit sheet saved to: {review_sheet_path}")
    print(f"Please review the trucks in {dest_dir} and fill out the 'human_verified_label' column in the CSV.")

if __name__ == "__main__":
    dataset_csv = r"d:\point cloud technology\Truck-PointCloud\ml_model\training_dataset.csv"
    zenodo_matches = r"d:\point cloud technology\Truck-PointCloud\zenodo_exact_matches"
    audit_dest = r"d:\point cloud technology\Truck-PointCloud\ml_model\audit_sample"
    
    generate_audit_sample(dataset_csv, zenodo_matches, audit_dest)
