import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def split_dataset(csv_path, out_train_path, out_test_path, test_size=0.2):
    """
    Splits the dataset into train and test sets, grouping by truck_id to prevent data leakage.
    If the same physical truck passes multiple times, all its passes will end up entirely
    in either the train set or the test set, but not split across both.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Dataset is empty.")
        return
        
    print(f"Loaded {len(df)} samples.")
    print("Class Balance (Original):")
    print(df['geometric_label'].value_counts(normalize=True) * 100)
    print("\n")
    
    from sklearn.model_selection import StratifiedGroupKFold
    
    # Initialize StratifiedGroupKFold to group by truck_id and stratify by geometric_label
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, test_idx = next(sgkf.split(df, y=df['geometric_label'], groups=df['truck_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Train set: {len(train_df)} samples")
    print("Class Balance (Train):")
    print(train_df['geometric_label'].value_counts(normalize=True) * 100)
    
    print(f"\nTest set: {len(test_df)} samples")
    print("Class Balance (Test):")
    print(test_df['geometric_label'].value_counts(normalize=True) * 100)
    
    train_df.to_csv(out_train_path, index=False)
    test_df.to_csv(out_test_path, index=False)
    
    print(f"\nSaved split datasets to:\n- {out_train_path}\n- {out_test_path}")

if __name__ == "__main__":
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    dataset_csv = os.path.join(base_dir, "training_dataset.csv")
    train_csv = os.path.join(base_dir, "train_dataset.csv")
    test_csv = os.path.join(base_dir, "test_dataset.csv")
    
    split_dataset(dataset_csv, train_csv, test_csv)
