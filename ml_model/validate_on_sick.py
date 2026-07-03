import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    dataset_csv = os.path.join(base_dir, "sick_validation_dataset.csv")
    model_path = os.path.join(base_dir, "models", "fill_classifier_v2.pkl")
    scaler_path = os.path.join(base_dir, "models", "fill_scaler_v2.pkl")
    
    if not os.path.exists(dataset_csv):
        print(f"Validation dataset not found: {dataset_csv}")
        return
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Trained model or scaler v2 not found in ml_model/models/.")
        return
        
    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f"Loaded {len(df)} SICK validation samples.")
    print("Class counts in validation data:")
    print(df['geometric_label'].value_counts())
    
    # Load model and scaler
    with open(model_path, 'rb') as f:
        rf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Prepare features and true labels
    feature_cols = ['max_heap_height', 'mean_bed_height', 'height_variance', 'coverage_ratio', 'height_skewness']
    X = df[feature_cols]
    y_true = df['geometric_label']
    
    # Predict using the model and the fallback rule (replicating infer.py)
    y_pred = []
    probs_list = []
    
    for idx, row in df.iterrows():
        if row['mean_bed_height'] < 10.0:
            pred_lbl = "Empty"
            prob = 1.0
        else:
            feat_df = pd.DataFrame([row[feature_cols]])
            feat_scaled = scaler.transform(feat_df)
            pred_lbl = rf.predict(feat_scaled)[0]
            prob = rf.predict_proba(feat_scaled)[0][np.where(rf.classes_ == pred_lbl)[0][0]]
            
        y_pred.append(pred_lbl)
        probs_list.append(prob)
        
    y_pred = np.array(y_pred)
    probs = np.array(probs_list)
    
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- SICK Domain Transfer Accuracy: {acc:.2%} ---")
    
    # Print classification report
    # Handle potentially missing classes in predictions/true labels gracefully
    labels_in_data = sorted(list(set(y_true) | set(y_pred)))
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=labels_in_data))
    
    # Print Confusion Matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels_in_data)
    print(pd.DataFrame(cm, index=[f"True {l}" for l in labels_in_data], columns=[f"Pred {l}" for l in labels_in_data]))
    
    # Print side-by-side comparison
    print("\n--- Detailed Predictions ---")
    print(f"{'Filename':<30} | {'True Label':<10} | {'Predicted':<10} | {'Confidence':<10} | {'Status':<8}")
    print("-" * 80)
    
    mismatches = []
    for idx, row in df.iterrows():
        true_lbl = row['geometric_label']
        pred_lbl = y_pred[idx]
        prob = probs[idx]
        status = "MATCH" if true_lbl == pred_lbl else "MISMATCH"
        
        print(f"{row['filename']:<30} | {true_lbl:<10} | {pred_lbl:<10} | {prob:<10.2%} | {status:<8}")
        if status == "MISMATCH":
            mismatches.append((row['filename'], true_lbl, pred_lbl, prob))
            
    print("-" * 80)
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches:")
        for fn, t, p, c in mismatches:
            print(f"  * {fn}: True = {t}, Predicted = {p} (confidence {c:.2%})")
    else:
        print("\nAll predictions match the geometric rule labels perfectly!")

if __name__ == "__main__":
    main()
