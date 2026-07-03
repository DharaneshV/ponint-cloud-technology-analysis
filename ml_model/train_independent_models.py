import pandas as pd
import numpy as np
import joblib
import os
import argparse
import subprocess
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, mean_absolute_error

def main():
    parser = argparse.ArgumentParser(description="Train Independent Models")
    parser.add_argument("--scanner", type=str, choices=["SL1", "SL2"], default="SL1", help="Scanner type (SL1 or SL2)")
    args = parser.parse_args()
    
    data_path = f"d:\\point cloud technology\\Truck-PointCloud\\ml_model\\independent_dataset_{args.scanner.lower()}.csv"
    model_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model\models"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    print(f"\n==========================================")
    print(f"TRAINING v3 MODEL FOR SCANNER: {args.scanner}")
    print(f"==========================================")
    print(f"Loaded {len(df)} samples.")
    print("Class distribution:")
    print(df['geometric_label'].value_counts())
    
    features = ['mean_z', 'median_z', 'std_z', 'skew_z', 'coverage_proxy', 'point_density']
    
    X = df[features].values
    y_class = df['geometric_label'].values
    y_reg = df['target_volume_m3'].values
    
    # --- 1. Baseline: Majority Class ---
    majority_class = df['geometric_label'].mode()[0]
    baseline_preds = [majority_class] * len(y_class)
    baseline_acc = accuracy_score(y_class, baseline_preds)
    print(f"\n--- MAJORITY-CLASS BASELINE ---")
    print(f"Predicting '{majority_class}' every time yields Accuracy: {baseline_acc*100:.2f}%")
    
    # --- 2. Leave-One-Out (LOO) Validation (Two-Stage Model) ---
    print("\n--- LEAVE-ONE-OUT (LOO) VALIDATION (TWO-STAGE) ---")
    loo = LeaveOneOut()
    
    y_true_class = []
    y_pred_class = []
    
    y_true_reg = []
    y_pred_reg = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train_c, y_test_c = y_class[train_index], y_class[test_index]
        y_train_r, y_test_r = y_reg[train_index], y_reg[test_index]
        
        # Scale on train folds
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Stage 1: Classifier (all train samples)
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
        clf.fit(X_train_scaled, y_train_c)
        pred_c = clf.predict(X_test_scaled)[0]
        
        y_true_class.append(y_test_c[0])
        y_pred_class.append(pred_c)
        
        # Train Stage 2: Regressor (only Partial train samples)
        partial_mask = (y_train_c == 'Partial')
        X_train_part = X_train_scaled[partial_mask]
        y_train_part_r = y_train_r[partial_mask]
        
        if len(y_train_part_r) > 2:
            reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
            reg.fit(X_train_part, y_train_part_r)
            
            # Predict Stage 2
            if pred_c == 'Empty':
                pred_r = 0.0
            else:
                pred_r = reg.predict(X_test_scaled)[0]
        else:
            # Fallback if too few samples
            pred_r = 0.0 if pred_c == 'Empty' else np.mean(y_train_part_r)
            
        y_true_reg.append(y_test_r[0])
        y_pred_reg.append(pred_r)
        
    loo_acc = accuracy_score(y_true_class, y_pred_class)
    print(f"\nClassifier LOO Accuracy: {loo_acc*100:.2f}%")
    
    df_results = pd.DataFrame({
        'True_Class': y_true_class,
        'True_Vol': y_true_reg,
        'Pred_Vol': y_pred_reg
    })
    df_results['Abs_Err'] = (df_results['True_Vol'] - df_results['Pred_Vol']).abs()
    
    print("\nRegressor MAE per class:")
    for cls in df_results['True_Class'].unique():
        cls_df = df_results[df_results['True_Class'] == cls]
        mae = cls_df['Abs_Err'].mean()
        print(f"  {cls} ({len(cls_df)} samples): {mae:.4f} m3")
        
    overall_mae = df_results['Abs_Err'].mean()
    partial_mae = df_results[df_results['True_Class'] == 'Partial']['Abs_Err'].mean()
    print(f"  Overall MAE: {overall_mae:.4f} m3")
    print(f"  Partial MAE (Regression only): {partial_mae:.4f} m3")
    
    # --- 3. Quality Gate check ---
    print("\n--- QUALITY GATE ---")
    quality_pass = True
    
    if loo_acc < 0.90:
        print(f"  [FAIL] Classifier accuracy {loo_acc*100:.1f}% is below quality gate of 90.0%!")
        quality_pass = False
    else:
        print(f"  [PASS] Classifier accuracy exceeds quality gate.")
        
    if partial_mae > 1.5:
        print(f"  [FAIL] Regressor MAE on partials {partial_mae:.4f} m3 is above quality gate of 1.5 m3!")
        quality_pass = False
    else:
        print(f"  [PASS] Regressor MAE is within quality gate.")
        
    if not quality_pass:
        print("\nModel failed the quality gate check! Final model will not be saved.")
        import sys
        sys.exit(1)
        
    # --- 4. Train Final Models on All Data ---
    print("\n--- TRAINING FINAL v3 MODELS ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Final Classifier
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
    final_clf.fit(X_scaled, y_class)
    
    # Final Regressor (only Partial samples)
    partial_mask_all = (y_class == 'Partial')
    X_scaled_part = X_scaled[partial_mask_all]
    y_reg_part = y_reg[partial_mask_all]
    
    final_reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
    final_reg.fit(X_scaled_part, y_reg_part)
    
    # Fetch Git commit dynamically
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r"d:\point cloud technology\Truck-PointCloud").decode().strip()
    except Exception as e:
        git_commit = "unknown"
        
    # Bundle Package
    model_package = {
        "classifier": final_clf,
        "regressor": final_reg,
        "scaler": scaler,
        "metadata": {
            "features": features,
            "training_date": str(pd.Timestamp.now()),
            "sample_counts": dict(df['geometric_label'].value_counts()),
            "git_commit": git_commit,
            "scanner_type": args.scanner,
            "validation_accuracy": loo_acc,
            "validation_partial_mae": partial_mae
        }
    }
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"fill_model_{args.scanner.lower()}.pkl")
    joblib.dump(model_package, model_path)
    
    print(f"\nSaved v3 model package to:\n  {model_path}")
    print("Bundle contains classifier, regressor, scaler, and metadata package.")

if __name__ == "__main__":
    main()
