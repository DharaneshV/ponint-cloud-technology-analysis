import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

def main():
    data_path = r"d:\point cloud technology\Truck-PointCloud\ml_model\independent_dataset.csv"
    model_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model\models"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")
    print("\nClass distribution:")
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
    
    # --- 2. Leave-One-Out Cross Validation (LOO) ---
    print("\n--- LEAVE-ONE-OUT (LOO) VALIDATION ---")
    loo = LeaveOneOut()
    
    y_true_class = []
    y_pred_class = []
    
    y_true_reg = []
    y_pred_reg = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train_c, y_test_c = y_class[train_index], y_class[test_index]
        y_train_r, y_test_r = y_reg[train_index], y_reg[test_index]
        
        # Scale
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Classifier
        clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
        clf.fit(X_train_scaled, y_train_c)
        pred_c = clf.predict(X_test_scaled)
        y_true_class.append(y_test_c[0])
        y_pred_class.append(pred_c[0])
        
        # Train Regressor
        reg = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
        reg.fit(X_train_scaled, y_train_r)
        pred_r = reg.predict(X_test_scaled)
        y_true_reg.append(y_test_r[0])
        y_pred_reg.append(pred_r[0])
        
    loo_acc = accuracy_score(y_true_class, y_pred_class)
    print(f"\nV3 Independent Classifier LOO Accuracy: {loo_acc*100:.2f}%")
    
    print("\n--- EXPERIMENTAL REGRESSOR VALIDATION ---")
    
    df_results = pd.DataFrame({
        'True_Class': y_true_class,
        'True_Vol': y_true_reg,
        'Pred_Vol': y_pred_reg
    })
    
    df_results['Abs_Err'] = np.abs(df_results['True_Vol'] - df_results['Pred_Vol'])
    
    print("LOO Mean Absolute Error (MAE) per class:")
    for cls in df_results['True_Class'].unique():
        cls_df = df_results[df_results['True_Class'] == cls]
        mae = cls_df['Abs_Err'].mean()
        count = len(cls_df)
        print(f"  {cls} ({count} samples): {mae:.4f} m3")
        
    overall_mae = df_results['Abs_Err'].mean()
    print(f"  Overall MAE: {overall_mae:.4f} m3")
    
    print("\nWarning: The regression MAE is likely high because learning continuous capacity with extremely limited data (10 partial scans) is precarious. This model is marked EXPERIMENTAL.")
        
    # --- 3. Train Final Models on All Data ---
    print("\n--- TRAINING FINAL v3 MODELS ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    final_clf.fit(X_scaled, y_class)
    
    final_reg = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    final_reg.fit(X_scaled, y_reg)
        
    os.makedirs(model_dir, exist_ok=True)
    classifier_path = os.path.join(model_dir, "fill_classifier_v3.pkl")
    regressor_path = os.path.join(model_dir, "fill_regressor_v3.pkl")
    scaler_path = os.path.join(model_dir, "fill_scaler_v3.pkl")
    
    joblib.dump(final_clf, classifier_path)
    joblib.dump(final_reg, regressor_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nSaved v3 models to:\n  {classifier_path}\n  {regressor_path}\n  {scaler_path}")

if __name__ == "__main__":
    main()
