import os
import json
import pandas as pd
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

def train_and_save_model(train_csv, test_csv, model_dir):
    """
    Trains a Random Forest classifier on the grouped training set.
    Performs grid search hyperparameter tuning, prints cross-validation metrics,
    logs feature importances, and saves version v2 of the model and scaler.
    """
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Error: Train or test CSV not found.")
        return
        
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    features = [
        'max_heap_height', 
        'mean_bed_height', 
        'height_variance',
        'coverage_ratio',
        'height_skewness'
    ]
    
    X_train = train_df[features]
    y_train = train_df['geometric_label']
    
    X_test = test_df[features]
    y_test = test_df['geometric_label']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Tuning Random Forest Classifier with Grid Search ---")
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 5],
        'class_weight': ['balanced', None]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf_base, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    print(f"Best parameters found: {grid.best_params_}")
    print(f"Best 5-fold cross-validation Macro F1: {grid.best_score_:.4f}")
    
    rf = grid.best_estimator_
    
    # Evaluate on test set
    y_pred = rf.predict(X_test_scaled)
    
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=rf.classes_))
    print(f"Classes order: {rf.classes_}")
    
    print("\n--- Feature Importances ---")
    importances = rf.feature_importances_
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.4f}")
        
    # Save the model and scaler v2
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fill_classifier_v2.pkl")
    scaler_path = os.path.join(model_dir, "fill_scaler_v2.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\nSaved v2 model to {model_path}")
    print(f"Saved v2 scaler to {scaler_path}")
    
    # Update registry
    registry_path = os.path.join(model_dir, "model_registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
    registry["v2"] = {
        "training_date": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features": features,
        "classes": rf.classes_.tolist(),
        "macro_f1": float(macro_f1),
        "best_params": grid.best_params_,
        "feature_importances": {feat: float(imp) for feat, imp in zip(features, importances)},
        "description": "Tuned Random Forest classifier with corrected ground Z and sky codes (v2)."
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
        
    print(f"Updated {registry_path}")

if __name__ == "__main__":
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    train_csv = os.path.join(base_dir, "train_dataset.csv")
    test_csv = os.path.join(base_dir, "test_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    
    train_and_save_model(train_csv, test_csv, models_dir)

