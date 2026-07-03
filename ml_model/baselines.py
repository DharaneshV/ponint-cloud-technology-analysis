import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

def evaluate_baselines(train_csv, test_csv):
    """
    Evaluates trivial baselines on the grouped dataset.
    This proves that the model is just approximating our geometric rule,
    as expected due to label circularity.
    """
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("Error: Train or test CSV not found. Run dataset_split.py first.")
        return
        
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Features we'll use for the baselines
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
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Baseline 1: Dummy Classifier (Predicts Most Frequent Class) ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_scaled, y_train)
    y_pred_dummy = dummy.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_dummy, zero_division=0))
    print(f"Macro F1: {f1_score(y_test, y_pred_dummy, average='macro'):.4f}")
    
    print("\n--- Baseline 2: Logistic Regression (Simple Linear Thresholding) ---")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_lr, zero_division=0))
    print(f"Macro F1: {f1_score(y_test, y_pred_lr, average='macro'):.4f}")
    print("\nLogistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_))
    print(f"Labels: {logreg.classes_}")
    
    print("\nConclusion: Because our labels are mathematically derived from the same Z-heights")
    print("used to create these features, a simple Logistic Regression should score very high.")
    print("If it scores >90%, it proves the ML model is successfully acting as a fast approximation")
    print("of our geometric pipeline, rather than a novel independent intelligence.")

if __name__ == "__main__":
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    train_csv = os.path.join(base_dir, "train_dataset.csv")
    test_csv = os.path.join(base_dir, "test_dataset.csv")
    
    evaluate_baselines(train_csv, test_csv)
