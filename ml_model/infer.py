import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew

def get_ml_prediction(surface_grid, floor_z, grid_resolution_cm=5.0):
    """
    Extracts the 5 ML features from the surface grid and runs the trained
    Random Forest to output a fast fill-state estimation.
    """
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    model_path = os.path.join(base_dir, "models", "fill_classifier_v2.pkl")
    scaler_path = os.path.join(base_dir, "models", "fill_scaler_v2.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "MODEL NOT FOUND", 0.0
        
    with open(model_path, 'rb') as f:
        rf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # surface_grid contains nan for empty cells. 
    # Calculate cargo heights relative to floor
    valid_mask = ~np.isnan(surface_grid)
    cargo_heights = (surface_grid[valid_mask] - floor_z)
    
    if len(cargo_heights) == 0:
        return "Empty", 1.0
        
    max_heap_height = np.max(cargo_heights)
    mean_bed_height = np.mean(cargo_heights)
    
    # Fast geometric check: if mean cargo height is under 10cm, it is Empty
    if mean_bed_height < 10.0:
        return "Empty", 1.0
        
    height_variance = np.var(cargo_heights)
    
    covered_cells = np.sum(cargo_heights > 10.0) # >10cm
    total_cells = surface_grid.size
    coverage_ratio = covered_cells / total_cells if total_cells > 0 else 0
    
    # Skewness requires length profile
    # Replace nans with floor_z for profile calculation
    filled_grid = np.nan_to_num(surface_grid, nan=floor_z)
    cargo_grid = filled_grid - floor_z
    length_profile = np.mean(cargo_grid, axis=0)
    height_skewness = skew(length_profile) if len(length_profile) > 3 else 0
    
    # Density proxy
    valid_point_count = np.sum(valid_mask)
    
    # Reject if data is too sparse
    if valid_point_count < 100:
        return "UNCERTAIN - recommend manual review", 0.0
        
    features = pd.DataFrame([{
        'max_heap_height': max_heap_height,
        'mean_bed_height': mean_bed_height,
        'height_variance': height_variance,
        'coverage_ratio': coverage_ratio,
        'height_skewness': height_skewness
    }])
    
    X_scaled = scaler.transform(features)
    
    probs = rf.predict_proba(X_scaled)[0]
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]
    
    # Configurable rejection floor
    if confidence < 0.70:
        return "UNCERTAIN - recommend manual review", confidence
        
    predicted_class = rf.classes_[class_idx]
    return predicted_class, confidence

def get_independent_ml_prediction(pcd):
    """
    Extracts the 6 independent scale-invariant shape and density features 
    from a single unaligned truck point cloud and runs the v3 classifier and regressor.
    Returns: (predicted_class, confidence, predicted_volume_m3)
    """
    base_dir = r"d:\point cloud technology\Truck-PointCloud\ml_model"
    classifier_path = os.path.join(base_dir, "models", "fill_classifier_v3.pkl")
    regressor_path = os.path.join(base_dir, "models", "fill_regressor_v3.pkl")
    scaler_path = os.path.join(base_dir, "models", "fill_scaler_v3.pkl")
    
    import joblib
    
    if not os.path.exists(classifier_path) or not os.path.exists(scaler_path) or not os.path.exists(regressor_path):
        return "MODEL NOT FOUND", 0.0, 0.0
        
    rf_clf = joblib.load(classifier_path)
    rf_reg = joblib.load(regressor_path)
    scaler = joblib.load(scaler_path)
        
    points = np.asarray(pcd.points)
    if len(points) < 100:
        return "UNCERTAIN - Sparse", 0.0, 0.0
        
    z_coords = points[:, 2]
    floor_z = np.percentile(z_coords, 95)
    
    mean_z = np.mean(z_coords)
    median_z = np.median(z_coords)
    std_z = np.std(z_coords)
    skew_z = skew(z_coords)
    
    cargo_points = np.sum(z_coords < (floor_z - 10.0))  # > 10cm above floor (Z points down)
    coverage_proxy = cargo_points / len(points)
    
    import open3d as o3d
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    vol_cm3 = extent[0] * extent[1] * extent[2]
    density = len(points) / vol_cm3 if vol_cm3 > 0 else 0
    
    features = pd.DataFrame([{
        'mean_z': mean_z,
        'median_z': median_z,
        'std_z': std_z,
        'skew_z': skew_z,
        'coverage_proxy': coverage_proxy,
        'point_density': density
    }])
    
    # The scaler for v3 was fitted on an array of 6 features
    X_scaled = scaler.transform(features.values)
    
    probs = rf_clf.predict_proba(X_scaled)[0]
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]
    
    pred_vol = rf_reg.predict(X_scaled)[0]
    
    if confidence < 0.70:
        return "UNCERTAIN - recommend manual review", confidence, pred_vol
        
    return rf_clf.classes_[class_idx], confidence, pred_vol
