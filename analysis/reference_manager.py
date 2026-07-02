import os
import open3d as o3d
from registration import pca_alignment

def save_reference_model(truck_unrotated, reference_filepath):
    """
    Saves the extracted (unrotated) empty truck as a reference model.
    """
    os.makedirs(os.path.dirname(reference_filepath), exist_ok=True)
    o3d.io.write_point_cloud(reference_filepath, truck_unrotated)
    print(f"Saved reference model to {reference_filepath}")
    return reference_filepath

def load_reference_model(reference_filepath):
    """
    Loads a reference model point cloud and computes its PCA alignment.
    Returns: (empty_aligned, empty_unrot)
    """
    if not os.path.exists(reference_filepath):
        print(f"Error: Reference model not found at {reference_filepath}")
        return None, None
        
    empty_unrot = o3d.io.read_point_cloud(reference_filepath)
    if not empty_unrot.has_points():
        print(f"Error: Empty reference model at {reference_filepath}")
        return None, None
        
    print(f"Loaded reference model from {reference_filepath}")
    
    # Generate the aligned version for dimensions/ICP
    empty_aligned, _, _ = pca_alignment.align_truck_pca(empty_unrot)
    
    return empty_aligned, empty_unrot
