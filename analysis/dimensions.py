import open3d as o3d
import json
import os

def calculate_dimensions(pcd, output_file=None):
    """
    Computes Axis-Aligned and Oriented Bounding Box dimensions.
    """
    print("Calculating dimensions...")
    
    # Axis-aligned (assumes PCA alignment was already done)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    extent_aabb = aabb.get_extent()
    
    # OBB (just in case)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    extent_obb = obb.extent
    
    dims = {
        "AABB": {
            "length_x": extent_aabb[0],
            "width_y": extent_aabb[1],
            "height_z": extent_aabb[2],
            "center": aabb.get_center().tolist()
        },
        "OBB": {
            "extent": extent_obb.tolist(),
            "center": obb.get_center().tolist()
        }
    }
    
    print(f"Dimensions (AABB): L={extent_aabb[0]:.2f}, W={extent_aabb[1]:.2f}, H={extent_aabb[2]:.2f} cm")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(dims, f, indent=4)
            
    return dims, aabb, obb
