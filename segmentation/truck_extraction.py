import open3d as o3d
import numpy as np
import config

def extract_truck_cluster(pcd, labels):
    """
    Extracts the truck cluster based on envelope constraints and largest volume.
    """
    print("Extracting truck cluster...")
    max_label = labels.max()
    if max_label < 0:
        print("Warning: No clusters found.")
        return None

    best_cluster_pcd = None
    max_points = -1
    
    for i in range(max_label + 1):
        # Extract cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < config.DBSCAN_MIN_POINTS:
            continue
            
        cluster_pcd = pcd.select_by_index(cluster_indices)
        
        # Check envelope constraints
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        length, width, height = extent[0], extent[1], extent[2]
        
        # Sort dimensions if orientation is arbitrary, but assuming X is roughly length
        # For our pipeline, X is length, Y is width, Z is height.
        if (config.TRUCK_ENVELOPE_MIN_LENGTH <= length <= config.TRUCK_ENVELOPE_MAX_LENGTH and
            config.TRUCK_ENVELOPE_MIN_WIDTH <= width <= config.TRUCK_ENVELOPE_MAX_WIDTH and
            config.TRUCK_ENVELOPE_MIN_HEIGHT <= height <= config.TRUCK_ENVELOPE_MAX_HEIGHT):
            
            print(f"Cluster {i}: valid envelope ({length:.1f} x {width:.1f} x {height:.1f} mm), points: {len(cluster_indices)}")
            
            if len(cluster_indices) > max_points:
                max_points = len(cluster_indices)
                best_cluster_pcd = cluster_pcd
        else:
            print(f"Cluster {i}: invalid envelope ({length:.1f} x {width:.1f} x {height:.1f} mm), rejected.")
            
    if best_cluster_pcd is None:
        print("Warning: No cluster matched the truck envelope constraints.")
        return None
        
    print(f"Extracted truck cluster with {len(best_cluster_pcd.points)} points.")
    return best_cluster_pcd
