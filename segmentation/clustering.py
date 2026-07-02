import open3d as o3d
import numpy as np
import config

def cluster_points(pcd):
    """
    Applies DBSCAN clustering on the point cloud.
    Returns an array of labels for each point.
    """
    print(f"Applying DBSCAN clustering (eps={config.DBSCAN_EPS}, min_points={config.DBSCAN_MIN_POINTS})...")
    labels = np.array(pcd.cluster_dbscan(
        eps=config.DBSCAN_EPS, 
        min_points=config.DBSCAN_MIN_POINTS, 
        print_progress=False
    ))
    
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters.")
    return labels
