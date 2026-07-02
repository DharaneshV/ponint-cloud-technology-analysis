import open3d as o3d
import config

def remove_statistical_outliers(pcd):
    """
    Removes statistical outliers from a point cloud.
    """
    print(f"Applying Statistical Outlier Removal (k={config.SOR_NB_NEIGHBORS}, std={config.SOR_STD_RATIO})...")
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=config.SOR_NB_NEIGHBORS, 
        std_ratio=config.SOR_STD_RATIO
    )
    pcd_clean = pcd.select_by_index(ind)
    print(f"Points remaining after SOR: {len(pcd_clean.points)}")
    return pcd_clean
