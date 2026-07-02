import open3d as o3d
import numpy as np
import config

def filter_gantry(pcd):
    """
    Filters out the gantry walls (points outside Y-bounds).
    """
    print(f"Filtering gantry on Y-axis ([{config.GANTRY_Y_MIN}, {config.GANTRY_Y_MAX}])...")
    points = np.asarray(pcd.points)
    mask = (points[:, 1] >= config.GANTRY_Y_MIN) & (points[:, 1] <= config.GANTRY_Y_MAX)
    
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    print(f"Points remaining after gantry filter: {len(pcd_filtered.points)}")
    return pcd_filtered
