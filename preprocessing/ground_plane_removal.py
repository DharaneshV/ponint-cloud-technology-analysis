import open3d as o3d
import numpy as np
import config

def remove_ground_plane(pcd):
    """
    Removes the ground plane (road) by removing points with Z > ROAD_Z_THRESHOLD.
    (In SICK gantry rig, Z represents distance downwards, so ground has large Z value).
    """
    print(f"Removing ground plane (Z > {config.ROAD_Z_THRESHOLD} cm)...")
    pts = np.asarray(pcd.points)
    mask = pts[:, 2] <= config.ROAD_Z_THRESHOLD
    
    pcd_no_ground = pcd.select_by_index(np.where(mask)[0])
    print(f"Points remaining after ground removal: {len(pcd_no_ground.points)}")
    return pcd_no_ground
