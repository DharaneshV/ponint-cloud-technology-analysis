import os
import pandas as pd
import numpy as np
import open3d as o3d

TARGET_LENGTH_CM = 1200.0 # 12 meters

def load_and_preprocess(filepath, name, target_len):
    # Read XYZ: [scan_index, Y, Z]
    df = pd.read_csv(filepath, sep='\s+', header=None)
    
    # Isolate truck points in Y to count the actual number of scans the truck spans
    df_obj = df[~np.isclose(df[1].abs(), 210.0, atol=1.0)]
    scan_min = df_obj[0].min()
    scan_max = df_obj[0].max()
    num_scans = scan_max - scan_min + 1
    
    # Calculate the speed to make the truck exactly target_len
    pitch_cm = target_len / float(num_scans)
    
    # Map X to centimeters
    x_vals = df[0] * pitch_cm
    y_vals = df[1]
    z_vals = df[2]
    
    points = np.vstack((x_vals, y_vals, z_vals)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Statistical Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    # Gantry Filter (Y range [-205, 205])
    pts = np.asarray(pcd.points)
    mask = (pts[:, 1] >= -205.0) & (pts[:, 1] <= 205.0)
    pcd = pcd.select_by_index(np.where(mask)[0])
    
    # Ground Removal (RANSAC)
    plane_model, inliers = pcd.segment_plane(distance_threshold=5.0, ransac_n=3, num_iterations=1000)
    # We keep the ground plane height to understand where the floor is
    [a, b, c, d] = plane_model
    ground_z = -d / c
    print(f"[{name}] Ground plane height Z: {ground_z:.2f} cm")
    
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)
    
    # Extract Truck using DBSCAN
    pcd_down = pcd_no_ground.voxel_down_sample(voxel_size=5.0)
    labels = np.array(pcd_down.cluster_dbscan(eps=50.0, min_points=20, print_progress=False))
    
    max_label = labels.max()
    best_cluster_pcd = None
    max_pts = -1
    for i in range(max_label + 1):
        c_idx = np.where(labels == i)[0]
        c_pcd = pcd_down.select_by_index(c_idx)
        aabb = c_pcd.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        # Filter by a wider envelope since it is in cm now
        if 500 <= extent[0] <= 1500 and 100 <= extent[1] <= 300 and 50 <= extent[2] <= 300:
            if len(c_idx) > max_pts:
                max_pts = len(c_idx)
                best_cluster_pcd = c_pcd
                
    if best_cluster_pcd is None:
        print(f"[{name}] Cluster extraction failed, using full no_ground cloud.")
        best_cluster_pcd = pcd_no_ground
        
    # Align along X to start at 0
    pts_truck = np.asarray(best_cluster_pcd.points)
    x_min_val = pts_truck[:, 0].min()
    pts_truck[:, 0] -= x_min_val
    best_cluster_pcd.points = o3d.utility.Vector3dVector(pts_truck)
    
    return best_cluster_pcd, ground_z

def run_pipeline_real():
    empty_file = "d:/point cloud technology/bgtest-3/empty/2026-04-07/2026-04-07_10-12-23.xyz"
    load_file = "d:/point cloud technology/bgtest-3/load/2026-04-07_11-27-22.xyz"
    
    empty_pcd, ground_empty = load_and_preprocess(empty_file, "empty", TARGET_LENGTH_CM)
    load_pcd, ground_load = load_and_preprocess(load_file, "load", TARGET_LENGTH_CM)
    
    # Shift loaded cloud to have same ground height as empty cloud
    # (suspension compression compensation)
    # The loaded truck ground plane was at ground_load, empty at ground_empty
    # So we shift Z of load by (ground_empty - ground_load)
    z_shift = ground_empty - ground_load
    print(f"Shifting loaded truck Z by {z_shift:.2f} cm to align ground heights.")
    load_pts = np.asarray(load_pcd.points)
    load_pts[:, 2] += z_shift
    load_pcd.points = o3d.utility.Vector3dVector(load_pts)
    
    # Run ICP to align them in X and Y (since they are already centered/started at X=0)
    print("Running ICP alignment...")
    if not load_pcd.has_normals():
        load_pcd.estimate_normals()
    if not empty_pcd.has_normals():
        empty_pcd.estimate_normals()
        
    reg = o3d.pipelines.registration.registration_icp(
        load_pcd, empty_pcd, 30.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    load_aligned = load_pcd.transform(reg.transformation)
    
    # Compute volume using heightmap difference
    empty_pts = np.asarray(empty_pcd.points)
    load_pts = np.asarray(load_aligned.points)
    
    # Find overlapping region
    x_min = max(empty_pts[:, 0].min(), load_pts[:, 0].min())
    x_max = min(empty_pts[:, 0].max(), load_pts[:, 0].max())
    y_min = max(empty_pts[:, 1].min(), load_pts[:, 1].min())
    y_max = min(empty_pts[:, 1].max(), load_pts[:, 1].max())
    
    res = 5.0 # 5 cm grid resolution
    x_bins = np.arange(x_min, x_max + res, res)
    y_bins = np.arange(y_min, y_max + res, res)
    
    empty_x_idx = np.digitize(empty_pts[:, 0], x_bins) - 1
    empty_y_idx = np.digitize(empty_pts[:, 1], y_bins) - 1
    load_x_idx = np.digitize(load_pts[:, 0], x_bins) - 1
    load_y_idx = np.digitize(load_pts[:, 1], y_bins) - 1
    
    grid_shape = (len(x_bins), len(y_bins))
    empty_grid = np.full(grid_shape, np.nan)
    load_grid = np.full(grid_shape, np.nan)
    
    df_e = pd.DataFrame({'x': empty_x_idx, 'y': empty_y_idx, 'z': empty_pts[:, 2]})
    df_l = pd.DataFrame({'x': load_x_idx, 'y': load_y_idx, 'z': load_pts[:, 2]})
    
    # In scanner coords, Z is height from ground, so max Z in each cell is the top surface
    e_max = df_e.groupby(['x', 'y'])['z'].max()
    l_max = df_l.groupby(['x', 'y'])['z'].max()
    
    for (x_idx, y_idx), z_val in e_max.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            empty_grid[x_idx, y_idx] = z_val
    for (x_idx, y_idx), z_val in l_max.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            load_grid[x_idx, y_idx] = z_val
            
    # Calculate volume of the load.
    # The load raises the surface height, so load_grid Z should be higher than empty_grid Z in the bed area.
    # If a cell is inside the truck bed:
    #   empty_grid has the bed floor (e.g. Z ~ 182)
    #   load_grid has the cargo surface (e.g. Z ~ 240)
    #   diff = 240 - 182 = 58 cm.
    # If a cell is on the side walls:
    #   empty_grid has the wall top (e.g. Z ~ 360)
    #   load_grid has the wall top (e.g. Z ~ 360)
    #   diff = 360 - 360 = 0 cm.
    
    valid_mask = ~np.isnan(empty_grid) & ~np.isnan(load_grid)
    height_diff = load_grid[valid_mask] - empty_grid[valid_mask]
    
    # Filter negative differences and also threshold small differences to ignore noise
    height_diff[height_diff < 5.0] = 0.0 # Ignore changes less than 5 cm
    
    # Calculate volume in cubic meters
    cell_area_cm2 = res * res
    volume_cm3 = np.sum(height_diff * cell_area_cm2)
    volume_m3 = volume_cm3 / 1e6 # convert cm^3 to m^3
    
    print("\n=========================================")
    print(f"Calibrated Volume Calculation (Length = {TARGET_LENGTH_CM/100:.2f} m)")
    print(f"Empty points: {len(empty_pts)}, Load points: {len(load_pts)}")
    print(f"Grid size: {grid_shape[0]} x {grid_shape[1]} cells")
    print(f"Average Height Diff in filled cells: {np.mean(height_diff[height_diff > 0]):.2f} cm")
    print(f"Computed Cargo Volume: {volume_m3:.4f} cubic meters")
    print("=========================================")

if __name__ == "__main__":
    run_pipeline_real()
