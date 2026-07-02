import os
import pandas as pd
import numpy as np
import open3d as o3d

TARGET_LENGTH_CM = 1200.0 # 12 meters

def load_and_preprocess_v2(filepath, name, target_len):
    # Read XYZ: [scan_index, Y, Z]
    df = pd.read_csv(filepath, sep='\s+', header=None)
    
    # Isolate truck points in Y to count the actual number of scans the truck spans
    df_obj = df[~np.isclose(df[1].abs(), 210.0, atol=1.0)]
    scan_min = df_obj[0].min()
    scan_max = df_obj[0].max()
    num_scans = scan_max - scan_min + 1
    
    # Calculate the speed to make the truck exactly target_len
    pitch_cm = target_len / float(num_scans)
    print(f"[{name}] Spans {num_scans} scans. Pitch: {pitch_cm:.4f} cm/scan.")
    
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
    
    # Ground Removal via Pass-through Z <= 330.0 (since Z=385 is ground, Z=143 is scanner top)
    pts = np.asarray(pcd.points)
    mask_ground = pts[:, 2] <= 330.0
    pcd_no_ground = pcd.select_by_index(np.where(mask_ground)[0])
    
    # Align along X to start at 0
    pts_truck = np.asarray(pcd_no_ground.points)
    x_min_val = pts_truck[:, 0].min()
    pts_truck[:, 0] -= x_min_val
    pcd_no_ground.points = o3d.utility.Vector3dVector(pts_truck)
    
    return pcd_no_ground

def run_pipeline_real_v2():
    empty_file = "d:/point cloud technology/bgtest-3/empty/2026-04-07/2026-04-07_10-12-23.xyz"
    load_file = "d:/point cloud technology/bgtest-3/load/2026-04-07_11-27-22.xyz"
    
    empty_pcd = load_and_preprocess_v2(empty_file, "empty", TARGET_LENGTH_CM)
    load_pcd = load_and_preprocess_v2(load_file, "load", TARGET_LENGTH_CM)
    
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
    print(f"ICP Transformation Matrix:\n{reg.transformation}")
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
    
    print(f"Empty unique cells: {len(df_e.groupby(['x', 'y']))}")
    print(f"Load unique cells: {len(df_l.groupby(['x', 'y']))}")
    
    # Since Z is distance downwards, the top surface is the MINIMUM Z in each cell
    e_min = df_e.groupby(['x', 'y'])['z'].min()
    l_min = df_l.groupby(['x', 'y'])['z'].min()
    
    for (x_idx, y_idx), z_val in e_min.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            empty_grid[x_idx, y_idx] = z_val
    for (x_idx, y_idx), z_val in l_min.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            load_grid[x_idx, y_idx] = z_val
            
    print(f"Empty grid non-NaNs: {np.sum(~np.isnan(empty_grid))}")
    print(f"Load grid non-NaNs: {np.sum(~np.isnan(load_grid))}")
    print(f"Overlap non-NaNs: {np.sum(~np.isnan(empty_grid) & ~np.isnan(load_grid))}")
            
    # Calculate volume of the load (Z_empty - Z_load)
    valid_mask = ~np.isnan(empty_grid) & ~np.isnan(load_grid)
    height_diff = empty_grid[valid_mask] - load_grid[valid_mask]
    
    print("\nRaw height differences statistics (Z_empty - Z_load):")
    print(pd.Series(height_diff).describe())
    
    print(f"Cells with diff > 5cm: {np.sum(height_diff > 5.0)}")
    print(f"Cells with diff > 10cm: {np.sum(height_diff > 10.0)}")
    print(f"Cells with diff > 30cm: {np.sum(height_diff > 30.0)}")
    print(f"Cells with diff > 50cm: {np.sum(height_diff > 50.0)}")
    
    # Filter negative differences and small noise
    height_diff[height_diff < 5.0] = 0.0 # Ignore differences < 5 cm
    
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
    run_pipeline_real_v2()
