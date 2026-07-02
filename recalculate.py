import os
import sys
import pandas as pd
import numpy as np
import open3d as o3d

# Target dimensions in centimeters
TARGET_LENGTH_CM = 1200.0 # 12 meters

def process_and_align_to_length(filepath, name, target_len):
    # Read XYZ: [scan_index, Y, Z]
    df = pd.read_csv(filepath, sep='\s+', header=None)
    
    # Isolate truck points in Y to count the actual number of scans the truck spans
    df_obj = df[~np.isclose(df[1].abs(), 210.0, atol=1.0)]
    scan_min = df_obj[0].min()
    scan_max = df_obj[0].max()
    num_scans = scan_max - scan_min + 1
    
    # Calculate the speed to make the truck exactly target_len
    # scan_interval = 0.010 s
    # length = num_scans * 0.010 * speed -> speed = length / (num_scans * 0.010)
    # X = scan_index * 0.010 * speed = scan_index * (length / num_scans)
    pitch_cm = target_len / float(num_scans)
    print(f"[{name}] Spans {num_scans} scans. Calibrated pitch: {pitch_cm:.4f} cm/scan.")
    
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
    pcd = pcd.select_by_index(inliers, invert=True)
    
    # Voxel Downsample
    pcd = pcd.voxel_down_sample(voxel_size=10.0)
    
    # DBSCAN Clustering
    labels = np.array(pcd.cluster_dbscan(eps=150.0, min_points=20, print_progress=False))
    
    # Extract Truck
    max_label = labels.max()
    best_cluster_pcd = None
    max_pts = -1
    for i in range(max_label + 1):
        c_idx = np.where(labels == i)[0]
        c_pcd = pcd.select_by_index(c_idx)
        aabb = c_pcd.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        # Filter by a wider envelope since it is in cm now
        # L ~ 1200 cm, W ~ 195 cm, H ~ 180 cm
        if 500 <= extent[0] <= 2000 and 100 <= extent[1] <= 300 and 50 <= extent[2] <= 300:
            if len(c_idx) > max_pts:
                max_pts = len(c_idx)
                best_cluster_pcd = c_pcd
                
    if best_cluster_pcd is None:
        print(f"[{name}] Error: Truck cluster extraction failed.")
        return None
        
    # PCA Alignment
    pts_truck = np.asarray(best_cluster_pcd.points)
    centroid = np.mean(pts_truck, axis=0)
    centered = pts_truck - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evecs = evecs[:, order]
    R = evecs.T
    if R[2, 2] < 0:
        R[2, :] = -R[2, :]
        R[1, :] = -R[1, :]
    best_cluster_pcd = best_cluster_pcd.rotate(R, center=centroid)
    
    return best_cluster_pcd

def run_test():
    empty_file = "d:/point cloud technology/bgtest-3/empty/2026-04-07/2026-04-07_10-12-23.xyz"
    load_file = "d:/point cloud technology/bgtest-3/load/2026-04-07_11-27-22.xyz"
    
    empty_pcd = process_and_align_to_length(empty_file, "empty", TARGET_LENGTH_CM)
    load_pcd = process_and_align_to_length(load_file, "load", TARGET_LENGTH_CM)
    
    if empty_pcd is None or load_pcd is None:
        return
        
    # ICP
    if not load_pcd.has_normals():
        load_pcd.estimate_normals()
    if not empty_pcd.has_normals():
        empty_pcd.estimate_normals()
    reg = o3d.pipelines.registration.registration_icp(
        load_pcd, empty_pcd, 50.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    load_aligned = load_pcd.transform(reg.transformation)
    
    # Calculate volume
    empty_pts = np.asarray(empty_pcd.points)
    load_pts = np.asarray(load_aligned.points)
    
    x_min = min(empty_pts[:, 0].min(), load_pts[:, 0].min())
    x_max = max(empty_pts[:, 0].max(), load_pts[:, 0].max())
    y_min = min(empty_pts[:, 1].min(), load_pts[:, 1].min())
    y_max = max(empty_pts[:, 1].max(), load_pts[:, 1].max())
    
    res = 20.0 # cm
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
    
    e_max = df_e.groupby(['x', 'y'])['z'].max()
    l_max = df_l.groupby(['x', 'y'])['z'].max()
    
    for (x_idx, y_idx), z_val in e_max.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            empty_grid[x_idx, y_idx] = z_val
    for (x_idx, y_idx), z_val in l_max.items():
        if 0 <= x_idx < grid_shape[0] and 0 <= y_idx < grid_shape[1]:
            load_grid[x_idx, y_idx] = z_val
            
    valid_mask = ~np.isnan(empty_grid) & ~np.isnan(load_grid)
    height_diff = load_grid[valid_mask] - empty_grid[valid_mask]
    
    # Filter negative differences (noise)
    height_diff[height_diff < 0] = 0
    
    # Volume calculation in m^3
    cell_area_cm2 = res * res
    volume_cm3 = np.sum(height_diff * cell_area_cm2)
    volume_m3 = volume_cm3 / 1e6 # 1 m^3 = 10^6 cm^3
    
    print(f"\n=========================================")
    print(f"Target Length: {TARGET_LENGTH_CM / 100:.2f} m")
    aabb = load_aligned.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    print(f"Computed dimensions: {extent[0]/100:.2f} x {extent[1]/100:.2f} x {extent[2]/100:.2f} m")
    print(f"Computed Cargo Volume: {volume_m3:.4f} cubic meters")
    print(f"=========================================")

if __name__ == "__main__":
    run_test()
