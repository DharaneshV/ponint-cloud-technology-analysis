import os
import sys
import glob
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.stats import skew

# Ensure modules can be imported
sys.path.append(r"d:\point cloud technology\Truck-PointCloud")

from preprocessing import io_utils, noise_filter, gantry_filter, ground_plane_removal, downsample
from segmentation import clustering, truck_extraction
from registration import icp_registration
from analysis import volume
import config

def process_empty_composite(empty_files):
    """Processes first 3 empty files and merges them using ICP to build a robust composite reference."""
    print("--- Building composite empty reference from first 3 scans ---")
    pcds = []
    for i, path in enumerate(empty_files[:3]):
        print(f"Loading empty reference scan {i+1}: {os.path.basename(path)}")
        pcd = io_utils.read_point_cloud(path)
        pcd = noise_filter.remove_statistical_outliers(pcd)
        pcd = gantry_filter.filter_gantry(pcd)
        pcd = ground_plane_removal.remove_ground_plane(pcd)
        pcd = downsample.downsample_point_cloud(pcd)
        
        # Segment truck
        labels = clustering.cluster_points(pcd)
        truck_pcd = truck_extraction.extract_truck_cluster(pcd, labels)
        if truck_pcd is not None:
            # Shift X to start at 0
            pts = np.asarray(truck_pcd.points)
            pts[:, 0] -= pts[:, 0].min()
            truck_pcd.points = o3d.utility.Vector3dVector(pts)
            pcds.append(truck_pcd)
            print(f"Parsed empty truck scan {i+1} with {len(truck_pcd.points)} points.")
        else:
            print(f"Warning: Failed to extract empty truck for {os.path.basename(path)}")

    if not pcds:
        raise ValueError("Failed to extract empty truck point clouds.")

    # Align scans 2 and 3 to scan 1 using ICP
    base_pcd = pcds[0]
    aligned_points = [np.asarray(base_pcd.points)]
    
    for i in range(1, len(pcds)):
        aligned_pcd, _ = icp_registration.register_icp(pcds[i], base_pcd)
        aligned_points.append(np.asarray(aligned_pcd.points))
        
    merged_pts = np.vstack(aligned_points)
    composite_empty = o3d.geometry.PointCloud()
    composite_empty.points = o3d.utility.Vector3dVector(merged_pts)
    
    # Downsample merged point cloud to standard resolution
    composite_empty = composite_empty.voxel_down_sample(voxel_size=0.5) # 5mm
    print(f"Composite empty reference model constructed with {len(composite_empty.points)} points.")
    return composite_empty

def process_single_file(filepath):
    """Helper to load and preprocess a single SICK SL1 file."""
    pcd = io_utils.read_point_cloud(filepath)
    pcd = noise_filter.remove_statistical_outliers(pcd)
    pcd = gantry_filter.filter_gantry(pcd)
    pcd = ground_plane_removal.remove_ground_plane(pcd)
    pcd = downsample.downsample_point_cloud(pcd)
    
    labels = clustering.cluster_points(pcd)
    truck_pcd = truck_extraction.extract_truck_cluster(pcd, labels)
    if truck_pcd is None:
        return None
        
    # Shift X to start at 0
    pts = np.asarray(truck_pcd.points)
    pts[:, 0] -= pts[:, 0].min()
    truck_pcd.points = o3d.utility.Vector3dVector(pts)
    return truck_pcd

def compute_grid_heights(empty_pcd, load_pcd):
    """Computes the 2D height difference grid between empty and load point clouds."""
    # 1. Compute empty capacity and grid bounds
    res = config.HEIGHTMAP_GRID_RESOLUTION # 5.0 cm
    empty_pts = np.asarray(empty_pcd.points)
    x_min, x_max = empty_pts[:, 0].min(), empty_pts[:, 0].max()
    y_min, y_max = empty_pts[:, 1].min(), empty_pts[:, 1].max()
    
    x_bins = np.arange(x_min, x_max + res, res)
    y_bins = np.arange(y_min, y_max + res, res)
    grid_shape = (len(x_bins) - 1, len(y_bins) - 1)
    
    # Empty floor grid
    empty_x_idx = np.digitize(empty_pts[:, 0], x_bins) - 1
    empty_y_idx = np.digitize(empty_pts[:, 1], y_bins) - 1
    valid_empty = (empty_x_idx >= 0) & (empty_x_idx < grid_shape[0]) & (empty_y_idx >= 0) & (empty_y_idx < grid_shape[1])
    
    df_empty = pd.DataFrame({
        'x': empty_x_idx[valid_empty],
        'y': empty_y_idx[valid_empty],
        'z': empty_pts[valid_empty, 2]
    })
    empty_floor_grid = np.full(grid_shape, np.nan)
    for (xi, yi), zval in df_empty.groupby(['x', 'y'])['z'].max().items():
        empty_floor_grid[xi, yi] = zval
        
    # Load floor grid
    load_pts = np.asarray(load_pcd.points)
    load_x_idx = np.digitize(load_pts[:, 0], x_bins) - 1
    load_y_idx = np.digitize(load_pts[:, 1], y_bins) - 1
    valid_load = (load_x_idx >= 0) & (load_x_idx < grid_shape[0]) & (load_y_idx >= 0) & (load_y_idx < grid_shape[1])
    
    df_load = pd.DataFrame({
        'x': load_x_idx[valid_load],
        'y': load_y_idx[valid_load],
        'z': load_pts[valid_load, 2]
    })
    load_floor_grid = np.full(grid_shape, np.nan)
    for (xi, yi), zval in df_load.groupby(['x', 'y'])['z'].max().items():
        load_floor_grid[xi, yi] = zval
        
    # Difference heightmap (empty Z depth - load Z depth)
    valid_both = ~np.isnan(empty_floor_grid) & ~np.isnan(load_floor_grid)
    cargo_heightmap = np.full(grid_shape, np.nan)
    cargo_heightmap[valid_both] = empty_floor_grid[valid_both] - load_floor_grid[valid_both]
    
    return cargo_heightmap, empty_floor_grid, load_floor_grid

def extract_ml_features(cargo_heightmap, empty_floor_grid):
    """Extracts the 5 ML shape features from a height difference grid."""
    valid_mask = ~np.isnan(cargo_heightmap)
    cargo_heights = cargo_heightmap[valid_mask]
    
    if len(cargo_heights) == 0:
        return None
        
    max_heap_height = np.max(cargo_heights)
    mean_bed_height = np.mean(cargo_heights)
    height_variance = np.var(cargo_heights)
    
    covered_cells = np.sum(cargo_heights > 10.0) # >10cm
    total_cells = cargo_heightmap.size
    coverage_ratio = covered_cells / total_cells if total_cells > 0 else 0
    
    # Skewness along length profile
    filled_grid = np.nan_to_num(cargo_heightmap, nan=0.0)
    length_profile = np.mean(filled_grid, axis=0)
    height_skewness = skew(length_profile) if len(length_profile) > 3 else 0
    
    return {
        'max_heap_height': max_heap_height,
        'mean_bed_height': mean_bed_height,
        'height_variance': height_variance,
        'coverage_ratio': coverage_ratio,
        'height_skewness': height_skewness
    }

def main():
    empty_dir = r"d:\point cloud technology\bgtest-3\empty"
    load_dir = r"d:\point cloud technology\bgtest-3\load"
    dest_file = r"d:\point cloud technology\Truck-PointCloud\ml_model\sick_validation_dataset.csv"
    
    empty_files = sorted(glob.glob(os.path.join(empty_dir, "SL1*_RD.txt")))
    load_files = sorted(glob.glob(os.path.join(load_dir, "SL1*_RD.txt")))
    
    if not empty_files or not load_files:
        print("SICK SL1 scan files not found.")
        return
        
    # Build Master Empty Composite
    composite_empty = process_empty_composite(empty_files)
    
    dataset_records = []
    
    # 1. Process all Empty scans to extract realistic noise features
    print("\n--- STEP 1: Processing Empty Scans for Noise Features ---")
    for i, path in enumerate(empty_files):
        filename = os.path.basename(path)
        print(f"Processing Empty {i+1}/{len(empty_files)}: {filename}")
        
        empty_truck = process_single_file(path)
        if empty_truck is None:
            print(f"Skipped empty: {filename} (failed extraction)")
            continue
            
        # Align empty scan to the composite empty reference
        aligned_empty, _ = icp_registration.register_icp(empty_truck, composite_empty)
        
        # Height map differencing (representing scan noise/alignment jitter)
        cargo_heightmap, empty_floor_grid, _ = compute_grid_heights(composite_empty, aligned_empty)
        
        # Clean small noise (less than -20cm or very large outliers)
        # Note: Do not truncate all small heights to 0.0 because we WANT to measure noise
        cargo_heightmap[cargo_heightmap < -20.0] = np.nan
        
        feats = extract_ml_features(cargo_heightmap, empty_floor_grid)
        if feats is not None:
            feats['filename'] = filename
            feats['geometric_label'] = 'Empty'
            feats['fill_percentage'] = 0.0
            dataset_records.append(feats)
            print(f"Extracted noise features: Mean Ht={feats['mean_bed_height']:.2f}cm, Var={feats['height_variance']:.2f}")
            
    # 2. Process all Load scans
    print("\n--- STEP 2: Processing Load Scans ---")
    for i, path in enumerate(load_files):
        filename = os.path.basename(path)
        print(f"Processing Load {i+1}/{len(load_files)}: {filename}")
        
        load_truck = process_single_file(path)
        if load_truck is None:
            print(f"Skipped load: {filename} (failed extraction)")
            continue
            
        # Align load scan to the composite empty reference
        aligned_load, _ = icp_registration.register_icp(load_truck, composite_empty)
        
        # Height map differencing
        cargo_heightmap, empty_floor_grid, _ = compute_grid_heights(composite_empty, aligned_load)
        
        # Run standard geometric pipeline volume checks to get actual fill % and capacity
        # For simplicity, calculate cargo volume directly from heights
        res = config.HEIGHTMAP_GRID_RESOLUTION
        cell_area = res * res
        
        # Standard volume ignores noise diffs < 5cm
        clean_heights = cargo_heightmap.copy()
        clean_heights[clean_heights < 5.0] = 0.0
        clean_heights[np.isnan(clean_heights)] = 0.0
        
        cargo_volume_cm3 = np.sum(clean_heights * cell_area)
        cargo_volume_m3 = cargo_volume_cm3 / 1e6
        
        # Compute bed capacity from composite empty reference
        empty_pts = np.asarray(composite_empty.points)
        wall_top_z = np.percentile(empty_pts[:, 2], 5) # 5th percentile
        
        valid_empty = ~np.isnan(empty_floor_grid)
        depth_grid = empty_floor_grid - wall_top_z
        depth_grid[depth_grid < 10.0] = 0.0
        
        bed_volume_cm3 = np.sum(depth_grid[valid_empty] * cell_area)
        bed_capacity_m3 = bed_volume_cm3 / 1e6
        
        fill_pct = (cargo_volume_m3 / bed_capacity_m3 * 100) if bed_capacity_m3 > 0 else 0
        
        # Extract shape features from heightmap
        # We clip negative values (noise) to np.nan for feature calculations
        feature_heightmap = cargo_heightmap.copy()
        feature_heightmap[feature_heightmap < 0.0] = 0.0
        
        feats = extract_ml_features(feature_heightmap, empty_floor_grid)
        if feats is not None:
            # Assign label based on geometric fill %
            if fill_pct < 20.0:
                label = 'Empty'
            elif fill_pct > 65.0:
                label = 'Full'
            else:
                label = 'Partial'
                
            feats['filename'] = filename
            feats['geometric_label'] = label
            feats['fill_percentage'] = fill_pct
            dataset_records.append(feats)
            print(f"Extracted load features: Fill={fill_pct:.1f}%, Label={label}, Mean Ht={feats['mean_bed_height']:.2f}cm")
            
    # Write SICK dataset CSV
    if dataset_records:
        df_sick = pd.DataFrame(dataset_records)
        df_sick.to_csv(dest_file, index=False)
        print(f"\nSuccessfully saved SICK features to: {dest_file}")
        print(f"Total samples: {len(df_sick)}")
        print(df_sick['geometric_label'].value_counts())
    else:
        print("No SICK features extracted.")

if __name__ == "__main__":
    main()
