import os
import glob
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.stats import skew
import sys
import argparse

# Ensure modules can be imported
sys.path.append(r"d:\point cloud technology\Truck-PointCloud")

from preprocessing import io_utils, noise_filter, ground_plane_removal, downsample
from segmentation import clustering, truck_extraction
import config

def extract_single_scan_features(pcd):
    """Extracts scale-invariant shape and density features from a single truck cluster."""
    points = np.asarray(pcd.points)
    if len(points) < 100:
        return None
        
    z_coords = points[:, 2]
    floor_z = np.percentile(z_coords, 95)
    
    mean_z = np.mean(z_coords)
    median_z = np.median(z_coords)
    std_z = np.std(z_coords)
    skew_z = skew(z_coords)
    
    cargo_points = np.sum(z_coords < (floor_z - 10.0))  # 10.0 cm
    coverage_proxy = cargo_points / len(points)
    
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent() # [x, y, z] in cm
    vol_cm3 = extent[0] * extent[1] * extent[2]
    density = len(points) / vol_cm3 if vol_cm3 > 0 else 0
    
    return {
        'mean_z': mean_z,
        'median_z': median_z,
        'std_z': std_z,
        'skew_z': skew_z,
        'coverage_proxy': coverage_proxy,
        'point_density': density
    }

def process_single_file(filepath, scanner_type="SL1"):
    print(f"\nProcessing {os.path.basename(filepath)}...")
    
    pcd = io_utils.read_point_cloud(filepath)
    pcd = noise_filter.remove_statistical_outliers(pcd)
    
    # Custom gantry bounds depending on the scanner configuration
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return None
        
    if scanner_type == "SL2":
        # The truck in Lane 1 is shifted to Y in [-450, -180] relative to SL2
        gantry_min, gantry_max = -450.0, -180.0
    else:
        # Default SL1 bounds
        gantry_min, gantry_max = config.GANTRY_Y_MIN, config.GANTRY_Y_MAX
        
    mask = (points[:, 1] >= gantry_min) & (points[:, 1] <= gantry_max)
    pcd = pcd.select_by_index(np.where(mask)[0])
    
    # Ground plane check
    points_filtered = np.asarray(pcd.points)
    if len(points_filtered) == 0:
        return None
    ground_z = np.max(points_filtered[:, 2])
    
    # Enforce ground sanity calibration check (both channels are ~590 cm base height)
    if abs(ground_z - 590.0) > 50.0:
        print(f"  WARNING: Ground Z ({ground_z:.1f}cm) differs wildly from expected 590.0cm. Skipping to avoid calibration risk.")
        return None
        
    pcd = ground_plane_removal.remove_ground_plane(pcd)
    pcd = downsample.downsample_point_cloud(pcd)
    
    # Apply custom min length envelope for SL2 scans
    orig_min_len = config.TRUCK_ENVELOPE_MIN_LENGTH
    if scanner_type == "SL2":
        config.TRUCK_ENVELOPE_MIN_LENGTH = 400.0
        
    labels = clustering.cluster_points(pcd)
    truck_pcd = truck_extraction.extract_truck_cluster(pcd, labels)
    
    config.TRUCK_ENVELOPE_MIN_LENGTH = orig_min_len
    
    return truck_pcd

def main():
    parser = argparse.ArgumentParser(description="Independent Dataset Generator")
    parser.add_argument("--scanner", type=str, choices=["SL1", "SL2"], default="SL1", help="Scanner type (SL1 or SL2)")
    args = parser.parse_args()
    
    empty_dir = r"d:\point cloud technology\bgtest-3\empty"
    load_dir = r"d:\point cloud technology\bgtest-3\load"
    
    out_csv = f"d:\\point cloud technology\\Truck-PointCloud\\ml_model\\independent_dataset_{args.scanner.lower()}.csv"
    reference_csv = r"d:\point cloud technology\Truck-PointCloud\ml_model\sick_validation_dataset.csv"
    
    ref_df = pd.read_csv(reference_csv)
    label_map = dict(zip(ref_df['filename'], ref_df['geometric_label']))
    fill_map = dict(zip(ref_df['filename'], ref_df['fill_percentage']))
    
    # Glob corresponding scanner files
    empty_files = sorted(glob.glob(os.path.join(empty_dir, f"{args.scanner}*_RD.txt")))
    load_files = sorted(glob.glob(os.path.join(load_dir, f"{args.scanner}*_RD.txt")))
    
    all_files = empty_files + load_files
    
    # Header validation verification
    for filepath in all_files:
        filename = os.path.basename(filepath)
        assert args.scanner in filename, f"File {filename} does not match scanner type {args.scanner}"
                
    dataset = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Cross-reference with the SL1 equivalent file in ref_df
        sl1_equivalent = filename.replace("SL2", "SL1")
        if sl1_equivalent not in label_map:
            print(f"Skipping {filename} - equivalent SL1 file not labeled.")
            continue
            
        truck_pcd = process_single_file(filepath, args.scanner)
        if truck_pcd is None:
            print(f"Skipping {filename} - point cloud extraction failed.")
            continue
            
        feats = extract_single_scan_features(truck_pcd)
        if feats is not None:
            feats['filename'] = filename
            feats['geometric_label'] = label_map[sl1_equivalent]
            fill_pct = fill_map[sl1_equivalent]
            feats['target_volume_m3'] = (fill_pct / 100.0) * 16.8814
            dataset.append(feats)
            print(f"  Extracted: {feats}")
            
    df = pd.DataFrame(dataset)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} independent scan features for {args.scanner} to {out_csv}")

if __name__ == "__main__":
    main()
