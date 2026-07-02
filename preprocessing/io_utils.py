import open3d as o3d
import numpy as np
import pandas as pd
import os
import config
from calibration import calibrate_mapping
from preprocessing import sl_parser

def read_point_cloud(filepath):
    """
    Reads a point cloud from .xyz, .pcd, or SICK SL1 telegram text file.
    If it's .xyz, dynamically maps scan_index to X (in cm) so the truck matches TRUCK_REFERENCE_LENGTH_CM.
    If it's an SL1 telegram file, parses hex telegrams and converts polar to Cartesian.
    """
    basename = os.path.basename(filepath)
    
    if filepath.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(filepath)
        return pcd
    
    # Detect SL1 telegram file: starts with "SL1" in filename or ends in _RD.txt with SL1 prefix
    is_sl1 = basename.upper().startswith('SL1') and filepath.endswith('.txt')
    
    if is_sl1:
        return _read_sl1_file(filepath)
    elif filepath.endswith('.xyz'):
        return _read_xyz_file(filepath)
    else:
        raise ValueError(f"Unsupported file format for '{basename}'. Use .xyz, .pcd, or SL1 telegram .txt")


def _read_sl1_file(filepath):
    """Reads a SICK SL1 telegram text file and returns an Open3D PointCloud."""
    points, total_scans = sl_parser.sl1_to_point_cloud(filepath)
    
    if points.shape[0] == 0:
        raise ValueError(f"No valid points parsed from SL1 file: {filepath}")
        
    # --- Truck Window Detection via Z-Profile ---
    # The SL1 file contains continuous scanning. We must extract only the scans 
    # where the truck is passing under the scanner.
    
    # 1. Analyze the central region to avoid gantry walls
    central_mask = np.abs(points[:, 1]) < 150.0
    central_pts = points[central_mask]
    
    scan_ids = np.unique(central_pts[:, 0])
    scan_median_z = []
    for sid in scan_ids:
        sp = central_pts[central_pts[:, 0] == sid]
        if len(sp) > 5:
            scan_median_z.append((sid, np.median(sp[:, 2])))
            
    scan_median_z = np.array(scan_median_z)
    
    # 2. Estimate the floor Z (90th percentile of median Z)
    if len(scan_median_z) > 0:
        floor_z = np.percentile(scan_median_z[:, 1], 90)
        
        # 3. Truck scans are those where median Z is at least 30 cm above the floor
        # (Remember: Z increases downwards, so "above" means smaller Z)
        truck_threshold = floor_z - 30.0
        truck_scans = scan_median_z[scan_median_z[:, 1] < truck_threshold, 0]
        
        if len(truck_scans) > 10:
            min_scan = truck_scans.min()
            max_scan = truck_scans.max()
            num_scans = int(max_scan - min_scan) + 1
            print(f"[{os.path.basename(filepath)}] Detected truck window: scans {int(min_scan)}-{int(max_scan)} ({num_scans} scans). Floor Z ~ {floor_z:.1f} cm.")
            
            # Crop the point cloud to ONLY the truck window
            window_mask = (points[:, 0] >= min_scan) & (points[:, 0] <= max_scan)
            points = points[window_mask]
            
            # Reset scan indices to start at 0
            points[:, 0] -= min_scan
        else:
            print(f"[{os.path.basename(filepath)}] Warning: Could not detect clear truck window. Using all {total_scans} scans.")
            num_scans = total_scans
    else:
        num_scans = total_scans
        
    # Calibrate X-pitch dynamically based on the detected truck window
    pitch_cm = config.TRUCK_REFERENCE_LENGTH_CM / float(num_scans)
    print(f"[{os.path.basename(filepath)}] Dynamically calibrated pitch: {pitch_cm:.4f} cm/scan.")
    
    # Scale X-axis
    points[:, 0] = points[:, 0] * pitch_cm
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _read_xyz_file(filepath):
    """Reads an XYZ file [scan_index, Y, Z] and returns an Open3D PointCloud."""
    df = pd.read_csv(filepath, sep='\s+', header=None)
    
    # Determine scan span for the truck (points not at gantry walls Y=±210)
    df_obj = df[~np.isclose(df[1].abs(), 210.0, atol=1.0)]
    scan_min = df_obj[0].min()
    scan_max = df_obj[0].max()
    num_scans = scan_max - scan_min + 1
    
    # Calibrate pitch on the fly (distance in cm per scan line)
    pitch_cm = config.TRUCK_REFERENCE_LENGTH_CM / float(num_scans)
    print(f"[{os.path.basename(filepath)}] Spans {num_scans} scans. Dynamically calibrated pitch: {pitch_cm:.4f} cm/scan.")
    
    x_vals = df[0] * pitch_cm
    y_vals = df[1]
    z_vals = df[2]
    
    points = np.vstack((x_vals, y_vals, z_vals)).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def save_point_cloud(pcd, filepath):
    o3d.io.write_point_cloud(filepath, pcd)

