"""
Parser for SICK LMS SL1 LiDAR telegram files.

Each SL1 file contains multiple scan lines (telegrams), one per non-empty line.
Each telegram is a SICK SOPAS binary-ASCII message of the form:
    STX sSN LMDscandata ... DIST1 <scale> <offset> <start_angle> <angular_step> <num_points> <hex_dist...> ... ETX

The scanner sweeps a 180-degree vertical fan per telegram. 
Successive telegrams correspond to successive positions along the X-axis (truck travel direction).

Output: an Open3D PointCloud with coordinates in centimeters [X, Y, Z].
"""

import numpy as np
import os


def parse_sl1_telegram(line):
    """
    Parses a single SL1 telegram line and returns an array of (Y, Z) distances in mm.
    
    Returns:
        distances_mm: np.array of shape (N,) – radial distances in mm
        start_angle_deg: float – start angle in degrees
        angular_step_deg: float – angular step in degrees
        num_points: int – number of measurements
        
    Returns (None, ...) if the line cannot be parsed.
    """
    # Strip control characters (STX=0x02, ETX=0x03) and whitespace
    line = line.replace('\x02', '').replace('\x03', '').strip()
    if not line:
        return None, 0, 0, 0
    
    tokens = line.split()
    
    # Find the DIST1 keyword
    if 'DIST1' not in tokens:
        return None, 0, 0, 0
    
    dist_idx = tokens.index('DIST1')
    
    # Tokens after DIST1:
    #   [0] Scale factor (IEEE float hex, e.g., 3F800000 = 1.0)
    #   [1] Offset (IEEE float hex, e.g., 00000000 = 0.0)
    #   [2] Start angle (hex, in 1/10000 degrees)
    #   [3] Angular step (hex, in 1/10000 degrees)
    #   [4] Number of data points (hex)
    #   [5..5+N] Distance values (hex, in mm)
    
    start_angle_raw = int(tokens[dist_idx + 3], 16)  # 1/10000 degrees
    angular_step_raw = int(tokens[dist_idx + 4], 16)  # 1/10000 degrees
    num_points = int(tokens[dist_idx + 5], 16)
    
    start_angle_deg = start_angle_raw / 10000.0
    angular_step_deg = angular_step_raw / 10000.0
    
    data_start = dist_idx + 6
    data_end = data_start + num_points
    
    if data_end > len(tokens):
        return None, 0, 0, 0
    
    distances_hex = tokens[data_start:data_end]
    distances_mm = np.array([int(h, 16) for h in distances_hex], dtype=np.float64)
    
    return distances_mm, start_angle_deg, angular_step_deg, num_points


def sl1_to_point_cloud(filepath, truck_length_cm=None):
    """
    Reads a SICK SL1 telegram file and converts it to a 3D point cloud.
    
    The scanner is mounted above the road pointing downward.
    Each telegram sweeps a 180-degree vertical fan (Y-Z plane).
    Successive telegrams advance along the X-axis (truck travel direction).
    
    Coordinate system:
        X = travel direction (scan_index * pitch)
        Y = lateral (horizontal, derived from polar conversion)
        Z = vertical (depth/height, derived from polar conversion)
    
    Args:
        filepath: Path to the SL1 text file.
        truck_length_cm: If set, dynamically calibrate X-pitch to this total length.
    
    Returns:
        points: np.array of shape (M, 3), coordinates in centimeters [X, Y, Z]
        num_scans: total number of scan lines processed
    """
    basename = os.path.basename(filepath)
    
    with open(filepath, 'r') as f:
        raw_lines = f.readlines()
    
    # Collect all valid telegrams
    all_points = []
    scan_index = 0
    
    for line in raw_lines:
        distances_mm, start_angle_deg, angular_step_deg, num_points = parse_sl1_telegram(line)
        if distances_mm is None:
            continue
        
        # Build angle array (in radians)
        # The scanner sweeps 0° to 180° — 0° is one side, 90° is straight down, 180° is the other side
        angles_deg = start_angle_deg + np.arange(num_points) * angular_step_deg
        angles_rad = np.deg2rad(angles_deg)
        
        # Convert polar (distance, angle) to Cartesian (Y, Z)
        # Convention: 90° = straight down
        #   Y = distance * cos(angle)   (lateral, positive to one side)
        #   Z = distance * sin(angle)   (vertical depth, downward is positive)
        # Distances are in mm; convert to cm
        dist_cm = distances_mm / 10.0
        
        # Filter out zero/invalid readings
        valid_mask = dist_cm > 0.5  # ignore distances < 0.5 cm (noise/zero)
        
        y_vals = dist_cm[valid_mask] * np.cos(angles_rad[valid_mask])
        z_vals = dist_cm[valid_mask] * np.sin(angles_rad[valid_mask])
        
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            scan_index += 1
            continue
        
        # X is just the scan index for now (will be scaled later)
        x_vals = np.full(n_valid, float(scan_index))
        
        scan_points = np.vstack((x_vals, y_vals, z_vals)).T
        all_points.append(scan_points)
        
        scan_index += 1
    
    if len(all_points) == 0:
        print(f"[{basename}] WARNING: No valid telegrams found!")
        return np.empty((0, 3)), 0
    
    points = np.vstack(all_points)
    total_scans = scan_index
    
    print(f"[{basename}] Parsed {total_scans} scan lines, {points.shape[0]} total points.")
    
    return points, total_scans
