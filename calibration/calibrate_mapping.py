import json
import os
import warnings

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), '..', 'calibration.json')

def calibrate_from_reference(num_scans, actual_length_units):
    """
    Calibrates the speed or pitch based on a known truck length.
    
    Args:
        num_scans (int): Number of scans the truck spanned in the point cloud.
        actual_length_units (float): Known physical length of the truck.
        
    Returns:
        dict: Calibration dictionary.
    """
    # From config.py
    # SCAN_INTERVAL_S = 0.010 (10 ms)
    # If encoder: pitch = length / num_scans
    # If timed: speed = length / (num_scans * 0.010)
    
    scan_interval_s = 0.010
    
    pitch = actual_length_units / float(num_scans)
    speed = actual_length_units / (float(num_scans) * scan_interval_s)
    
    calib_data = {
        "encoder_pitch": pitch,
        "speed": speed,
        "reference_scans": num_scans,
        "reference_length": actual_length_units
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"Calibration successful. Pitch: {pitch:.4f} units/scan. Speed: {speed:.4f} units/s.")
    return calib_data

def load_calibration():
    """
    Loads calibration data if it exists.
    """
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return None
