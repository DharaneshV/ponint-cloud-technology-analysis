import os
import numpy as np
import pandas as pd
import glob
from scipy.stats import skew

# We use the SL1 reference volume to calculate the normalized fill percentage
SL1_BED_CAPACITY_M3 = 14.25 # Derived from empty_report.json earlier
Z_RES_CM = 2.5 # Our scaling assumption for Zenodo

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def extract_ml_features(filepath):
    try:
        matrix = np.loadtxt(filepath)
    except:
        return None
        
    # Filter out sky/error points before ground estimation
    valid_pts = matrix[matrix < config.SKY_CODE_THRESHOLD]
    if len(valid_pts) == 0:
        return None
        
    # Use 95th percentile of valid points as the road surface depth
    ground_z = np.percentile(valid_pts, 95)
    
    # Assert sanity check to make sure ground Z is not contaminated by sky codes
    assert ground_z < 50000.0, f"Sky contamination detected in ground Z estimation ({ground_z:.1f} mm)!"
    
    height_map = ground_z - matrix
    
    vehicle_mask = height_map > 500 # >50cm tall
    if not np.any(vehicle_mask):
        return None
        
    rows = np.any(vehicle_mask, axis=1)
    cols = np.any(vehicle_mask, axis=0)
    
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        return None
        
    veh_width = (rmax - rmin) * Z_RES_CM
    veh_length = (cmax - cmin) * Z_RES_CM
    
    # Extract the bed area (assuming cab is at the front, so we look at the back 70%)
    # Let's define the bed footprint more robustly. The bed is usually the lower section.
    # To keep it simple, we use the central 50% of the vehicle width, and the back 60% of length.
    bed_cmin = cmin + int((cmax - cmin) * 0.4) # skip front 40% (cab)
    bed_cmax = cmax
    bed_rmin = rmin + int((rmax - rmin) * 0.2)
    bed_rmax = rmax - int((rmax - rmin) * 0.2)
    
    if bed_cmax <= bed_cmin or bed_rmax <= bed_rmin:
        return None
        
    bed_region = height_map[bed_rmin:bed_rmax, bed_cmin:bed_cmax]
    
    # Filter out negative sky/error cells in the bed region before finding floor_z
    valid_bed = bed_region[bed_region > -100]
    if len(valid_bed) == 0:
        return None
        
    # The lowest point in the bed is the floor
    floor_z = np.min(valid_bed)
    
    # The height of the cargo relative to the floor (for flat statistics)
    cargo_heights = (valid_bed - floor_z) / 10.0 # cm
    
    # 1. max_heap_height
    max_heap_height = np.max(cargo_heights)
    
    # 2. mean_bed_height
    mean_bed_height = np.mean(cargo_heights)
    
    # 3. height_variance
    height_variance = np.var(cargo_heights)
    
    # 4. coverage_ratio
    # Assuming any material > 10cm above floor is "covered"
    covered_cells = np.sum(cargo_heights > 10.0)
    total_cells = bed_region.size
    coverage_ratio = covered_cells / total_cells if total_cells > 0 else 0
    
    # 5. height_skewness (front vs back loaded)
    # Replace sky cells (<= -100) with floor_z, then subtract floor_z
    filled_bed = np.where(bed_region > -100, bed_region, floor_z)
    cargo_grid = (filled_bed - floor_z) / 10.0
    length_profile = np.mean(cargo_grid, axis=0)
    height_skewness = skew(length_profile) if len(length_profile) > 3 else 0
    
    # 6. valid_point_count (density)
    valid_point_count = np.sum(vehicle_mask)
    
    # 7. normalized_fill_pct (relative to this truck's own walls!)
    # The walls are the highest point of the vehicle overall
    max_vehicle_h = np.max(height_map)
    max_wall_height = (max_vehicle_h - floor_z) / 10.0 # cm
    
    # Calculate geometric volume of the cargo
    cell_area_cm2 = Z_RES_CM ** 2
    cargo_volume_cm3 = np.sum(cargo_heights) * cell_area_cm2
    
    # Calculate max possible capacity of this specific bed region
    bed_area_cm2 = bed_region.size * cell_area_cm2
    max_capacity_cm3 = bed_area_cm2 * max_wall_height
    
    # Calculate percentage (scaling units cancel out completely!)
    if max_capacity_cm3 > 0:
        normalized_fill_pct = (cargo_volume_cm3 / max_capacity_cm3) * 100.0
    else:
        normalized_fill_pct = 0.0
        
    # Cap at 120% (for heaped sand)
    normalized_fill_pct = min(normalized_fill_pct, 120.0)
    
    # Generate labels using calibrated thresholds
    if normalized_fill_pct < 20.0:
        label = "Empty"
    elif normalized_fill_pct > 65.0:
        label = "Full"
    else:
        label = "Partial"
        
    # Generate a pseudo truck_id based on physical dimensions 
    # (assuming vehicles with identical length/width might be the same physical truck returning)
    truck_id = f"truck_{int(veh_length)}_{int(veh_width)}"
    
    return {
        "filename": os.path.basename(filepath),
        "truck_id": truck_id,
        "veh_length_cm": veh_length,
        "veh_width_cm": veh_width,
        "max_heap_height": max_heap_height,
        "mean_bed_height": mean_bed_height,
        "height_variance": height_variance,
        "coverage_ratio": coverage_ratio,
        "height_skewness": height_skewness,
        "valid_point_count": valid_point_count,
        "geometric_volume_m3": cargo_volume_cm3 / 1000000.0,
        "normalized_fill_pct": normalized_fill_pct,
        "geometric_label": label
    }

def main():
    src_dir = r"d:\point cloud technology\Truck-PointCloud\zenodo_exact_matches"
    dest_file = r"d:\point cloud technology\Truck-PointCloud\ml_model\training_dataset.csv"
    
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    
    files = glob.glob(os.path.join(src_dir, "*.out"))
    print(f"Found {len(files)} matches. Extracting features...")
    
    data = []
    for i, file in enumerate(files):
        features = extract_ml_features(file)
        if features:
            data.append(features)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(files)} files...")
            
    df = pd.DataFrame(data)
    df.to_csv(dest_file, index=False)
    
    print(f"\nFeature extraction complete! Saved to {dest_file}")
    print(f"Total valid samples: {len(df)}")
    print("\nClass Balance:")
    print(df['geometric_label'].value_counts())

if __name__ == "__main__":
    main()
