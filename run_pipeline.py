import os
import sys
import numpy as np
import open3d as o3d

# Add the current directory to sys.path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import io_utils, noise_filter, gantry_filter, ground_plane_removal, downsample
from segmentation import clustering, truck_extraction
from registration import pca_alignment, icp_registration
from analysis import dimensions, volume
from debug import visualize_stage
import config

def process_single_scan(filepath, name="scan", debug_dir="results/debug"):
    print(f"\n--- Processing {name} ---")
    
    # 1. Read
    pcd = io_utils.read_point_cloud(filepath)
    visualize_stage.dump_stage(pcd, f"{name}_01_raw", output_dir=debug_dir)
    
    # 2. Noise Filter
    pcd = noise_filter.remove_statistical_outliers(pcd)
    visualize_stage.dump_stage(pcd, f"{name}_02_sor", output_dir=debug_dir)
    
    # 3. Gantry Filter
    pcd = gantry_filter.filter_gantry(pcd)
    visualize_stage.dump_stage(pcd, f"{name}_03_gantry_filtered", output_dir=debug_dir)
    
    # 4. Ground Plane Removal (Pass-through Z <= 330)
    pcd = ground_plane_removal.remove_ground_plane(pcd)
    visualize_stage.dump_stage(pcd, f"{name}_04_no_ground", output_dir=debug_dir)
    
    # 5. Downsample
    pcd = downsample.downsample_point_cloud(pcd)
    visualize_stage.dump_stage(pcd, f"{name}_05_downsampled", output_dir=debug_dir)
    
    # 6. Clustering
    labels = clustering.cluster_points(pcd)
    
    # 7. Truck Extraction
    truck_pcd = truck_extraction.extract_truck_cluster(pcd, labels)
    
    if truck_pcd is None:
        print(f"Failed to extract truck for {name}.")
        return None, None
        
    visualize_stage.dump_stage(truck_pcd, f"{name}_06_truck_extracted", output_dir=debug_dir)
    
    # Align along X to start at 0 (for volume calculation)
    pts_truck = np.asarray(truck_pcd.points)
    x_min_val = pts_truck[:, 0].min()
    pts_truck_shifted = pts_truck.copy()
    pts_truck_shifted[:, 0] -= x_min_val
    
    truck_unrotated = o3d.geometry.PointCloud()
    truck_unrotated.points = o3d.utility.Vector3dVector(pts_truck_shifted)
    
    # 8. PCA Alignment (for dimensions only)
    truck_aligned, R, centroid = pca_alignment.align_truck_pca(truck_pcd)
    visualize_stage.dump_stage(truck_aligned, f"{name}_07_pca_aligned", output_dir=debug_dir)
    
    return truck_aligned, truck_unrotated

def run_pipeline_comparison(empty_aligned, empty_unrot, load_aligned, load_unrot, output_dir="results/runs", debug_dir="results/debug", ml_classify=False, ml_mode="reference_diff"):
    if empty_aligned is None or load_aligned is None:
        print("Pipeline aborted due to extraction failure.")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
        
    # ICP Registration for Dimensions (Load onto Empty PCA-aligned)
    print("\n--- ICP Registration (PCA Frame) ---")
    load_icp, transform = icp_registration.register_icp(load_aligned, empty_aligned)
    visualize_stage.dump_stage(load_icp, "load_08_icp_aligned", output_dir=debug_dir)
    
    # ICP Registration for Volume (Scanner Frame preserving Z)
    print("\n--- ICP Registration (Volume Frame) ---")
    load_unrot_aligned, transform_unrot = icp_registration.register_icp(load_unrot, empty_unrot)
    visualize_stage.dump_stage(load_unrot_aligned, "load_08_unrot_icp_aligned", output_dir=debug_dir)
    
    # Dimensions (calculated on aligned PCA frame)
    print("\n--- Dimensions ---")
    dim_file = os.path.join(output_dir, "dimensions_load.json")
    dims, aabb, obb = dimensions.calculate_dimensions(load_icp, output_file=dim_file)
    
    # Volume (calculated on aligned unrotated scanner frame to preserve Z distance)
    print("\n--- Volume Calculation ---")
    volume_report, empty_floor_grid, load_floor_grid = volume.compute_volume(empty_unrot, load_unrot_aligned)
    
    print(f"\nPipeline completed successfully!")
    
    results = {
        "dimensions": dims,
        "volume": volume_report
    }
    
    if ml_classify:
        print(f"\n--- ML Classification (Mode: {ml_mode}) ---")
        try:
            import ml_model.infer
            
            if ml_mode == "independent":
                class_pred, conf, ml_vol = ml_model.infer.get_independent_ml_prediction(load_unrot)
            else:
                # floor_z is the median level of the empty truck bed floor
                floor_z = np.nanmedian(empty_floor_grid)
                class_pred, conf = ml_model.infer.get_ml_prediction(load_floor_grid, floor_z)
            
            print(f"Fill estimate: {class_pred} (confidence {conf:.2%})")
            results["ml_prediction"] = class_pred
            results["ml_confidence"] = conf
            results["mode"] = ml_mode
            if ml_mode == "independent":
                results["ml_volume_m3"] = round(ml_vol, 4)
        except Exception as e:
            print(f"ML Classification failed: {e}")
            
    # Save a combined report
    import json
    report_file = os.path.join(output_dir, "report.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved full report to {report_file}")
    
    return results

if __name__ == "__main__":
    empty_file = "d:/point cloud technology/bgtest-3/empty/2026-04-07/2026-04-07_10-12-23.xyz"
    load_file = "d:/point cloud technology/bgtest-3/load/2026-04-07_11-27-22.xyz"
    
    print(f"Configuration Mode: {config.MAPPING_MODE}")
    
    empty_aligned, empty_unrot = process_single_scan(empty_file, "empty")
    load_aligned, load_unrot = process_single_scan(load_file, "load")
    
    run_pipeline_comparison(empty_aligned, empty_unrot, load_aligned, load_unrot)

