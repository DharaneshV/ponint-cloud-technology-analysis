import numpy as np
import open3d as o3d
import sys
import os

sys.path.append(r"d:\point cloud technology\Truck-PointCloud")
from analysis.volume import compute_single_volume

def process_zenodo_out(file_path):
    print(f"Loading Zenodo depth matrix: {file_path}")
    data = np.loadtxt(file_path)
    
    # data is a 2D depth matrix (e.g. 240x543)
    rows, cols = data.shape
    
    # Generate X and Y grid
    # Let's assume a standard 5cm (50mm) resolution between LiDAR points
    res = 5.0 
    
    x = np.linspace(0, cols * res, cols)
    y = np.linspace(0, rows * res, rows)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the arrays to create XYZ points
    Z = data.flatten()
    X = X.flatten()
    Y = Y.flatten()
    
    # Filter out invalid points (e.g. Z == 1.0 or Z > 60000 might be errors/sky in LiDAR)
    valid = (Z > 100) & (Z < 50000)
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    
    # Convert Z from mm to cm if the values are very large (like 24000)
    if np.mean(Z) > 5000:
        Z = Z / 10.0
        
    pts = np.vstack((X, Y, Z)).T
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    print(f"Generated Point Cloud with {len(pcd.points)} points.")
    
    # Run our volume calculator!
    vol, wall_top_z, floor_grid, grid_shape, x_bins, y_bins = compute_single_volume(pcd)
    
    print("\n--- VOLUMETRIC ANALYSIS ---")
    print(f"Truck Ground (Floor) Mean Z: {np.nanmean(floor_grid):.2f} cm")
    print(f"Truck Wall (Top) Z:   {float(wall_top_z):.2f} cm")
    print(f"Detected Dimensions: Grid {grid_shape[0]}x{grid_shape[1]} cells")
    print(f"Calculated Empty Volume Space: {vol:.2f} cubic meters (m3)")
    print("---------------------------\n")

if __name__ == "__main__":
    test_file = r"d:\point cloud technology\zenodo dataset\20230426\14\2023_04_26_14_16_44_D_4.out"
    process_zenodo_out(test_file)
