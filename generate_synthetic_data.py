import numpy as np
import open3d as o3d
import os

def generate_synthetic_trucks(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Truck Dimensions (cm)
    L, W, H = 650.0, 250.0, 160.0
    wall_z = 250.0 # Z height of the top of the walls (scanner is above, at Z=0)
    floor_z = wall_z + H # Z height of the floor (410.0)
    
    # Grid resolution for generation
    res = 2.0 
    x_range = np.arange(0, L, res)
    y_range = np.arange(0, W, res)
    
    empty_pts = []
    flat_pts = []
    heaped_pts = []
    
    for x in x_range:
        for y in y_range:
            # Generate the floor and loads as before
            empty_pts.append([x, y, floor_z])
            flat_pts.append([x, y, wall_z])
            
            cx, cy = L/2, W/2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_dist = np.sqrt((L/2)**2 + (W/2)**2)
            heap_z = 200.0 + (dist / max_dist) * (floor_z - 200.0)
            heap_z = min(heap_z, floor_z)
            heaped_pts.append([x, y, heap_z])

            # Generate the WALLS (perimeter) so 5th percentile Z finds them!
            is_wall = (x < 10) or (x > L - 10) or (y < 10) or (y > W - 10)
            if is_wall:
                # Add points along the vertical wall
                for wz in np.arange(wall_z, floor_z, res * 2):
                    empty_pts.append([x, y, wz])
                    flat_pts.append([x, y, wz])
                    heaped_pts.append([x, y, wz])

            
    # Convert to Open3D point clouds and save
    def save_pcd(pts, filename):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(pts))
        o3d.io.write_point_cloud(os.path.join(out_dir, filename), pcd)
        print(f"Saved {filename}")

    save_pcd(empty_pts, "synthetic_empty.pcd")
    save_pcd(flat_pts, "synthetic_flat.pcd")
    save_pcd(heaped_pts, "synthetic_heaped.pcd")

if __name__ == "__main__":
    generate_synthetic_trucks(r"d:\point cloud technology\Truck-PointCloud\synthetic_data")
