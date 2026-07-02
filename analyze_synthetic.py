import open3d as o3d
from analysis.volume import compute_single_volume

def analyze(pcd_path, name):
    pcd = o3d.io.read_point_cloud(pcd_path)
    vol, _, _, _, _, _ = compute_single_volume(pcd)
    print(f"{name} -> Scanned Internal Volume: {vol:8.2f} m3")

if __name__ == "__main__":
    analyze(r"d:\point cloud technology\Truck-PointCloud\synthetic_data\synthetic_empty.pcd", "1. EMPTY TRUCK")
    analyze(r"d:\point cloud technology\Truck-PointCloud\synthetic_data\synthetic_flat.pcd", "2. PERFECTLY FLAT LOAD")
    analyze(r"d:\point cloud technology\Truck-PointCloud\synthetic_data\synthetic_heaped.pcd", "3. HEAPED/SLOPED LOAD")
