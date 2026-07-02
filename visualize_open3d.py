import open3d as o3d
import sys

def main():
    filepath = "results/debug/load_08_icp_aligned.pcd"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
    print(f"Loading point cloud: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        print("Error: Point cloud is empty or file not found.")
        return
        
    print("Opening 3D viewer. Press 'q' to close the window.")
    o3d.visualization.draw_geometries([pcd], window_name="Truck Point Cloud Viewer")

if __name__ == "__main__":
    main()
