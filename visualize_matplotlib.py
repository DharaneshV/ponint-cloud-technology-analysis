import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_projections(pcd_path, save_path="results/truck_projections.png"):
    print(f"Loading {pcd_path} for Matplotlib visualization...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    
    if len(pts) == 0:
        print("Point cloud is empty.")
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Side View (X vs Z)
    axes[0].scatter(pts[:, 0], pts[:, 2], s=1, c=pts[:, 2], cmap='viridis')
    axes[0].set_title("Side View (X-Z)")
    axes[0].set_xlabel("Length (X) mm")
    axes[0].set_ylabel("Height (Z) mm")
    axes[0].axis('equal')
    
    # Top View (X vs Y)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=1, c=pts[:, 2], cmap='viridis')
    axes[1].set_title("Top View (X-Y)")
    axes[1].set_xlabel("Length (X) mm")
    axes[1].set_ylabel("Width (Y) mm")
    axes[1].axis('equal')
    
    # Front View (Y vs Z)
    axes[2].scatter(pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap='viridis')
    axes[2].set_title("Front View (Y-Z)")
    axes[2].set_xlabel("Width (Y) mm")
    axes[2].set_ylabel("Height (Z) mm")
    axes[2].axis('equal')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Successfully saved projections image to {save_path}")

if __name__ == "__main__":
    generate_projections("results/debug/load_08_icp_aligned.pcd")
