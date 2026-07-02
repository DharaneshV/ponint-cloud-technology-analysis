import open3d as o3d
import numpy as np
import pandas as pd

# Load empty and load PCDs that we saved during recalculate
empty_pcd = o3d.io.read_point_cloud("results/debug/empty_07_pca_aligned.pcd")
load_pcd = o3d.io.read_point_cloud("results/debug/load_08_icp_aligned.pcd")

empty_pts = np.asarray(empty_pcd.points)
load_pts = np.asarray(load_pcd.points)

print("Empty Z min/max:", empty_pts[:, 2].min(), empty_pts[:, 2].max(), "mean:", empty_pts[:, 2].mean())
print("Load Z min/max:", load_pts[:, 2].min(), load_pts[:, 2].max(), "mean:", load_pts[:, 2].mean())

print("Empty X min/max:", empty_pts[:, 0].min(), empty_pts[:, 0].max())
print("Empty Y min/max:", empty_pts[:, 1].min(), empty_pts[:, 1].max())
print("Empty Z min/max:", empty_pts[:, 2].min(), empty_pts[:, 2].max())

print("\nLoad X min/max:", load_pts[:, 0].min(), load_pts[:, 0].max())
print("Load Y min/max:", load_pts[:, 1].min(), load_pts[:, 1].max())
print("Load Z min/max:", load_pts[:, 2].min(), load_pts[:, 2].max())

print("\nEmpty first 5 points:")
print(empty_pts[:5])

