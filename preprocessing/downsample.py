import open3d as o3d
import config

def downsample_point_cloud(pcd):
    """
    Downsamples the point cloud using a voxel grid.
    """
    print(f"Applying Voxel Grid Downsampling (voxel_size={config.VOXEL_SIZE}mm)...")
    pcd_down = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE)
    print(f"Points remaining after downsampling: {len(pcd_down.points)}")
    return pcd_down
