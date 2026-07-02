import open3d as o3d
import numpy as np

def register_icp(source, target, threshold=50.0):
    """
    Registers source point cloud to target using ICP.
    Typically source is the loaded truck, target is the empty truck.
    """
    print(f"Applying ICP Registration (threshold={threshold}mm)...")
    
    # Compute normals if missing
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
        
    # Initial alignment is assumed to be identity (since they are already roughly aligned by PCA or just scanner coords)
    trans_init = np.eye(4)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    print(f"ICP complete. Fitness: {reg_p2p.fitness:.4f}, Inlier RMSE: {reg_p2p.inlier_rmse:.4f}")
    source_aligned = source.transform(reg_p2p.transformation)
    return source_aligned, reg_p2p.transformation
