import open3d as o3d
import numpy as np

def align_truck_pca(pcd):
    """
    Aligns the truck using Principal Component Analysis (PCA).
    Rotates the point cloud so its longest variance axis is aligned with X.
    """
    print("Aligning truck using PCA...")
    points = np.asarray(pcd.points)
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Compute covariance matrix and its eigenvectors
    cov = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by descending eigenvalues
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    
    # PCA rotation matrix
    # The principal axis (largest variance) will be the new X-axis
    R = eigenvectors.T
    
    # Check orientation to keep Z pointing UP
    # If the Z axis flipped, invert the 3rd row and maybe the 2nd to keep det=1
    if R[2, 2] < 0:
        R[2, :] = -R[2, :]
        R[1, :] = -R[1, :]
        
    pcd_aligned = pcd.rotate(R, center=centroid)
    print("PCA alignment complete.")
    return pcd_aligned, R, centroid
