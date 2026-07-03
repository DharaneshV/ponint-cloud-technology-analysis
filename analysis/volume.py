import numpy as np
import open3d as o3d
import config

def compute_single_volume(pcd):
    """
    Computes the internal volume of a single scan (from wall tops down to the visible surface).
    For an empty truck, this is the total bed capacity.
    For a loaded truck, this is the remaining empty space above the cargo.
    """
    pts = np.asarray(pcd.points)
    
    # Grid setup
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    res = config.HEIGHTMAP_GRID_RESOLUTION
    
    x_bins = np.arange(x_min, x_max + res, res)
    y_bins = np.arange(y_min, y_max + res, res)
    grid_shape = (len(x_bins), len(y_bins))
    
    import pandas as pd
    x_idx = np.digitize(pts[:, 0], x_bins) - 1
    y_idx = np.digitize(pts[:, 1], y_bins) - 1
    
    df = pd.DataFrame({'x': x_idx, 'y': y_idx, 'z': pts[:, 2]})
    z_max = df.groupby(['x', 'y'])['z'].max()
    
    floor_grid = np.full(grid_shape, np.nan)
    for (xi, yi), zval in z_max.items():
        if 0 <= xi < grid_shape[0] and 0 <= yi < grid_shape[1]:
            floor_grid[xi, yi] = zval
            
    wall_top_z = np.percentile(pts[:, 2], 5)
    
    valid_cells = ~np.isnan(floor_grid)
    depth_grid = np.zeros(grid_shape)
    depth_grid[valid_cells] = floor_grid[valid_cells] - wall_top_z
    
    MIN_DEPTH_CM = 10.0
    interior_mask = depth_grid > MIN_DEPTH_CM
    
    cell_area = res * res
    volume_cm3 = np.sum(depth_grid[interior_mask] * cell_area)
    volume_m3  = volume_cm3 / 1e6
    
    return volume_m3, wall_top_z, floor_grid, grid_shape, x_bins, y_bins


def compute_volume(empty_pcd, load_pcd):
    """
    Computes:
      1. Bed Capacity   – total empty-truck bed volume (wall-top to floor)
      2. Cargo Volume    – volume occupied by the load
      3. Remaining Capacity – bed capacity minus cargo volume
    """
    print(f"Computing volume (grid resolution = {config.HEIGHTMAP_GRID_RESOLUTION} cm)...")

    # 1. Get empty capacity and grid
    bed_capacity_m3, wall_top_z, empty_floor_grid, grid_shape, x_bins, y_bins = compute_single_volume(empty_pcd)
    print(f"  Wall-top reference Z (5th pct): {wall_top_z:.2f} cm")

    # 2. Process load scan
    load_pts  = np.asarray(load_pcd.points)
    import pandas as pd
    
    load_x_idx  = np.digitize(load_pts[:, 0], x_bins) - 1
    load_y_idx  = np.digitize(load_pts[:, 1], y_bins) - 1
    
    # Filter bounds to match empty grid size safely
    valid_x = (load_x_idx >= 0) & (load_x_idx < grid_shape[0])
    valid_y = (load_y_idx >= 0) & (load_y_idx < grid_shape[1])
    valid_idx = valid_x & valid_y
    
    df_load  = pd.DataFrame({
        'x': load_x_idx[valid_idx],  
        'y': load_y_idx[valid_idx],  
        'z': load_pts[valid_idx, 2]
    })
    
    l_max = df_load.groupby(['x', 'y'])['z'].max()
    
    load_floor_grid  = np.full(grid_shape, np.nan)
    for (xi, yi), zval in l_max.items():
        load_floor_grid[xi, yi] = zval

    # 3. Cargo volume (height-map differencing: empty floor – load floor)
    valid_empty = ~np.isnan(empty_floor_grid)
    valid_both = valid_empty & ~np.isnan(load_floor_grid)
    
    cargo_height = np.zeros(grid_shape)
    cargo_height[valid_both] = empty_floor_grid[valid_both] - load_floor_grid[valid_both]

    # Ignore small or negative differences (noise)
    cargo_height[cargo_height < 5.0] = 0.0

    res = config.HEIGHTMAP_GRID_RESOLUTION
    cell_area = res * res  # cm²
    cargo_volume_cm3 = np.sum(cargo_height * cell_area)
    cargo_volume_m3  = cargo_volume_cm3 / 1e6

    # 4. Remaining capacity
    remaining_m3 = bed_capacity_m3 - cargo_volume_m3
    if remaining_m3 < 0:
        remaining_m3 = 0.0

    # 5. Summary
    print(f"\n  +------------------------------------------+")
    print(f"  |  Bed Capacity (empty) : {bed_capacity_m3:8.4f} m3   |")
    print(f"  |  Cargo Volume (load)  : {cargo_volume_m3:8.4f} m3   |")
    print(f"  |  Remaining Capacity   : {remaining_m3:8.4f} m3   |")
    print(f"  |  Fill Percentage      : {(cargo_volume_m3 / bed_capacity_m3 * 100) if bed_capacity_m3 > 0 else 0:7.2f} %    |")
    print(f"  +------------------------------------------+")

    volume_report = {
        "bed_capacity_m3":     round(bed_capacity_m3, 4),
        "cargo_volume_m3":     round(cargo_volume_m3, 4),
        "remaining_capacity_m3": round(remaining_m3, 4),
        "fill_percentage":     round((cargo_volume_m3 / bed_capacity_m3 * 100) if bed_capacity_m3 > 0 else 0, 2),
        "wall_top_ref_z_cm":   round(wall_top_z, 2),
        "grid_resolution_cm":  res
    }

    return volume_report, empty_floor_grid, load_floor_grid

