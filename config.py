import warnings

# --- Mapping Configuration ---
# 'encoder' assumes each scan index represents a fixed physical distance.
# 'timed' assumes each scan index represents a fixed time interval.
MAPPING_MODE = "timed" 

# Reference length of the truck in centimeters (including cabin/frame)
# A standard semi-trailer + tractor has a total length of approx 16.75 meters (1675 cm).
# This length scales the X-axis so that the truck spans exactly this dimension.
TRUCK_REFERENCE_LENGTH_CM = 650.0

# Speed is used only in 'timed' mode. Speed in cm/s (e.g. 100 cm/s = 1 m/s).
DEFAULT_SPEED_CM_S = 100.0

# In SICK SL1, we observed a 10 ms interval (100 Hz).
SCAN_INTERVAL_MS = 10.0
SCAN_INTERVAL_S = SCAN_INTERVAL_MS / 1000.0

# --- Preprocessing Configuration ---
# Values are in centimeters (cm)
VOXEL_SIZE = 5.0 # 5 cm voxel size

# SOR (Statistical Outlier Removal)
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Gantry Filter (Y-axis bounds in cm)
GANTRY_Y_MIN = -205.0
GANTRY_Y_MAX = 205.0

# Ground Plane Removal Threshold
# In the SICK gantry rig, Z represents distance downwards from the scanner.
# The gantry floor is at Z ~ 587 cm in SL1 files.
ROAD_Z_THRESHOLD = 500.0

# --- Segmentation Configuration ---
# DBSCAN Clustering
DBSCAN_EPS = 50.0 # cm
DBSCAN_MIN_POINTS = 20

# Truck Envelope (cm)
TRUCK_ENVELOPE_MIN_LENGTH = 500.0   # 5 meters
TRUCK_ENVELOPE_MAX_LENGTH = 2000.0  # 20 meters
TRUCK_ENVELOPE_MIN_WIDTH = 100.0    # 1.0 meter
TRUCK_ENVELOPE_MAX_WIDTH = 300.0    # 3.0 meters
TRUCK_ENVELOPE_MIN_HEIGHT = 50.0    # 0.5 meters
TRUCK_ENVELOPE_MAX_HEIGHT = 300.0   # 3.0 meters

# --- Analysis Configuration ---
HEIGHTMAP_GRID_RESOLUTION = 5.0 # 5 cm grid resolution

# --- Scanner Constants ---
SKY_CODE_THRESHOLD = 50000.0 # Raw Z values above this are sky/error misses


