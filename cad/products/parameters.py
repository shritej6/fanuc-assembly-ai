"""Shared parameters for the printed assembly models, in millimetres.
Authored by Shritej Gunda.
"""

# --- FDM process ---------------------------------------------------------
NOZZLE_DIAMETER = 0.40
LAYER_HEIGHT = 0.20

WALL = 2.40
MIN_WALL = 1.60
MIN_FLOOR = 2.40

# --- Fits ----------------------------------------------------------------
ROBOT_CLEARANCE_PER_SIDE = 0.50
LOCATING_CLEARANCE_PER_SIDE = 0.50
ENTRY_CHAMFER = 1.50
EDGE_FILLET = 1.00
GRIP_EDGE_RADIUS = 0.40

# --- Standard grip pad ---------------------------------------------------
GRIP_WIDTH = 7.00
GRIP_LENGTH = 10.00
GRIP_HEIGHT = 8.00
GRIP_ROOT_FILLET = 1.50
GRIP_TOP_CHAMFER = 0.50

# --- Fixture -------------------------------------------------------------
FIXTURE_X = 60.00
FIXTURE_Y = 50.00
FIXTURE_Z = 8.00
FIXTURE_CORNER_R = 3.00
FIXTURE_HOLE_D = 4.50
FIXTURE_HOLE_X = 25.00
FIXTURE_HOLE_Y = 20.00

FRONT_STOP_WALL = 3.00
FRONT_STOP_LEG = 8.00
STOP_HEIGHT = 3.00
SIDE_STOP_X = 3.00
SIDE_STOP_Y = 6.00
SIDE_STOP_CENTER_Y = 10.00
SIDE_STOP_GAP = 2.00  # side-stop centre sits housing X/2 + this

PEG_A_D = 4.00
PEG_B_D = 3.00
PEG_HEIGHT = 2.50
PEG_TOP_CHAMFER = 0.50
SOCKET_A_D = 5.00
SOCKET_B_D = 4.00
SOCKET_DEPTH = 2.80
SOCKET_MOUTH_CHAMFER = 0.50
PEG_A_X = -12.00
PEG_A_Y = -10.00
PEG_B_X = 11.00
PEG_B_Y = 9.00

# --- Export --------------------------------------------------------------
STL_LINEAR_TOLERANCE = 0.08
STL_ANGULAR_TOLERANCE = 0.12

# --- Control family (Models 1-3) -----------------------------------------
CTRL_HOUSING_Z = 16.0
CTRL_POCKET_FLOOR_Z = 7.0
CTRL_BOSS_Y0 = 11.0
CTRL_BOSS_Y1 = 24.0
CTRL_BOSS_Z = 8.0
GUARD_SOCKET_X = 6.0
GUARD_SOCKET_Y = 9.0
GUARD_SOCKET_BOTTOM_Z = 4.0
GUARD_CENTER_Y = 17.5
GUARD_ARM_X = 5.0
GUARD_ARM_Y = 8.0
GUARD_ARM_Z = 26.0
GUARD_BRIDGE_Z = 4.0
GUARD_PAD_Y = 8.0
GUARD_FOOT_CHAMFER = 1.0
HOUSING_PAD_CENTER_Y = -20.0
HOUSING_PAD_Z0 = 8.0

CARRIER_FLANGE_Z0 = 16.0
CARRIER_FLANGE_H = 2.0
CARRIER_BOTTOM_CHAMFER = 1.0
KEY_POST_X = 7.0
KEY_POST_Y = 10.0
# Cap/knob sockets are post height + this headroom,
# so every cap seats on its carrier flange instead of riding on the post top.
SOCKET_HEADROOM = 0.5
SOCKET_ROOF = 1.5
SOCKET_ENTRY_CHAMFER = 1.0

# --- Connector family (Models 4-6) ---------------------------------------
CONN_HOUSING_X = 38.0
CONN_HOUSING_Y = 30.0
CONN_HOUSING_Z = 18.0
CONN_FLOOR_Z = 5.0
CONN_RIB_PROJECTION = 1.0
CONN_RIB_WIDTH = 1.6
CONN_PAD_CENTER_X = 23.0
CONN_PAD_Z0 = 10.0
INSERT_Z0 = 5.0
INSERT_H = 12.0
INSERT_BOTTOM_CHAMFER = 1.5
POCKET_DEPTH = 4.0
POCKET_ROOF = 1.0
MARKER_H = 0.8

GUIDE_X = 36.0
GUIDE_Y = 34.0
GUIDE_Z0 = 18.0
GUIDE_T = 5.0
GUIDE_OPEN_X = 19.0
GUIDE_OPEN_Y1 = 7.0
GUIDE_HOLE = 4.0
GUIDE_HOLE_X = 13.0
GUIDE_HOLE_Y = 12.0
GUIDE_PEG = 3.0
GUIDE_PEG_H = 2.0
GUIDE_PAD_CENTER_Y = 18.0
LOCK_HOLE = 4.0
LOCK_X = 11.5
LOCK_Y = -11.5
LOCK_LEG = 3.0
LOCK_BRIDGE_X = 26.0
LOCK_BRIDGE_Y = 6.0
LOCK_BRIDGE_Z = 4.0
LOCK_BRIDGE_CHAMFER = 1.5
LOCK_PAD_TAPER_H = 2.0
LOCK_GUSSET_RUN = 5.0
SMALL_LEAD_IN = 0.5  # lead-in on 3-4 mm pegs/holes where a 1.5 mm chamfer would not fit

# --- Material (mass estimate only) ---------------------------------------
PLA_DENSITY_G_PER_CM3 = 1.24
MAX_PART_MASS_G = 60.0
