"""Feature names and dimensional conventions used by ValGraphNet."""

NODE_FEATURE_NAMES = [
    "x0",
    "y0",
    "z0",
    "u",
    "v",
    "w",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "fixed_mask",
    "pressure_mask",
    "leaflet_id",
    "normal_x",
    "normal_y",
    "normal_z",
    "nodal_area",
    "thickness",
    "pressure_k",
    "pressure_next",
    "pressure_rate",
    "phase_sin",
    "phase_cos",
    "traction_x",
    "traction_y",
    "traction_z",
]

EDGE_FEATURE_NAMES = [
    "ref_dx",
    "ref_dy",
    "ref_dz",
    "ref_len",
    "cur_dx",
    "cur_dy",
    "cur_dz",
    "cur_len",
    "normal_dot",
    "pressure_pair_mean",
    "fixed_pair",
    "gap",
    "same_leaflet",
    "is_world_edge",
    "reserved_0",
    "reserved_1",
]

BASE_OUTPUT_NAMES = [
    "delta_u_x",
    "delta_u_y",
    "delta_u_z",
    "delta_v_x",
    "delta_v_y",
    "delta_v_z",
    "accel_x",
    "accel_y",
    "accel_z",
]

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
BASE_OUTPUT_DIM = len(BASE_OUTPUT_NAMES)

