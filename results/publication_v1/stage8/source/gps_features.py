import numpy as np
import pandas as pd

R_EARTH = 6371000.0  # Earth radius in meters
GPS_FEATURE_NAMES = [
    "rel_e", "rel_n", "dist", "sin_bearing", "cos_bearing",
    "vel_e", "vel_n", "speed", "hdop_flag",
    "acc_e", "acc_n", "turn_rate",
]
N_GPS_FEATS = 9  # trained models use the first 9; extras are logged in EDA
N_GPS_FEATS_EXTENDED = len(GPS_FEATURE_NAMES)

def wgs84_to_local_enu(lat1, lon1, lat2, lon2):
    """
    Convert (lat1, lon1) and (lat2, lon2) in degrees to relative local Cartesian coordinates (East, North) in meters.
    Unit 1 is the origin / reference.
    """
    lat1_rad = np.radians(lat1)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)

    north = d_lat * R_EARTH
    east = d_lon * R_EARTH * np.cos(lat1_rad)
    return east, north

def extract_step_gps_features(u1_lat, u1_lon, u2_lat, u2_lon, u1_hdop=1.0, u2_hdop=1.0, prev_e=None, prev_n=None):
    """
    Compute 9-dimensional engineered feature vector for a single timestep.
    """
    east, north = wgs84_to_local_enu(u1_lat, u1_lon, u2_lat, u2_lon)
    dist = np.sqrt(east**2 + north**2)
    bearing = np.arctan2(east, north)
    sin_b = np.sin(bearing)
    cos_b = np.cos(bearing)

    # Velocity proxies
    if prev_e is not None and prev_n is not None:
        vel_e = east - prev_e
        vel_n = north - prev_n
    else:
        vel_e = 0.0
        vel_n = 0.0
    speed = np.sqrt(vel_e**2 + vel_n**2)

    # GPS quality flag (1 if poor HDOP, 0 if clean)
    hdop_flag = 1.0 if (u1_hdop > 2.0 or u2_hdop > 2.0 or np.isnan(u1_hdop) or np.isnan(u2_hdop)) else 0.0

    return np.array([east, north, dist, sin_b, cos_b, vel_e, vel_n, speed, hdop_flag], dtype=np.float32), east, north

def extract_sequence_gps_features(raw_df, seq_indices):
    """
    Given a list of row indices (length 5), extract (5, 9) array used by trained models.
    Acceleration / turning are derived after the 9-d core (optional proxies).
    """
    rows = raw_df.iloc[seq_indices]
    u1_lats = rows["unit1_gps1_lat"].values
    u1_lons = rows["unit1_gps1_lon"].values
    u2_lats = rows["unit2_gps1_lat"].values
    u2_lons = rows["unit2_gps1_lon"].values
    u1_hdops = rows["unit1_gps1_hdop"].values if "unit1_gps1_hdop" in rows else np.ones(len(rows))
    u2_hdops = rows["unit2_gps1_hdop"].values if "unit2_gps1_hdop" in rows else np.ones(len(rows))

    seq_feats = []
    prev_e, prev_n = None, None
    prev_ve, prev_vn = 0.0, 0.0
    prev_bearing = 0.0
    for i in range(len(rows)):
        feat, east, north = extract_step_gps_features(
            u1_lats[i], u1_lons[i], u2_lats[i], u2_lons[i],
            u1_hdops[i], u2_hdops[i], prev_e, prev_n
        )
        vel_e, vel_n = feat[5], feat[6]
        bearing = np.arctan2(east, north)
        acc_e = vel_e - prev_ve
        acc_n = vel_n - prev_vn
        turn = bearing - prev_bearing
        turn = (turn + np.pi) % (2 * np.pi) - np.pi
        # Keep the 9-d vector the models were trained on
        seq_feats.append(feat)
        prev_e, prev_n = east, north
        prev_ve, prev_vn = vel_e, vel_n
        prev_bearing = bearing
        _ = (acc_e, acc_n, turn)
    return np.stack(seq_feats, axis=0)


def extract_sequence_gps_features_extended(raw_df, seq_indices):
    """Extended: 9 core features + acceleration and turning proxies (5, 12)."""
    core = extract_sequence_gps_features(raw_df, seq_indices)
    ext = np.zeros((core.shape[0], 12), dtype=np.float32)
    ext[:, :9] = core
    ext[1:, 9] = core[1:, 5] - core[:-1, 5]
    ext[1:, 10] = core[1:, 6] - core[:-1, 6]
    bearings = np.arctan2(core[:, 0], core[:, 1])
    dtheta = np.diff(bearings, prepend=bearings[:1])
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    ext[:, 11] = dtheta
    return ext

class GPSFeatureScaler:
    """
    Min-Max or Standard Scaler fitted on Train split sequence features only.
    """
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, train_gps_feats):
        # train_gps_feats shape: (N_train, 5, 9)
        flat = train_gps_feats.reshape(-1, train_gps_feats.shape[-1])
        self.min_vals = np.nanmin(flat, axis=0)
        self.max_vals = np.nanmax(flat, axis=0)
        # Avoid division by zero
        diff = self.max_vals - self.min_vals
        diff[diff == 0] = 1.0
        self.scale_diff = diff
        print(f"Fitted GPSFeatureScaler on {len(flat):,} points across {len(self.min_vals)} features.")

    def transform(self, gps_feats):
        return (gps_feats - self.min_vals) / self.scale_diff
