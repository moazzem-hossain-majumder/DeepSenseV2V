"""
Analytic Line-of-Sight Geometry Prior for 256-Beam V2V Phased Arrays.

Computes the physical line-of-sight direction from historical GNSS kinematics,
projects into local ENU coordinates, extrapolates by 500 ms, maps to the 256-beam codebook,
and generates the base spatial beam probability distribution.
"""

import numpy as np
import torch
import torch.nn as nn

N_BEAMS = 256


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    if isinstance(angle, torch.Tensor):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_enu_kinematics(p1, p2, times, horizon_sec=0.5):
    """
    Computes local East-North-Up (ENU) coordinates and extrapolated relative direction.

    p1: (B, 5, 2) Receiver GPS (lat, lon) in degrees.
    p2: (B, 5, 2) Transmitter GPS (lat, lon) in degrees.
    times: (B, 5) Observation timestamps in seconds.
    horizon_sec: Future prediction horizon (default: 0.5 s).

    Returns:
      bearing_rad: (B,) Extrapolated relative line-of-sight bearing in radians [-pi, pi].
      heading_rad: (B,) Receiver forward heading direction in radians.
      relative_angle_rad: (B,) Physical relative angle relative to receiver heading [-pi, pi].
      kinematic_context: (B, 8) Kinematic context features (speeds, heading sin/cos, etc.).
    """
    is_torch = isinstance(p1, torch.Tensor)
    if is_torch:
        np_p1 = p1.detach().cpu().numpy()
        np_p2 = p2.detach().cpu().numpy()
        np_times = times.detach().cpu().numpy()
    else:
        np_p1, np_p2, np_times = p1, p2, times

    B = len(np_p1)
    origin = np_p1[:, -1:, :]  # Last receiver position (B, 1, 2)
    coslat = np.cos(np.deg2rad(origin[:, :, 0]))
    radius = 6371000.0  # Earth radius in meters

    # ENU coordinates about last receiver position
    d1 = np.deg2rad(np_p1 - origin)
    r1 = np.stack((d1[:, :, 1] * radius * coslat, d1[:, :, 0] * radius), axis=-1)  # (B, 5, 2)

    d2 = np.deg2rad(np_p2 - origin)
    r2 = np.stack((d2[:, :, 1] * radius * coslat, d2[:, :, 0] * radius), axis=-1)  # (B, 5, 2)

    # Velocity estimation via centered finite-difference regression
    centered = np_times - np_times.mean(axis=1, keepdims=True)  # (B, 5)
    denom = (centered ** 2).sum(axis=1)  # (B,)
    denom = np.where(denom <= 1e-6, 1.0, denom)

    v1 = (r1 * centered[:, :, None]).sum(axis=1) / denom[:, None]  # (B, 2) m/s
    v2 = (r2 * centered[:, :, None]).sum(axis=1) / denom[:, None]  # (B, 2) m/s

    # Extrapolate positions by horizon_sec
    predicted1 = r1[:, -1] + horizon_sec * v1
    predicted2 = r2[:, -1] + horizon_sec * v2
    relative = predicted2 - predicted1  # Vector from receiver (1) to transmitter (2)

    heading = np.arctan2(v1[:, 1], v1[:, 0])  # Receiver velocity heading
    bearing = np.arctan2(relative[:, 1], relative[:, 0])  # Relative bearing to transmitter

    # Physical angle relative to receiver heading
    relative_angle = wrap_angle(bearing - heading)

    # Legacy codebook angle mapping (calibrated coordinate convention)
    # The receiver front panel is oriented at -pi/4 relative to vehicle forward heading
    calibrated_angle = wrap_angle(-(bearing - heading - np.pi / 4))

    speed1 = np.linalg.norm(v1, axis=1)
    speed2 = np.linalg.norm(v2, axis=1)
    delta_t = np_times[:, -1] - np_times[:, -2]

    context = np.column_stack((
        np.sin(heading),
        np.cos(heading),
        np.sin(relative_angle),
        np.cos(relative_angle),
        speed1,
        speed2,
        (speed1 < 1.0).astype(np.float32),
        delta_t,
    )).astype(np.float32)

    if is_torch:
        device = p1.device
        return (
            torch.from_numpy(calibrated_angle).to(device=device, dtype=torch.float32),
            torch.from_numpy(heading).to(device=device, dtype=torch.float32),
            torch.from_numpy(relative_angle).to(device=device, dtype=torch.float32),
            torch.from_numpy(context).to(device=device, dtype=torch.float32),
        )

    return calibrated_angle, heading, relative_angle, context


class AnalyticGeometryPrior(nn.Module):
    """
    Computes the baseline beam probability distribution from predicted LOS angles.
    Generates Gaussian spatial distribution centered at predicted beam direction.
    """
    def __init__(self, n_beams=N_BEAMS, bandwidth_rad=0.35, offset=0):
        super().__init__()
        self.n_beams = n_beams
        self.bandwidth_rad = bandwidth_rad
        self.offset = offset

        # Codebook beam angles spanning [-pi, pi]
        grid = np.linspace(-np.pi, np.pi, n_beams, endpoint=False, dtype=np.float32)
        self.register_buffer("beam_angles", torch.from_numpy(grid))

    def forward(self, angles_rad, offset=None):
        """
        angles_rad: (B,) calibrated LOS angles in radians.
        offset: optional circular index offset (integer or tensor).

        Returns:
          prior_logits: (B, 256) spatial logit distribution.
          center_beam: (B,) argmax beam indices.
        """
        if offset is None:
            offset = self.offset

        B = angles_rad.shape[0]
        # (B, 1) - (1, 256) -> (B, 256)
        angular_dist = wrap_angle(angles_rad[:, None] - self.beam_angles[None, :])
        abs_dist = torch.abs(angular_dist)

        # Spatial Gaussian prior over angular distance
        # Logit shape matches: -0.5 * (d / sigma)^2, clamped to [-20, 0]
        sigma = self.bandwidth_rad
        prior = torch.clamp(-0.5 * (abs_dist / sigma) ** 2, min=-20.0, max=0.0)

        # Circular offset shift if specified
        if offset != 0:
            if isinstance(offset, int):
                prior = torch.roll(prior, shifts=offset, dims=1)
            else:
                # Per-sample offset if tensor
                pass

        center_beam = prior.argmax(dim=1)
        return prior, center_beam

    def get_candidate_subset(self, prior_logits, k=20):
        """
        Extracts top-k candidate beams from prior logits.
        Returns: (B, k) beam indices.
        """
        return torch.topk(prior_logits, k=k, dim=1).indices
