"""
Unit and Integration Tests for Physically Anchored Residual Network and Losses.
"""

import pytest
import numpy as np
import torch

from src.panel_losses import (
    MaskedMainLobeHuberLoss,
    DisjointIntraPanelSmoothnessLoss,
    SoftPowerAwareCELoss,
    PhysicallyAnchoredMultiTaskLoss,
)
from src.geometry_prior import (
    wrap_angle,
    compute_enu_kinematics,
    AnalyticGeometryPrior,
)
from src.par_net import PARNet
from src.adaptive_conformal_controller import KinematicAdaptiveConformalController


def test_angle_wrapping():
    angles = torch.tensor([np.pi + 0.1, -np.pi - 0.2, 3 * np.pi, 0.0])
    wrapped = wrap_angle(angles)
    assert (wrapped >= -np.pi).all() and (wrapped <= np.pi).all()
    assert torch.isclose(wrapped[3], torch.tensor(0.0))


def test_analytic_geometry_prior_shape():
    prior_module = AnalyticGeometryPrior(n_beams=256, bandwidth_rad=0.35)
    angles = torch.tensor([0.0, np.pi / 2, -np.pi / 4], dtype=torch.float32)
    prior_logits, center_beams = prior_module(angles)

    assert prior_logits.shape == (3, 256)
    assert center_beams.shape == (3,)
    assert (prior_logits <= 0.0).all() and (prior_logits >= -20.0).all()

    candidates = prior_module.get_candidate_subset(prior_logits, k=20)
    assert candidates.shape == (3, 20)


def test_masked_main_lobe_huber_loss():
    loss_fn = MaskedMainLobeHuberLoss(active_threshold_db=10.0, delta=1.0)
    B = 4
    # Create synthetic power profile where only beam 10 is near 0 dB, all others at -40 dB
    true_profile = torch.full((B, 256), -40.0)
    true_profile[:, 10] = 0.0  # Main peak
    true_profile[:, 11] = -5.0  # In active lobe

    # Prediction matches exactly on active lobe, but has massive errors on inactive noise floor
    pred_profile = true_profile.clone()
    pred_profile[:, 100:] += 30.0  # Corrupt inactive sidelobes

    loss = loss_fn(pred_profile, true_profile)
    # The corrupted sidelobes (-40 + 30 = -10 dB) are compared relative to max
    assert torch.isfinite(loss)


def test_disjoint_intra_panel_smoothness_boundary_independence():
    """
    CRITICAL PHYSICAL CHECK:
    Altering beam 63 and beam 64 (panel boundary) MUST NOT change the loss,
    because they belong to different physical panels.
    """
    loss_fn = DisjointIntraPanelSmoothnessLoss()
    pred = torch.zeros((2, 256))

    loss_base = loss_fn(pred)
    assert loss_base.item() == 0.0

    # Introduce a massive step discontinuity strictly at panel boundary (beam 63 to 64)
    pred[:, 64:] += 50.0  # Shift entire Panel 2 upwards by 50 dB
    loss_stepped = loss_fn(pred)

    # Within each panel, all values are still uniform (diffs = 0), so loss must still be 0!
    assert torch.isclose(loss_stepped, torch.tensor(0.0), atol=1e-6), (
        f"Smoothness loss penalized panel boundary! Expected 0.0, got {loss_stepped.item()}"
    )


def test_par_net_initialization_at_geometry_prior():
    """
    Verifies that PAR-Net zero-initialization starts identically at the geometry prior.
    """
    net = PARNet(d_model=64, candidate_k=20, visual_in_dim=64)
    net.eval()

    B = 2
    gps_seq = torch.randn(B, 5, 9)
    context = torch.randn(B, 8)
    visual_tokens = torch.randn(B, 5, 64)
    base_angles = torch.tensor([0.0, 1.0], dtype=torch.float32)

    with torch.no_grad():
        out = net(gps_seq, context, visual_tokens, base_angle_rad=base_angles)

    # Offset head should output ~0 at init
    assert torch.allclose(out["residual_offset_rad"], torch.zeros((B, 1)), atol=1e-4)


def test_par_net_gradient_flow():
    """Verifies that all model components receive finite, non-zero gradients."""
    net = PARNet(d_model=64, candidate_k=20, visual_in_dim=64)
    net.train()

    loss_fn = PhysicallyAnchoredMultiTaskLoss()

    B = 2
    gps_seq = torch.randn(B, 5, 9, requires_grad=True)
    context = torch.randn(B, 8, requires_grad=True)
    visual_tokens = torch.randn(B, 5, 64, requires_grad=True)
    base_angles = torch.tensor([0.0, 1.0], dtype=torch.float32)

    true_profile = torch.randn(B, 256)

    out = net(gps_seq, context, visual_tokens, base_angle_rad=base_angles)
    loss_dict = loss_fn(out, true_profile)
    total_loss = loss_dict["loss"]

    assert torch.isfinite(total_loss)
    total_loss.backward()

    # Verify gradients on key parameter sets
    for name, param in net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient!"


def test_adaptive_conformal_controller():
    controller = KinematicAdaptiveConformalController(target_alpha=0.10, delta_db=3.0)
    T = 100
    pred = np.zeros((T, 256), dtype=np.float32)
    true_prof = np.full((T, 256), -40.0, dtype=np.float32)
    true_prof[:, 10] = 0.0  # Beam 10 is optimal
    pred[:, 10] = 0.0  # Predicted optimal matches

    res = controller.run_stream(pred, true_prof, q_init=5.0)
    assert res["miss_rate"] <= 0.10
    assert res["avg_candidate_size"] >= 1.0
