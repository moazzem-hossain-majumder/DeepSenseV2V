"""
Physically Informed and Hardware-Aware Loss Functions for mmWave Beam Forecasting.

Key features:
1. Masked Main-Lobe Huber Loss:
   Focuses gradient capacity strictly on beams within Delta_active dB (default: 10 dB)
   of the maximum measured power, preventing noise-floor gradient dilution across inactive beams.

2. Disjoint Intra-Panel Smoothness Regularization:
   Enforces spatial angular continuity STRICTLY WITHIN each physical 64-beam panel:
     Panel 1: Beams 0..63
     Panel 2: Beams 64..127
     Panel 3: Beams 128..191
     Panel 4: Beams 192..255
   NEVER penalizes transitions across physical panel boundaries (63->64, 127->128, 191->192).

3. Soft-Margin Power-Aware Cross-Entropy:
   Rewards near-optimal alternative beams proportional to their measured dB power gap,
   avoiding the single-label over-sharpening pathology that degrades 3 dB coverage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BEAMS = 256
BEAMS_PER_PANEL = 64
N_PANELS = 4


class MaskedMainLobeHuberLoss(nn.Module):
    """
    Computes Huber loss between predicted and ground-truth power profiles
    ONLY on beams within active_threshold_db of the optimal beam.
    """
    def __init__(self, active_threshold_db=10.0, delta=1.0):
        super().__init__()
        self.active_threshold_db = active_threshold_db
        self.delta = delta

    def forward(self, pred_profile, true_profile):
        """
        pred_profile: (B, 256) predicted relative power profile in dB.
        true_profile: (B, 256) measured true power profile in dB (or relative dB <= 0).
        """
        max_true = true_profile.max(dim=1, keepdim=True).values
        relative_true = true_profile - max_true  # (B, 256), range [-inf, 0]

        # Active mask: beams within active_threshold_db of the maximum
        mask = relative_true >= -self.active_threshold_db  # (B, 256) bool

        # Normalise pred_profile to relative scale if not already
        max_pred = pred_profile.max(dim=1, keepdim=True).values
        relative_pred = pred_profile - max_pred

        diff = relative_pred - relative_true

        # Huber loss
        abs_diff = torch.abs(diff)
        huber = torch.where(
            abs_diff <= self.delta,
            0.5 * (diff ** 2),
            self.delta * (abs_diff - 0.5 * self.delta)
        )

        masked_huber = huber * mask.float()
        active_counts = mask.float().sum(dim=1).clamp(min=1.0)
        sample_losses = masked_huber.sum(dim=1) / active_counts
        return sample_losses.mean()


class DisjointIntraPanelSmoothnessLoss(nn.Module):
    """
    Penalizes roughness strictly WITHIN each physical 64-beam array panel.
    Never crosses panel boundaries (e.g., beam 63 to 64).
    """
    def __init__(self, n_panels=N_PANELS, beams_per_panel=BEAMS_PER_PANEL):
        super().__init__()
        self.n_panels = n_panels
        self.beams_per_panel = beams_per_panel

    def forward(self, pred_profile):
        """
        pred_profile: (B, 256)
        """
        B, N = pred_profile.shape
        assert N == self.n_panels * self.beams_per_panel, f"Expected {self.n_panels * self.beams_per_panel} beams, got {N}"

        # Reshape to (B, n_panels, beams_per_panel)
        panels = pred_profile.view(B, self.n_panels, self.beams_per_panel)

        # Compute differences along the beam dimension within each panel
        diffs = panels[:, :, 1:] - panels[:, :, :-1]  # (B, n_panels, 63)

        # Mean squared difference across all intra-panel adjacent pairs
        panel_smoothness = (diffs ** 2).mean()
        return panel_smoothness


class SoftPowerAwareCELoss(nn.Module):
    """
    Soft-label cross entropy where target distribution is smoothed by measured power gap.
    p_soft(b) ~ exp(-gap(b) / temperature_db)
    """
    def __init__(self, temperature_db=3.0, label_smoothing=0.0):
        super().__init__()
        self.temperature_db = temperature_db
        self.label_smoothing = label_smoothing

    def forward(self, logits, true_profile):
        """
        logits: (B, 256) predicted classification logits.
        true_profile: (B, 256) true measured powers in dB.
        """
        max_true = true_profile.max(dim=1, keepdim=True).values
        gaps = (max_true - true_profile).clamp(min=0.0, max=40.0)

        # Soft probability target
        soft_targets = F.softmax(-gaps / self.temperature_db, dim=1)

        if self.label_smoothing > 0.0:
            uniform = torch.full_like(soft_targets, 1.0 / N_BEAMS)
            soft_targets = (1.0 - self.label_smoothing) * soft_targets + self.label_smoothing * uniform

        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return ce_loss


class PhysicallyAnchoredMultiTaskLoss(nn.Module):
    """
    Combined loss for Physically Anchored Residual Network.
    """
    def __init__(
        self,
        lambda_ce=1.0,
        lambda_main_lobe=0.25,
        lambda_panel_smooth=0.02,
        lambda_residual_reg=0.05,
        temperature_db=3.0,
        active_threshold_db=10.0,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_main_lobe = lambda_main_lobe
        self.lambda_panel_smooth = lambda_panel_smooth
        self.lambda_residual_reg = lambda_residual_reg

        self.ce_loss = SoftPowerAwareCELoss(temperature_db=temperature_db)
        self.main_lobe_loss = MaskedMainLobeHuberLoss(active_threshold_db=active_threshold_db)
        self.panel_smooth_loss = DisjointIntraPanelSmoothnessLoss()

    def forward(self, outputs, true_profile, true_labels=None):
        """
        outputs: dict containing:
          - 'logits': (B, 256)
          - 'pred_profile': (B, 256)
          - 'residual_offset_rad': (B, 1) or None
        true_profile: (B, 256) true powers in dB
        true_labels: (B,) optimal beam index (optional)
        """
        logits = outputs["logits"]
        pred_profile = outputs.get("pred_profile", logits)

        loss_ce = self.ce_loss(logits, true_profile)
        loss_lobe = self.main_lobe_loss(pred_profile, true_profile)
        loss_smooth = self.panel_smooth_loss(pred_profile)

        total_loss = self.lambda_ce * loss_ce + self.lambda_main_lobe * loss_lobe + self.lambda_panel_smooth * loss_smooth

        if "residual_offset_rad" in outputs and outputs["residual_offset_rad"] is not None:
            res_reg = (outputs["residual_offset_rad"] ** 2).mean()
            total_loss = total_loss + self.lambda_residual_reg * res_reg
        else:
            res_reg = torch.tensor(0.0, device=logits.device)

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_lobe": loss_lobe.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_residual_reg": res_reg.detach(),
        }
