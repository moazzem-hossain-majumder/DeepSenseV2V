"""
Physically Anchored Residual Network (PAR-Net) for mmWave V2V Beam Forecasting.

Key architectural design:
1. Anchored on the physical Analytic Geometry Prior:
   The model does NOT learn absolute beam directions from scratch.
   It predicts continuous angular corrections (delta_theta) and candidate-subspace
   residual logits around the physical line-of-sight vector.

2. Heading-Conditioned Cross-Attention Fusion:
   Kinematic GPS tokens act as Queries (Q) attending over Visual (K, V) tokens,
   preventing visual background clutter from overwhelming the high-SNR kinematic state.

3. Candidate Subspace Reranking:
   Residual logit adjustments are localized around the top-K candidate beams,
   eliminating degradation on distant sidelobes and guaranteeing physical containment.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BEAMS = 256


class KinematicEncoder(nn.Module):
    """Encodes 5-step GPS kinematics and ENU context features."""
    def __init__(self, in_dim=9, context_dim=8, d_model=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.fusion = nn.Linear(d_model * 2 + d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gps_seq, context):
        """
        gps_seq: (B, 5, 9) historical GPS sequence
        context: (B, 8) ENU kinematic context features
        Returns: (B, d_model) summary kinematic token
        """
        gru_out, _ = self.gru(gps_seq)  # (B, 5, 2 * d_model)
        last_state = gru_out[:, -1, :]   # (B, 2 * d_model)

        ctx_feat = self.context_mlp(context)  # (B, d_model)

        combined = torch.cat([last_state, ctx_feat], dim=-1)
        fused = self.norm(self.fusion(combined))
        return fused


class VisualFeatureEncoder(nn.Module):
    """
    Lightweight visual feature encoder over 5 frames.
    Accepts raw RGB clip (B, 5, 3, H, W) or pre-extracted features (B, 5, in_dim).
    """
    def __init__(self, d_model=128, in_dim=512, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_pos = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)

    def forward(self, visual_tokens):
        """
        visual_tokens: (B, 5, in_dim)
        Returns: (B, 5, d_model)
        """
        return self.proj(visual_tokens) + self.temporal_pos


class KinematicCrossAttention(nn.Module):
    """
    Kinematic Query (Q) attends to Visual Tokens (K, V).
    Forces visual representations to filter through kinematic relevance.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.15):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, q_kinematic, kv_visual):
        """
        q_kinematic: (B, d_model)
        kv_visual: (B, 5, d_model)
        Returns: (B, d_model) multimodal fused token
        """
        q = q_kinematic.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.mha(query=q, key=kv_visual, value=kv_visual)
        x = self.norm1(q + attn_out)
        out = self.norm2(x + self.mlp(x))
        return out.squeeze(1)


class PARNet(nn.Module):
    """
    Physically Anchored Residual Network.
    Combines analytic geometry prior with bounded residual neural corrections.
    """
    def __init__(
        self,
        d_model=128,
        n_beams=N_BEAMS,
        max_angular_offset_rad=0.45,  # ~25.8 degrees
        max_candidate_correction_db=6.0,
        candidate_k=20,
        dropout=0.15,
        visual_in_dim=128,
    ):
        super().__init__()
        self.n_beams = n_beams
        self.max_angular_offset_rad = max_angular_offset_rad
        self.max_candidate_correction_db = max_candidate_correction_db
        self.candidate_k = candidate_k

        # Beam angle grid [-pi, pi]
        grid = np.linspace(-np.pi, np.pi, n_beams, endpoint=False, dtype=np.float32)
        self.register_buffer("beam_angles", torch.from_numpy(grid))

        # Encoders
        self.kinematic_enc = KinematicEncoder(in_dim=9, context_dim=8, d_model=d_model, dropout=dropout)
        self.visual_enc = VisualFeatureEncoder(d_model=d_model, in_dim=visual_in_dim, dropout=dropout)
        self.cross_attn = KinematicCrossAttention(d_model=d_model, nhead=4, dropout=dropout)

        # Residual Prediction Heads
        # 1. Continuous Angular Offset Head: Delta theta in radians
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # Bounded to [-1, 1] * max_angular_offset_rad
        )

        # 2. Local Candidate Residual Head: Delta z_b over candidate subspace
        self.candidate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, candidate_k),
            nn.Tanh(),  # Bounded to [-1, 1] * max_candidate_correction_db
        )

        # 3. Confidence Gate: learned balance factor between geometry prior and neural correction
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),  # in [0, 1]
        )

        # Initialize residual weights close to zero so model starts exactly at the geometry baseline
        nn.init.zeros_(self.offset_head[-2].weight)
        nn.init.zeros_(self.offset_head[-2].bias)
        nn.init.zeros_(self.candidate_head[-2].weight)
        nn.init.zeros_(self.candidate_head[-2].bias)

    def forward(self, gps_seq, context, visual_tokens, prior_logits=None, base_angle_rad=None):
        """
        gps_seq: (B, 5, 9) historical GPS sequence
        context: (B, 8) ENU kinematic context features
        visual_tokens: (B, 5, visual_in_dim) visual feature tokens
        prior_logits: (B, 256) precomputed geometry prior logits (optional)
        base_angle_rad: (B,) precomputed base LOS angle in radians (optional)

        Returns dict:
          - 'logits': (B, 256) calibrated final prediction logits
          - 'pred_profile': (B, 256) predicted relative power profile in dB
          - 'residual_offset_rad': (B, 1) continuous angular correction in radians
          - 'gate': (B, 1) confidence gate weight
          - 'prior_logits': (B, 256) input or computed base geometry prior
        """
        B = gps_seq.shape[0]

        # 1. Feature encoding & multimodal fusion
        h_kin = self.kinematic_enc(gps_seq, context)
        h_vis = self.visual_enc(visual_tokens)
        h_fused = self.cross_attn(h_kin, h_vis)

        # 2. Predict continuous angular correction Delta theta
        delta_theta = self.offset_head(h_fused) * self.max_angular_offset_rad  # (B, 1)

        # 3. Predict candidate residual corrections Delta z
        delta_cand = self.candidate_head(h_fused) * self.max_candidate_correction_db  # (B, candidate_k)

        # 4. Confidence gate
        gate = self.gate_head(h_fused)  # (B, 1)

        # If base_angle_rad is provided, compute shifted geometry distribution
        if base_angle_rad is not None:
            # Shift the base angle by the learned residual offset
            adjusted_angle = (base_angle_rad[:, None] + gate * delta_theta).squeeze(1)
            # Wrap to [-pi, pi]
            adjusted_angle = (adjusted_angle + torch.pi) % (2 * torch.pi) - torch.pi

            # Generate shifted spatial Gaussian prior
            ang_dist = (adjusted_angle[:, None] - self.beam_angles[None, :] + torch.pi) % (2 * torch.pi) - torch.pi
            abs_dist = torch.abs(ang_dist)
            shifted_prior = torch.clamp(-0.5 * (abs_dist / 0.35) ** 2, min=-20.0, max=0.0)
        elif prior_logits is not None:
            shifted_prior = prior_logits
        else:
            shifted_prior = torch.zeros((B, self.n_beams), device=gps_seq.device)

        # 5. Local candidate subspace refinement
        # Identify top-K candidate beams from shifted prior
        topk_candidates = torch.topk(shifted_prior, k=self.candidate_k, dim=1).indices  # (B, K)

        # Add candidate residual adjustments
        final_logits = shifted_prior.clone()
        for b in range(B):
            final_logits[b, topk_candidates[b]] += gate[b, 0] * delta_cand[b]

        # Power profile is directly modeled from final logits
        pred_profile = final_logits

        return {
            "logits": final_logits,
            "pred_profile": pred_profile,
            "residual_offset_rad": delta_theta,
            "gate": gate,
            "prior_logits": shifted_prior,
        }
