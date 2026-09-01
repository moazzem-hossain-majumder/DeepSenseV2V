import torch
import torch.nn as nn
import torchvision.models as models

N_BEAMS = 256
N_GPS_FEATS = 9


def _mlp_head(d_model, n_beams, dropout, head_hidden):
    return nn.Sequential(
        nn.Linear(d_model, head_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, n_beams),
    )


class ResNet18VisionEncoder(nn.Module):
    """
    ResNet-18 encoder over a 5-frame clip.
    freeze_until='layer2' trains layer2–4; 'layer3' trains layer3–4; 'none' trains all.
    """
    def __init__(self, d_model=256, freeze_until="layer2"):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(weights=None)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.proj = nn.Linear(512, d_model)

        frozen = []
        if freeze_until == "layer3":
            frozen = [self.conv1, self.bn1, self.layer1, self.layer2]
        elif freeze_until == "layer2":
            frozen = [self.conv1, self.bn1, self.layer1]
        elif freeze_until == "layer1":
            frozen = [self.conv1, self.bn1]
        for layer in frozen:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.reshape(B * S, C, H, W)
        if x_flat.is_cuda:
            x_flat = x_flat.contiguous(memory_format=torch.channels_last)

        x_feat = self.conv1(x_flat)
        x_feat = self.bn1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)
        x_feat = self.layer1(x_feat)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)
        x_feat = self.layer4(x_feat)
        x_feat = self.avgpool(x_feat).flatten(1)
        return self.proj(x_feat).view(B, S, -1)


class BiGRUPositionEncoder(nn.Module):
    def __init__(self, in_dim=N_GPS_FEATS, d_model=256, num_layers=3, dropout=0.12):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)


class PreLNTransformerFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.12):
        super().__init__()
        self.modality_gps = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_rgb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rgb_tokens, gps_tokens):
        gps_tok = gps_tokens + self.modality_gps
        rgb_tok = rgb_tokens + self.modality_rgb
        seq = torch.cat([gps_tok, rgb_tok], dim=1) + self.pos_embed
        return torch.mean(self.norm(self.transformer(seq)), dim=1)


def _common_kwargs(kwargs):
    return {
        "d_model": kwargs.get("d_model", 256),
        "fusion_heads": kwargs.get("fusion_heads", 8),
        "fusion_layers": kwargs.get("fusion_layers", 3),
        "freeze_until": kwargs.get("freeze_until", "layer2"),
        "dropout": kwargs.get("dropout", 0.12),
        "n_beams": kwargs.get("n_beams", N_BEAMS),
        "gru_layers": kwargs.get("gru_layers", 3),
        "head_hidden": kwargs.get("head_hidden", 512),
    }


class P3_MultiTaskProfile(nn.Module):
    """Wider multi-task RGB+GPS model sized for an 8 GB consumer GPU."""

    def __init__(
        self,
        d_model=256,
        fusion_heads=8,
        fusion_layers=3,
        freeze_until="layer2",
        dropout=0.12,
        n_beams=N_BEAMS,
        gru_layers=3,
        head_hidden=512,
        **_ignored,
    ):
        super().__init__()
        self.rgb_encoder = ResNet18VisionEncoder(d_model=d_model, freeze_until=freeze_until)
        self.gps_encoder = BiGRUPositionEncoder(d_model=d_model, num_layers=gru_layers, dropout=dropout)
        self.fusion = PreLNTransformerFusion(
            d_model=d_model,
            nhead=fusion_heads,
            num_layers=fusion_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.cls_head = _mlp_head(d_model, n_beams, dropout, head_hidden)
        self.profile_head = _mlp_head(d_model, n_beams, dropout, head_hidden)

    def forward(self, rgb, gps):
        fused_rep = self.fusion(self.rgb_encoder(rgb), self.gps_encoder(gps))
        return {
            "logits": self.cls_head(fused_rep),
            "pred_profile": self.profile_head(fused_rep),
            "fused_rep": fused_rep,
        }


class B0_Geometric(nn.Module):
    """Stretch B0: map last-step Tx-Rx bearing to a codebook index. No learned weights."""

    def __init__(self, n_beams=N_BEAMS, **_ignored):
        super().__init__()
        self.n_beams = n_beams
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, gps):
        # gps: (B, 5, 9) with [rel_e, rel_n, ...]
        rel_e = gps[:, -1, 0]
        rel_n = gps[:, -1, 1]
        bearing = torch.atan2(rel_e, rel_n)
        beam = ((bearing + torch.pi) / (2 * torch.pi) * self.n_beams).long() % self.n_beams
        logits = torch.full((gps.size(0), self.n_beams), -50.0, device=gps.device, dtype=gps.dtype)
        logits.scatter_(1, beam.unsqueeze(1), 10.0)
        return {"logits": logits}


class B2_RGBOnly(nn.Module):
    """Stretch B2: RGB-only temporal encoder."""

    def __init__(self, **kwargs):
        super().__init__()
        p = _common_kwargs(kwargs)
        self.rgb_enc = ResNet18VisionEncoder(d_model=p["d_model"], freeze_until=p["freeze_until"])
        self.head = _mlp_head(p["d_model"], p["n_beams"], p["dropout"], p["head_hidden"])

    def forward(self, rgb, gps):
        logits = self.head(torch.mean(self.rgb_enc(rgb), dim=1))
        return {"logits": logits}


class B4_MultimodalTransformer(nn.Module):
    """Stretch B4: deeper Pre-LN multimodal Transformer fusion."""

    def __init__(self, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        kwargs.setdefault("fusion_layers", 4)
        kwargs.setdefault("fusion_heads", 8)
        p = _common_kwargs(kwargs)
        self.rgb_enc = ResNet18VisionEncoder(d_model=p["d_model"], freeze_until=p["freeze_until"])
        self.gps_enc = BiGRUPositionEncoder(d_model=p["d_model"], num_layers=p["gru_layers"], dropout=p["dropout"])
        self.fusion = PreLNTransformerFusion(
            d_model=p["d_model"],
            nhead=p["fusion_heads"],
            num_layers=max(p["fusion_layers"], 4),
            dim_feedforward=p["d_model"] * 4,
            dropout=p["dropout"],
        )
        self.head = _mlp_head(p["d_model"], p["n_beams"], p["dropout"], p["head_hidden"])

    def forward(self, rgb, gps):
        logits = self.head(self.fusion(self.rgb_enc(rgb), self.gps_enc(gps)))
        return {"logits": logits}


class B1_GPSOnly(nn.Module):
    def __init__(self, d_model=256, n_beams=N_BEAMS, gru_layers=3, dropout=0.12, head_hidden=512, **_ignored):
        super().__init__()
        self.gps_enc = BiGRUPositionEncoder(d_model=d_model, num_layers=gru_layers, dropout=dropout)
        self.head = _mlp_head(d_model, n_beams, dropout, head_hidden)

    def forward(self, rgb, gps):
        logits = self.head(torch.mean(self.gps_enc(gps), dim=1))
        return {"logits": logits}


class GatedConcatFusion(nn.Module):
    """B3: gate RGB vs GPS then concatenate (gated/concat fusion)."""

    def __init__(self, d_model=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out = nn.Linear(d_model * 2, d_model)

    def forward(self, rgb_tokens, gps_tokens):
        rgb_p = torch.mean(rgb_tokens, dim=1)
        gps_p = torch.mean(gps_tokens, dim=1)
        g = self.gate(torch.cat([rgb_p, gps_p], dim=-1))
        mixed = torch.cat([g * rgb_p, (1.0 - g) * gps_p], dim=-1)
        return self.out(mixed)


class B3_Fusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        p = _common_kwargs(kwargs)
        self.rgb_enc = ResNet18VisionEncoder(d_model=p["d_model"], freeze_until=p["freeze_until"])
        self.gps_enc = BiGRUPositionEncoder(d_model=p["d_model"], num_layers=p["gru_layers"], dropout=p["dropout"])
        self.fusion = GatedConcatFusion(d_model=p["d_model"])
        self.head = _mlp_head(p["d_model"], p["n_beams"], p["dropout"], p["head_hidden"])

    def forward(self, rgb, gps):
        logits = self.head(self.fusion(self.rgb_enc(rgb), self.gps_enc(gps)))
        return {"logits": logits}


class P1_ClassificationOnly(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        p = _common_kwargs(kwargs)
        self.rgb_enc = ResNet18VisionEncoder(d_model=p["d_model"], freeze_until=p["freeze_until"])
        self.gps_enc = BiGRUPositionEncoder(d_model=p["d_model"], num_layers=p["gru_layers"], dropout=p["dropout"])
        self.fusion = PreLNTransformerFusion(
            d_model=p["d_model"],
            nhead=p["fusion_heads"],
            num_layers=p["fusion_layers"],
            dim_feedforward=p["d_model"] * 4,
            dropout=p["dropout"],
        )
        self.cls_head = _mlp_head(p["d_model"], p["n_beams"], p["dropout"], p["head_hidden"])

    def forward(self, rgb, gps):
        logits = self.cls_head(self.fusion(self.rgb_enc(rgb), self.gps_enc(gps)))
        return {"logits": logits}


class P2_ProfileOnly(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        p = _common_kwargs(kwargs)
        self.rgb_enc = ResNet18VisionEncoder(d_model=p["d_model"], freeze_until=p["freeze_until"])
        self.gps_enc = BiGRUPositionEncoder(d_model=p["d_model"], num_layers=p["gru_layers"], dropout=p["dropout"])
        self.fusion = PreLNTransformerFusion(
            d_model=p["d_model"],
            nhead=p["fusion_heads"],
            num_layers=p["fusion_layers"],
            dim_feedforward=p["d_model"] * 4,
            dropout=p["dropout"],
        )
        self.profile_head = _mlp_head(p["d_model"], p["n_beams"], p["dropout"], p["head_hidden"])

    def forward(self, rgb, gps):
        pred_profile = self.profile_head(self.fusion(self.rgb_enc(rgb), self.gps_enc(gps)))
        return {"pred_profile": pred_profile, "logits": pred_profile}


def compute_profile_smoothness_loss(pred_profile):
    diffs = pred_profile[:, 1:] - pred_profile[:, :-1]
    return torch.mean(diffs ** 2)


def pairwise_ranking_loss(pred_profile, true_profile, n_pairs=32):
    """Optional P3 ranking term: sampled pairwise order of beams should match measured power."""
    B, K = pred_profile.shape
    i = torch.randint(0, K, (B, n_pairs), device=pred_profile.device)
    j = torch.randint(0, K, (B, n_pairs), device=pred_profile.device)
    pred_d = pred_profile.gather(1, i) - pred_profile.gather(1, j)
    true_d = true_profile.gather(1, i) - true_profile.gather(1, j)
    sign = torch.sign(true_d)
    return torch.mean(torch.relu(1.0 - sign * pred_d))


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_prof=0.1, lambda_smooth=0.01, lambda_rank=0.05, use_huber=False, class_weights=None):
        super().__init__()
        self.lambda_prof = lambda_prof
        self.lambda_smooth = lambda_smooth
        self.lambda_rank = lambda_rank
        self.use_huber = use_huber
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, outputs, target_beam, target_profile_db):
        details = {}
        total = 0.0
        if "logits" in outputs and not self.use_huber:
            loss_ce = self.ce_loss(outputs["logits"], target_beam)
            total = total + loss_ce
            details["loss_ce"] = loss_ce.item()
        if "pred_profile" in outputs:
            crit = self.huber if self.use_huber else self.mse_loss
            loss_prof = crit(outputs["pred_profile"], target_profile_db)
            loss_smooth = compute_profile_smoothness_loss(outputs["pred_profile"])
            loss_rank = pairwise_ranking_loss(outputs["pred_profile"], target_profile_db)
            total = total + self.lambda_prof * loss_prof + self.lambda_smooth * loss_smooth + self.lambda_rank * loss_rank
            details["loss_prof"] = loss_prof.item()
            details["loss_smooth"] = loss_smooth.item()
            details["loss_rank"] = loss_rank.item()
        if isinstance(total, float):
            total = outputs["logits"].sum() * 0.0
        details["loss_total"] = total.item() if hasattr(total, "item") else float(total)
        return total, details


def create_model(model_name="P3", **kwargs):
    models_dict = {
        "B0": B0_Geometric,
        "B1": B1_GPSOnly,
        "B2": B2_RGBOnly,
        "B3": B3_Fusion,
        "B4": B4_MultimodalTransformer,
        "P1": P1_ClassificationOnly,
        "P2": P2_ProfileOnly,
        "P3": P3_MultiTaskProfile,
    }
    if model_name not in models_dict:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from {list(models_dict.keys())}")
    return models_dict[model_name](**kwargs)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
