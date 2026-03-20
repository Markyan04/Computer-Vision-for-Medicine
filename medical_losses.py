import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).float()


def _label_smoothing_one_hot(target: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        true_dist = torch.zeros(target.size(0), num_classes, device=target.device)
        true_dist.fill_(smoothing / max(num_classes - 1, 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return true_dist


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


class ClassBalancedFocalCELoss(nn.Module):
    """
    CE + focal modulation + effective-number reweighting + optional label smoothing.
    Good default for imbalanced medical classification.
    """
    def __init__(self, class_counts, beta: float = 0.9999, gamma: float = 2.0, smoothing: float = 0.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.smoothing > 0:
            soft_t = _label_smoothing_one_hot(target, logits.size(1), self.smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            ce_per_sample = -(soft_t * log_probs).sum(dim=1)
        else:
            ce_per_sample = F.cross_entropy(logits, target, reduction="none")

        sample_weights = self.class_weights[target]
        loss = sample_weights * focal_factor * ce_per_sample
        return loss.mean()


class OrdinalFocalMSELoss(nn.Module):
    """
    For ordinal labels (e.g., KL 0-4):
    focal CE + expected-grade regression penalty.
    Penalizes far-away mistakes more strongly.
    """
    def __init__(self, num_classes: int, alpha_ce: float = 1.0, alpha_mse: float = 0.3,
                 gamma: float = 2.0, class_weights: Optional[list] = None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_ce = alpha_ce
        self.alpha_mse = alpha_mse
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None
        self.register_buffer("grade_values", torch.arange(num_classes, dtype=torch.float))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.class_weights)
        focal_ce = (focal_factor * ce).mean()

        pred_grade = (probs * self.grade_values.unsqueeze(0)).sum(dim=1)
        target_grade = target.float()
        mse = F.mse_loss(pred_grade, target_grade)
        return self.alpha_ce * focal_ce + self.alpha_mse * mse


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Robust to noisy labels: CE + Reverse CE.
    Useful when medical labels have inter-observer disagreement.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, num_classes: int = 5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = _one_hot(target, self.num_classes).clamp(min=1e-4, max=1.0)
        rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    GCE bridges CE and MAE; often used for noisy labels.
    q in (0,1]. Smaller q => more robust.
    """
    def __init__(self, q: float = 0.7):
        super().__init__()
        assert 0 < q <= 1.0
        self.q = q

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        if abs(self.q - 1.0) < 1e-8:
            return (-torch.log(pt)).mean()
        return ((1.0 - pt.pow(self.q)) / self.q).mean()


class DistanceAwareSoftTargetLoss(nn.Module):
    """
    Novel loss 1:
    build distance-decayed soft targets for ordinal grades, then apply soft CE.
    Neighbor grades receive small probability mass; far-away grades are strongly discouraged.
    """
    def __init__(self, num_classes: int, tau: float = 1.0, gamma: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.gamma = gamma
        self.register_buffer("class_ids", torch.arange(num_classes, dtype=torch.float))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f = target.float().unsqueeze(1)
        dist = torch.abs(self.class_ids.unsqueeze(0) - target_f)
        soft_targets = torch.exp(-dist / self.tau)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        per_sample = -(soft_targets * log_probs).sum(dim=1)

        if self.gamma > 0:
            pt_soft = (probs * soft_targets).sum(dim=1).clamp(min=1e-8, max=1.0)
            per_sample = ((1.0 - pt_soft).pow(self.gamma)) * per_sample
        return per_sample.mean()


class PrototypeConsistencyOrdinalLoss(nn.Module):
    """
    Novel loss 2:
    classification loss + feature-to-class-prototype consistency + ordinal spacing regularization.
    Requires passing intermediate features of shape [B, D].
    This can create more orderly feature geometry for grading tasks.
    """
    def __init__(self, num_classes: int, feat_dim: int, lambda_proto: float = 0.2, lambda_order: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_proto = lambda_proto
        self.lambda_order = lambda_order
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.02)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)

        proto_target = self.prototypes[target]
        proto_loss = F.mse_loss(features, proto_target)

        if self.num_classes > 1:
            gaps = self.prototypes[1:] - self.prototypes[:-1]
            gap_norms = gaps.norm(dim=1)
            order_loss = gap_norms.var()
        else:
            order_loss = torch.tensor(0.0, device=logits.device)

        return ce + self.lambda_proto * proto_loss + self.lambda_order * order_loss


class AdaptiveOrdinalMarginLoss(nn.Module):
    """
    Novel loss 3:
    add a distance-dependent margin against confusing nearby/far classes.
    Works on logits only, easy to plug in.
    """
    def __init__(self, num_classes: int, margin_base: float = 0.15, power: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin_base = margin_base
        self.power = power
        self.register_buffer("class_ids", torch.arange(num_classes, dtype=torch.float))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        target_ids = target.float().unsqueeze(1)
        dist = torch.abs(self.class_ids.unsqueeze(0) - target_ids)
        margins = self.margin_base * (dist.pow(self.power) / max((self.num_classes - 1) ** self.power, 1e-8))

        adjusted_logits = logits.clone()
        adjusted_logits = adjusted_logits - margins
        adjusted_logits[torch.arange(B, device=logits.device), target] = logits[torch.arange(B, device=logits.device), target]
        return F.cross_entropy(adjusted_logits, target)


# -----------------------------
# Example factory for quick use
# -----------------------------
def build_loss(name: str, **kwargs):
    name = name.lower()
    if name == "cb_focal_ce":
        return ClassBalancedFocalCELoss(**kwargs)
    if name == "ordinal_focal_mse":
        return OrdinalFocalMSELoss(**kwargs)
    if name == "sce":
        return SymmetricCrossEntropyLoss(**kwargs)
    if name == "gce":
        return GeneralizedCrossEntropyLoss(**kwargs)
    if name == "dast":
        return DistanceAwareSoftTargetLoss(**kwargs)
    if name == "pcol":
        return PrototypeConsistencyOrdinalLoss(**kwargs)
    if name == "aom":
        return AdaptiveOrdinalMarginLoss(**kwargs)
    raise ValueError(f"Unknown loss name: {name}")


if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, D = 8, 5, 128
    logits = torch.randn(B, C)
    target = torch.randint(0, C, (B,))
    features = torch.randn(B, D)

    loss1 = ClassBalancedFocalCELoss(class_counts=[500, 300, 120, 60, 20])(logits, target)
    loss2 = OrdinalFocalMSELoss(num_classes=C)(logits, target)
    loss3 = SymmetricCrossEntropyLoss(num_classes=C)(logits, target)
    loss4 = GeneralizedCrossEntropyLoss(q=0.7)(logits, target)
    loss5 = DistanceAwareSoftTargetLoss(num_classes=C, tau=1.0, gamma=1.5)(logits, target)
    loss6 = PrototypeConsistencyOrdinalLoss(num_classes=C, feat_dim=D)(logits, target, features)
    loss7 = AdaptiveOrdinalMarginLoss(num_classes=C)(logits, target)

    print("cb_focal_ce:", float(loss1))
    print("ordinal_focal_mse:", float(loss2))
    print("sce:", float(loss3))
    print("gce:", float(loss4))
    print("dast:", float(loss5))
    print("pcol:", float(loss6))
    print("aom:", float(loss7))
