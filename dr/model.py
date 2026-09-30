"""Pretrained CNN backbones (EfficientNet-B0, ResNet50, ConvNeXt-Tiny) with a new grading head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ("efficientnet_b0", "resnet50", "convnext_tiny")

# cut points that turn a regression output into grades (tunable on validation)
DEFAULT_THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


def outputs_to_grades(outputs: torch.Tensor, thresholds=DEFAULT_THRESHOLDS) -> torch.Tensor:
    """Head outputs to grades 0-4: argmax for 5 outputs, threshold cuts for 1."""
    outputs = outputs.detach().cpu()
    if outputs.shape[1] == 1:
        cuts = torch.tensor(thresholds, dtype=outputs.dtype)
        return torch.bucketize(outputs[:, 0].contiguous(), cuts)
    return outputs.argmax(1)


def build_model(
    name: str = "efficientnet_b0",
    n_classes: int = 5,
    dropout: float = 0.3,
    pretrained: bool = True,
    regression: bool = False,
) -> nn.Module:
    """Load a pretrained backbone and replace its final layer with dropout + linear(n_out)."""
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}, expected one of {ARCHITECTURES}")

    weights = "IMAGENET1K_V1" if pretrained else None
    n_out = 1 if regression else n_classes

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        # classifier = Sequential(Dropout, Linear(1280, 1000))
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_out))
    elif name == "convnext_tiny":
        # classifier = Sequential(LayerNorm2d, Flatten, Linear(768, 1000)); swap only the last layer
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_out))
    else:
        model = models.resnet50(weights=weights)
        # fc = Linear(2048, 1000)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_out))

    model.arch = name  # so evaluate/app can rebuild the same model
    model.regression = regression
    return model


def head(model: nn.Module) -> nn.Module:
    """The new classification layers (always trainable)."""
    return model.classifier if hasattr(model, "classifier") else model.fc


def backbone_stages(model: nn.Module) -> list[nn.Module]:
    """Backbone stages from shallow to deep (EfficientNet/ConvNeXt .features, ResNet layer1..4)."""
    if hasattr(model, "features"):
        return list(model.features)
    return [model.layer1, model.layer2, model.layer3, model.layer4]


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Phase 1: freeze every backbone weight, train only the head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in head(model).parameters():
        param.requires_grad = True
    return model


def unfreeze_top(model: nn.Module, n_stages: int = 3) -> nn.Module:
    """Phase 2: also train the deepest n_stages; batch-norm layers stay frozen and in eval mode."""
    for stage in backbone_stages(model)[-n_stages:]:
        for param in stage.parameters():
            param.requires_grad = True

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return model


def freeze_batchnorm(model: nn.Module) -> None:
    """Put batch-norm layers back in eval mode (model.train() undoes it each epoch)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block, used by Grad-CAM."""
    if hasattr(model, "features"):
        return model.features[-1]
    return model.layer4[-1]


def param_counts(model: nn.Module) -> dict[str, int]:
    """Trainable / frozen / total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float) -> list[dict]:
    """Optimiser groups: small LR for the backbone, larger LR for the head."""
    head_params = set(id(p) for p in head(model).parameters())
    backbone = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_params
    ]
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": [p for p in head(model).parameters() if p.requires_grad], "lr": lr_head},
    ]


if __name__ == "__main__":
    # self-check without downloading ImageNet weights
    for arch in ("efficientnet_b0", "resnet50"):
        model = build_model(arch, pretrained=False)

        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 5), f"{arch}: expected [2, 5] logits, got {tuple(out.shape)}"

        freeze_backbone(model)
        phase1 = param_counts(model)
        head_only = sum(p.numel() for p in head(model).parameters())
        assert phase1["trainable"] == head_only, f"{arch}: phase 1 should train the head only"

        unfreeze_top(model, n_stages=3)
        phase2 = param_counts(model)
        assert phase2["trainable"] > phase1["trainable"], f"{arch}: phase 2 unfroze nothing"
        assert phase2["trainable"] < phase2["total"], f"{arch}: phase 2 unfroze everything"

        # batch-norm stays frozen after unfreezing
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert bns, f"{arch}: expected batch-norm layers"
        assert not any(m.training for m in bns), f"{arch}: batch-norm left in train mode"
        assert not any(p.requires_grad for m in bns for p in m.parameters()), (
            f"{arch}: batch-norm parameters left trainable"
        )

        groups = param_groups(model, 1e-4, 1e-3)
        assert len(groups) == 2 and groups[0]["lr"] < groups[1]["lr"], (
            f"{arch}: backbone should get the smaller learning rate"
        )
        assert target_layer(model) is not None

    values = torch.tensor([[-0.3], [0.7], [1.49], [2.6], [9.0]])
    assert outputs_to_grades(values).tolist() == [0, 1, 1, 3, 4], "threshold cut is wrong"
    assert outputs_to_grades(torch.eye(5)).tolist() == [0, 1, 2, 3, 4], "argmax path is wrong"

    cx = build_model("convnext_tiny", pretrained=False)
    assert cx(torch.randn(2, 3, 224, 224)).shape == (2, 5)
    freeze_backbone(cx)
    p1 = param_counts(cx)["trainable"]
    unfreeze_top(cx, 3)
    p2 = param_counts(cx)
    assert p1 < p2["trainable"] < p2["total"], "convnext phase 2 must unfreeze part, not all"
    assert p2["trainable"] > 0.6 * p2["total"], "stages 3-4 hold most ConvNeXt weights"
    cam_layer = target_layer(cx)
    assert cam_layer is cx.features[-1]

    reg = build_model("resnet50", pretrained=False, regression=True)
    assert reg(torch.randn(2, 3, 224, 224)).shape == (2, 1), "regression head must output one value"
    assert reg.regression and not build_model("resnet50", pretrained=False).regression

    print("model.py self-check passed")
