from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_dinov2_backbone(name: str) -> bool:
    return str(name).lower() == "vit_base_patch14_dinov2"


def backbone_interpolation(name: str) -> InterpolationMode:
    return InterpolationMode.BICUBIC if is_dinov2_backbone(name) else InterpolationMode.BILINEAR


def backbone_eval_resize(name: str, image_size: int) -> int:
    return image_size if is_dinov2_backbone(name) else 256


def build_image_transforms(backbone: str, image_size: int, train_aug: bool) -> T.Compose:
    interp = backbone_interpolation(backbone)
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train_aug:
        ops: list[nn.Module] = [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=interp),
            T.RandomHorizontalFlip(),
        ]
        if not is_dinov2_backbone(backbone):
            ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))
        ops.extend([T.ToTensor(), norm])
        return T.Compose(ops)

    resize_size = backbone_eval_resize(backbone, image_size)
    return T.Compose([
        T.Resize(resize_size, interpolation=interp),
        T.CenterCrop(image_size),
        T.ToTensor(),
        norm,
    ])


def build_pil_transforms(backbone: str, image_size: int) -> T.Compose:
    interp = backbone_interpolation(backbone)
    resize_size = backbone_eval_resize(backbone, image_size)
    return T.Compose([
        T.Resize(resize_size, interpolation=interp),
        T.CenterCrop(image_size),
    ])


def iter_head_modules(model: nn.Module) -> Iterable[nn.Module]:
    for name in ("fc", "head", "classifier"):
        mod = getattr(model, name, None)
        if isinstance(mod, nn.Module):
            yield mod
    heads = getattr(model, "heads", None)
    if isinstance(heads, nn.Module):
        yield heads


def set_backbone_trainable(model: nn.Module, train_backbone: bool) -> None:
    for p in model.parameters():
        p.requires_grad = train_backbone
    for head in iter_head_modules(model):
        for p in head.parameters():
            p.requires_grad = True


def split_backbone_and_head_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    head_param_ids = set()
    head_params: list[nn.Parameter] = []
    for head in iter_head_modules(model):
        for p in head.parameters():
            if id(p) not in head_param_ids:
                head_params.append(p)
                head_param_ids.add(id(p))

    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return backbone_params, head_params


def forward_with_features(model: nn.Module, backbone: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, penultimate features) for the supported Waterbirds backbones."""
    name = str(backbone).lower()

    if name in {"resnet50", "resnet18"}:
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)
        h = model.layer1(h)
        h = model.layer2(h)
        h = model.layer3(h)
        h = model.layer4(h)
        h = model.avgpool(h)
        feats = torch.flatten(h, 1)
        logits = model.fc(feats)
        return logits, feats

    if name == "convnext_tiny":
        h = model.features(x)
        h = model.avgpool(h)
        h = torch.flatten(h, 1)
        feats = h
        for layer in model.classifier[:-1]:
            feats = layer(feats)
        logits = model.classifier[-1](feats)
        return logits, feats

    if name == "vit_b_16":
        n = x.shape[0]
        h = model._process_input(x)
        cls = model.class_token.expand(n, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = model.encoder(h)
        feats = h[:, 0]
        logits = model.heads(feats)
        return logits, feats

    if name == "vit_base_patch14_dinov2":
        tokens = model.forward_features(x)
        feats = model.forward_head(tokens, pre_logits=True)
        logits = model.forward_head(tokens)
        return logits, feats

    raise ValueError(f"Unsupported backbone for feature extraction: {backbone}")
