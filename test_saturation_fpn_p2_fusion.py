#!/usr/bin/env python3
"""Structural smoke test for the single-use saturation FPN P3→P2 fusion case."""
from __future__ import annotations

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops


CASE_CFG = "model_cfg/yolov8_baseline_gap_factorized_k15_rgb_saturation_fpn_p2_fusion.yaml"
RGB_CFG = "model_cfg/yolov8_baseline_gap_factorized_k15.yaml"


def _loss(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float().mean()
    if isinstance(value, dict):
        return sum((_loss(v) for v in value.values()), torch.zeros(()))
    return sum((_loss(v) for v in value), torch.zeros(()))


def main() -> None:
    torch.manual_seed(42)
    model = DetectionModel(CASE_CFG, ch=3, nc=1, verbose=False).train()
    rgb_model = DetectionModel(RGB_CFG, ch=3, nc=1, verbose=False)
    cue = model.model[11]
    tensors = {}
    for index, name in ((18, "up_f3"), (3, "p2_backbone"), (19, "fusion_concat"), (20, "f2")):
        model.model[index].register_forward_hook(lambda _m, _i, out, key=name: tensors.__setitem__(key, out.detach()))

    output = model(torch.rand(1, 3, 128, 128))
    _loss(output).backward()
    shapes = {key: tuple(value.shape) for key, value in tensors.items()}
    expected = {
        "up_f3": (1, 64, 32, 32), "p2_backbone": (1, 32, 32, 32),
        "fusion_concat": (1, 112, 32, 32), "f2": (1, 32, 32, 32),
    }
    assert shapes == expected, shapes
    assert tuple(cue.last_s2.shape) == (1, 16, 32, 32)
    s2_shape = tuple(cue.last_s2.shape)
    assert abs(cue.alpha.item() - 0.1) < 1e-6
    assert cue.alpha.grad is not None and cue.alpha.grad.abs().item() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in cue.stem.parameters())
    head = model.model[-1]
    assert head.stride.tolist() == [4.0] and head.nl == 1
    assert any(module.__class__.__name__ == "SPPF" for module in model.modules())

    params = sum(parameter.numel() for parameter in model.parameters())
    rgb_params = sum(parameter.numel() for parameter in rgb_model.parameters())
    print({
        "forward_backward": "OK", "params": params, "extra_params_vs_rgb": params - rgb_params,
        "gflops_512": get_flops(model, imgsz=512), "alpha": cue.alpha.item(),
        "alpha_grad": cue.alpha.grad.item(), "s2": s2_shape, "shapes": shapes,
    })


if __name__ == "__main__":
    main()
