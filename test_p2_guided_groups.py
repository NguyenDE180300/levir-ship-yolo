#!/usr/bin/env python3
"""Structural and identity tests for the four P2'-guided ablation groups."""

from pathlib import Path

import torch

from ultralytics.nn.modules import (
    P2GuidedDeformConv,
    P2GuidedDynamicConv,
    P2GuidedGatedModulation,
    P2GuidedLocalCrossAttention,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "input_guided_context_refine_p2p3": "yolov8_input_guided_context_refine_p2p3.yaml",
    "p2guided_gate_p3p4": "yolov8_p2guided_gate_p3p4.yaml",
    "p2guided_dynamic_conv_p3p4": "yolov8_p2guided_dynamic_conv_p3p4.yaml",
    "p2guided_cross_attention_p3p4": "yolov8_p2guided_cross_attention_p3p4.yaml",
    "p2guided_deform_conv_p3p4": "yolov8_p2guided_deform_conv_p3p4.yaml",
}
BLOCKS = (
    P2GuidedGatedModulation,
    P2GuidedDynamicConv,
    P2GuidedLocalCrossAttention,
    P2GuidedDeformConv,
)


def check_identity() -> None:
    target = torch.randn(2, 64, 16, 16)
    guidance = torch.randn(2, 32, 32, 32)
    for block_type in BLOCKS:
        block = block_type([64, 32])
        output = block([target, guidance])
        torch.testing.assert_close(output, target, rtol=0, atol=0)
        output.mean().backward()
        assert block.gamma.grad is not None
        print(f"identity_ok {block_type.__name__} shape={tuple(output.shape)}")


def check_models() -> None:
    for case, filename in CONFIGS.items():
        model = DetectionModel(ROOT / "model_cfg" / filename, ch=3, nc=1, verbose=False)
        model.train()
        image = torch.randn(1, 3, 128, 128, requires_grad=True)
        outputs = model(image)
        tensors = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
        tensors = [output for output in tensors if isinstance(output, torch.Tensor)]
        sum(output.float().mean() for output in tensors).backward()
        shapes = [tuple(output.shape) for output in tensors]
        parameters = sum(parameter.numel() for parameter in model.parameters())
        gflops = get_flops(model, imgsz=512)
        print(f"model_ok {case} params={parameters} gflops_512={gflops:.6f} shapes={shapes}")


if __name__ == "__main__":
    check_identity()
    check_models()
