#!/usr/bin/env python3
"""Focused structural smoke test for the four post-FPN SRM guidance ablations.

This intentionally uses synthetic tensors only.  It validates graph wiring and
the zero-init safety contract before an expensive detector training run.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.nn.modules.srm_f2_guidance import SRMF2Guidance
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "srm3_f2": "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance.yaml",
    "k4k12k16_f2": "yolov8_baseline_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance.yaml",
    "srm3_f2_reg": "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg.yaml",
    "srm3_cls": "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance.yaml",
    "srm3_saturation_cls": "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance.yaml",
}


def _loss(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float().mean()
    if isinstance(value, dict):
        return sum((_loss(v) for v in value.values()), torch.zeros(()))
    if isinstance(value, (list, tuple)):
        return sum((_loss(v) for v in value), torch.zeros(()))
    raise TypeError(type(value))


def _branch_probe(module: SRMF2Guidance) -> tuple[float, int]:
    """Return identity error and number of finite SRM-encoder gradients.

    The zero-initialized gamma intentionally blocks branch gradients at exact
    initialization.  We therefore check identity first, then use a temporary
    nonzero scale solely to prove the branch is differentiably connected.
    """
    f2 = torch.randn(2, module.f2_channels, 32, 32, requires_grad=True)
    rgb = torch.rand(2, 3, 128, 128)
    module.train()
    out = module([f2, rgb])
    identity_error = (out - f2).abs().max().item()
    assert identity_error < 1e-6, identity_error
    assert module.last_residuals.shape == (2, 3, 32, 32)
    assert module.last_mask.shape == (2, 1, 32, 32)
    assert module.last_gamma.item() == 0.0
    if module.use_saturation_guidance:
        assert module.last_saturation_mask.shape == (2, 1, 32, 32)
        assert module.last_saturation_gamma.item() == 0.0

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.gamma_raw.fill_(0.1)
    module([f2, rgb]).square().mean().backward()
    grads = [p.grad for p in module.mask_net.parameters() if p.requires_grad]
    finite = sum(g is not None and torch.isfinite(g).all() for g in grads)
    assert finite == len(grads), (finite, len(grads))
    assert module.gamma_raw.grad is not None and torch.isfinite(module.gamma_raw.grad).all()
    if module.use_saturation_guidance:
        assert module.gamma_saturation_raw.grad is not None and torch.isfinite(module.gamma_saturation_raw.grad).all()
    with torch.no_grad():
        module.gamma_raw.zero_()
    return identity_error, finite


@torch.no_grad()
def _conv_gflops(model: torch.nn.Module, image_size: int, guidance: SRMF2Guidance) -> float:
    """Count Conv2d FLOPs (2 MACs) plus the fixed functional SRM convolution.

    This deliberately avoids optional ``thop`` and the local broken Kineto
    profiler. Normalization/activation/interpolation are excluded, matching the
    usual lightweight detector convention that reports convolution FLOPs.
    """
    total = 0

    def count_conv(module, _inputs, output):
        nonlocal total
        if not isinstance(output, torch.Tensor):
            return
        batch, out_channels, height, width = output.shape
        kh, kw = module.kernel_size
        total += batch * out_channels * height * width * (module.in_channels // module.groups) * kh * kw * 2

    hooks = [m.register_forward_hook(count_conv) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    was_training = model.training
    model.eval()
    try:
        model(torch.zeros(1, 3, image_size, image_size))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    # Functional fixed SRM: grayscale [B,1,H,W] convolved by three 5x5 filters.
    total += 1 * 3 * image_size * image_size * 1 * 5 * 5 * 2
    return total / 1e9


def main() -> None:
    torch.manual_seed(42)
    for name, yaml_name in CONFIGS.items():
        model = DetectionModel(str(ROOT / "model_cfg" / yaml_name), ch=3, nc=1, verbose=False)
        model.train()
        guidance = next(m for m in model.modules() if isinstance(m, SRMF2Guidance))
        identity_error, branch_grad_tensors = _branch_probe(guidance)

        # Whole-model forward/backward verifies the YAML graph, FPN and head.
        captured = {}
        hook = None
        if name in {"srm3_cls", "srm3_saturation_cls"}:
            hook = model.model[21].register_forward_hook(lambda _m, _i, out: captured.setdefault("f2_base", out.detach()))
        model.zero_grad(set_to_none=True)
        output = model(torch.randn(1, 3, 128, 128))
        if hook is not None:
            hook.remove()
        _loss(output).backward()
        head = model.model[-1]
        assert head.nl == 1 and head.stride.tolist() == [4.0]
        assert any(m.__class__.__name__ == "SPPF" for m in model.modules())
        assert guidance.last_residuals.shape == (1, 3, 32, 32)
        assert guidance.last_mask.shape == (1, 1, 32, 32)
        assert guidance.last_gamma.item() == 0.0
        assert guidance.gamma_raw.grad is not None and torch.isfinite(guidance.gamma_raw.grad).all()
        gamma_grad = guidance.gamma_raw.grad.item()

        if name in {"srm3_cls", "srm3_saturation_cls"}:
            assert torch.equal(head.last_reg_feature, captured["f2_base"]), "regression must receive untouched F2_base"
            assert torch.equal(head.last_reg_feature, head.last_cls_feature), "gamma=0 should preserve cls identity"
            assert head.last_reg_feature.shape == (1, 32, 32, 32)

        params = sum(p.numel() for p in model.parameters())
        gflops = _conv_gflops(model, image_size=512, guidance=guidance)
        f2_512_shape = (1, guidance.f2_channels, *guidance.last_mask.shape[-2:])
        print(
            f"{name}: build/forward/backward=OK params={params:,} gflops_512={gflops:.4f} "
            f"stride={head.stride.tolist()} nl={head.nl} F2@512={f2_512_shape} "
            f"SRM={tuple(guidance.last_residuals.shape)} A2={tuple(guidance.last_mask.shape)} "
            f"gamma={guidance.last_gamma.item():.1f} identity_max={identity_error:.1e} "
            f"gamma_grad={gamma_grad:.2e} branch_grad_tensors={branch_grad_tensors}"
        )


if __name__ == "__main__":
    main()
