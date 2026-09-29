"""Architecture checks for the two corrected local-chroma ablations."""

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import (
    ASFHighResFusion,
    ContrastClsFTSCDetect,
    FTSCDetect,
    LocalChromaClsGuidance,
    LocalChromaContrast9x9,
    LocalChromaSharedGuidance,
    SENetV2EnSimAM,
)


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "model_cfg/yolov8_baseline_gap_factorized_k15_local_chroma_contrast_p2p3_nosasf_ftsc_f5_p2p3p4.yaml"
CLS_ONLY = ROOT / "model_cfg/yolov8_baseline_gap_factorized_k15_local_chroma_contrast_cls_guidance_p2p3_nosasf_ftsc_f5_p2p3p4.yaml"


def _tensor_sum(value):
    if isinstance(value, torch.Tensor):
        return value.float().sum()
    if isinstance(value, dict):
        return sum((_tensor_sum(v) for v in value.values()), torch.tensor(0.0))
    if isinstance(value, (list, tuple)):
        return sum((_tensor_sum(v) for v in value), torch.tensor(0.0))
    return torch.tensor(0.0)


def _build(path):
    return YOLO(str(path)).model


def test_shared_local_chroma_graph_and_gradients():
    model = _build(SHARED)
    modules = list(model.modules())
    guides = [m for m in modules if isinstance(m, LocalChromaSharedGuidance)]
    cues = [m for m in modules if isinstance(m, LocalChromaContrast9x9)]
    assert len(cues) == 1 and len(guides) == 2
    assert not any(isinstance(m, ASFHighResFusion) for m in modules)
    assert isinstance(model.model[-1], FTSCDetect) and not isinstance(model.model[-1], ContrastClsFTSCDetect)
    assert model.stride.tolist() == [4.0, 8.0, 16.0]
    assert all(float(m.gamma) == 0.0 for m in guides)

    calls = [0]
    hook = cues[0].register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    for guide in guides:
        guide.gamma.data.fill_(0.1)
    model.train()
    loss = _tensor_sum(model(torch.rand(1, 3, 128, 128)))
    loss.backward()
    hook.remove()
    assert calls[0] == 1
    for guide in guides:
        assert guide.gamma.grad is not None and torch.isfinite(guide.gamma.grad)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.encoder.parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.gate.parameters())


def test_cls_only_regression_isolation_and_gradients():
    model = _build(CLS_ONLY).eval()
    modules = list(model.modules())
    cues = [m for m in modules if isinstance(m, LocalChromaContrast9x9)]
    guides = [m for m in modules if isinstance(m, LocalChromaClsGuidance)]
    refiners = [m for m in modules if isinstance(m, SENetV2EnSimAM)]
    head = model.model[-1]
    assert len(cues) == 1 and len(guides) == 3 and len(refiners) == 2
    assert isinstance(head, ContrastClsFTSCDetect)
    assert not any(isinstance(m, ASFHighResFusion) for m in modules)
    assert model.stride.tolist() == [4.0, 8.0, 16.0]
    assert all(float(m.gamma) == 0.0 for m in guides)

    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        model(image)
        reg_zero = tuple(x.clone() for x in head.last_reg_features)
        cls_zero = tuple(x.clone() for x in head.last_cls_features)
        for guide in guides:
            guide.gamma.fill_(0.5)
        model(image)
        reg_mod = head.last_reg_features
        cls_mod = head.last_cls_features
    assert all(torch.equal(a, b) for a, b in zip(reg_zero, reg_mod))
    assert any(not torch.equal(a, b) for a, b in zip(cls_zero, cls_mod))

    model.train().zero_grad(set_to_none=True)
    loss = _tensor_sum(model(image))
    loss.backward()
    for guide in guides:
        assert guide.gamma.grad is not None and torch.isfinite(guide.gamma.grad)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.encoder.parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.gate.parameters())
