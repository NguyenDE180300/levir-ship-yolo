#!/usr/bin/env python3
"""Focused checks for ASRM candidate verification and sparse refinement."""
from copy import deepcopy

import torch

from ultralytics.nn.modules.asrm import ASRMCandidateSparseRefine


def main() -> None:
    torch.manual_seed(42)
    module = ASRMCandidateSparseRefine(
        [32, 64, 128, 256, 3], embed_channels=32,
        num_prototypes=8, debug=True,
    ).train()
    inputs = [
        torch.randn(1, 32, 128, 128),
        torch.randn(1, 64, 64, 64),
        torch.randn(1, 128, 32, 32),
        torch.randn(1, 256, 16, 16),
        torch.rand(1, 3, 512, 512),
    ]
    output, mask, confidence = module(inputs)
    assert output.shape == inputs[0].shape
    assert torch.equal(output, inputs[0]), "gamma=0 must be an exact identity"
    assert mask.shape == (1, 1, 128, 128)
    assert confidence.shape == (1, 96)
    active_fraction = (mask > 0).float().mean().item()
    assert 0.0 < active_fraction < 0.5, active_fraction
    deepcopy(module).eval()

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.gamma.fill_(0.1)
    enhanced, _, _ = module(inputs)
    enhanced.square().mean().backward()
    required = {
        "srm": module.srm.conv.weight.grad,
        "prototype_assignment": module.prototype_assignment.weight.grad,
        "verifier": module.verifier[-1].weight.grad,
        "refine": module.refine[-1].weight.grad,
        "gamma": module.gamma.grad,
    }
    assert all(value is not None and torch.isfinite(value).all() for value in required.values()), required
    print(f"OK shape={tuple(output.shape)} active_mask={active_fraction:.4f} "
          f"valid_candidates={(confidence > 0).sum().item()}/96")


if __name__ == "__main__":
    main()
