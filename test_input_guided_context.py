#!/usr/bin/env python3
"""Focused test: P3 refinement is guided by image+P2, never image+P3."""
import torch

from ultralytics.nn.modules.asrm import InputGuidedContextRefine, InputGuidedP2P3ContextRefine


def main() -> None:
    torch.manual_seed(42)
    # target=P3, peer=P2, guidance=P2, P4, P5, image
    module = InputGuidedContextRefine(
        [64, 32, 32, 128, 256, 3], embed_channels=16, debug=True
    ).train()
    inputs = [
        torch.randn(1, 64, 32, 32), torch.randn(1, 32, 64, 64),
        torch.randn(1, 32, 64, 64), torch.randn(1, 128, 16, 16),
        torch.randn(1, 256, 8, 8), torch.rand(1, 3, 256, 256),
    ]
    output, gate, attention_p2 = module(inputs)
    assert torch.equal(output, inputs[0]), "zero-init gamma must be exact identity"
    assert gate.shape == (1, 64, 32, 32)
    assert attention_p2.shape == (1, 1, 64, 64), "guidance must stay at P2 resolution"
    with torch.no_grad():
        module.gamma.fill_(0.1)
    module(inputs)[0].square().mean().backward()
    gradients = (
        module.guidance.srm.conv.weight.grad,
        module.guidance.p2_project.weight.grad,
        module.context_gate[-1].weight.grad,
        module.gamma.grad,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    print("OK: image+P2 guidance refines P3 through context with finite gradients")

    # New P2+P3-guided version: its P3 refiner must produce a genuine A3 map.
    p3_guided = InputGuidedP2P3ContextRefine(
        [64, 32, 64, 128, 256, 3], embed_channels=16, debug=True
    ).train()
    p3_inputs = [inputs[0], inputs[1], inputs[0], inputs[3], inputs[4], inputs[5]]
    p3_output, p3_gate, attention_p3 = p3_guided(p3_inputs)
    assert torch.equal(p3_output, p3_inputs[0])
    assert p3_gate.shape == (1, 64, 32, 32)
    assert attention_p3.shape == (1, 1, 32, 32), "P3 must use its own image+P3 prior"
    print("OK: independent image+P3 guidance produces A3 at P3 resolution")


if __name__ == "__main__":
    main()
