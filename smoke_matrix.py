#!/usr/bin/env python3
"""Build all 4 YOLO matrix cases and verify a forward pass, before spending GPU time."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def _sum_all(x):
    if torch.is_tensor(x):
        return x.float().sum()
    if isinstance(x, dict):
        return sum(_sum_all(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_sum_all(v) for v in x)
    raise TypeError(f"unexpected output type {type(x)}")

sys.path.insert(0, str(ROOT))
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

MODEL_CFG_DIR = ROOT / "model_cfg"
CONFIGS = {
    "baseline_yolov8n_dbss_full": str(MODEL_CFG_DIR / "yolov8_baseline_yolov8n_dbss_full.yaml"),
    "baseline_gap_factorized_k15": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15.yaml"),
    "baseline_gap_factorized_k15_nomosaic": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15.yaml"),
    "baseline_gap_factorized_k15_hsv_input": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_hsv_input.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_fpn_p2_fusion": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_fpn_p2_fusion.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_quality": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_quality.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_stem": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_stem.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_guided_p3_residual": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_guided_p3_residual.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_feature_filter": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_feature_filter.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_stem_f2": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_stem_f2.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_guided_p3_f2": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_guided_p3_f2.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_regularized_f2_fusion": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_regularized_f2_fusion.yaml"),
    "baseline_gap_factorized_k15_rgb_labb": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_labb.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_labb": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_labb.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance.yaml"),
    "asrm_spg_fusion_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_asrm_spg_fusion_p2_gap_ftal_k15.yaml"
    ),
    "asrm_p4verify_residual_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_asrm_p4verify_residual_p2_gap_ftal_k15.yaml"
    ),
    "baseline_p2": str(MODEL_CFG_DIR / "yolov8n_p2_levir_baseline.yaml"),
    "p2_asrm_fusion": str(MODEL_CFG_DIR / "yolov8_p2_asrm_fusion.yaml"),
    "p2p3_asrm_fusion": str(MODEL_CFG_DIR / "yolov8_p2p3_asrm_fusion.yaml"),
    "input_asrm_auxiliary": str(MODEL_CFG_DIR / "yolov8_input_asrm_auxiliary.yaml"),
    "input_asrm_guided_p2": str(MODEL_CFG_DIR / "yolov8_input_asrm_guided_p2.yaml"),
    "input_asrm_guided_p2p3": str(MODEL_CFG_DIR / "yolov8_input_asrm_guided_p2p3.yaml"),
    "input_guided_context_refine_p2p3": str(MODEL_CFG_DIR / "yolov8_input_guided_context_refine_p2p3.yaml"),
    "input_guided_p34_context_refine_p2p3": str(
        MODEL_CFG_DIR / "yolov8_input_guided_p34_context_refine_p2p3.yaml"
    ),
    "input_guided_context_refine_p2p3_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_input_guided_context_refine_p2p3_gap_ftal_k15.yaml"
    ),
    "input_guided_context_refine_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_input_guided_context_refine_p2_gap_ftal_k15.yaml"
    ),
    "input_guided_p34_context_refine_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_input_guided_p34_context_refine_p2_gap_ftal_k15.yaml"
    ),
    "input_guided_p2p3_context_refine_p2p3": str(MODEL_CFG_DIR / "yolov8_input_guided_p2p3_context_refine_p2p3.yaml"),
    "asrm_detail_context_refine_p2p3": str(
        MODEL_CFG_DIR / "yolov8_asrm_detail_context_refine_p2p3.yaml"
    ),
    "asrm_candidate_sparse_refine_p2p3": str(
        MODEL_CFG_DIR / "yolov8_asrm_candidate_sparse_refine_p2p3.yaml"
    ),
    "carafe_p2": str(MODEL_CFG_DIR / "yolov8n_p2_carafe_p2.yaml"),
    "asrm_context_carafe_p2": str(MODEL_CFG_DIR / "yolov8_asrm_context_carafe_p2.yaml"),
    "asrm_context_p2guided_carafe_p2": str(MODEL_CFG_DIR / "yolov8_asrm_context_p2guided_carafe_p2.yaml"),
    "asrm_context_p2guided_carafe_spd_neck": str(MODEL_CFG_DIR / "yolov8_asrm_context_p2guided_carafe_spd_neck.yaml"),
    "p2guided_gate_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_gate_p3p4.yaml"),
    "p2guided_dynamic_conv_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_dynamic_conv_p3p4.yaml"),
    "p2guided_cross_attention_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_cross_attention_p3p4.yaml"),
    "p2guided_deform_conv_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_deform_conv_p3p4.yaml"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for name, cfg in CONFIGS.items():
        model = DetectionModel(cfg, ch=3, nc=1, verbose=False).to(device)
        model.train()
        x = torch.rand(2, 3, args.image_size, args.image_size, device=device)
        out = model(x)
        _sum_all(out).backward()
        n_params = sum(p_.numel() for p_ in model.parameters())
        print(f"OK {name:24s} params={n_params:,}")


if __name__ == "__main__":
    main()
