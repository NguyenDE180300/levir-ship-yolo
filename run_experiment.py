#!/usr/bin/env python3
"""Train and test one YOLO matrix cell on LEVIR-Ship, writing machine-readable JSON.

Mirrors Levir_ship_training/run_experiment.py (the mmdet matrix runner), adapted to
Ultralytics: one case is the stock P2-enabled YOLOv8 baseline, the other three are the
ASRM variants from model_cfg/ (see that folder's README for the architecture details).
All four cases share data, image size, batch size, epochs and model scale so results
are directly comparable; run once per (case, seed) and aggregate across seeds with
collect_results.py.

Hyperparameter defaults (imgsz=512, batch=8, pretrained=yolov8n.pt, 3 seeds via the
notebook loop) match the reference yolov8n LEVIR-Ship setup this builds on top of
(https://github.com/aduy2408/yolo_code, train_all_levir_yolov8n_p2_routing.py) so any
mAP delta is attributable to the ASRM architecture change, not a hyperparameter or
init-weights difference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # `ultralytics/` lives right next to this file
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, intersect_dicts  # noqa: E402

MODEL_CFG_DIR = ROOT / "model_cfg"


def profile_gflops(model, image_size: int) -> float:
    """Return 2*MACs in GFLOPs, with a THOP fallback for custom heads."""
    value = float(get_flops(model.model, imgsz=image_size))
    if value > 0:
        return value
    try:
        from thop import profile
        device = next(model.model.parameters()).device
        macs, _ = profile(model.model.eval(), inputs=(torch.zeros(1, 3, image_size, image_size, device=device),), verbose=False)
        return float(2.0 * macs / 1e9)
    except Exception:
        return 0.0
# All four share the same `scales:` block (defaults to 'n' when loaded bare, see
# model_cfg/README.md) so the matrix stays single-scale and comparable, matching how
# the mmdet matrix fixes ResNet-50 across all 8 cases instead of sweeping backbone size.
CONFIGS = {
    "aduy_yolov8n_dbss_full": str(MODEL_CFG_DIR / "yolov8_aduy_yolov8n_dbss_full.yaml"),
    "aduy_gap_factorized_k15": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15.yaml"),
    "aduy_gap_factorized_k15_nomosaic": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15.yaml"),
    "aduy_gap_factorized_k15_hsv_input": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_hsv_input.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_fpn_p2_fusion": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_fpn_p2_fusion.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_quality": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_quality.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_stem": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_stem.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_guided_p3_residual": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_guided_p3_residual.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_feature_filter": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_feature_filter.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_stem_f2": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_stem_f2.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_guided_p3_f2": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_guided_p3_f2.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_regularized_f2_fusion": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_regularized_f2_fusion.yaml"),
    "aduy_gap_factorized_k15_rgb_labb": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_labb.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_labb": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_labb.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance": str(
        MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance.yaml"
    ),
    "aduy_gap_factorized_k15_rgb_saturation_sagri_fpn_p2": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_sagri_fpn_p2.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_mssen_f2": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_mssen_f2.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_semantic_gate": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_cls_semantic_gate.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_edge_refine": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_cls_edge_refine.yaml"),
    "asrm_spg_fusion_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_asrm_spg_fusion_p2_gap_ftal_k15.yaml"
    ),
    "asrm_p4verify_residual_p2_gap_ftal_k15": str(
        MODEL_CFG_DIR / "yolov8_asrm_p4verify_residual_p2_gap_ftal_k15.yaml"
    ),
    "baseline_p2": str(MODEL_CFG_DIR / "yolov8n_p2_levir_baseline.yaml"),
    "p2_asrm_fusion": str(MODEL_CFG_DIR / "yolov8_p2_asrm_fusion.yaml"),  # V1, P2 only
    "p2p3_asrm_fusion": str(MODEL_CFG_DIR / "yolov8_p2p3_asrm_fusion.yaml"),  # V1, P2 + P3
    "input_asrm_auxiliary": str(MODEL_CFG_DIR / "yolov8_input_asrm_auxiliary.yaml"),  # V2, P2 only
    "input_asrm_guided_p2": str(MODEL_CFG_DIR / "yolov8_input_asrm_guided_p2.yaml"),  # V3, P2 only
    "input_asrm_guided_p2p3": str(MODEL_CFG_DIR / "yolov8_input_asrm_guided_p2p3.yaml"),  # V3, P2 + P3
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
    "asrm_detail_context_refine_p2p3": str(MODEL_CFG_DIR / "yolov8_asrm_detail_context_refine_p2p3.yaml"),
    "asrm_candidate_sparse_refine_p2p3": str(MODEL_CFG_DIR / "yolov8_asrm_candidate_sparse_refine_p2p3.yaml"),
    "carafe_p2": str(MODEL_CFG_DIR / "yolov8n_p2_carafe_p2.yaml"),
    "asrm_context_carafe_p2": str(MODEL_CFG_DIR / "yolov8_asrm_context_carafe_p2.yaml"),
    "asrm_context_p2guided_carafe_p2": str(MODEL_CFG_DIR / "yolov8_asrm_context_p2guided_carafe_p2.yaml"),
    "asrm_context_p2guided_carafe_spd_neck": str(MODEL_CFG_DIR / "yolov8_asrm_context_p2guided_carafe_spd_neck.yaml"),
    "p2guided_gate_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_gate_p3p4.yaml"),
    "p2guided_dynamic_conv_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_dynamic_conv_p3p4.yaml"),
    "p2guided_cross_attention_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_cross_attention_p3p4.yaml"),
    "p2guided_deform_conv_p3p4": str(MODEL_CFG_DIR / "yolov8_p2guided_deform_conv_p3p4.yaml"),
    "baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance_ftsc_f5_p2p3p4": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance_ftsc_f5_p2p3p4.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_contrast_input_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_input_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_rgb_scharr_input_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_scharr_input_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_rgb_scharr_p2p3p4_guidance_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_scharr_p2p3p4_guidance_ftsc_f5_p2p3p4.yaml"),
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_p2p3p4.yaml"),
    "baseline_factorized_k15_rgb_contrast_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4.yaml"),
    "baseline_factorized_k15_rgb_scharr_input_repc2f_gap_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_scharr_input_repc2f_gap_ftsc_f5_p2p3p4.yaml"),
    "baseline_factorized_k15_rgb_scharr_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_scharr_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4.yaml"),
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3": str(
        MODEL_CFG_DIR / "yolov8_baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr.yaml"
    ),
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr_cp2": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr.yaml"
    ),
    "baseline_gap_factorized_k15_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"
    ),
    "baseline_gap_factorized_k15_local_chroma_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_local_chroma_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"
    ),
    "baseline_gap_factorized_k15_contrast_cls_p2p3p4_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(
        MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_contrast_cls_p2p3p4_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"
    ),
    # Exact FTSC H2+ES1 graph. Its upstream YAML spells the F5-calibrated
    # head `Detect`; this fork exposes the equivalent implementation as FTSCDetect.
    "es1_ftsc_original_mosaic_p2p3": str(
        MODEL_CFG_DIR / "yolov8_es1_ftsc_original_mosaic_p2p3.yaml"
    ),
    "es1_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3.yaml"),
}

# Match aduy2408/yolo_code's Varroa `...gap_factorized_k15` supervision while
# preserving this repository's input-guided context-refine topology and P2--P5 Detect.
CASE_TRAIN_OVERRIDES = {
    "baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance_ftsc_f5_p2p3p4": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "optimizer": "AdamW", "lr0": 0.002, "mosaic": 0.0, "close_mosaic": 0,
    },
    "aduy_gap_factorized_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    # Augmentation ablation only: the exact P2-only GAP + FTAL k=1.5 case,
    # but without composing four samples into each mosaic training image.
    "aduy_gap_factorized_k15_nomosaic": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "mosaic": 0.0,
    },
    "aduy_gap_factorized_k15_hsv_input": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_fpn_p2_fusion": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_labb": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_quality": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_stem": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_guided_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_feature_filter": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_stem_f2": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_guided_p3_f2": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_regularized_f2_fusion": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_labb": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    # RGB+Saturation FPN-only/P2-only reference plus the exact image-guided
    # context-refine topology. The runner-wide mosaic=0 remains in force.
    "aduy_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "aduy_gap_factorized_k15_rgb_saturation_sagri_fpn_p2": {"factorized_tal_target": True, "factorized_tal_tau": .75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": .5, "factorized_tal_s_max": 32., "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True},
    "aduy_gap_factorized_k15_rgb_saturation_mssen_f2": {"factorized_tal_target": True, "factorized_tal_tau": .75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": .5, "factorized_tal_s_max": 32., "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True},
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_semantic_gate": {"factorized_tal_target": True, "factorized_tal_tau": .75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": .5, "factorized_tal_s_max": 32., "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": False},
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_edge_refine": {"factorized_tal_target": True, "factorized_tal_tau": .75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": .5, "factorized_tal_s_max": 32., "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True},
    "asrm_spg_fusion_p2_gap_ftal_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "asrm_p4verify_residual_p2_gap_ftal_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "input_guided_context_refine_p2p3_gap_ftal_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "input_guided_context_refine_p2_gap_ftal_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "input_guided_p34_context_refine_p2_gap_ftal_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
}

# RGB+contrast input ablation with the same GAP + Factorized TAL K15 protocol
# as the other GAP cases. The YAML applies ChannelAttention to P2/P3 and keeps
# the P4 semantic feature unchanged before FTSCDetect.
CASE_TRAIN_OVERRIDES["baseline_gap_factorized_k15_rgb_contrast_input_ftsc_f5_p2p3p4"] = {
    "factorized_tal_target": True, "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    "optimizer": "AdamW", "lr0": 0.002, "momentum": 0.9,
    "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "nbs": 64,
    "mosaic": 0.0, "close_mosaic": 0,
}

# Current RGB+contrast guidance experiment: explicitly enable Mosaic for the
# requested TinyPerson run while retaining its FTAL/GAP/FTSC protocol.
CASE_TRAIN_OVERRIDES["baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4"] = {
    "factorized_tal_target": True, "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    "optimizer": "MuSGD", "lr0": 0.01, "momentum": 0.9,
    "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "nbs": 64,
    "mosaic": 1.0, "close_mosaic": 50,
    "nms_method": "soft-gaussian", "soft_nms_sigma": 0.5, "soft_nms_min_score": 0.001,
}

# ASF + multi-scale Scharr/EnSimAM neck ablation. This keeps the current
# input-guided RGB/contrast backbone and F5 FTSC protocol, while replacing the
# neck with neighbor-scale ASF plus a BiFPN-style top-down/bottom-up path.
_ASF_MS_SCHARR_PROTOCOL = {
    "factorized_tal_target": True, "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    "optimizer": "AdamW", "lr0": 0.002, "momentum": 0.9,
    "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "nbs": 64, "amp": True,
    "mosaic": 0.0, "close_mosaic": 0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "rect": False,
}
CASE_TRAIN_OVERRIDES["baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr"] = dict(
    _ASF_MS_SCHARR_PROTOCOL,
    copy_paste_enabled=False,
)
CASE_TRAIN_OVERRIDES["baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr_cp2"] = dict(
    _ASF_MS_SCHARR_PROTOCOL,
    copy_paste_enabled=True, copy_paste_mode="cp2", copy_paste_unit="single",
    copy_paste_copies=2, copy_paste_p=0.5, copy_paste_scale=1.0,
    copy_paste_padding=0.0, copy_paste_blend="hard",
    copy_paste_placement="random", copy_paste_max_overlap=0.0,
    copy_paste_max_trials=30, copy_paste_allow_empty_target=True,
    copy_paste_allow_same_source=True,
)

# Clean deterministic-cue ablations retain their original P2/P3/P4 setup.
for _cue_case in (
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_contrast_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_scharr_input_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_scharr_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4",
):
    CASE_TRAIN_OVERRIDES[_cue_case] = {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "optimizer": "AdamW", "lr0": 0.002,
        "mosaic": 0.0, "close_mosaic": 0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "rect": False,
    }

# One isolated H2/A0 case.  It alone uses P2/P3 prediction and locks every
# optimizer, scheduler and augmentation value from the H2/ES1 protocol.
CASE_TRAIN_OVERRIDES["baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3"] = {
    "factorized_tal_target": True, "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    "optimizer": "AdamW", "lr0": 0.002, "momentum": 0.9,
    "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "nbs": 64,
    "mosaic": 0.0, "close_mosaic": 0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "rect": False,
}

# Exact FTSC H2+ES1 protocol: no standalone H2 run.  The ES1 graph is copied
# from the FTSC repository; later cases add one component cumulatively.
_ES1_A0 = {
    "optimizer": "AdamW", "lr0": 0.002, "momentum": 0.9,
    "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0, "nbs": 64,
    "amp": True,
    "mosaic": 0.0, "close_mosaic": 0,
    # A0 geometric/colour policy from the FTSC runner. Mosaic remains off
    # deliberately: all new project experiments use the no-mosaic protocol.
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "rect": False,
}
# Exact optimization/augmentation mode used by FTSC's ES1_MOSAIC control. It
# deliberately remains separate from the project's no-mosaic ES1 ablations.
_ES1_FTSC_ORIGINAL = {
    **_ES1_A0,
    "mosaic": 1.0,
    "close_mosaic": 10,
}
CASE_TRAIN_OVERRIDES["es1_ftsc_original_mosaic_p2p3"] = _ES1_FTSC_ORIGINAL
for _es1_case in (
    "es1_ftsc_f5_p2p3",
    "es1_ftal_k15_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3",
):
    CASE_TRAIN_OVERRIDES[_es1_case] = dict(_ES1_A0)
for _es1_ftal_case in (
    "es1_ftal_k15_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3",
    "es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3",
):
    CASE_TRAIN_OVERRIDES[_es1_ftal_case].update({
        "factorized_tal_target": True, "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    })
# Cases with parameter-free input transforms/taps shift backbone layer indices versus
# a stock yolov8n.pt checkpoint (by one or, for the RGB+ASRM case, two layers; see
# model_cfg/README.md). Ultralytics' own `Model.load()` matches pretrained weights by
# exact layer name + shape (`ultralytics.nn.tasks.BaseModel.load` -> `intersect_dicts`),
# so a plain `.load()` on a shifted model transfers zero backbone tensors. `baseline_p2`
# and `p2_asrm_fusion` are NOT shifted (their P2ASRMFusion tap is appended after the
# backbone, layers 0-9 stay at stock indices) and use the plain path instead.
def load_pretrained(model: YOLO, case: str, pretrained: str) -> dict[str, int]:
    """Transfer-learn from a stock Ultralytics checkpoint (e.g. yolov8n.pt).

    Mirrors the `remap_dbss_backbone` pattern in aduy2408/yolo_code's
    train_all_levir_yolov8n_p2_routing.py: when a custom module shifts backbone layer
    indices, pretrained weights must be remapped by name before loading, otherwise
    Ultralytics' name+shape intersection silently transfers nothing for the backbone.
    """
    if not pretrained:
        return {"total": 0}
    source = YOLO(pretrained)
    source_state = source.model.float().state_dict()
    target_state = model.model.state_dict()
    matched = intersect_dicts(source_state, target_state, smart_transfer=True)
    model.load(pretrained, smart_transfer=True)
    remapped = {}
    input_conv_adapted = 0
    repc2f_reparameterized = 0
    # Parameter-free input transforms/taps shift stock backbone layers. Load
    # their tensors explicitly after the standard direct load. Most historical
    # configs have one prefix layer; the RGB+Saturation+ASRM case has both an
    # Identity raw-image tap and RGBColorAugment, hence a two-layer shift.
    if model.model.model[0].__class__.__name__ in {"Identity", "RGBToHSV", "RGBColorAugment"}:
        backbone_offset = 0
        for layer in model.model.model:
            if layer.__class__.__name__ in {"Identity", "RGBToHSV", "RGBColorAugment", "RGBExplicitCue"}:
                backbone_offset += 1
            else:
                break
        for source_key, value in source_state.items():
            parts = source_key.split(".")
            if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
                continue
            layer_index = int(parts[1])
            if not 0 <= layer_index <= 9:
                continue
            parts[1] = str(layer_index + backbone_offset)
            target_key = ".".join(parts)
            # RepC2f replaces Bottleneck.cv1 by RepConv. Transfer the stock
            # 3x3 Conv+BN into RepConv's primary branch, then disable its
            # auxiliary 1x1/identity branches. This makes its initial forward
            # equivalent to the pretrained C2f branch instead of silently
            # leaving most of the backbone random.
            rep_target_key = None
            if "repc2f_gap" in case:
                if ".m." in target_key and ".cv1.conv." in target_key:
                    rep_target_key = target_key.replace(".cv1.conv.", ".cv1.conv1.conv.")
                elif ".m." in target_key and ".cv1.bn." in target_key:
                    rep_target_key = target_key.replace(".cv1.bn.", ".cv1.conv1.bn.")
            selected_key = target_key if target_key in target_state else rep_target_key
            if selected_key not in target_state:
                continue
            if target_state[selected_key].shape == value.shape:
                remapped[selected_key] = value
                if rep_target_key is not None:
                    repc2f_reparameterized += 1
                    if rep_target_key.endswith(".cv1.conv1.conv.weight"):
                        prefix = rep_target_key.removesuffix(".conv1.conv.weight")
                        for suffix in (".conv2.conv.weight", ".conv2.bn.weight", ".bn.weight"):
                            reset_key = prefix + suffix
                            if reset_key in target_state:
                                remapped[reset_key] = torch.zeros_like(target_state[reset_key])
            elif (target_key == f"model.{backbone_offset}.conv.weight" and value.ndim == 4
                  and value.shape[0] == target_state[target_key].shape[0]
                  and value.shape[1] == 3 and value.shape[2:] == target_state[target_key].shape[2:]
                  and target_state[target_key].shape[1] > 3):
                # Preserve RGB filters exactly; initialize each added cue with their mean response.
                expanded = target_state[target_key].clone()
                expanded[:, :3] = value
                expanded[:, 3:] = value.mean(dim=1, keepdim=True).expand_as(expanded[:, 3:])
                remapped[target_key] = expanded
                input_conv_adapted += 1
        model.model.load_state_dict(remapped, strict=False)
        if not remapped:
            raise RuntimeError(f"{case}: shifted backbone received no pretrained remap tensors")
    report = {
        "total": len(matched) + len(remapped),
        "direct": len(matched),
        "backbone_remapped": len(remapped),
        "input_conv_adapted": input_conv_adapted,
        "repc2f_reparameterized": repc2f_reparameterized,
        "method": "aduy_smart_transfer_plus_shifted_backbone_remap",
    }
    print(f"Pretrained transfer from {pretrained} (case={case}): {report}")
    return report


def collect_learned_guidance_scales(model: torch.nn.Module) -> dict[str, float]:
    """Expose learned zero-init guidance scales in each completed run JSON."""
    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SRMF2Guidance":
            scales[f"{name}.gamma_srm"] = float(module.effective_gamma().detach().cpu())
            if getattr(module, "use_saturation_guidance", False):
                scales[f"{name}.gamma_saturation"] = float(module.gamma_saturation_raw.detach().cpu())
        elif module.__class__.__name__ == "ExplicitCueGuidance":
            scales[f"{name}.gamma"] = float(module.gamma.detach().cpu())
            for stat, value in getattr(module, "last_attention_stats", {}).items():
                scales[f"{name}.attention_{stat}"] = float(value)
    return scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--pretrained", default="yolov8n.pt",
                        help="checkpoint to transfer-learn the backbone from; pass '' for from-scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=None,
                        help="optional initial learning-rate override; uses the standard hyperparameter value when omitted")
    parser.add_argument("--force-rerun", action="store_true",
                        help="retrain even when a previous best.pt exists")
    parser.add_argument("--tiny-data-root", type=Path, default=None,
                        help="optional extracted TinyPerson root; enables merged-window AP50/AP75/Tiny1/2/3/Tiny/Small/Medium")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")

    work_root = args.work_root.resolve()
    run_dir = work_root / args.case / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model = YOLO(CONFIGS[args.case])
    transfer_report = load_pretrained(model, args.case, args.pretrained)
    # Profile the configured architecture before training. This uses the same
    # image size as the experiment and is therefore directly comparable across cases.
    gflops = profile_gflops(model, args.image_size)
    # Experiment policy from 2026-09-08 onward: train every newly launched case
    # without mosaic. Historical result folders retain their original settings.
    train_overrides = {"mosaic": 0.0, **CASE_TRAIN_OVERRIDES.get(args.case, {})}
    if args.lr0 is not None:
        train_overrides["lr0"] = args.lr0
    existing_best = run_dir / "weights" / "best.pt"
    if existing_best.is_file() and not args.force_rerun:
        print(f"Reusing completed checkpoint for post-train evaluation: {existing_best}")
    else:
        model.train(
            data=str(args.data_yaml.resolve()),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.image_size,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            project=str(work_root / args.case),
            name=f"seed_{args.seed}",
            exist_ok=True,
            plots=False,
            verbose=True,
            **train_overrides,
        )
    parameter_count = sum(p.numel() for p in model.model.parameters())
    trainable_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    best_path = existing_best if existing_best.is_file() else Path(model.trainer.best)
    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(
        data=str(args.data_yaml.resolve()),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        iou=0.5,  # Match aduy2408/yolo_code's LEVIR GAP+FTAL evaluation protocol.
        nms_method=train_overrides.get("nms_method", "hard"),
        soft_nms_sigma=train_overrides.get("soft_nms_sigma", 0.5),
        soft_nms_min_score=train_overrides.get("soft_nms_min_score", 0.001),
        project=str(work_root / args.case),
        name=f"seed_{args.seed}_val",
        exist_ok=True,
        plots=False,
    )
    test_metrics = best_model.val(
        data=str(args.data_yaml.resolve()),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        iou=0.5,  # Match aduy2408/yolo_code's LEVIR GAP+FTAL evaluation protocol.
        nms_method=train_overrides.get("nms_method", "hard"),
        soft_nms_sigma=train_overrides.get("soft_nms_sigma", 0.5),
        soft_nms_min_score=train_overrides.get("soft_nms_min_score", 0.001),
        project=str(work_root / args.case),
        name=f"seed_{args.seed}_test",
        exist_ok=True,
        plots=False,
    )
    merged_tiny_metrics = {}
    if args.tiny_data_root is not None:
        import subprocess
        evaluator_out = run_dir / "tinybenchmark_merged"
        subprocess.run([
            sys.executable, str(ROOT / "evaluate_tinybenchmark.py"),
            "--weights", str(best_path),
            "--test-images", str(args.data_yaml.resolve().parent / "images" / "test"),
            "--data-root", str(args.tiny_data_root.resolve()), "--out-dir", str(evaluator_out),
            "--image-size", str(args.image_size), "--batch-size", str(args.batch_size),
        ], check=True)
        merged_tiny_metrics = json.loads((evaluator_out / "tinybenchmark_merged_metrics.json").read_text())
    learned_guidance = collect_learned_guidance_scales(best_model.model)
    payload = {
        "case": args.case,
        "seed": args.seed,
        "pretrained": args.pretrained or None,
        "pretrained_transfer": transfer_report,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "gflops": gflops,
        "tinybenchmark_merged": merged_tiny_metrics,
        "runtime_seconds": time.time() - started,
        "best_checkpoint": str(best_path),
        "learned_guidance": learned_guidance,
        "metrics": {
            "val": {
                "bbox_mAP": float(val_metrics.box.map),
                "bbox_mAP_50": float(val_metrics.box.map50),
                "bbox_mAP_75": float(val_metrics.box.map75),
                "precision": float(val_metrics.box.mp),
                "recall": float(val_metrics.box.mr),
            },
            "test": {
                "bbox_mAP": float(test_metrics.box.map),
                "bbox_mAP_50": float(test_metrics.box.map50),
                "bbox_mAP_75": float(test_metrics.box.map75),
                "precision": float(test_metrics.box.mp),
                "recall": float(test_metrics.box.mr),
            },
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    # Keep the historical filename as the completion marker while storing both
    # validation and held-out test metrics in the same self-contained payload.
    (run_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
