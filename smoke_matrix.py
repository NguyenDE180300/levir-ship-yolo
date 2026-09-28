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
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance.yaml"),
    "aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance": str(MODEL_CFG_DIR / "yolov8_aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance.yaml"),
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
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3": str(MODEL_CFG_DIR / "yolov8_baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3.yaml"),
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr.yaml"),
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr_cp2": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr.yaml"),
    "baseline_gap_factorized_k15_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_local_chroma_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_local_chroma_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_local_chroma_contrast_p2p3_nosasf_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_local_chroma_contrast_p2p3_nosasf_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_local_chroma_contrast_cls_guidance_p2p3_nosasf_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_local_chroma_contrast_cls_guidance_p2p3_nosasf_ftsc_f5_p2p3p4.yaml"),
    "baseline_gap_factorized_k15_contrast_cls_p2p3p4_senetv2_ensimam_asf_ftsc_f5_p2p3p4": str(MODEL_CFG_DIR / "yolov8_baseline_gap_factorized_k15_contrast_cls_p2p3p4_senetv2_ensimam_asf_ftsc_f5_p2p3p4.yaml"),
    "es1_ftsc_original_mosaic_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftsc_original_mosaic_p2p3.yaml"),
    "es1_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_repc2f_ftsc_f5_p2p3.yaml"),
    "es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3": str(MODEL_CFG_DIR / "yolov8_es1_ftal_k15_gap_repc2f_rgb_contrast_ftsc_f5_p2p3.yaml"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--cases", nargs="+", choices=sorted(CONFIGS), default=None)
    args = p.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selected = args.cases or CONFIGS.keys()
    for name in selected:
        cfg = CONFIGS[name]
        model = DetectionModel(cfg, ch=3, nc=1, verbose=False).to(device)
        model.train()
        head = model.model[-1]
        expected_strides = [4.0, 8.0] if (name.endswith("_a0_h2_p2p3") or name.startswith("es1_")) else [4.0, 8.0, 16.0]
        assert list(head.stride.float().tolist()) == expected_strides
        assert getattr(head, "ftsc_calibrator", None) is not None
        # Guidance cases keep RGB as a three-channel input; only explicit
        # input-channel variants adapt the stem to four channels.
        is_input = "_input_" in name and "_guidance_" not in name
        is_guidance = "_guidance_" in name
        first_conv = next(module for module in model.modules() if module.__class__.__name__ == "Conv")
        if is_input:
            assert first_conv.conv.in_channels == 4, (name, first_conv.conv.in_channels)
        if is_guidance:
            guidance = [module for module in model.modules() if module.__class__.__name__ == "ExplicitCueGuidance"]
            assert len(guidance) == 3 and all(float(module.gamma.detach()) == 0.0 for module in guidance)
        if "repc2f_gap" in name or "gap_repc2f" in name:
            gaps = [module for module in model.modules() if module.__class__.__name__ == "ChannelAttention"]
            assert len(gaps) == 2, (name, len(gaps))
        x = torch.rand(2, 3, args.image_size, args.image_size, device=device)
        out = model(x)
        _sum_all(out).backward()
        if is_guidance:
            assert all(module.gamma.grad is not None for module in guidance), name
        n_params = sum(p_.numel() for p_ in model.parameters())
        print(f"OK {name:24s} params={n_params:,}")


if __name__ == "__main__":
    main()
