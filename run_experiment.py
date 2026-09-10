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
(the reference implementation, train_all_levir_yolov8n_p2_routing.py) so any
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
# All four share the same `scales:` block (defaults to 'n' when loaded bare, see
# model_cfg/README.md) so the matrix stays single-scale and comparable, matching how
# the mmdet matrix fixes ResNet-50 across all 8 cases instead of sweeping backbone size.
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
}

# Match reference implementation's Varroa `...gap_factorized_k15` supervision while
# preserving this repository's input-guided context-refine topology and P2--P5 Detect.
CASE_TRAIN_OVERRIDES = {
    "baseline_gap_factorized_k15": {
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
    "baseline_gap_factorized_k15_nomosaic": {
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
    "baseline_gap_factorized_k15_hsv_input": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_labb": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_quality": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_stem": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_guided_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_feature_filter": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_stem_f2": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_guided_p3_f2": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_regularized_f2_fusion": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0, "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    "baseline_gap_factorized_k15_rgb_saturation_labb": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True,
    },
    # RGB+Saturation FPN-only/P2-only reference plus the exact image-guided
    # context-refine topology. The runner-wide mosaic=0 remains in force.
    "baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine": {
        "factorized_tal_target": True, "factorized_tal_tau": 0.75, "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5, "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
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
# Cases with parameter-free input transforms/taps shift backbone layer indices versus
# a stock yolov8n.pt checkpoint (by one or, for the RGB+ASRM case, two layers; see
# model_cfg/README.md). Ultralytics' own `Model.load()` matches pretrained weights by
# exact layer name + shape (`ultralytics.nn.tasks.BaseModel.load` -> `intersect_dicts`),
# so a plain `.load()` on a shifted model transfers zero backbone tensors. `baseline_p2`
# and `p2_asrm_fusion` are NOT shifted (their P2ASRMFusion tap is appended after the
# backbone, layers 0-9 stay at stock indices) and use the plain path instead.
def load_pretrained(model: YOLO, case: str, pretrained: str) -> dict[str, int]:
    """Transfer-learn from a stock Ultralytics checkpoint (e.g. yolov8n.pt).

    Mirrors the `remap_dbss_backbone` pattern in reference implementation's
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
    # Parameter-free input transforms/taps shift stock backbone layers. Load
    # their tensors explicitly after the standard direct load. Most historical
    # configs have one prefix layer; the RGB+Saturation+ASRM case has both an
    # Identity raw-image tap and RGBColorAugment, hence a two-layer shift.
    if model.model.model[0].__class__.__name__ in {"Identity", "RGBToHSV", "RGBColorAugment"}:
        backbone_offset = 0
        for layer in model.model.model:
            if layer.__class__.__name__ in {"Identity", "RGBToHSV", "RGBColorAugment"}:
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
            if target_key not in target_state:
                continue
            if target_state[target_key].shape == value.shape:
                remapped[target_key] = value
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
        "method": "reference_smart_transfer_plus_shifted_backbone_remap",
    }
    print(f"Pretrained transfer from {pretrained} (case={case}): {report}")
    return report


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
    gflops = float(get_flops(model.model, imgsz=args.image_size))
    # Experiment policy from 2026-09-08 onward: train every newly launched case
    # without mosaic. Historical result folders retain their original settings.
    train_overrides = {"mosaic": 0.0, **CASE_TRAIN_OVERRIDES.get(args.case, {})}
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

    best_model = YOLO(str(model.trainer.best))
    val_metrics = best_model.val(
        data=str(args.data_yaml.resolve()),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        iou=0.5,  # Match reference implementation's LEVIR GAP+FTAL evaluation protocol.
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
        iou=0.5,  # Match reference implementation's LEVIR GAP+FTAL evaluation protocol.
        project=str(work_root / args.case),
        name=f"seed_{args.seed}_test",
        exist_ok=True,
        plots=False,
    )
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
        "runtime_seconds": time.time() - started,
        "best_checkpoint": str(Path(model.trainer.best)),
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
