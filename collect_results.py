#!/usr/bin/env python3
"""Collect all YOLO matrix runs (4 cases x N seeds) into raw + aggregate CSVs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

CASES = ("aduy_yolov8n_dbss_full", "aduy_gap_factorized_k15", "aduy_gap_factorized_k15_nomosaic", "aduy_gap_factorized_k15_hsv_input", "aduy_gap_factorized_k15_rgb_saturation", "aduy_gap_factorized_k15_rgb_saturation_fpn_p2_fusion", "aduy_gap_factorized_k15_rgb_saturation_quality", "aduy_gap_factorized_k15_rgb_saturation_stem", "aduy_gap_factorized_k15_rgb_saturation_guided_p3_residual", "aduy_gap_factorized_k15_rgb_saturation_feature_filter", "aduy_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual", "aduy_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual", "aduy_gap_factorized_k15_rgb_saturation_stem_f2", "aduy_gap_factorized_k15_rgb_saturation_guided_p3_f2", "aduy_gap_factorized_k15_rgb_saturation_regularized_f2_fusion", "aduy_gap_factorized_k15_rgb_labb", "aduy_gap_factorized_k15_rgb_saturation_labb", "aduy_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine", "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance", "aduy_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance", "aduy_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg", "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_guidance", "aduy_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance", "asrm_spg_fusion_p2_gap_ftal_k15", "asrm_p4verify_residual_p2_gap_ftal_k15", "baseline_p2", "p2_asrm_fusion", "p2p3_asrm_fusion",
         "input_asrm_auxiliary", "input_asrm_guided_p2", "input_asrm_guided_p2p3",
         "input_guided_context_refine_p2p3", "input_guided_p2p3_context_refine_p2p3",
         "input_guided_context_refine_p2p3_gap_ftal_k15",
         "input_guided_context_refine_p2_gap_ftal_k15",
         "asrm_detail_context_refine_p2p3", "asrm_candidate_sparse_refine_p2p3",
         "carafe_p2", "asrm_context_carafe_p2", "asrm_context_p2guided_carafe_p2",
         "asrm_context_p2guided_carafe_spd_neck", "p2guided_gate_p3p4",
         "p2guided_dynamic_conv_p3p4", "p2guided_cross_attention_p3p4",
         "p2guided_deform_conv_p3p4")
CASES += (
    "baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance_ftsc_f5_p2p3p4",
    "aduy_gap_factorized_k15_rgb_saturation_sagri_fpn_p2",
    "aduy_gap_factorized_k15_rgb_saturation_mssen_f2",
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_semantic_gate",
    "aduy_gap_factorized_k15_rgb_saturation_srm3_cls_edge_refine",
    "baseline_gap_factorized_k15_rgb_contrast_input_ftsc_f5_p2p3p4",
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4",
    "baseline_gap_factorized_k15_rgb_scharr_input_ftsc_f5_p2p3p4",
    "baseline_gap_factorized_k15_rgb_scharr_p2p3p4_guidance_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_contrast_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_scharr_input_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_scharr_p2p3p4_guidance_repc2f_gap_ftsc_f5_p2p3p4",
    "baseline_factorized_k15_rgb_contrast_input_repc2f_gap_ftsc_f5_a0_h2_p2p3",
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr",
    "baseline_gap_factorized_k15_rgb_contrast_p2p3p4_guidance_ftsc_f5_p2p3p4_asf_ms_scharr_cp2",
    "baseline_gap_factorized_k15_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4",
    "baseline_gap_factorized_k15_local_chroma_contrast_shared_p2p3_senetv2_ensimam_asf_ftsc_f5_p2p3p4",
    "baseline_gap_factorized_k15_contrast_cls_p2p3p4_senetv2_ensimam_asf_ftsc_f5_p2p3p4",
)
METRICS = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "precision", "recall")
# Metrics emitted by evaluate_tinybenchmark.py. Keep these separate from the
# COCO-style bbox_mAP fields because TinyBenchmark uses its own merged-window
# evaluator and area buckets.
TINY_METRICS = (
    "AP50", "AP75", "mAP50-75",
    "AP50_tiny1", "AP_tiny1", "AP50_tiny2", "AP_tiny2",
    "AP50_tiny3", "AP_tiny3", "AP50_tiny", "AP_tiny",
    "AP50_small", "AP_small", "AP50_medium", "AP_medium",
)
SPLITS = ("val", "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES,
                        help="subset of cases to collect; default is all registered cases. "
                             "delta_bbox_mAP_mean is only computed if baseline_p2 is included.")
    parser.add_argument("--output", type=Path, required=True,
                        help="raw per-run CSV path; the aggregate CSV is written next to it as "
                             "<output>_aggregate.csv")
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        for seed in args.seeds:
            path = args.work_root / case / f"seed_{seed}" / "test_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing completed run: {path}")
            raw = json.loads(path.read_text())
            row = {k: raw[k] for k in ("case", "seed", "pretrained", "epochs", "image_size",
                   "batch_size", "parameters", "trainable_parameters", "runtime_seconds",
                   "best_checkpoint", "torch", "cuda", "gpu")}
            row["gflops"] = raw.get("gflops")
            for key, value in raw.get("learned_guidance", {}).items():
                row[f"learned_{key.replace('.', '_')}"] = value
            metrics = raw["metrics"]
            if "val" not in metrics or "test" not in metrics:
                raise ValueError(
                    f"{path} predates val+test collection; rerun this seed with the current runner"
                )
            for split in SPLITS:
                row.update({f"{split}_{key}": metrics[split].get(key) for key in METRICS})
            tiny = raw.get("tinybenchmark_merged", {}) or {}
            for key in TINY_METRICS:
                value = tiny.get(key)
                row[f"test_tiny_{key}"] = float(value) if value is not None else None
            rows.append(row)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved raw runs: {args.output}")

    aggregate_rows = []
    baseline_values = frame.loc[frame.case == "baseline_p2", "test_bbox_mAP"]
    baseline_mean = statistics.fmean(baseline_values) if len(baseline_values) else None
    reference_values = frame.loc[
        frame.case == "input_guided_context_refine_p2p3", "test_bbox_mAP"
    ]
    reference_mean = statistics.fmean(reference_values) if len(reference_values) else None
    for case in args.cases:
        group = frame[frame.case == case]
        record = {
            "case": case,
            "runs": len(group),
            "parameters": int(group["parameters"].iloc[0]),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
            "gflops": float(group["gflops"].iloc[0]),
        }
        for split in SPLITS:
            for key in METRICS:
                column = f"{split}_{key}"
                values = group[column].astype(float).tolist()
                record[f"{column}_mean"] = statistics.fmean(values)
                record[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for key in TINY_METRICS:
            column = f"test_tiny_{key}"
            values = group[column].dropna().astype(float).tolist()
            if not values:
                record[f"{column}_mean"] = None
                record[f"{column}_std"] = None
            else:
                record[f"{column}_mean"] = statistics.fmean(values)
                record[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for column in (c for c in group.columns if c.startswith("learned_") and group[c].notna().any()):
            values = group[column].dropna().astype(float).tolist()
            record[f"{column}_mean"] = statistics.fmean(values)
            record[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        record["delta_test_bbox_mAP_mean"] = (
            record["test_bbox_mAP_mean"] - baseline_mean if baseline_mean is not None else None
        )
        record["delta_vs_input_guided_context_test_bbox_mAP_mean"] = (
            record["test_bbox_mAP_mean"] - reference_mean if reference_mean is not None else None
        )
        aggregate_rows.append(record)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = args.output.with_name(args.output.stem + "_aggregate" + args.output.suffix)
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.to_string(index=False))
    print(f"\nSaved aggregate (mean +/- std over seeds): {aggregate_path}")


if __name__ == "__main__":
    main()
