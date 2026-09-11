#!/usr/bin/env python3
"""Collect all YOLO matrix runs (4 cases x N seeds) into raw + aggregate CSVs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

CASES = ("baseline_yolov8n_dbss_full", "baseline_gap_factorized_k15", "baseline_gap_factorized_k15_nomosaic", "baseline_gap_factorized_k15_hsv_input", "baseline_gap_factorized_k15_rgb_saturation", "baseline_gap_factorized_k15_rgb_saturation_fpn_p2_fusion", "baseline_gap_factorized_k15_rgb_saturation_quality", "baseline_gap_factorized_k15_rgb_saturation_stem", "baseline_gap_factorized_k15_rgb_saturation_guided_p3_residual", "baseline_gap_factorized_k15_rgb_saturation_feature_filter", "baseline_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual", "baseline_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual", "baseline_gap_factorized_k15_rgb_saturation_stem_f2", "baseline_gap_factorized_k15_rgb_saturation_guided_p3_f2", "baseline_gap_factorized_k15_rgb_saturation_regularized_f2_fusion", "baseline_gap_factorized_k15_rgb_labb", "baseline_gap_factorized_k15_rgb_saturation_labb", "baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine", "baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance", "baseline_gap_factorized_k15_rgb_saturation_k4k12k16_f2_guidance", "baseline_gap_factorized_k15_rgb_saturation_srm3_f2_guidance_reg", "baseline_gap_factorized_k15_rgb_saturation_srm3_cls_guidance", "baseline_gap_factorized_k15_rgb_saturation_srm3_saturation_cls_guidance", "asrm_spg_fusion_p2_gap_ftal_k15", "asrm_p4verify_residual_p2_gap_ftal_k15", "baseline_p2", "p2_asrm_fusion", "p2p3_asrm_fusion",
         "input_asrm_auxiliary", "input_asrm_guided_p2", "input_asrm_guided_p2p3",
         "input_guided_context_refine_p2p3", "input_guided_p2p3_context_refine_p2p3",
         "input_guided_context_refine_p2p3_gap_ftal_k15",
         "input_guided_context_refine_p2_gap_ftal_k15",
         "asrm_detail_context_refine_p2p3", "asrm_candidate_sparse_refine_p2p3",
         "carafe_p2", "asrm_context_carafe_p2", "asrm_context_p2guided_carafe_p2",
         "asrm_context_p2guided_carafe_spd_neck", "p2guided_gate_p3p4",
         "p2guided_dynamic_conv_p3p4", "p2guided_cross_attention_p3p4",
         "p2guided_deform_conv_p3p4")
METRICS = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "precision", "recall")
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
