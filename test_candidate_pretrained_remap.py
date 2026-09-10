#!/usr/bin/env python3
"""Verify the exact A Duy fork + smart-transfer basis for all eight cases."""
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts

CONFIGS = {
    "baseline_p2": "model_cfg/yolov8n_p2_levir_baseline.yaml",
    "p2_asrm_fusion": "model_cfg/yolov8_p2_asrm_fusion.yaml",
    "p2p3_asrm_fusion": "model_cfg/yolov8_p2p3_asrm_fusion.yaml",
    "input_asrm_auxiliary": "model_cfg/yolov8_input_asrm_auxiliary.yaml",
    "input_asrm_guided_p2": "model_cfg/yolov8_input_asrm_guided_p2.yaml",
    "input_asrm_guided_p2p3": "model_cfg/yolov8_input_asrm_guided_p2p3.yaml",
    "input_guided_context_refine_p2p3": "model_cfg/yolov8_input_guided_context_refine_p2p3.yaml",
    "input_guided_p2p3_context_refine_p2p3": "model_cfg/yolov8_input_guided_p2p3_context_refine_p2p3.yaml",
    "asrm_detail_context_refine_p2p3": "model_cfg/yolov8_asrm_detail_context_refine_p2p3.yaml",
    "asrm_candidate_sparse_refine_p2p3": "model_cfg/yolov8_asrm_candidate_sparse_refine_p2p3.yaml",
    "carafe_p2": "model_cfg/yolov8n_p2_carafe_p2.yaml",
    "asrm_context_carafe_p2": "model_cfg/yolov8_asrm_context_carafe_p2.yaml",
    "asrm_context_p2guided_carafe_p2": "model_cfg/yolov8_asrm_context_p2guided_carafe_p2.yaml",
    "asrm_context_p2guided_carafe_spd_neck": "model_cfg/yolov8_asrm_context_p2guided_carafe_spd_neck.yaml",
    "p2guided_gate_p3p4": "model_cfg/yolov8_p2guided_gate_p3p4.yaml",
    "p2guided_dynamic_conv_p3p4": "model_cfg/yolov8_p2guided_dynamic_conv_p3p4.yaml",
    "p2guided_cross_attention_p3p4": "model_cfg/yolov8_p2guided_cross_attention_p3p4.yaml",
    "p2guided_deform_conv_p3p4": "model_cfg/yolov8_p2guided_deform_conv_p3p4.yaml",
}


def main() -> None:
    source = DetectionModel("yolov8n.yaml", ch=3, nc=80, verbose=False)
    baseline_params = None
    for case, config in CONFIGS.items():
        model = DetectionModel(str(config), ch=3, nc=1, verbose=False)
        matched = intersect_dicts(
            source.state_dict(), model.state_dict(), smart_transfer=True
        )
        params = sum(parameter.numel() for parameter in model.parameters())
        assert matched, case
        if case == "baseline_p2":
            baseline_params = params
            assert params == 3_352_396, params
            assert len(matched) == 358, len(matched)
        model.load(source, smart_transfer=True)
        print(case, {"parameters": params, "smart_transfer": len(matched)})
    assert baseline_params == 3_352_396


if __name__ == "__main__":
    main()
