# GitHub publication manifest

This directory is a source-only snapshot of the LEVIR-Ship YOLO experiments.

Included:

- custom local `ultralytics/` fork required by the model YAMLs;
- `model_cfg/` for all registered experiments, including
  `baseline_gap_factorized_k15_rgb_saturation_asrm_k4k12k16_context_refine`;
- dataset preparation, training, smoke-test and result-collection scripts;
- unit/smoke-oriented test scripts and the training notebook;
- dependency and usage documentation.

Excluded intentionally:

- LEVIR-Ship and Varroa data;
- Hugging Face/API credentials;
- checkpoints, run folders, result CSVs and visualizations;
- package ZIPs, Python cache files, virtual environments and editor state.

The repository expects users to obtain data independently, run
`prepare_yolo_data.py`, then invoke `run_experiment.py` or the notebook.
