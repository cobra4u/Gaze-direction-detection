Multimodal Gaze Estimation with Demographic-Aware Calibration

ETH-XGaze — Full Evaluation, Calibration, & Demographic Pseudo-Labeling Pipeline

This repository contains the full implementation of a gaze-estimation research pipeline integrating:

Appearance-based gaze regression (PyTorch)

Facial landmarking + demographic extraction (ONNX InsightFace)

Subgroup analysis (gender/age)

Multi-stage calibration

Ablation experiments and fairness evaluation

The system achieves a final champion model with ~13.2° MAE, improving significantly over the raw baseline (21.9° MAE) on the ETH-XGaze dataset.

Key Features:

Complete evaluation for ETH-XGaze: pose binning, histograms, CDF curves, MAE tables.

Demographic pseudo-labeling using InsightFace ONNX models.

Full calibration suite:

Group-offset correction

Residual 2D grid smoothing

Isotonic regression refinement

Subgroup fairness metrics for gender and age.

Automated figure & table generation for research/thesis writing.

Reproducible results, including raw baseline and calibrated champion outputs.

Repository Structure
code/
  gazemain/
    make_figs_ch5_part*.py
    make_tables_ch5_part*.py
    eval/infer_xgaze_h5_fixed.py
    tools/calibration/*.py
    demo/run_folder_demo_v3.py

results/
  eth_eval/
    figs/        # F1–F13 research figures
    tables/      # T5.1–T5.7 research tables
    preds_test_gaze_exacthead_stride1.csv
    eth_metrics.json
    eth_withinK.csv

results/eth_eval_calib/
  test_CHAMPION.csv   # ~13.2° MAE calibrated model
  test_CHAMP_BLEND.csv
  residual_outputs/*.csv

models/
  gaze/gaze_best.pth
  insightface/buffalo_l/*.onnx   # detection, landmarks, gender/age

Core Pipeline

Face detection & landmarking (InsightFace ONNX)

Demographic pseudo-labeling (gender/age from ONNX models)

Gaze regression (PyTorch checkpoint: gaze_best.pth)

Error decomposition & pose-bin analysis

Calibration pipeline

Group offsets

Residual 2D correction

Isotonic smoothing

Champion model selection

Visualization: F1–F13 & T5.1–T5.7

Environment Setup
source /users/project1/pt01281/miniconda3/etc/profile.d/conda.sh
conda activate /users/project1/pt01281/conda_envs/gaze_train

Results (ETH-XGaze):
Model Stage	Avg MAE (deg)
Raw Baseline	21.92°
After Demo Offsets	~21° (subgroup-corrected)
After Residual Grid	~15°
Final Champion	≈13.2°
Demo for Personal Images
python demo/run_folder_demo_v3.py \
    --input_dir my_faces/ \
    --output_dir my_faces/results/

License & Citation:

This repository is part of a university research project.
Citations for ETH-XGaze and InsightFace should be included.
