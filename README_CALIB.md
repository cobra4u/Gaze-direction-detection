# Gaze ETH-XGaze — Inference + Calibration (Single Source of Truth)

## Folder map
- tools/calibration/ : all post-hoc calibrators (affine, isotonic, iso+affine, per-bin poly, vector-rot)
- tools/diagnostics/ : quick sanity scripts (correlation, sign/swap scan)
- metadata/          : demographics or split metadata (CSV) — untouched
- my_gaze_outputs/   : your personal demo outputs — not used for ETH results
- eval/              : place ETH eval/inference entrypoint here if needed
- archive/           : everything swept out of the root (safe, reversible)

## 2-command result recipe (after raw TTA export):
1) Affine:
python -u tools/calibration/calibrate_affine_ethxgaze.py \
  --val  /users/project1/pt01281/gaze_outputs/eth_eval_calib/val_raw_tta.csv \
  --test /users/project1/pt01281/gaze_outputs/eth_eval_calib/test_raw_tta.csv \
  --out-dir /users/project1/pt01281/gaze_outputs/eth_eval_calib

2) Isotonic + Affine:
python -u tools/calibration/calibrate_iso_affine_ethxgaze.py \
  --val  /users/project1/pt01281/gaze_outputs/eth_eval_calib/val_raw_tta.csv \
  --test /users/project1/pt01281/gaze_outputs/eth_eval_calib/test_raw_tta.csv \
  --out-dir /users/project1/pt01281/gaze_outputs/eth_eval_calib
