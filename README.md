# CMPE 492 — Deepfake detection under distribution shift

Fork of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) for a final-year project on threshold-aware evaluation across dataset shifts.

**Student:** Hande Karabul  
**Course:** CMPE 492

## Scope

- Xception baseline, trained on FaceForensics++ (c23)
- Evaluation: in-distribution FF++, internal shift (FF-F2F, FF-DF), external shift (Celeb-DF-v2)
- Evaluation and failure analysis only — no new detector architecture

## Setup

1. Install dependencies (Python 3.9+, PyTorch; see [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)).
2. Download datasets separately: FF++, Celeb-DF-v2 (not included in this repo).
3. Preprocess from `preprocessing/`, then run `rearrange.py`.
4. Train and test from the repo root:

```bash
python training/train.py --detector_path training/config/detector/xception.yaml

python training/test.py --detector_path training/config/detector/xception.yaml \
  --test_dataset FF-DF --weights_path <checkpoint.pth> \
  --save_preds logs/calibration/ffdf_baseline.npz

python training/calibration_analysis.py \
  --preds logs/calibration/ffdf_baseline_FF-DF.npz
```

## Project changes

| Path | Purpose |
|------|---------|
| `training/metrics/utils.py` | Optimal threshold + balanced accuracy |
| `training/analyze_threshold.py` | Threshold stats from saved metric pickles |
| `training/calibration_analysis.py` | Temperature scaling + reliability diagram |
| `training/loss/weighted_cross_entropy_loss.py` | Weighted CE ablation |
| `training/config/detector/xception_*.yaml` | Eval and WCE configs |
| `report/` | Final report (LaTeX) and figures |

## Results (summary)

- **FF-DF:** high AUC, Acc@0.5 collapses — threshold / calibration issue
- **Celeb-DF-v2:** low AUC — ranking / generalization failure
- **Weighted CE:** no clear gain over baseline on FF-DF
- **Temperature scaling (FF-DF):** ECE and NLL improve; AUC unchanged (`logs/calibration/ffdf_temperature_scaling.json`)

## Not included

- Raw videos and preprocessed frames (`datasets/`)
- Training checkpoints (`*.pth`)
- Full training logs

## Acknowledgement

Built on [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (Yan et al., NeurIPS 2023 D&B).
