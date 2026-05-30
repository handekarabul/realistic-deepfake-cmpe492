# Threshold stats from metric_dict_best.pickle (pred + label fields)import argparse
import pickle

import numpy as np
from sklearn import metrics


def analyze(y_pred, y_true):
    y_true = np.clip(y_true, 0, 1)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    youden_idx = np.argmax(tpr - fpr)
    opt_thr = thresholds[youden_idx]
    pred = (y_pred > opt_thr).astype(int)
    real_idx = np.where(y_true == 0)[0]
    fake_idx = np.where(y_true == 1)[0]
    acc_real = (pred[real_idx] == 0).mean() if len(real_idx) else float("nan")
    acc_fake = (pred[fake_idx] == 1).mean() if len(fake_idx) else float("nan")
    acc_05 = (y_pred > 0.5).astype(int)
    acc_at_05 = (acc_05 == y_true).mean()
    return {
        "auc": metrics.auc(fpr, tpr),
        "acc_at_0.5": acc_at_05,
        "opt_threshold": float(opt_thr),
        "balanced_acc_opt": 0.5 * (acc_real + acc_fake),
        "acc_real_opt": acc_real,
        "acc_fake_opt": acc_fake,
        "pred_mean_real": float(y_pred[real_idx].mean()) if len(real_idx) else float("nan"),
        "pred_mean_fake": float(y_pred[fake_idx].mean()) if len(fake_idx) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_pickle", type=str, required=True)
    args = parser.parse_args()
    with open(args.metric_pickle, "rb") as f:
        d = pickle.load(f)
    stats = analyze(d["pred"], d["label"])
    print("=== Threshold analysis ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
