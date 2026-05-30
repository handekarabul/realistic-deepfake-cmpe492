# Temperature scaling on saved test outputs (.npz from test.py --save_preds)
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score, roc_curve


def fit_temperature(logits, y_true, max_iter=50):
    """Fit scalar T by minimizing cross-entropy on the given split."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(z / temperature.clamp(min=1e-6), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def softmax_fake_prob(logits, temperature=1.0):
    z = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(z / max(float(temperature), 1e-6), dim=1)[:, 1].numpy()


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if not mask.any():
            continue
        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += mask.mean() * abs(acc_bin - conf_bin)
    return float(ece)


def threshold_metrics(y_true, y_prob):
    y_true = np.clip(np.asarray(y_true, dtype=int), 0, 1)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    youden_idx = int(np.argmax(tpr - fpr))
    opt_threshold = float(thresholds[youden_idx])
    pred = (y_prob > opt_threshold).astype(int)
    real_idx = np.where(y_true == 0)[0]
    fake_idx = np.where(y_true == 1)[0]
    acc_real = float((pred[real_idx] == 0).mean()) if len(real_idx) else float("nan")
    acc_fake = float((pred[fake_idx] == 1).mean()) if len(fake_idx) else float("nan")
    balanced_acc = 0.5 * (acc_real + acc_fake)
    acc_at_05 = float(((y_prob > 0.5).astype(int) == y_true).mean())
    return {
        "acc_at_0.5": acc_at_05,
        "opt_threshold": opt_threshold,
        "balanced_acc_opt": float(balanced_acc),
        "acc_real_opt": acc_real,
        "acc_fake_opt": acc_fake,
    }


def summarize(y_true, y_prob):
    y_true = np.clip(np.asarray(y_true, dtype=int), 0, 1)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    stats = {
        "n_samples": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "nll": float(log_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    stats.update(threshold_metrics(y_true, y_prob))
    return stats


def plot_reliability(y_true, y_prob_before, y_prob_after, out_fig, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    for ax, probs, title in zip(
        axes,
        [y_prob_before, y_prob_after],
        ["Uncalibrated", "Temperature-scaled"],
    ):
        prob_true, prob_pred = calibration_curve(
            y_true, probs, n_bins=n_bins, strategy="uniform"
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0)
        ax.plot(prob_pred, prob_true, marker="o", color="#4C6A8A", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted fake probability")
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("Fraction of positives (fake)")
    fig.suptitle("Reliability diagram — FF-DF baseline", fontsize=13)
    fig.tight_layout()
    out_fig = Path(out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True, help="Path to .npz from test.py")
    parser.add_argument("--out_dir", type=str, default="logs/calibration")
    parser.add_argument("--out_fig", type=str, default="report/figures/fig_reliability_ffdf.png")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.preds)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    y_logits = data["y_logits"]
    dataset = str(data["dataset"]) if "dataset" in data else "unknown"

    temperature = fit_temperature(y_logits, y_true)
    y_prob_cal = softmax_fake_prob(y_logits, temperature)

    before = summarize(y_true, y_prob)
    after = summarize(y_true, y_prob_cal)

    results = {
        "preds_file": str(args.preds),
        "dataset": dataset,
        "temperature": temperature,
        "note": "T fit post-hoc on the same FF-DF split; AUC unchanged.",
        "before": before,
        "after": after,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ffdf_temperature_scaling.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_reliability(y_true, y_prob, y_prob_cal, args.out_fig, n_bins=args.n_bins)

    print("=== Temperature scaling (FF-DF) ===")
    print(f"dataset: {dataset}")
    print(f"T = {temperature:.4f}")
    print(f"n = {before['n_samples']}")
    for key in ["auc", "nll", "ece", "acc_at_0.5", "opt_threshold", "balanced_acc_opt"]:
        print(f"{key:18s}  before={before[key]:.4f}  after={after[key]:.4f}")
    print(f"Saved JSON  -> {out_json}")
    print(f"Saved figure -> {args.out_fig}")


if __name__ == "__main__":
    main()
