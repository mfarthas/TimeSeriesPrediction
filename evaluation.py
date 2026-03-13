import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DATA_DIR = "./synthetic"
OUT_DIR = "./evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"lstm": "#2563eb", "baseline": "#dc2626"}


def load():
    y_true = np.load(f"{DATA_DIR}/y_true.npy")
    y_prob_lstm = np.load(f"{DATA_DIR}/y_prob_lstm.npy")
    y_prob_baseline = np.load(f"{DATA_DIR}/y_prob_baseline.npy")
    return y_true, y_prob_lstm, y_prob_baseline


def plot_pr_curve(y_true, y_prob_lstm, y_prob_baseline):
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, y_prob in [("lstm", y_prob_lstm), ("baseline", y_prob_baseline)]:
        prec, rec, thresh = precision_recall_curve(y_true, y_prob)
        auc = average_precision_score(y_true, y_prob)
        ax.plot(
            rec, prec,
            color=COLORS[name],
            lw=2,
            label=f"{name.upper()}  PR-AUC={auc:.4f}",
        )
        idx = np.argmin(np.abs(thresh - 0.5)) if len(thresh) else 0
        ax.scatter(rec[idx], prec[idx], color=COLORS[name], s=60, zorder=5)

    baseline_rate = y_true.mean()
    ax.axhline(baseline_rate, color="grey", lw=1, linestyle="--",
               label=f"No-skill  ({baseline_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pr_curve.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob_lstm, y_prob_baseline):
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, y_prob in [("lstm", y_prob_lstm), ("baseline", y_prob_baseline)]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(
            fpr, tpr,
            color=COLORS[name],
            lw=2,
            label=f"{name.upper()}  ROC-AUC={auc:.4f}",
        )

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
            label="No-skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_prob, name):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    labels = ["Normal", "Incident"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix : {name.upper()}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def plot_pr_vs_threshold(y_true, y_prob_lstm):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob_lstm)
    prec, rec = prec[:-1], rec[:-1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresh, prec, color=COLORS["lstm"], lw=2, label="Precision")
    ax.plot(thresh, rec, color="#16a34a", lw=2, label="Recall")
    ax.axvline(0.5, color="grey", lw=1, linestyle="--", label="Threshold=0.5")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Threshold: LSTM")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pr_vs_threshold.png", dpi=150)
    plt.close(fig)


def main():
    y_true, y_prob_lstm, y_prob_baseline = load()

    plot_pr_curve(y_true, y_prob_lstm, y_prob_baseline)
    plot_roc_curve(y_true, y_prob_lstm, y_prob_baseline)
    plot_confusion_matrix(y_true, y_prob_lstm, "lstm")
    plot_confusion_matrix(y_true, y_prob_baseline, "baseline")
    plot_pr_vs_threshold(y_true, y_prob_lstm)

    print(f"Saved 6 plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
