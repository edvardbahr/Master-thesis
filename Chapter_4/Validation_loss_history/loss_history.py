"""Plot validation-loss histories for the standard-SV neural estimators."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent


PRIORS = ("default", "finance")
ARCHITECTURES = ("TCN", "Summary NN")
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUTPUT_PATH = HERE / "gaussian_loss_history.pdf"

CHECKPOINT_NAMES = {
    ("Summary NN", "default"): "summary_nn_default_arima.pt",
    ("Summary NN", "finance"): "summary_nn_finance_arima.pt",
    ("TCN", "default"): "tcn_default.pt",
    ("TCN", "finance"): "tcn_finance.pt",
}

COLORS = {
    "TCN": "#ff0000",
    "Summary NN": "#008000",
}


def load_checkpoints() -> dict[tuple[str, str], dict[str, object]]:
    checkpoints = {}
    for prior in PRIORS:
        for architecture in ARCHITECTURES:
            checkpoint_path = WEIGHTS_DIR / CHECKPOINT_NAMES[(architecture, prior)]
            checkpoints[(architecture, prior)] = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
    return checkpoints


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )


def plot_loss_histories(
    checkpoints: dict[tuple[str, str], dict[str, object]],
    output_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    legend_handles = {}

    for column, prior in enumerate(PRIORS):
        ax = axes[column]

        for architecture in ARCHITECTURES:
            checkpoint = checkpoints[(architecture, prior)]
            marginal_losses = np.asarray(
                checkpoint["val_marginal_loss_history"],
                dtype=np.float64,
            )
            losses = np.mean(marginal_losses, axis=1)
            epochs = np.arange(1, len(losses) + 1)

            legend_handles[architecture] = ax.plot(
                epochs,
                losses,
                color=COLORS[architecture],
                label=architecture,
            )[0]
            legend_handles["Final epoch"] = ax.plot(
                epochs[-1],
                losses[-1],
                color="black",
                marker="x",
                markersize=8,
                markeredgewidth=1.8,
                linestyle="none",
                label="Final epoch",
            )[0]

        ax.set_title(f"{prior.capitalize()} prior")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean marginal Gaussian loss")
    legend_order = ("TCN", "Summary NN", "Final epoch")
    fig.legend(
        [legend_handles[name] for name in legend_order],
        legend_order,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    checkpoints = load_checkpoints()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_loss_histories(checkpoints, OUTPUT_PATH)
    print(f"Saved the Gaussian loss histories to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
