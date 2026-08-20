"""Plot the validation-loss history for the five-parameter SV-GHST TCN."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent

CHECKPOINT_PATH = PROJECT_DIR / "weights" / "svghst_tcn_default.pt"
OUTPUT_PATH = HERE / "five_parameter_validation_loss_history.pdf"
TCN_COLOR = "#ff0000"


def load_checkpoint() -> dict[str, object]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary at {CHECKPOINT_PATH}.")
    return checkpoint


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


def plot_loss_history(
    checkpoint: dict[str, object],
    output_path: Path,
) -> None:
    apply_plot_style()
    marginal_losses = np.asarray(
        checkpoint["val_marginal_loss_history"],
        dtype=np.float64,
    )
    losses = np.mean(marginal_losses, axis=1)
    epochs = np.arange(1, len(losses) + 1)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    tcn_handle = axis.plot(
        epochs,
        losses,
        color=TCN_COLOR,
        label="TCN",
    )[0]
    final_handle = axis.plot(
        epochs[-1],
        losses[-1],
        color="black",
        marker="x",
        markersize=8,
        markeredgewidth=1.8,
        linestyle="none",
        label="Final epoch",
    )[0]
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean marginal Gaussian loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)

    fig.legend(
        [tcn_handle, final_handle],
        ["TCN", "Final epoch"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_loss_history(load_checkpoint(), OUTPUT_PATH)
    print(f"Saved validation loss history to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
