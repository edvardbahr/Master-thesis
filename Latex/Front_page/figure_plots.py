import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_SIZE = (4.0, 3.0)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from simulation import sim_5_param_data as sim
from evaluation import test_sv_nn_model as eval


def save_clean_figure(
    figure: Figure,
    axis: Axes,
    title: str,
    filename: str,
) -> None:
    axis.set_title(title)
    axis.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    axis.grid(False)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.84)
    figure.savefig(OUTPUT_DIR / filename, format="pdf")
    plt.close(figure)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 15,
        "axes.titlesize": 26,
        "axes.labelsize": 30,
        "legend.fontsize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
    }
)

mu = [-3]
phi = [0.95]
sigma = [0.3]
r = [0]
nu = [np.inf]
# 8
rng = np.random.default_rng(12)


checkpoint_path = PROJECT_DIR / "weights" / "tcn_default.pt"

model, checkpoint = eval.load_model(checkpoint_path)



y = sim.simulate_sv_chunk(mu, phi, sigma, r, nu, 253, rng)

y_figure, y_axis = plt.subplots(figsize=FIGURE_SIZE)
y_axis.plot(y[0], color="black")
save_clean_figure(
    y_figure,
    y_axis,
    r"$y \sim \mathrm{SV}(\mu, \psi, \rho)$",
    "y.pdf",
)

mean, var = eval.predict(model, checkpoint, y)

mean = mean[0]
var = var[0]

true_values = np.array(
    [
        mu[0],
        2.0 * np.arctanh(phi[0]),
        np.log(sigma[0]),
    ]
)
parameter_labels = [
    r"$\mu$",
    r"$\psi$",
    r"$\rho$",
]
output_filenames = ["mu.pdf", "psi.pdf", "rho.pdf"]

standard_deviations = np.sqrt(var[:3])
x_min = min(np.min(mean[:3] - 3.0 * standard_deviations), np.min(true_values))
x_max = max(np.max(mean[:3] + 3.0 * standard_deviations), np.max(true_values))
x_padding = 0.05 * (x_max - x_min)
x = np.linspace(x_min - x_padding, x_max + x_padding, 1_000)

for label, filename, predicted_mean, predicted_var, true_value in zip(
    parameter_labels,
    output_filenames,
    mean[:3],
    var[:3],
    true_values,
):
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    density = np.exp(-0.5 * (x - predicted_mean) ** 2 / predicted_var) / np.sqrt(
        2.0 * np.pi * predicted_var
    )
    peak_density = 1.0 / np.sqrt(2.0 * np.pi * predicted_var)
    axis.plot(x, density, color="black")
    axis.vlines(
        predicted_mean,
        ymin=0.0,
        ymax=peak_density,
        color="black",
        linestyle="--",
    )
    axis.axvline(true_value, color="red")
    axis.set_xlim(x[0], x[-1])
    save_clean_figure(figure, axis, label, filename)

print(mean[:3])
