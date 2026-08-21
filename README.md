# Neural Posterior Estimation for Stochastic Volatility Models

This repository contains the simulation, neural posterior estimation, MCMC
benchmarking, evaluation, and figure-generation code used for the master's
thesis *Neural Posterior Estimation for Stochastic Volatility Models*.

The project studies amortized posterior estimation for stochastic volatility
(SV) models. It includes a three-parameter standard SV model and a
five-parameter extension with generalized hyperbolic skew Student-t (GHST)
innovations. Summary-statistic neural networks and temporal convolutional
networks (TCNs) are evaluated against posterior estimates produced with the R
package `stochvol`.

## Repository structure

| Path | Contents |
| --- | --- |
| `simulation/` | SV simulation and the Python/R interface to `stochvol` MCMC |
| `training/` | Summary-network and TCN architectures and training functions |
| `evaluation/` | Checkpoint loading, preprocessing, and neural prediction utilities |
| `Chapter_3/` | Prior, MCMC, effective-sample-size, and tuning analyses |
| `Chapter_4/` | Three-parameter model comparisons and diagnostics |
| `Chapter_5/` | Five-parameter SV-GHST analyses and diagnostics |
| `weights/` | Final trained model checkpoints used by the evaluation scripts |
| `requirements.txt` | Exact Python environment recovered from IDUN training |

The files named `test_*.py` in `evaluation/` are evaluation utilities rather
than an automated test suite.

## Computational environments

Training and post-training evaluation were performed in different
environments. This distinction matters both for software reproducibility and
for the resources needed to rerun the project.

### Training on IDUN

Neural-network training and online simulation of training data were performed
on NTNU's IDUN HPC cluster through SLURM. The recorded jobs used the following
configuration:

| Component | IDUN configuration |
| --- | --- |
| GPU | NVIDIA Tesla V100 PCIe, 32 GB |
| CPUs | 16 CPUs allocated to the job |
| Memory | 64 GB RAM |
| Maximum job time | 16 hours |
| Python | 3.11.3 |
| PyTorch | 2.12.1+cu126 |
| CUDA build used by PyTorch | 12.6 |
| cuDNN | 9.10.2 |

The complete Python package set recovered from the IDUN virtual environment is
pinned in `requirements.txt`. It includes Linux- and CUDA-specific packages and
should be understood as an exact training-environment record, not as a
portable CPU-only dependency file.

### Evaluation on the local workstation

The trained checkpoints were copied to `weights/`. Model evaluation, MCMC
comparisons, diagnostic calculations, tables, and figures were then produced
locally on the following reference machine:

| Component | Local reference configuration |
| --- | --- |
| Computer | Lenovo Legion Pro 5 16IRX8 |
| Operating system | Pop!_OS 24.04 LTS, x86-64 |
| CPU | 13th Gen Intel Core i7-13700HX, 16 cores / 24 threads |
| Memory | 31.1 GiB RAM |
| Swap | 16 GiB |
| Dedicated GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB |
| Integrated GPU | Intel UHD Graphics 770 |

The principal local software versions were:

| Software | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| NumPy | 2.3.4 |
| pandas | 2.3.3 |
| SciPy | 1.16.3 |
| statsmodels | 0.14.6 |
| Matplotlib | 3.10.7 |
| scikit-learn | 1.7.2 |
| R | 4.6.1 |
| R package `stochvol` | 3.2.9 |

Neural evaluation automatically uses CUDA when PyTorch reports an available
CUDA device and otherwise falls back to the CPU. Access to IDUN is therefore
not required to use the committed checkpoints. Runtime measurements should,
however, only be compared when the hardware and device are also reported.

## Installation

Run all commands from the repository root. To recreate the recorded IDUN
training environment on a compatible Linux/CUDA system:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact IDUN PyTorch build and CUDA packages may require a compatible NVIDIA
driver and access to the package index from which they were originally
installed. A CPU-only or local evaluation setup may instead use compatible
versions of NumPy, pandas, SciPy, statsmodels, Matplotlib, and PyTorch; the
versions used for the reported local evaluation are listed above.

Several comparisons call R through `Rscript`. Install R and the `stochvol`
package before running those analyses:

```r
install.packages("stochvol")
```

For non-interactive or headless sessions, a non-GUI Matplotlib backend can be
selected before running plotting scripts:

```bash
export MPLBACKEND=Agg
```

## Training

The training configurations are located at:

- `Chapter_3/Training_configs/init_training.py` for the three-parameter model;
- `Chapter_5/Training_config/init_training.py` for the five-parameter SV-GHST
  model.

The training calls in these files are intentionally protected by `if False`
conditions to prevent accidental multi-hour runs. To retrain a model, review
the configuration, change only the intended condition to `True`, and submit
the corresponding command through an appropriately provisioned SLURM job:

```bash
python Chapter_3/Training_configs/init_training.py
python Chapter_5/Training_config/init_training.py
```

The configurations perform live simulation rather than loading a fixed
training dataset. They use multiprocessing for simulation and CUDA automatic
mixed precision for the TCN. The final checkpoints used by the evaluation code
are stored in `weights/`.

Reproducing training locally is not recommended with the recorded batch sizes,
validation sizes, and worker counts. Review these values carefully when using
different hardware.

## Evaluation and figure generation

The repository includes the final checkpoints and many of the generated CSV,
PDF, and PNG outputs. Retraining is therefore not necessary to inspect the
saved results or rerun an evaluation.

Each analysis script can be executed directly from the repository root. For
example:

```bash
python Chapter_3/Prior_plots/prior_plots.py
python Chapter_4/Validation_loss_history/loss_history.py
python Chapter_5/SV_plots/sv_plots.py
```

The full Chapter 4 benchmark is generated by:

```bash
python Chapter_4/RSD_and_metrics/rsd_and_metrics.py
python Chapter_4/Empirical_coverage_rate/empirical_coverage_rate.py
```

The empirical-coverage script uses the posterior-moment file generated by the
first command, so the order is significant. Other chapter scripts are
self-contained unless they report a missing prerequisite file.

## Resource and memory considerations

The local machine above is a reference configuration, not a tested minimum
requirement. Some plotting and checkpoint-inspection scripts use little memory,
but simulations involving thousands of time series and many MCMC draws are
substantially more demanding.

In particular:

- `Chapter_5/Metrics/metrics.py` uses 5,000 benchmark series of length 2,530
  and 20,000 MCMC draws. On the 31.1 GiB reference system, running more than
  three concurrent MCMC workers caused out-of-memory failures, so
  `MCMC_MAX_CORES` is deliberately set to `3`.
- Chapter 3 and Chapter 4 MCMC analyses may use negative worker settings such
  as `-2`, meaning all detected logical CPUs except two. More workers also mean
  more concurrent R processes and greater peak memory use.
- Large neural prediction batches and long simulated series also increase RAM
  or GPU-memory use.

On a machine with less memory, edit the clearly named module-level constants
before running an expensive script. The most useful controls are
`MCMC_MAX_CORES`, `BENCHMARK_SIZE`, `MCMC_DRAWS`, `N_SERIES`, `DRAWS`, and
`PREDICTION_BATCH_SIZE`. Start with one MCMC worker and a smaller benchmark,
then increase them while monitoring memory. Swap may prevent a crash but can
make MCMC evaluation considerably slower.

## Reproducibility notes

- Random seeds and experiment sizes are defined near the top of the analysis
  and training scripts.
- Checkpoint loading maps weights through the CPU before moving a model to the
  selected evaluation device, making the committed weights usable without the
  original IDUN GPU.
- Small numerical differences can arise from hardware, operating-system,
  BLAS, R, CUDA, and PyTorch differences.
- Wall-clock runtimes are hardware-dependent. Statistical outputs should be
  reproducible up to the numerical differences above, but runtime tables are
  not hardware-independent results.
- `requirements.txt` records the exact IDUN Python environment. The separate
  local software table documents the environment used for post-training
  evaluation.

## Author and citation

Edvard Kjesbu Bahr<br>
Master's thesis, Norwegian University of Science and Technology (NTNU), 2026.

If you use this repository or its results, please cite the accompanying thesis:

> Edvard Kjesbu Bahr. *Neural Posterior Estimation for Stochastic Volatility
> Models*. Master's thesis, Norwegian University of Science and Technology,
> 2026.

```bibtex
@mastersthesis{kjesbubahr2026neural,
  author = {Edvard Kjesbu Bahr},
  title  = {Neural Posterior Estimation for Stochastic Volatility Models},
  school = {Norwegian University of Science and Technology},
  year   = {2026}
}
```
