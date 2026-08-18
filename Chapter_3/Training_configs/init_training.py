import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.sbt_summary_nn import train_summary_nn
from training.sbt_tcn import train_tcn


def main() -> None:
    for prior in ["default", "finance"]:
        if False:
            checkpoint_path = HERE / f"summary_nn_{prior}_arima.pt"
            latest_checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.latest{checkpoint_path.suffix}"
            )
            train_summary_nn(
                sequence_length=253,
                prior=prior,
                hidden_dims_shared_trunk=(128, 64),
                hidden_dims_head=(16, 16),
                checkpoint_path=checkpoint_path,
                resume_from=(
                    latest_checkpoint_path
                    if latest_checkpoint_path.is_file()
                    else None
                ),
                seed=2,
                batch_size=1024,
                n_batches=10,
                val_size=20_000,
                n_quantiles=19,
                compute_arima_coeff=True,
                n_epochs=2000,
                patience=100,
                min_delta=1e-5,
                min_var=1e-12,
                layer_norm=True,
                n_workers=-2,
                verbose=True,
                lr=5e-4,
            )

        if False:
            checkpoint_path = HERE / f"tcn_{prior}.pt"
            latest_checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.latest{checkpoint_path.suffix}"
            )
            train_tcn(
                sequence_length=253,
                prior=prior,
                fixed_r=0,
                fixed_nu=np.inf,
                tcn_channels=(16, 32, 32, 64, 64),
                kernel_size=(9, 9, 7, 5, 5),
                dilations=None,
                hidden_dims_head=(32, 32),
                topk_pool_fraction=0.05,
                activation=nn.ReLU,
                checkpoint_path=checkpoint_path,
                resume_from=(
                    latest_checkpoint_path
                    if latest_checkpoint_path.is_file()
                    else None
                ),
                seed=2,
                batch_size=1024 * 4,
                n_batches=100,  # Number of batches done before each validation
                val_size=1024 * 2 * 100,
                lr=5e-4,
                n_epochs=2000,
                # Live validation is noisier than validation on fixed datasets.
                patience=75,
                min_delta=1e-5,
                # Minimum variance for numerical stability in loss and gradients.
                min_var=1e-12,
                use_amp=True,  # Save VRAM with automatic mixed precision.
                grad_clip_norm=5.0,
                deterministic_torch=True,
                n_workers=-4,  # Use all but four CPU cores for simulation.
                out_dtype=np.float32,
                verbose=True,
            )


if __name__ == "__main__":
    main()
