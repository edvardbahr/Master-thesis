import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.sbt_tcn import train_tcn


def main() -> None:
    if False:
        checkpoint_path = HERE / "svghst_tcn_default.pt"
        latest_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.latest{checkpoint_path.suffix}"
        )
        train_tcn(
            # Use a longer series when fitting GHST with r and nu unfixed.
            sequence_length=253 * 10,
            prior="default",
            tcn_channels=(16, 32, 32, 64, 64, 64),
            kernel_size=(9, 9, 7, 5, 5, 5),
            dilations=(1, 2, 4, 16, 64, 256),
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
