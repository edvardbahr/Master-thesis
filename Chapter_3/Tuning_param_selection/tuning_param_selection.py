from pathlib import Path

import torch


SHARED_TRUNK_DIMS = [(64, 32), (128, 64), (256, 128)]
HEAD_DIMS = [(16, 16), (32, 32), (64, 64)]


def main():
    checkpoint_dir = Path(__file__).resolve().parent
    final_nll = {}

    for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        grid_position = (
            tuple(checkpoint["hidden_dims_shared_trunk"]),
            tuple(checkpoint["hidden_dims_head"]),
        )
        final_nll[grid_position] = float(checkpoint["final_val_loss"])

    print("Final validation NLL:")
    print(f"{'shared trunk / head':>20}", end="")
    for head_dims in HEAD_DIMS:
        print(f"{str(head_dims):>14}", end="")
    print()

    for trunk_dims in SHARED_TRUNK_DIMS:
        print(f"{str(trunk_dims):>20}", end="")
        for head_dims in HEAD_DIMS:
            print(f"{final_nll[(trunk_dims, head_dims)]/3:>14.6f}", end="")
        print()


if __name__ == "__main__":
    main()



    best_checkpoint_dir = Path(__file__).resolve().parent / "trial_004_hidden_dims_shared_trunk-128-64_hidden_dims_head-16-16.pt"
    checkpoint = torch.load(
        best_checkpoint_dir,
        map_location="cpu",
        weights_only=False,
    )

    print("Prior used: " + checkpoint["prior"])

