import copy
import numpy as np
import matplotlib.pyplot as plt
import simulateData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Model definition
# ============================================================

def make_mlp(
    input_dim,
    hidden_dims,
    output_dim=None,
    activation=nn.ReLU,
    dropout=0.0,
    layer_norm=False,
):
    layers = []
    d_prev = input_dim

    for d_hidden in hidden_dims:
        layers.append(nn.Linear(d_prev, d_hidden))

        if layer_norm:
            layers.append(nn.LayerNorm(d_hidden))

        layers.append(activation())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        d_prev = d_hidden

    # If output_dim is specified we add a final linear layer without activation or dropout.
    if output_dim is not None:
        layers.append(nn.Linear(d_prev, output_dim))
        d_prev = output_dim
    
    # If no hidden layers and no output layer, the function returns the identity function
    # which is useful for the shared trunk when we want to skip it and directly connect the input to the heads.
    if len(layers) == 0:
        return nn.Identity(), input_dim

    return nn.Sequential(*layers), d_prev

class SVPosteriorNN(nn.Module):
    """
    Neural network for amortized inference in the standard SV model.

    Input:
        z: summary statistics, shape (batch_size, input_dim)

    Output:
        mean: shape (batch_size, 3)
              columns: [mu_mean, psi_mean, log_sigma_mean]

        var: shape (batch_size, 3)
             columns: [mu_var, psi_var, log_sigma_var]

    where:
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """

    param_names = ("mu", "psi", "log_sigma")

    def __init__(
        self,
        input_dim,
        hidden_dims_shared_trunk=(128, 128),
        hidden_dims_head=(64,),
        activation=nn.ReLU,
        min_var=1e-12,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims_shared_trunk = hidden_dims_shared_trunk
        self.hidden_dims_head = hidden_dims_head
        self.min_var = min_var

        self.shared_trunk, trunk_output_dim = make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims_shared_trunk,
            output_dim=None,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        
        # pytorch's ModuleDict is a dictionary that properly registers its contents as submodules.
        self.heads = nn.ModuleDict()

        for name in self.param_names:
            head, _ = make_mlp(
                input_dim=trunk_output_dim,
                hidden_dims=hidden_dims_head,
                output_dim=2,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            self.heads[name] = head

    def forward(self, z):
        h = self.shared_trunk(z)

        means = []
        variances = []

        for name in self.param_names:
            out = self.heads[name](h)

            mean = out[:, 0:1]    # Preserves shape (batch_size, 1) which is
            raw_var = out[:, 1:2] # important for torch.cat later.

            var = F.softplus(raw_var) + self.min_var

            means.append(mean)
            variances.append(var)

        mean = torch.cat(means, dim=1)
        var = torch.cat(variances, dim=1)

        # The mean and var posterior parameters for each main parameter is returned as we assume a diagonal Gaussian posterior.
        return mean, var  
    

def theta_to_target_numpy(theta, eps=1e-6):
    """
    Converts original SV parameters to transformed training targets.

    theta has shape (n_samples, 3), with columns:
        theta[:, 0] = mu
        theta[:, 1] = phi
        theta[:, 2] = sigma

    Returns target with columns:
        mu
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """
    mu = theta[:, 0]
    phi = theta[:, 1]
    sigma = theta[:, 2]

    phi = np.clip(phi, -1.0 + eps, 1.0 - eps)
    sigma = np.clip(sigma, eps, None)

    psi = 2 * np.arctanh(phi)
    log_sigma = np.log(sigma)

    target = np.column_stack([mu, psi, log_sigma])

    return target.astype(np.float32)


def diagonal_gaussian_nll(mean, var, target):
    """
    Computes marginal negative log scores under a diagonal Gaussian.

    mean:
        shape (batch_size, param_dim)

    var:
        shape (batch_size, param_dim)

    target:
        shape (batch_size, param_dim)

    Returns:
        losses:
            shape (param_dim,), one mean NLL per transformed parameter.
    """
    elementwise_nll = F.gaussian_nll_loss(
        input=mean,
        target=target,
        var=var,
        full=True,
        reduction="none",
    )

    # Average over the batch, keeping one marginal NLL per transformed parameter.
    losses = elementwise_nll.mean(dim=0)

    return losses



def train_summary_nn(
    data_path,
    hidden_dims_shared_trunk=(128, 128),
    hidden_dims_head=(64, 64),
    activation=nn.ReLU,
    checkpoint_path="sv_posterior_nn.pt",
    seed=1,
    val_fraction=0.2,
    batch_size=1024,
    lr= 0.2e-3,
    n_epochs=1000,
    patience=50,
    min_delta=1e-4,
    min_var=1e-12,
    verbose=True,
    plot=True,
    num_workers=0,
):
    """
    Train a neural network to predict transformed SV parameters from summary statistics.

    Parameters
    ----------
    data_path:
        Path to .npz file containing arrays with keys "summaries" and "params".

    hidden_dims_shared_trunk:
        Hidden dimensions for the shared trunk of the neural network.

    hidden_dims_head:
        Hidden dimensions for each posterior head.

    activation:
        PyTorch activation class, e.g. nn.ReLU, nn.Tanh.

    checkpoint_path:
        Path where the trained checkpoint is saved.

    seed:
        Random seed used for train/validation split and PyTorch initialization.

    val_fraction:
        Fraction of data used for validation.

    batch_size:
        Mini-batch size for training and validation.
        If batch_size=None, full-batch training is used.

    lr:
        Learning rate for Adam.

    n_epochs:
        Maximum number of epochs.

    patience:
        Number of epochs without validation improvement before early stopping.

    min_delta:
        Minimum validation improvement required to reset patience.

    min_var:
        Lower bound added/used inside model variance output.

    verbose:
        Whether to print progress.

    plot:
        Whether to plot training and validation loss curves.

    num_workers:
        Number of subprocesses for DataLoader. Keep at 0 unless you need more.
    """

    # ============================================================
    # Reproducibility
    # ============================================================

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("Using device:", device)

    pin_memory = device.type == "cuda"

    # ============================================================
    # Evaluation helper
    # ============================================================

    @torch.no_grad()
    def evaluate(model, loader, device):
        """
        Return one mean validation NLL per transformed parameter.
        """
        model.eval()

        total_losses = None
        total_n = 0

        for z_batch, target_batch in loader:
            z_batch = z_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            mean, var = model(z_batch)
            losses = diagonal_gaussian_nll(mean, var, target_batch)

            batch_n = z_batch.shape[0]

            if total_losses is None:
                total_losses = torch.zeros_like(losses)

            total_losses += losses * batch_n
            total_n += batch_n

        return total_losses / total_n

    # ============================================================
    # Load data
    # ============================================================

    data = np.load(data_path, allow_pickle=True)

    Z = data["summaries"].astype(np.float32)
    theta = data["params"].astype(np.float32)

    if verbose:
        print("Z shape:", Z.shape)
        print("theta shape:", theta.shape)

    # Transform constrained parameters to unconstrained training targets.
    target = theta_to_target_numpy(theta).astype(np.float32)

    if verbose:
        print("target shape:", target.shape)

    # ============================================================
    # Train-validation split
    # ============================================================

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")

    N = len(Z)

    if N == 0:
        raise ValueError("Dataset is empty.")

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(N)

    n_val = int(val_fraction * N)

    if n_val == 0:
        raise ValueError("Validation set is empty. Increase val_fraction or dataset size.")

    if n_val == N:
        raise ValueError("Training set is empty. Decrease val_fraction.")

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    Z_train = Z[train_idx]
    Z_val = Z[val_idx]

    target_train = target[train_idx]
    target_val = target[val_idx]

    # ============================================================
    # Standardize summary statistics using training set only
    # ============================================================

    if(False):

        z_mean = Z_train.mean(axis=0, keepdims=True)
        z_std = Z_train.std(axis=0, keepdims=True)

        # Avoid division by zero if a summary statistic is constant.
        z_std = np.where(z_std < 1e-8, 1.0, z_std)

        Z_train_scaled = (Z_train - z_mean) / z_std
        Z_val_scaled = (Z_val - z_mean) / z_std
    else:
        Z_train_scaled = Z_train
        Z_val_scaled = Z_val
        z_mean = np.zeros((1, Z.shape[1]), dtype=np.float32)
        z_std = np.ones((1, Z.shape[1]), dtype=np.float32)

    # ============================================================
    # Create PyTorch datasets
    # ============================================================

    train_dataset = TensorDataset(
        torch.from_numpy(Z_train_scaled).float(),
        torch.from_numpy(target_train).float(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(Z_val_scaled).float(),
        torch.from_numpy(target_val).float(),
    )

    # ============================================================
    # Decide batch sizes
    # ============================================================

    if batch_size is None:
        effective_batch_size = len(train_dataset)
        train_shuffle = False

        if verbose:
            print("Training mode: full-batch")
    else:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        effective_batch_size = batch_size
        train_shuffle = True

        if verbose:
            print(f"Training mode: mini-batch, batch_size={batch_size}")


    # ============================================================
    # Initialize model
    # ============================================================

    input_dim = Z.shape[1]

    model = SVPosteriorNN(
        input_dim=input_dim,
        hidden_dims_shared_trunk=hidden_dims_shared_trunk,
        hidden_dims_head=hidden_dims_head,
        activation=activation,
        min_var=min_var,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)



    # ============================================================
    # Create DataLoaders
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=train_shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if verbose:
        print("Train size:", len(train_dataset))
        print("Validation size:", len(val_dataset))
        print("Batch size:", effective_batch_size)
        print("Number of train batches per epoch:", len(train_loader))
        print("Number of validation batches:", len(val_loader))


    # ============================================================
    # Training loop with early stopping
    # ============================================================

    target_names = ["mu", "psi", "log_sigma"]

    train_marginal_loss_history = []
    val_marginal_loss_history = []
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()

        total_train_losses = None
        total_train_n = 0

        for z_batch, target_batch in train_loader:
            z_batch = z_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            mean, var = model(z_batch)
            train_marginal_losses = diagonal_gaussian_nll(mean, var, target_batch)
            train_loss = train_marginal_losses.sum()

            optimizer.zero_grad() # Clears old gradients from the last step.
            train_loss.backward() # Computes gradient of train_loss w.r.t. model parameters.
            optimizer.step()      # Updates model parameters using the computed gradient.

            batch_n = z_batch.shape[0]

            if total_train_losses is None:
                total_train_losses = torch.zeros_like(train_marginal_losses)

            total_train_losses += train_marginal_losses.detach() * batch_n
            total_train_n += batch_n

        train_marginal_losses_value = total_train_losses / total_train_n
        val_marginal_losses_value = evaluate(model, val_loader, device)

        train_marginal_losses_np = train_marginal_losses_value.cpu().numpy()
        val_marginal_losses_np = val_marginal_losses_value.cpu().numpy()

        train_loss_value = float(train_marginal_losses_np.sum())
        val_loss_value = float(val_marginal_losses_np.sum())

        train_loss_history.append(train_loss_value)
        val_loss_history.append(val_loss_value)
        train_marginal_loss_history.append(train_marginal_losses_np.tolist())
        val_marginal_loss_history.append(val_marginal_losses_np.tolist())

        if val_loss_value < best_val_loss - min_delta:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            train_parts = ", ".join(
                f"{name}={loss:.4f}"
                for name, loss in zip(target_names, train_marginal_losses_np)
            )
            val_parts = ", ".join(
                f"{name}={loss:.4f}"
                for name, loss in zip(target_names, val_marginal_losses_np)
            )

            print(
                f"Epoch {epoch + 1:4d}: "
                f"train NLL = {train_loss_value:.4f}, "
                f"val NLL = {val_loss_value:.4f}"
            )
            print(
                f"             "
                f"train marginal NLLs: {train_parts}"
            )
            print(
                f"             "
                f"val marginal NLLs:   {val_parts}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    # ============================================================
    # Restore best model
    # ============================================================

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: best_state is None, using current model.")

    model.eval()

    final_val_marginal_losses = evaluate(model, val_loader, device)
    final_val_marginal_losses_np = final_val_marginal_losses.detach().cpu().numpy()
    final_val_loss = float(final_val_marginal_losses_np.sum())

    if verbose:
        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean negative joint log score: {best_val_loss:.6f}")
        print(f"Final validation mean negative joint log score: {final_val_loss:.6f}")
        print(
            "Final validation marginal NLLs:",
            {
                name: float(loss)
                for name, loss in zip(target_names, final_val_marginal_losses_np)
            },
        )

    # ============================================================
    # Save checkpoint
    # ============================================================

    activation_name = getattr(activation, "__name__", str(activation))

    # Store model weights on CPU to make checkpoint loading easier later.
    model_state_cpu = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": model_state_cpu,

        "input_dim": input_dim,
        "hidden_dims_shared_trunk": hidden_dims_shared_trunk,
        "hidden_dims_head": hidden_dims_head,
        "activation": activation_name,
        "min_var": min_var,

        "z_mean": z_mean.astype(np.float32),
        "z_std": z_std.astype(np.float32),

        "target_names": target_names,
        "target_transform": {
            "mu": "mu",
            "psi": "2 * atanh(phi)",
            "log_sigma": "log(sigma)",
        },

        "loss": "mean negative joint Gaussian log score, diagonal covariance",
        "loss_components": "mean marginal Gaussian negative log scores for mu, psi, and log_sigma",
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_marginal_losses": final_val_marginal_losses_np.astype(np.float32),
        "best_epoch": best_epoch,

        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_marginal_loss_history": train_marginal_loss_history,
        "val_marginal_loss_history": val_marginal_loss_history,

        "data_path": data_path,
        "val_fraction": val_fraction,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "seed": seed,
    }

    torch.save(checkpoint, checkpoint_path)

    if verbose:
        print(f"Model saved to {checkpoint_path}")

    # ============================================================
    # Plot loss curves
    # ============================================================

    if plot:
        plt.plot(train_loss_history, label="train")
        plt.plot(val_loss_history, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Mean negative joint log score")
        plt.legend()
        plt.show()

        train_marginal_loss_array = np.array(train_marginal_loss_history)
        val_marginal_loss_array = np.array(val_marginal_loss_history)

        for i, name in enumerate(target_names):
            plt.plot(train_marginal_loss_array[:, i], label=f"train {name}")
            plt.plot(val_marginal_loss_array[:, i], label=f"validation {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Mean marginal negative log score")
        plt.legend()
        plt.show()

    return model, checkpoint

def main():
    """
    Z, theta, feature_names = simulateData.generate_sv_dataset_parallel(
        N=800,
        n=253,
        chunk_size=200,
        seed=1,
        prior="default",
        random_init=True,
        n_acvf_ratios=4,
        compute_arima_coeff=True,
        out_dtype=np.float32,
        show_progress=True,
        n_workers=4,
    )

    print(Z)
    """


    train_summary_nn(
    data_path = "sv_dataset_default_1M_ARIMA.npz",
    hidden_dims_shared_trunk=(128, 128),
    hidden_dims_head=(64, 64),
    activation=nn.ReLU,
    checkpoint_path="sv_posterior_nn_1M_ARIMA.pt",
    seed=1,
    val_fraction=0.2,
    batch_size=1024*16,
    lr=0.5e-3,    #Best step size found so far (I forgot standardazing tho :/)
    n_epochs=2000,
    patience=50,
    min_delta=1e-5,
    min_var=1e-12,
    verbose=True,
    plot=True,
    num_workers=0,
)


if __name__ == "__main__":
    main()
