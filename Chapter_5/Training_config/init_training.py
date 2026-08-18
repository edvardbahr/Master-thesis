def main() -> None:

    train_live_cnn(
            sequence_length=253*10, # times ten when fitting GHST model with r and nu unfixed
            prior="default",
            tcn_channels=(16, 32, 32, 64, 64, 64),
            kernel_size=(9, 9, 7, 5, 5, 5),
            dilations = (1, 2, 4, 16, 64, 256),
            hidden_dims_head=(32, 32),
            topk_pool_fraction=0.05,
            activation=nn.ReLU,
            checkpoint_path="sv_posterior_tcn_live_finance_n253_multiscale_topk.pt",
            resume_from="sv_posterior_tcn_live_finance_n253_multiscale_topk.latest.pt",  # Set to the n2530_multiscale_topk latest checkpoint to continue.
            seed=2,
            batch_size=1024 * 4,
            n_batches=100,    # Number of batches done before each validation
            val_size=1024 * 2 * 100,
            lr=5e-4,
            n_epochs=2000,
            patience=75, # A bit higher patience since live training is noisier than fixed datasets
            min_delta=1e-5,
            min_var=1e-12, # Minimum variance to ensure numerical stability in the loss and gradients
            use_amp=True,  # Use automatic mixed precision to save on vram
            grad_clip_norm=5.0,
            deterministic_torch=True,
            n_workers= -4, # Uses all but 4 cpu cores for data simulation
            out_dtype=np.float32,
            verbose=True,
        )