import simulateData as sim
import torch.nn.functional as F









def main():


    N = 800
    n = 253

    n_workers = sim.resolve_n_workers(-2)
    chunk_size = sim.resolve_chunk_size(N, n_workers, 4)

    data = sim.simulate_sv_log_y_squared_parallel(
        N=N,
        n=n,
        chunk_size=chunk_size,
        seed=1,)
    

    print(data)





if __name__ == "__main__":
    main()