
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
def baseline_sgd(R, M, lr=0.02, reg=0.02, epochs=15, seed=42, shuffle=True):
    rng = np.random.default_rng(seed)
    U, I = R.shape
    denom = max(M.sum(), 1.0)
    mu = float((R * M).sum() / denom)
    bu = np.zeros(U, dtype=np.float32)
    bi = np.zeros(I, dtype=np.float32)
    idx = np.array(np.where(M > 0)).T
    for _ in range(epochs):
        if shuffle: rng.shuffle(idx)
        for u, i in idx:
            e = R[u, i] - (mu + bu[u] + bi[i])
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
    return (mu + bu[:, None] + bi[None, :]).astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    M_out = (~np.isnan(table)).astype(np.float32)
    R_out = np.nan_to_num(table, nan=0.0).astype(np.float32)
    R_hat_fit = baseline_sgd(R_out, M_out, lr=0.02, reg=0.02, epochs=15, seed=42)

 
    completed = R_hat_fit.copy()
    obs = np.where(M_out > 0)
    completed[obs] = R_out[obs]
    table=completed


    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
