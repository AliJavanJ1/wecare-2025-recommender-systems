
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


def item_knn_residual_boost(
    R: np.ndarray,
    M_train: np.ndarray,
    pred_bias: np.ndarray,
    k: int = 50,
    shrink: float = 25.0,
    eps: float = 1e-8,
    use_pearson: bool = True,  
) -> np.ndarray:

    U, I = R.shape
    E = (R - pred_bias) * M_train  # U x I

    C = (M_train.T @ M_train).astype(np.float32)      # I x I

    if use_pearson:
        sum_E = E.sum(axis=0, keepdims=True)
        cnt_E = M_train.sum(axis=0, keepdims=True) + eps
        E_use = E - sum_E / cnt_E
    else:
        E_use = E

    S_num = (E_use.T @ E_use).astype(np.float32)      # I x I
    item_norm = np.sqrt((E_use**2).sum(axis=0)) + eps
    S_den = np.outer(item_norm, item_norm) + eps
    S = S_num / S_den                                 
   
    S *= (C / (C + shrink))
    np.fill_diagonal(S, 0.0)

    if k < I:
        for i in range(I):
            row = S[i]
            drop_idx = np.argpartition(np.abs(row), -k)[:-k]
            row[drop_idx] = 0.0

    row_sums = np.sum(np.abs(S), axis=1, keepdims=True) + eps
    S = S / row_sums


    num = (E @ S.T)                                     # U x I

    denom = (M_train @ (S.T != 0).astype(np.float32)) + eps
    boost = num / denom

    pred = pred_bias + boost

    return pred.astype(np.float32)

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

    R_hat_boost = item_knn_residual_boost(
        R=R_out,                 
        M_train=M_out,         
        pred_bias=R_hat_fit,
        k=50,
        shrink=25,
    )
    completed = R_hat_boost.copy()
    obs = np.where(M_out > 0)
    completed[obs] = R_out[obs]
    table=completed


    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
