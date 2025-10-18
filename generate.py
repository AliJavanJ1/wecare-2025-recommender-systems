
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

import re
import numpy as np
def ridge_item_residual_boost(R, M, pred_bias, X, lam=1.0):

    U, I = R.shape
    assert pred_bias.shape == (U, I)
    assert X.shape[0] == I


    E = (R - pred_bias) * M
    cnt_i = M.sum(axis=0).astype(np.float32)          # (I,)
    y = np.divide(E.sum(axis=0), np.maximum(cnt_i,1), where=cnt_i>0, out=np.zeros(I, dtype=np.float32))  # (I,)

    # Ridge ：beta = (X^T W X + lam I)^{-1} X^T W y，W=diag(cnt_i)

    # A = X^T (W X) = X^T (X * cnt_i[:,None])
    Xw = X * cnt_i[:, None]                    # (I,F)
    A = X.T @ Xw                               # (F,F)
    b = X.T @ (y * cnt_i)                      # (F,)

 
    F = X.shape[1]
    reg = np.eye(F, dtype=np.float32) * lam
    reg[0,0] = 0.0

    beta = np.linalg.solve(A + reg, b)         # (F,)
    item_adjust = X @ beta                      # (I,)

    return item_adjust.astype(np.float32), beta.astype(np.float32)
def build_item_features(
    genres_array,       
    M_train,            
    use_genre=True,
    normalize_multilabel=True,   
    use_year=True,
    use_log_count=True,
    use_title_len=True,
):
   
    if isinstance(genres_array, np.ndarray) and genres_array.ndim==2 and genres_array.shape[1]>=2:
        titles = [row[0] for row in genres_array]
        gstrs  = [row[1] for row in genres_array]
    else:
        titles = ["" for _ in range(len(genres_array))]
        gstrs  = list(genres_array)
    I = len(gstrs)

    feats = []
    feat_names = []

  
    if use_genre:
        allg = sorted({g for s in gstrs if s for g in s.split('|')})
        g2i = {g:i for i,g in enumerate(allg)}
        G = len(g2i)
        Xg = np.zeros((I, G), dtype=np.float32)
        for i, s in enumerate(gstrs):
            if not s: continue
            gs = [g2i[g] for g in s.split('|') if g in g2i]
            if not gs: continue
            if normalize_multilabel and len(gs) > 1:
                Xg[i, gs] = 1.0 / np.sqrt(len(gs))
            else:
                Xg[i, gs] = 1.0
        feats.append(Xg)
        feat_names += [f"G:{g}" for g in allg]

 
    if use_year:
        years = np.full(I, np.nan, dtype=np.float32)
        for i, t in enumerate(titles):
            m = re.search(r"\((\d{4})\)", t or "")
            if m:
                y = int(m.group(1))
                
                if 1900 <= y <= 2025:
                    years[i] = y
       
        if np.isnan(years).any():
            med = np.nanmedian(years)
            years[np.isnan(years)] = med
        # z-score
        y_mu, y_std = years.mean(), years.std() + 1e-8
        years_z = (years - y_mu) / y_std
        feats.append(years_z[:, None])
        feat_names.append("year(z)")

    
    if use_log_count:
        cnt = M_train.sum(axis=0).astype(np.float32)   # (I,)
        logc = np.log1p(cnt)
        lc_mu, lc_std = logc.mean(), logc.std() + 1e-8
        logc_z = (logc - lc_mu) / lc_std
        feats.append(logc_z[:, None])
        feat_names.append("log_count(z)")

   
    if use_title_len:
        tlen = np.array([len(t or "") for t in titles], dtype=np.float32)
        tl_mu, tl_std = tlen.mean(), tlen.std() + 1e-8
        tlen_z = (tlen - tl_mu) / tl_std
        feats.append(tlen_z[:, None])
        feat_names.append("title_len(z)")

   
    X = np.concatenate(feats, axis=1) if feats else np.zeros((I,0), dtype=np.float32)
    X = np.concatenate([np.ones((I,1), dtype=np.float32), X], axis=1)
    feat_names = ["bias"] + feat_names
    return X, feat_names
from neumf import  run_neumf
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    # print('Ratings Loaded.')
    

    # # Any method you want
    # M_out = (~np.isnan(table)).astype(np.float32)
    # R_out = np.nan_to_num(table, nan=0.0).astype(np.float32)
    # R_hat_base = baseline_sgd(R_out, M_out, lr=0.01, reg=0.05, epochs=15, seed=42)
    # genres_np=np.load("namesngenre.npy")
    # X, feat_names = build_item_features(genres_np, M_out,
    #                                 use_genre=True,
    #                                 use_year=True,
    #                                 use_log_count=True,
    #                                 use_title_len=False)
    
    # item_adj, beta = ridge_item_residual_boost(R_out, M_out, R_hat_base, X, lam=1)


    # R_hat_final = R_hat_base + item_adj[None, :]

    # completed = R_hat_final.copy()
    # obs = np.where(M_out > 0)
    # completed[obs] = R_out[obs]
    # table=completed

    output=run_neumf(table)
    

    # Save the completed table 
    np.save("output.npy", output) ## DO NOT CHANGE THIS LINE


        
