
import numpy as np
from typing import Tuple, Dict, Optional

# ------------------------------
# I/O and masks
# ------------------------------

import numpy as np

def load_data(
    train_path="ratings_train.npy",
    test_path="ratings_test.npy",

    namesgenres_path="namesngenre.npy",
):
    R_train_raw = np.load(train_path)
    R_test_raw  = np.load(test_path)


    # 1) masks: True=observed rating, False=missing
    M_train = (~np.isnan(R_train_raw)).astype(np.float32)
    M_test  = (~np.isnan(R_test_raw)).astype(np.float32)


    # 2) filled matrices for computation (missing -> 0)
    R_train = np.nan_to_num(R_train_raw, nan=0.0).astype(np.float32)
    R_test  = np.nan_to_num(R_test_raw,  nan=0.0).astype(np.float32)



    R_fit = R_train + R_test
    M_fit = (M_train + M_test) > 0
    M_fit = M_fit.astype(np.float32)

    try:
        namesgenres = np.load(namesgenres_path, allow_pickle=True)
    except Exception:
        namesgenres = None

    return R_train, R_test, M_train, M_test, R_fit, M_fit, namesgenres


def rmse_on(R_true: np.ndarray, R_pred: np.ndarray, M_mask: np.ndarray) -> float:

    R_true = np.nan_to_num(R_true, nan=0.0)
    diff = (R_true - R_pred) * M_mask
    denom = float(M_mask.sum())
    if denom == 0:
        return np.nan
    return np.sqrt((diff ** 2).sum() / denom)



def build_item2genres(namesgenres):

    item2genres = {}
    G = 0

    if hasattr(namesgenres, "dtype") and namesgenres.dtype == object:
        data = list(namesgenres)
    else:
        data = namesgenres

    if isinstance(data, dict):

        for i, gs in data.items():
            if len(gs) and isinstance(gs[0], str):

                pass

        data = [(i, gs) for i, gs in data.items()]


    all_g = {}
    for i, entry in enumerate(data):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            maybe_idx, gs = entry[0], entry[1]
            item_idx = int(maybe_idx) if isinstance(maybe_idx, (int, np.integer)) else i
            if len(gs) and isinstance(gs[0], str):
                ids = []
                for s in gs:
                    if s not in all_g:
                        all_g[s] = len(all_g)
                    ids.append(all_g[s])
                item2genres[item_idx] = ids
            else:
                item2genres[item_idx] = [int(x) for x in gs]
        else:
            item2genres[i] = []

    if not all_g:
        max_id = 0
        for gs in item2genres.values():
            if gs:
                max_id = max(max_id, max(gs))
        G = max_id + 1
    else:
        G = len(all_g)

    return item2genres, G



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






def train_and_eval(
    n_iters: int = 15,
    seed: int = 42,
    use_genre_linear: bool = True,   
    
):
    R_train, R_test, M_train, M_test, R_fit_tt, M_fit_tt, namesgenres = load_data()

    item2genres, G = build_item2genres(namesgenres)

    R_fit, M_fit = R_train, M_train
    R_target, M_target = R_test, M_test



 

    R_hat_base = baseline_sgd(R_fit, M_fit, lr=0.02, reg=0.02, epochs=n_iters, seed=seed)
    rmse_base = rmse_on(R_target, R_hat_base, M_target)
    print(f"[Baseline] RMSE on TEST = {rmse_base:.4f}")






if __name__ == "__main__":

    #train_and_eval(k=160, reg=0.05, n_iters=25, fit_on="train", eval_on="test")
    train_and_eval()

