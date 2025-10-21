import numpy as np

def clean_data(ratings):
    valid = np.isfinite(ratings)
    cleaned = np.where(valid, ratings, 0.0)
    mask = (valid & (ratings != 0))
    return cleaned, mask

def compute_rmse(true_ratings, predicted_ratings, mask):
    diff = (true_ratings - predicted_ratings) * mask
    return np.sqrt((diff ** 2).sum() / max(mask.sum(), 1))

def als_with_biases(ratings, mask, rank=150, reg_factor=0.9, reg_bias=0.5, iterations=30, seed=42):
    rng = np.random.default_rng(seed)
    
    num_users, num_items = ratings.shape

    user_factors = 0.01 * rng.standard_normal((num_users, rank))
    item_factors = 0.01 * rng.standard_normal((num_items, rank))

    global_bias = (ratings * mask).sum() / max(mask.sum(), 1)
    user_biases = np.zeros(num_users)
    item_biases = np.zeros(num_items)
    identity = np.eye(rank)

    user_items = [np.flatnonzero(mask[u]) for u in range(num_users)]
    item_users = [np.flatnonzero(mask[:, i]) for i in range(num_items)]

    for _ in range(iterations):
        for u in range(num_users):
            items = user_items[u]
            if items.size == 0:
                continue
            v = item_factors[items]
            residuals = ratings[u, items] - global_bias - user_biases[u] - item_biases[items]
            a = v.T @ v + reg_factor * identity
            b = v.T @ residuals
            user_factors[u] = np.linalg.solve(a, b)

        for i in range(num_items):
            users = item_users[i]
            if users.size == 0:
                continue
            u = user_factors[users]
            residuals = ratings[users, i] - global_bias - user_biases[users] - item_biases[i]
            a = u.T @ u + reg_factor * identity
            b = u.T @ residuals
            item_factors[i] = np.linalg.solve(a, b)

        for u in range(num_users):
            items = user_items[u]
            if items.size == 0:
                continue
            predictions = global_bias + item_biases[items] + user_factors[u] @ item_factors[items].T
            user_biases[u] = (ratings[u, items] - predictions).sum() / (len(items) + reg_bias)

        for i in range(num_items):
            users = item_users[i]
            if users.size == 0:
                continue
            predictions = global_bias + user_biases[users] + (user_factors[users] @ item_factors[i]).ravel()
            item_biases[i] = (ratings[users, i] - predictions).sum() / (len(users) + reg_bias)

    return (global_bias + user_biases[:, None] + item_biases[None, :] + user_factors @ item_factors.T)


ratings_train_raw = np.load("ratings_train.npy")
ratings_eval_raw = np.load("ratings_eval.npy")

ratings_train, mask_train = clean_data(ratings_train_raw)
ratings_eval, mask_eval = clean_data(ratings_eval_raw)

predictions = als_with_biases(ratings_train, mask_train, rank=150, reg_factor=0.9, reg_bias=0.5, iterations=30, seed=42)

print("Train RMSE:", compute_rmse(ratings_train, predictions, mask_train))
print("Eval  RMSE:", compute_rmse(ratings_eval, predictions, mask_eval))
