import torch
import numpy as np
import time
import re
import torch.nn as nn
from torch.nn import functional as F
import math
import tqdm
import os
from sklearn.model_selection import KFold
'''
Neural Matrix Factorization (NeuMF) Implementation with Metadata and JOINTLY TRAINED SDAE features.
FIXED VERSION: Enhanced numerical stability and bug fixes for joint training.
'''


class SDAE(nn.Module):
    """Stacked Denoising Autoencoder component (Encoder and Decoder)"""
    def __init__(self, input_dim, hidden_layers, latent_dim, corruption_rate=0.3):
        super().__init__()
        
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
            
        decoder_layers.append(nn.Linear(in_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers) 
        
        self.corruption_rate = corruption_rate

    def corrupt(self, x):
        """Denoising mechanism: Dropout on input"""
        if self.training:
            return F.dropout(x, p=self.corruption_rate, training=True)
        return x
    
    def forward(self, x):

        corrupted_x = self.corrupt(x)
        latent = self.encoder(corrupted_x)
        reconstruction = self.decoder(latent)

        reconstruction = torch.sigmoid(reconstruction) 
        return latent, reconstruction 

class GMF(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, sdae_feature_dim=0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_indices, item_indices, user_sdae_feat, item_sdae_feat):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        user_embed = torch.cat([user_embed, user_sdae_feat], dim=-1)
        item_embed = torch.cat([item_embed, item_sdae_feat], dim=-1)

        output = user_embed * item_embed 
        return output

class MLP(nn.Module):
    # ... (MLP类保持不变，但其 forward 方法签名必须匹配 NeuMF 的调用)
    def __init__(self, num_users, num_items, embedding_dim, item_metadata_dim=0, 
                  layers=[64,32,16,8], meta_dim_out=32, sdae_feature_dim=0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.meta_proj = None
        if item_metadata_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(item_metadata_dim, meta_dim_out),
                nn.ReLU()
            )
            meta_input_size = meta_dim_out
        else:
            meta_input_size = 0
            
        input_size = embedding_dim*2 + sdae_feature_dim*2 + meta_input_size 

        self.fc_layers = nn.ModuleList([nn.Linear(input_size, layers[0])])
        for i in range(1, len(layers)):
            self.fc_layers.append(nn.Linear(layers[i-1], layers[i]))

    def forward(self, u, i, meta=None, user_sdae_feat=None, item_sdae_feat=None):
        ue = self.user_embedding(u); ie = self.item_embedding(i)
        
        ue = torch.cat([ue, user_sdae_feat], dim=-1)
        ie = torch.cat([ie, item_sdae_feat], dim=-1)
            
        meta_feat = None
        if meta is not None and self.meta_proj is not None:
            meta_feat = self.meta_proj(meta)
        
        features = [ue, ie]
        if meta_feat is not None:
            features.append(meta_feat)
            
        x = torch.cat(features, dim=-1)
        
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_embedding_dim=64, mlp_embedding_dim=64, 
                 item_metadata_dim=0, mlp_layers=[64, 32, 16, 8], dropout=0.2, 
                 sdae_latent_dim=64, sdae_hidden_layers=[512, 128], corruption_rate=0.3):
        super().__init__()
        
        self.sdae_latent_dim = sdae_latent_dim
        
        # 1. SDAE Modules (Integrated for Joint Training)
        self.user_sdae = SDAE(num_items, sdae_hidden_layers, sdae_latent_dim, corruption_rate)
        self.item_sdae = SDAE(num_users, sdae_hidden_layers, sdae_latent_dim, corruption_rate)
        
        # 2. GMF and MLP Modules 
        self.gmf = GMF(num_users, num_items, gmf_embedding_dim, sdae_latent_dim)
        self.mlp = MLP(num_users, num_items, mlp_embedding_dim, item_metadata_dim, mlp_layers, sdae_feature_dim=sdae_latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 3. Final Prediction Layer
        final_gmf_dim = gmf_embedding_dim + sdae_latent_dim 
        final_mlp_dim = mlp_layers[-1]
        
        self.predict_layer = nn.Linear(final_gmf_dim + final_mlp_dim, 1)
        self.out_act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.zeros_(self.predict_layer.bias)
        nn.init.normal_(self.predict_layer.weight, std=0.01)
        nn.init.constant_(self.predict_layer.bias, 0)
        
    def forward(self, user_indices, item_indices, item_metadata=None, user_input_R_row=None, item_input_R_col=None):
        
        # FIX: Added check for None input to prevent crash during evaluation/output generation 
        # where SDAE inputs might be placeholders. 
        if user_input_R_row is None or item_input_R_col is None:
            raise ValueError("Missing full rating vector input for SDAE in forward pass.")

        # --- SDAE Feature Extraction and Reconstruction ---
        user_sdae_feat, user_recon = self.user_sdae(user_input_R_row)
        item_sdae_feat, item_recon = self.item_sdae(item_input_R_col)
        
        # --- NeuMF Prediction ---
        gmf_output = self.gmf(user_indices, item_indices, user_sdae_feat, item_sdae_feat)
        mlp_output = self.mlp(user_indices, item_indices, item_metadata, user_sdae_feat, item_sdae_feat)
        
        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        concat = self.dropout(concat)
        
        # Final prediction
        prediction = self.out_act(self.predict_layer(concat)).squeeze()
        
        return prediction.squeeze(), user_recon, item_recon

# ----------------------------------------------------
# Recommender 
# ----------------------------------------------------

class NeuMFRecommender:
    def __init__(self, R=None, metadata_path=None, learning_rate=1e-3, batch_size=256, 
                 gmf_dim=64, mlp_dim=64, mlp_layers=[64, 32, 16, 8],
                 dropout=0.2, use_metadata=True,
                 sdae_latent_dim=64, sdae_hidden_layers=[512, 128], corruption_rate=0.3):
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.sdae_latent_dim = sdae_latent_dim
        self.sdae_hidden_layers = sdae_hidden_layers
        self.corruption_rate = corruption_rate
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
 
        R_temp = torch.tensor(R, dtype=torch.float32)
        self.mask = ~torch.isnan(R_temp)
        R_temp = torch.nan_to_num(R_temp, nan=0.0)

     
        self.R_full = (R_temp / 5.0).float().to(self.device)
      
        self.R = R_temp.float().to(self.device)
        
        self.num_users, self.num_items = R.shape[0], R.shape[1]

        item_metadata_dim = 0
        self.item_metadata = None
        if use_metadata and metadata_path is not None:
            self.item_metadata = self.make_metadata(np.load(metadata_path))
            item_metadata_dim = self.item_metadata.shape[1]
        
    
        self.model = NeuMF(
            self.num_users, self.num_items, gmf_embedding_dim=gmf_dim,
            mlp_embedding_dim=mlp_dim, item_metadata_dim=item_metadata_dim,
            mlp_layers=mlp_layers, dropout=dropout,
            sdae_latent_dim=self.sdae_latent_dim, sdae_hidden_layers=self.sdae_hidden_layers,
            corruption_rate=self.corruption_rate
        ).to(self.device)
        
        self.R_output = None
        # print(f"NeuMF+SDAE (Joint Training) initialized on {self.device}")
    def get_state_dict(self):
        """Returns the state dictionary of the underlying NeuMF model."""
        return self.model.state_dict().copy()
    import re
    import torch
    from torch.nn import functional as F

    def make_metadata(self, meta_datas, rating_counts=None):

   
        regex = r'.*\(([0-9]{4})\)'
        dates = []  
        all_years = []  

        for m in meta_datas:
            title = m[0]
            match = re.search(regex, title)  
            if match:
                year = int(match.group(1))  #
                dates.append(year)
                all_years.append(year)
            else:
                dates.append(None)  


        if all_years:
            avg_year = int(round(sum(all_years) / len(all_years)))

        #print(f"Average movie year: {avg_year} (used for missing dates)")

        
        filled_years = [avg_year if year is None else year for year in dates]
        years_tensor = torch.tensor(filled_years, dtype=torch.int64, device=self.device)

       
        movies_date = F.one_hot(years_tensor).float()  


        genre_to_idx = {}
        for m in meta_datas:
            genres = m[1].split('|')
            for g in genres:
                if g not in genre_to_idx:
                    genre_to_idx[g] = len(genre_to_idx)  

        num_genres = len(genre_to_idx)
        movies_genres_matrix = torch.zeros((meta_datas.shape[0], num_genres), device=self.device)
        for i, m in enumerate(meta_datas):
            genres = m[1].split('|')
            for g in genres:
                if g in genre_to_idx:
                    movies_genres_matrix[i, genre_to_idx[g]] = 1


        if rating_counts is not None:
            if not isinstance(rating_counts, torch.Tensor):
                rating_counts = torch.tensor(rating_counts, device=self.device)
            else:
                rating_counts = rating_counts.to(self.device)
            log_counts = torch.log1p(rating_counts).unsqueeze(1)  # log(1 + count)
        else:
            log_counts = torch.zeros(meta_datas.shape[0], 1, device=self.device)
            #print("Warning: rating_counts not provided, using zeros for log_count.")


        item_metadata = torch.cat([movies_date, movies_genres_matrix, log_counts], dim=1)

        # print(f"Metadata processed - Date features: {movies_date.shape[1]}, "
        #     f"Genre features: {movies_genres_matrix.shape[1]}, "
        #     f"LogCount feature added: {log_counts.shape[1]}")

        return item_metadata
        
    def train(self, epochs=20, val_matrix=None, recon_lambda=0.001, weight_decay=1e-5):
        """Train the NeuMF model with Joint SDAE Reconstruction Loss"""
        print(f"\nTraining with Joint SDAE Loss (lambda={recon_lambda}) for {epochs} epochs...")
        
        train_idx = self.mask.nonzero(as_tuple=False)
        user_indices = train_idx[:, 0]
        item_indices = train_idx[:, 1]
        # ratings是原始 0-5 评分（用于 Huber 目标）
        ratings = self.R[user_indices, item_indices]
        user_indices = user_indices.to(self.device)  # <-- ADD THIS LINE
        item_indices = item_indices.to(self.device)  # <-- ADD THIS LINE
        # FIX: 使用 AdamW 和 Weight Decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        best_metric = float('inf')
        
        # ... (Validation Data preparation, similar to original)
        val_idx, val_ratings, val_ratings_normalized = None, None, None
        has_validation = val_matrix is not None
        if has_validation:
            val_matrix_tensor = torch.tensor(np.nan_to_num(val_matrix), dtype=torch.float32).to(self.device)
            val_idx = (val_matrix_tensor > 0).nonzero(as_tuple=False)
            val_ratings = val_matrix_tensor[val_idx[:, 0], val_idx[:, 1]]
            val_ratings_normalized = val_ratings / 5.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            perm = torch.randperm(len(train_idx))
            total_loss = 0
            num_batches = 0
            
            for start in range(0, len(train_idx), self.batch_size):
                end = min(start + self.batch_size, len(train_idx))
                batch_perm = perm[start:end]
                
                batch_users = user_indices[batch_perm]
                batch_items = item_indices[batch_perm]
                rating_raw = ratings[batch_perm]
                
                batch_metadata = self.item_metadata[batch_items] if self.item_metadata is not None else None
                
                # --- 1. Get SDAE Input (Full Vectors - 0-1 Normalized) ---
                user_sdae_input = self.R_full[batch_users] 
                item_sdae_input = self.R_full.T[batch_items] 
                
                # --- 2. Forward Pass ---
                predictions, user_recon, item_recon = self.model(
                    batch_users, batch_items, batch_metadata, 
                    user_sdae_input, item_sdae_input
                )
                
                # --- 3. Loss Calculation (Hybrid) ---
                
                # A. Prediction Loss (Huber on raw ratings)
                huber = torch.nn.SmoothL1Loss(beta=0.75)
                rating_norm = rating_raw / 5.0
                
                # predictions are already 0-1 normalized by Sigmoid in forward()
                pred_loss = F.binary_cross_entropy(predictions, rating_norm, reduction='mean')

                
                # B. Reconstruction Loss (Masked MSE on 0-1 normalized targets)
                
                # User Reconstruction Loss
                mask_u = (user_sdae_input > 0)
                loss_u = F.mse_loss(user_recon, user_sdae_input, reduction='none')
                loss_u = loss_u[mask_u].mean() # Mean over non-zero elements
                
                # Item Reconstruction Loss
                mask_i = (item_sdae_input > 0)
                loss_i = F.mse_loss(item_recon, item_sdae_input, reduction='none')
                loss_i = loss_i[mask_i].mean() # Mean over non-zero elements

                # Total Hybrid Loss
                loss = pred_loss + recon_lambda * (loss_u + loss_i)
                
                # --- 4. Backward Pass & Optimization ---
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # FIX: 降低裁剪范数到 1.0
                optimizer.step()
                
                total_loss += pred_loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # --- Evaluation ---
            if has_validation:
                val_rmse = self._evaluate_batch(val_idx, val_ratings, val_ratings_normalized)
                
                # Save best model
                if val_rmse < best_metric:
                    best_metric = val_rmse
                    best_model_state = self.model.state_dict().copy()
                
                print(f"Epoch {epoch}/{epochs} - Train Pred Loss: {avg_train_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}, Best Val RMSE: {best_metric:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} - Train Pred Loss: {avg_train_loss:.4f}")

        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nTraining completed! Best Validation RMSE: {best_metric:.4f}")
        
    def _get_sdae_input(self, user_indices, item_indices):
        """Helper to get SDAE input vectors."""
        user_features = self.R_full[user_indices]
        item_features = self.R_full.T[item_indices]
        return user_features, item_features

    def _evaluate_batch(self, val_idx, val_ratings, val_ratings_normalized, batch_size=256):
        """Internal evaluation function."""
        self.model.eval()
        val_user_indices = val_idx[:, 0]
        val_item_indices = val_idx[:, 1]
        val_predictions_list = []
        
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                end = min(start + batch_size, len(val_idx))
                
                batch_val_users = val_user_indices[start:end]
                batch_val_items = val_item_indices[start:end]
                
                batch_val_metadata = self.item_metadata[batch_val_items] if self.item_metadata is not None else None
                
                val_user_sdae_feat, val_item_sdae_feat = self._get_sdae_input(batch_val_users, batch_val_items)
                
                # FIX: 评估时只取 prediction，忽略 reconstruction outputs
                batch_val_pred, _, _ = self.model(
                    batch_val_users, batch_val_items, batch_val_metadata, 
                    val_user_sdae_feat, val_item_sdae_feat
                )
                val_predictions_list.append(batch_val_pred)
            
            val_predictions = torch.cat(val_predictions_list)
            
            # Calculate RMSE on original scale
            val_predictions_scaled = val_predictions * 5.0
            # FIX: 确保只在有评分的项上计算 RMSE
            val_rmse = torch.sqrt(F.mse_loss(val_predictions_scaled, val_ratings))
            return val_rmse.item()

    def predict(self, batch_size=256):
        """Generate predictions for all zero entries in the rating matrix"""
        print("\nGenerating predictions for missing ratings...")
        self.model.eval()
        
        zero_idx = (~self.mask).nonzero(as_tuple=False)
        predictions = torch.zeros(zero_idx.shape[0], device=self.device)
        
        with torch.no_grad():
            for start in tqdm.trange(0, zero_idx.shape[0], batch_size):
                end = min(start + batch_size, zero_idx.shape[0])
                
                batch_users = zero_idx[start:end, 0]
                batch_items = zero_idx[start:end, 1]
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_metadata = self.item_metadata[batch_items] if self.item_metadata is not None else None
                
                user_sdae_input, item_sdae_input = self._get_sdae_input(batch_users, batch_items)
                
                # FIX: 评估时只取 prediction
                batch_predictions, _, _ = self.model(
                    batch_users, batch_items, batch_metadata,
                    user_sdae_input, item_sdae_input
                )
                
                batch_predictions = torch.clamp(batch_predictions * 5.0, 0, 5) 
                predictions[start:end] = batch_predictions
        
        predicted_matrix = torch.sparse_coo_tensor(
            zero_idx.T.to(self.device), predictions.to(self.device), (self.num_users, self.num_items)
        )
        output = predicted_matrix.to_dense().cpu().numpy()
        print("Prediction completed!")
        return output
    
    def evaluate(self, test_matrix):
        """Evaluate model on test set"""
        print("\nEvaluating model on test set...")
        
        self.model.eval()
        
        test_matrix_tensor = torch.tensor(np.nan_to_num(test_matrix), dtype=torch.float32).to(self.device)
        test_idx = (test_matrix_tensor > 0).nonzero(as_tuple=False)
        test_ratings = test_matrix_tensor[test_idx[:, 0], test_idx[:, 1]]
        
        test_rmse = self._evaluate_batch(test_idx, test_ratings, None)
        
        # Recalculate MAE separately for the output log
        test_predictions_list = []
        with torch.no_grad():
            for start in range(0, len(test_idx), self.batch_size):
                end = min(start + self.batch_size, len(test_idx))
                
                batch_users = test_idx[start:end, 0]
                batch_items = test_idx[start:end, 1]
                batch_metadata = self.item_metadata[batch_items] if self.item_metadata is not None else None
                
                user_sdae_input, item_sdae_input = self._get_sdae_input(batch_users, batch_items)
                
                batch_predictions, _, _ = self.model(
                    batch_users, batch_items, batch_metadata,
                    user_sdae_input, item_sdae_input
                )
                batch_predictions = torch.clamp(batch_predictions * 5.0, 0, 5)
                test_predictions_list.append(batch_predictions)
            
            final_predictions = torch.cat(test_predictions_list)
            mae = F.l1_loss(final_predictions, test_ratings)
        
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {mae.item():.4f}")
        
        return test_rmse, mae.item()
        
    def save_model(self, path="neumf_sdae_joint.pt"):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path="neumf_sdae_joint.pt"):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
def set_seed(seed=42):
    """Sets the seed for reproducibility across numpy, torch, and Python standard library."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # 推荐使用 deterministic 模式（虽然可能会略微降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Python standard library random module
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
def run_neumf(table):
    
    METADATA_PATH = "namesngenre.npy"
    
   
    K_FOLDS = 5 
    GMF_DIM = 128
    MLP_DIM = 128
    MLP_LAYERS = [64, 32] 
    DROPOUT_RATE = 0.4      
    
    SDAE_LATENT_DIM = 128
    SDAE_HIDDEN_LAYERS = [128] 
    
    LEARNING_RATE = 8e-4    
    RECONSTRUCTION_LAMBDA = 2 
    JOINT_EPOCHS = 5     
    WEIGHT_DECAY = 1e-4    

    R_full = table.copy()
    M_train = ~np.isnan(R_full)
    

    obs_indices = M_train.nonzero()
    obs_indices_flat = np.array(list(zip(*obs_indices))) 
    N_obs = len(obs_indices_flat)


    best_model_states = []
    
 
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation Training ---")

    for fold, (train_idx_obs, val_idx_obs) in enumerate(kf.split(obs_indices_flat)):
        print(f"\nTraining Fold {fold + 1}/{K_FOLDS}...")
        
        
        R_k_train = R_full.copy()
        
        
        val_coords = obs_indices_flat[val_idx_obs] 
        
        
        R_val_matrix = np.full_like(R_full, np.nan, dtype=np.float32)
        
        for u, i in val_coords:
            R_val_matrix[u, i] = R_k_train[u, i] 
            R_k_train[u, i] = np.nan            
            
        
        model_k = NeuMFRecommender(
            R=R_k_train,
            metadata_path=METADATA_PATH,
            learning_rate=LEARNING_RATE,
            batch_size=256,
            gmf_dim=GMF_DIM, mlp_dim=MLP_DIM, mlp_layers=MLP_LAYERS,
            dropout=DROPOUT_RATE, use_metadata=True,
            sdae_latent_dim=SDAE_LATENT_DIM, sdae_hidden_layers=SDAE_HIDDEN_LAYERS
        )
        
        model_k.train(
            epochs=JOINT_EPOCHS, 
            val_matrix=R_val_matrix, 
            recon_lambda=RECONSTRUCTION_LAMBDA,
            weight_decay=WEIGHT_DECAY 
        )
        

        best_model_states.append(model_k.get_state_dict())
        del model_k
        torch.cuda.empty_cache()



    zero_idx = np.argwhere(np.isnan(R_full))
    

    prediction_models = []
    for state_dict in best_model_states:
        
        model_pred = NeuMFRecommender(
            R=R_full, 
            metadata_path=METADATA_PATH,

            gmf_dim=GMF_DIM, mlp_dim=MLP_DIM, mlp_layers=MLP_LAYERS,
            dropout=DROPOUT_RATE, use_metadata=True,
            sdae_latent_dim=SDAE_LATENT_DIM, sdae_hidden_layers=SDAE_HIDDEN_LAYERS
        )
      
        model_pred.model.load_state_dict(state_dict)
        prediction_models.append(model_pred)

  
    all_fold_predictions = []
    print("\n--- Starting K-Fold Ensemble Prediction ---")
    

    for model_pred in prediction_models:
      
        fold_output = model_pred.predict() 
        
      
        fold_predictions_flat = fold_output[zero_idx[:, 0], zero_idx[:, 1]]
        all_fold_predictions.append(fold_predictions_flat)

   
    all_fold_predictions = np.stack(all_fold_predictions, axis=0) 
    avg_predictions_flat = np.mean(all_fold_predictions, axis=0) 
    
   
    final_output = np.zeros_like(R_full, dtype=np.float32)
  
    for idx, (u, i) in enumerate(zero_idx):
        final_output[u, i] = avg_predictions_flat[idx]
        
    return final_output



