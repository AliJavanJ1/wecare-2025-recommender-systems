import numpy as np
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        return x


def run(table):
    ratings_train_np = table
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    user_movie_connections_np = np.argwhere(~np.isnan(ratings_train_np)).T
    
    namesngenre_np = np.load('namesngenre.npy')
    all_genres = set()
    for name, genres in namesngenre_np:
        for genre in genres.split('|'):
            all_genres.add(genre)
    all_genres = sorted(list(all_genres))
    no_genres = '(no genres listed)'
    all_genres = [g for g in all_genres if g != no_genres]
    genre_one_hot_np = np.apply_along_axis(lambda x: [1 if g in x[1].split('|') else 0 for g in all_genres], 1, namesngenre_np)
    
    x = torch.tensor(np.concatenate((np.random.random_sample((ratings_train_np.shape[0], genre_one_hot_np.shape[1])), genre_one_hot_np), axis=0), dtype=torch.float32)

    edge_index = torch.tensor(user_movie_connections_np, dtype=torch.long)
    edge_index[1] += ratings_train_np.shape[0]
    
    graph_data = Data(x=x, edge_index=edge_index)
    graph_data = graph_data.to(device)
    
    
    model = GCN(in_channels=x.shape[1], out_channels=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    actual_ratings = torch.tensor(ratings_train_np[~np.isnan(ratings_train_np)], dtype=torch.float32)
    actual_ratings = actual_ratings.to(device)
    
    
    print("Starting training...")
    for epoch in range(10001):
        optimizer.zero_grad()
        embeddings = model(graph_data.x, graph_data.edge_index)
        user_embeds = embeddings[graph_data.edge_index[0]]
        movie_embeds = embeddings[graph_data.edge_index[1]]
        logits = (user_embeds * movie_embeds).sum(dim=1)
        predictions = torch.sigmoid(logits) * 4.5 + .5
        loss = criterion(predictions, actual_ratings)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Training finished.")
    
    final_embeddings = model(graph_data.x, graph_data.edge_index).detach()

    num_users = ratings_train_np.shape[0]
    final_user_embeddings = final_embeddings[:num_users]
    final_movie_embeddings = final_embeddings[num_users:]
    
    full_prediction_matrix = torch.matmul(final_user_embeddings, final_movie_embeddings.T)
    
    return full_prediction_matrix.detach().cpu().numpy()