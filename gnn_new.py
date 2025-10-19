import numpy as np
import torch
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
import re


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * 2, heads=heads, dropout=0.1)
        self.conv3 = GATConv(hidden_channels * 2 * heads, out_channels, heads=1, concat=False, dropout=0.1)


    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


def run(table):

    SEED = 12345
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    VALIDATION_SPLIT = 0.05
    NUM_FEATURES = 32


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ratings_train_np = np.load('ratings_train.npy')
    ratings_train_np = table
    namesngenre_np = np.load('namesngenre.npy')
    ratings_test_np = np.load('ratings_test.npy')

    year_pattern = re.compile(r"\((\d{4})\)")
    all_decades = set(int(year_pattern.search(name.item()).group(1)) // 10 * 10 for name, _ in namesngenre_np if year_pattern.search(name.item()))
    num_decades = len(all_decades)


    sorted_decades = sorted(all_decades)
    decade_to_index = {decade: idx for idx, decade in enumerate(sorted_decades)}
    # Each edge connects a movie (row index in namesngenre_np) to its release decade
    decade_edges = []
    for movie_idx, (raw_name, _) in enumerate(namesngenre_np):
        match = year_pattern.search(raw_name.item())
        if match:
            decade = int(match.group(1)) // 10 * 10
            decade_edges.append((movie_idx, decade_to_index[decade]))
    decade_edges = torch.tensor(decade_edges, dtype=torch.long).t() if decade_edges else torch.empty((2, 0), dtype=torch.long)


    all_genres = set(g for _, genres in namesngenre_np for g in genres.split('|'))
    all_genres.discard('(no genres listed)')
    all_genres = sorted(list(all_genres))
    num_genres = len(all_genres)

    genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}
    # Each edge connects a movie to one of its genres without applying offsets
    genre_edges = []
    for movie_idx, (_, genres_str) in enumerate(namesngenre_np):
        for genre in genres_str.split('|'):
            genre = genre.strip()
            if genre in genre_to_index:
                genre_edges.append((movie_idx, genre_to_index[genre]))
    genre_edges = torch.tensor(genre_edges, dtype=torch.long).t() if genre_edges else torch.empty((2, 0), dtype=torch.long)


    num_users, num_movies = ratings_train_np.shape


    x = torch.rand(num_users + num_movies + num_genres + num_decades, NUM_FEATURES)


    user_ids, movie_ids = np.where(~np.isnan(ratings_train_np))
    ratings = ratings_train_np[~np.isnan(ratings_train_np)]

    test_user_ids, test_movie_ids = np.where(~np.isnan(ratings_test_np))
    test_ratings = ratings_test_np[~np.isnan(ratings_test_np)]

    edges = np.column_stack((user_ids, movie_ids))
    train_edges, val_edges, train_ratings, val_ratings = train_test_split(
        edges, ratings, test_size=VALIDATION_SPLIT, random_state=SEED, shuffle=True
    )
    train_edges, val_edges = train_edges.T, val_edges.T

    test_edges = np.column_stack((test_user_ids, test_movie_ids)).T
    ratings = ratings.astype(np.float32)
    test_ratings = test_ratings.astype(np.float32)



    def offset_edges(edge_tensor, src_offset=0, dst_offset=0):
        if edge_tensor.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        adjusted = edge_tensor.clone()
        adjusted[0, :] += src_offset
        adjusted[1, :] += dst_offset
        return adjusted

    movie_offset = num_users
    genre_offset = num_users + num_movies
    decade_offset = num_users + num_movies + num_genres
    
    train_edge_index = torch.tensor(train_edges, dtype=torch.long)
    val_edge_index = torch.tensor(val_edges, dtype=torch.long)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long)

    train_edge_index = offset_edges(train_edge_index, 0, movie_offset)
    val_edge_index = offset_edges(val_edge_index, 0, movie_offset)
    test_edge_index = offset_edges(test_edge_index, 0, movie_offset)

    train_actual_ratings = torch.tensor(train_ratings, dtype=torch.float32)
    val_actual_ratings = torch.tensor(val_ratings, dtype=torch.float32)
    test_actual_ratings = torch.tensor(test_ratings, dtype=torch.float32)

    genre_edge_index = offset_edges(genre_edges, movie_offset, genre_offset)
    decade_edge_index = offset_edges(decade_edges, movie_offset, decade_offset)


    graph_data = Data(x=x, edge_index=torch.concat([train_edge_index, genre_edge_index, decade_edge_index], axis=1), y=train_actual_ratings)
    graph_data = graph_data.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    val_actual_ratings = val_actual_ratings.to(device)
    test_actual_ratings = test_actual_ratings.to(device)



    OUTPUT_DIM = 4
    HIDDEN_CHANNELS = 32
    HEADS = 8
    SCALE_FUNCTION_NAME = 'sigmoid'  # Options: 'sigmoid' or 'clamp'
    scale_functions = {
        'sigmoid': lambda x: torch.sigmoid(x) * 4.5 + .5,
        'clamp': lambda x: x.clamp(.5, 5.0)
    }
    scale_function = scale_functions[SCALE_FUNCTION_NAME]
    model = GAT(in_channels=x.shape[1], hidden_channels=HIDDEN_CHANNELS, out_channels=OUTPUT_DIM, heads=HEADS)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None

    print("Starting training...")
    for epoch in range(81):
        model.train()
        optimizer.zero_grad()
        embeddings = model(graph_data.x, graph_data.edge_index)
        user_embeds = embeddings[train_edge_index[0]]
        movie_embeds = embeddings[train_edge_index[1]]
        logits = (user_embeds * movie_embeds).sum(dim=1)
        if SCALE_FUNCTION_NAME == 'sigmoid':
            predictions = scale_function(logits)
        else:
            predictions = logits
        loss = criterion(predictions, graph_data.y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            eval_embeddings = model(graph_data.x, graph_data.edge_index)
            val_user_embeds = eval_embeddings[val_edge_index[0]]
            val_movie_embeds = eval_embeddings[val_edge_index[1]]
            val_logits = (val_user_embeds * val_movie_embeds).sum(dim=1)
            val_predictions = scale_function(val_logits)
            val_loss = criterion(val_predictions, val_actual_ratings)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
    print("Training finished.")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index)
        
        val_user_embeds = embeddings[val_edge_index[0]]
        val_movie_embeds = embeddings[val_edge_index[1]]
        val_logits = (val_user_embeds * val_movie_embeds).sum(dim=1)
        val_predictions = scale_function(val_logits)
        val_loss = criterion(val_predictions, val_actual_ratings)
        print(f'Best Val Loss: {val_loss:.4f}, during training was {best_val_loss:.4f}')
        
        test_user_embeds = embeddings[test_edge_index[0]]
        test_movie_embeds = embeddings[test_edge_index[1]]
        test_logits = (test_user_embeds * test_movie_embeds).sum(dim=1)
        test_predictions = scale_function(test_logits)
        test_loss = criterion(test_predictions, test_actual_ratings)
        print(f'Test Loss: {test_loss:.4f}')
        
        user_embeds = embeddings[range(num_users)]
        movie_embeds = embeddings[range(num_users, num_users + num_movies)]
        logits = user_embeds @ movie_embeds.T
        all_predictions = scale_function(logits).cpu().numpy()
        
        # load test data and calculate test RMSE
        ratings_test_np = np.load('ratings_test.npy')
        from sklearn.metrics import root_mean_squared_error
        print(test_predictions)
        print(all_predictions[~np.isnan(ratings_test_np)])
        print("real test ratings")
        print(test_actual_ratings)
        print(ratings_test_np[~np.isnan(ratings_test_np)])
        print(root_mean_squared_error(ratings_test_np[~np.isnan(ratings_test_np)], all_predictions[~np.isnan(ratings_test_np)]))
    return all_predictions