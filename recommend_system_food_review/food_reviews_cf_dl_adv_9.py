# Improved Deep Learning
"""
with global mean | user bias | item bias | user embedding | item embedding | bounded prediction range
Ideas:
- global mean (u) - Some datasets just have an overall avg rating
- user bias (Bu) - Some uesrs tend to rate higher or lower than avg
- item bias (Bi) - some products are brodly liked or disliked
- r = u + Bu + Bi + f(u,i)
    - f(u,i) = neural interaction part from embeddings
- bounded pred range (1<=r<=5); without bounding, it might predict 5.8,0.2...
    - torch.clamp(pred, min=1.0, max=5.0)
Code thinking process
1. Keep the data preprocessing pipeline: load the data->keep 3 cols->clean & filter->split->map->DF & dataloader
2. Add explicit bias terms - for each user u, the model learns Bu, for item, the model learns Bi
    - since nn.Embedding is a learnable lookup table, using embedding dimension 1 is a clean way to represent scalar biases
3.Store the global mean inside the model: u = mean of all train ratings
4. Keep the neural interaction part: r = u = Bu + Bi + MLP(Pu,Qi)
    - Pu = user embedding vector | Qi = item embedding vector
Detailed explanation
- self.user_bias = nn.Embedding(num_users, 1) - this creates one learned # per user.
- self.register_buffer("global_mean", torch.tensor(float(global_mean))) - This stores the global mean inside the model as a tensor buffer.
    - Why buffer? It should move with the model to CPU/GPU; but it is not meant to be optimized like a normal param
The forward() formula - r = u + Bu + Bi + MLP([Pu, Qi])
1. look up user embedding & item embedding |2. look up user bias & item bias
3. compupte neural interaction | 4. add all parts together | 5. clamp the result to [1,5]
Why this can improve performance - bcz it separates simple effects from complex effects.
    - Instead of MLP learning everything, we let the model explicitly represent:
        - the avg score; user generosity/strictness; item popularity/quality
    - Then the nn onlyy has to learn the remaining interaction patter. (easier)
Improvement
- Compare the eariler nn, this verson has a better inductive bias for rating pred.
    - You may see: faster convergence; slightly better RMSE/MAE; more stable preds
Mental model - why it's stronger than the plain embedding + MLP
- Baseline layer: u + Bu + Bi  -- This says start with the avg, then adj for this user and this item. 
- Interaction layer: f(u,i) -- This says now add a learned interaction correction based on embeddings.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def load_food_reviews_stream():
    ds = load_dataset("jhan21/amazon-food-reviews-dataset", streaming=True)
    return ds

def get_train_split(ds):
    train_ds = ds["train"]
    return train_ds

def collect_n_rows(train_ds, n=20000):
    rows = []
    for i, row in enumerate(train_ds):
        rows.append(row)
        if i+1 >= n: break
    return rows

def rows_to_df(rows):
    df = pd.DataFrame(rows)
    return df

def keep_rec_cols(df):
    reco_df = df[["UserId", "ProductId", "Score"]].copy()
    return reco_df

def clean_rec_data(reco_df):
    reco_df = reco_df.copy()
    reco_df = reco_df.dropna(subset=["UserId", "ProductId", "Score"])
    reco_df = reco_df.groupby(["UserId", "ProductId"], as_index=False)["Score"].mean() # Collapse repeated user-item pairs into one avg score
    return reco_df

def filter_active_users_and_popular_items(reco_df, min_user_ratings=5, min_item_ratings=5):
    user_counts = reco_df["UserId"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    filtered_df = reco_df[reco_df["UserId"].isin(active_users)].copy()
    item_counts = filtered_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts >= min_item_ratings].index
    filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def train_test_split_random(reco_df, test_size=0.2, random_state=42):
    test_df = reco_df.sample(frac=test_size, random_state=random_state)
    train_df = reco_df.drop(index=test_df.index)
    train_df = train_df.reset_index(drop=True) # reset_index: indexes reset to 12345 instead of 135; drop-delete index col
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

def build_id_mappings(train_df):
    unique_users = train_df["UserId"].unique()
    unique_items = train_df["ProductId"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    return user_to_idx, item_to_idx

def apply_id_mappings(df, user_to_idx, item_to_idx):
    mapped_df = df.copy()
    mapped_df["user_idx"] = mapped_df["UserId"].map(user_to_idx)
    mapped_df["item_idx"] = mapped_df["ProductId"].map(item_to_idx)
    mapped_df = mapped_df.dropna(subset=["user_idx", "item_idx"]).copy()
    mapped_df["user_idx"] = mapped_df["user_idx"].astype(int)
    mapped_df["item_idx"] = mapped_df["item_idx"].astype(int)
    return mapped_df

class RatingsDataset(Dataset): # PyTorch Dataset for user-item-rating triples
    def __init__(self, df): # Stoer user indices, item indices, and rating targets
        self.user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_tensor = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating_tensor = torch.tensor(df["Score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.rating_tensor)

    def __getitem__(self, idx): # Return one training eg
        return (self.user_tensor[idx], self.item_tensor[idx], self.rating_tensor[idx])
    
class NeuralRecWithBias(nn.Module):
    # Improve nn with global mean | user & item bias | user/item embeddings | MLP | bounded predictions
    def __init__(self, num_users, num_items, global_mean, embedding_dim=32, hidden_dim=64, min_rating=1.0, max_rating=5.0): # min_rating: min allowed rating
        super().__init__() # Initialize the parent nn.Module
        self.register_buffer("global_mean", torch.tensor(float(global_mean))) # Save the global mean as a non-trainable tensor buffer
        self.min_rating = float(min_rating) # save rating bounds
        self.max_rating = float(max_rating)
        self.user_embedding = nn.Embedding(num_users, embedding_dim) # Create learned user embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1) # Create learned scalar user bias terms
        self.item_bias = nn.Embedding(num_items, 1) 
        self.mlp = nn.Sequential( # Build a small feedforward network for interaction effects
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 32), 
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, user_idx, item_idx): # Forward pass of the model
        user_vec = self.user_embedding(user_idx) # Look up user embedding vectors
        item_vec = self.item_embedding(item_idx)
        user_b = self.user_bias(user_idx).squeeze(1) # Look up scalar user biases
        item_b = self.item_bias(item_idx).squeeze(1)
        x = torch.cat([user_vec, item_vec], dim=1) # Concatenate user and item embeddings
        interaction = self.mlp(x).squeeze(1) # Compute nonlinear interaction term
        prediction = self.global_mean + user_b + item_b + interaction # Combine mean, biases, and interaction
        prediction = torch.clamp(prediction, min=self.min_rating, max=self.max_rating) # Clamp pred into the valid rating range
        return prediction
    
def create_dataloaders(train_df, test_df, batch_size=256):
    train_dataset = RatingsDataset(train_df)
    test_dataset = RatingsDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # set model to train mode (need it bcz certain layer behave differently on whether you train or eval)
    total_loss=0.0 # Track total loss
    for user_idx, item_idx, ratings in dataloader:
        user_idx = user_idx.to(device) # move batch to device
        item_idx = item_idx.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad() # Clear old gradients
        preds = model(user_idx, item_idx) # predict ratings
        loss = criterion(preds, ratings) # Compute loss
        loss.backward() # Backpropagate
        optimizer.step() # Update params
        total_loss += loss.item() # Add batch loss
    return total_loss/len(dataloader) # Return avg batch loss

def evaluate_model(model, dataloader, device):
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_targets=[]

    with torch.no_grad(): # Disable grad computation during eval
        for user_idx, item_idx, ratings in dataloader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            ratings = ratings.to(device)
            preds = model(user_idx, item_idx)
            all_preds.extend(preds.cpu().numpy()) # Move results to CPU and save them
                # preds.numpy(): tensor -> array; extend(): add array to list
            all_targets.extend(ratings.cpu().numpy())
    all_preds = np.array(all_preds) # list->array(for later rate computation)
    all_targets = np.array(all_targets)
    mse = np.mean((all_preds - all_targets)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    return mse, mae
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choose device
    print("Using device:", device)
    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)
    rows = collect_n_rows(train_ds)
    df = rows_to_df(rows)
    reco_df = keep_rec_cols(df)
    reco_df = clean_rec_data(reco_df)
    reco_df = filter_active_users_and_popular_items(reco_df, 5, 5)
    print("\nFiltered recommendation DataFrame shape:", reco_df.shape)
    train_df, test_df = train_test_split_random(reco_df, test_size=0.2, random_state=42)
    global_mean = train_df["Score"].mean()
    print("\nTraining global mean rating:", round(global_mean, 4))
    user_to_idx, item_to_idx = build_id_mappings(train_df) # Build int mapping
    train_df = apply_id_mappings(train_df, user_to_idx, item_to_idx) # apply mapping
    test_df = apply_id_mappings(test_df, user_to_idx, item_to_idx)
    print("\nMapped train shape:", train_df.shape)
    print("\nMapped test shape:", test_df.shape)
    train_loader, test_loader = create_dataloaders(train_df, test_df, batch_size=356)
    model = NeuralRecWithBias(num_users=len(user_to_idx), num_items=len(item_to_idx),
                              global_mean=global_mean, embedding_dim=32, hidden_dim=64,
                              min_rating=1.0, max_rating=5.0).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs=20
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        rmse, mae = evaluate_model(model, test_loader, device)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Train Loss:", round(train_loss, 4))
        print("Test RMSE:", round(rmse, 4))
        print("Test MAE:", round(mae, 4))

if __name__ == "__main__":
    main()
    


