# Neural CF with embeddings
"""
Idea: Because you have explicit ratings (Score), we can build a model that learns:
- a user embedding | an item embedding | a small nn that combines them | a predicted rating for each user-item pair
- nn.Embedding is designed to map integer IDs to dense embedding vectors
- nn.MSELoss measures MSE between predicted and true values, which fits a rating-prediction setup well. 
Pipeline
1. load and clean the data
2. keep users/items with enough ratings
3. split into train/test
4. map UserId and ProductId into integer indices
5. build a PyTorch Dataset and DataLoader
6. build a neural recommender model
7. train it with MSELoss
8. evaluate with RMSE and MAE
Code thinking process
1. Keep the model simple: embedding layers for users & items-> combine them-> pass through a MLP -> predict
2. Convert raw IDs into integer indices - eg user_to_idx["A3SGXH7AUHU8GW"] = 0 | item_to_idx["B001E4KFG0"] = 0
3. Keep train/test split explicit - train_df; test_df
4. Predict ratings directly
Steps
1. Collect interaction data: UserId | ProductId | Score
2. Filter the data
3. split train and test
4. Convert Ids into integer indices (for embeddings)
    - eg. user "A3SGXH7AUHU8GW" becomes 17; then the embedding layers can look up a dense vector for user 17. 
    - PyTorch embedding layer is designed as a lookup table from integer indicees to dense vectors. 
5. Learn user and item embeddings - A user/item is no longer just a raw ID-he model leans a compact numeric representation of it. 
6. Combine them with a nn - e concatenate the user and item vectos, then feed them through a small multilayer perceptron
    - So the model learns (u,i)->r - That is more flexible than simple dot-product-style classical CF.
7. Train with MSE loss: MSE = avg(sum(predn - actualn)^2)
    - PyTorch documents MSELoss as mean squared error between pred and target
8. Evaluate with RMSE and MAE
    - RMSE =  √MSE -- (This penalizes larger errors more strongly, sensitive to outliers)
    - MAE = avg(sum(abs(predn - actualn))) -- (treating all errors equally and offering better robustness to outliers)
Takeaways
When training you might see:
- training loss decreases over epoches | test RMSE and MAE improve at first | then maybe level off
    - This means the model is learning. 
- If the test metrics get worse while training loss keeps improving, that may sugget overfitting.    
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
        if i+1 > n: break
    return rows

def rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    return df

def keep_reco_cols(df):
    reco_df = df[["UserId", "ProductId", "Score"]].copy()
    return reco_df

def clean_reco_data(reco_df):
    reco_df = reco_df.copy()
    reco_df = reco_df.dropna(subset=["UserId", "ProductId", "Score"])
    reco_df = reco_df.drop_duplicates()
    return reco_df

def filter_ative_users_and_popular_items(reco_df, min_user_ratings=5, min_item_ratings=5):
    user_counts = reco_df["UserId"].value_counts()
    active_users = user_counts[user_counts>=min_user_ratings].index
    filtered_df = reco_df[reco_df["UserId"].isin(active_users)].copy()
    item_counts = filtered_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts>=min_item_ratings].index
    filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def train_test_split_random(reco_df, test_size=0.2, random_state=42):
    test_df = reco_df.sample(frac=test_size, random_state=random_state)
    train_df = reco_df.drop(index=test_df.index) # use remaining rows as train set
    # Reset indexes
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

def build_id_mappings(train_df):
    # Build mappings from raw uer/item IDs to integer indices.
    unique_users = train_df["UserId"].unique()
    unique_items = train_df["ProductId"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)} # dict that set user_id as key
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    return user_to_idx, item_to_idx

def apply_id_mappings(df, user_to_idx, item_to_idx):
    # Apply user/item integer mappings and keep only rows that exist in both mappings. 
    # Returns: mapped_df: Df with integer user/item indices
    mapped_df = df.copy()
    mapped_df["user_idx"] = mapped_df["UserId"].map(user_to_idx)
    mapped_df["item_idx"] = mapped_df["ProductId"].map(item_to_idx)
    mapped_df = mapped_df.dropna(subset=["user_idx", "item_idx"]).copy()
    mapped_df["user_idx"] = mapped_df["user_idx"].astype(int) # convert mapped indices to integers
    mapped_df["item_idx"] = mapped_df["item_idx"].astype(int)
    return mapped_df

class RatingsDataset(Dataset): # PyTorch Dataset for user-item-rating triples
    def __init__(self, df): # Store user indices, item indices, and targets
        self.user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long) # Store he user indices as a tensor
        self.item_tensor = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating_tensor = torch.tensor(df["Score"].values, dtype=torch.float32) # Store the raing targets as float tensors

    def __len__(self): # Return the # of rows in the dataset
        return len(self.rating_tensor)
    
    def __getitem__(self, idx): # return one training eg
        return (self.user_tensor[idx], self.item_tensor[idx], self.rating_tensor[idx])
    
class NeuralRecommender(nn.Module): # Simple neural collaborative filtering model
    # Architecture: user embedding -> item embedding -> concatenate both -> small multilayer perceptron -> output one predicted rating
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64): # Initialize the model layers
        super().__init__() # Initialize the parent nn.Module
        self.user_embedding = nn.Embedding(num_users, embedding_dim) # Create a learned embedding table for user
        self.item_embedding = nn.Embedding(num_items, embedding_dim) # Creeate a learned embedding table for items
        self.mlp = nn.Sequential( # Build a small feedforward network for rating prediction
            nn.Linear(embedding_dim*2, hidden_dim), # first linear transformation: z=wx+b
            #dim*2 becaause user emb and item emb eash has size 32 total is 64
            nn.ReLU(), #max(0,x) - if >0, keep it; if<0, change it to 0; it makes the model nonlinear
            nn.Linear(hidden_dim, 32), # 2nd linear transformation - compress the hidden from 64 to 32
            # 1st layer: mix user & item info; 2nd layer: refine the important signals
            nn.ReLU(),
            nn.Linear(32, 1) # final linear transformation - it takes 32 learned features and outputs one #
        )
    
    def forward(self, user_idx, item_idx): # Forward pass of the model
        user_vec = self.user_embedding(user_idx) # Look up user embeddings
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec], dim=1) # Concatenate user and item vectors
        out = self.mlp(x) # Pass through the MLP
        return out.squeeze(1) # Squeeze the output to shape [batch_size]
       
def create_dataloaders(train_df, test_df, batch_size=256): # Create PyTorch dataloaders for training and testing
    # The model does not train directly from a pandas DF, it needs data in PyTorch batch form.
    train_dataset = RatingsDataset(train_df) # Build the training dataset obj
    test_dataset = RatingsDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create the training Dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, dataloader, optimizer, criterion, device): # Train the model for one epoch - Return: avg_loss
    model.train()
    total_loss = 0.0 # Track total loss across all batches
    for user_idx, item_idx, ratings in dataloader:
        user_idx = user_idx.to(device) # Move tensors to device
        item_idx = item_idx.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad() # Clear previous gradients
        predictions = model(user_idx, item_idx) # model = NeuralRecommender() It's the same as model.forward(user_idx, item_idx)
        loss = criterion(predictions, ratings) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step() # Update model params
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader) # Compute avg loss
    return avg_loss

def evaluate_model(model, dataloader, device): # Evaluate the model using RMSE and MAE
    model.eval() # Set the model to eval mode
    all_preds = []
    all_targets = []
    with torch.no_grad(): # Disable gradient computation during evaluation - it's only for training
        for user_idx, item_idx, ratings in dataloader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            ratings = ratings.to(device)
            pred = model(user_idx, item_idx)
            all_preds.extend(pred.cpu().numpy()) # PyTorch tensors -> Numpy array for mse calculation
            all_targets.extend(ratings.cpu().numpy())
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        mse = np.mean((all_preds - all_targets)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets))
        return rmse, mae
                            

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Choose GPU is available, otherwise CPU
    print("Using device:", device)
    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)
    rows = collect_n_rows(train_ds, n=20000)
    df = rows_to_dataframe(rows)
    reco_df = keep_reco_cols(df)
    reco_df = clean_reco_data(reco_df)
    reco_df = filter_ative_users_and_popular_items(reco_df, 5, 5)
    print("\nFiltered recommendation DataFrame shape:")
    print(reco_df.shape)
    train_df, test_df = train_test_split_random(reco_df, test_size=0.2, random_state=42)
    user_to_idx, item_to_idx = build_id_mappings(train_df)
    train_df = apply_id_mappings(train_df, user_to_idx, item_to_idx)
    test_df = apply_id_mappings(test_df, user_to_idx, item_to_idx)
    print("\nMapped train shape:", train_df.shape)
    train_loader, test_loader = create_dataloaders(train_df, test_df, batch_size=256)
    model = NeuralRecommender(num_users=len(user_to_idx), num_items=len(item_to_idx), embedding_dim=32, hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs=5
    for epoch in range(num_epochs):
        train_loss=train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
        rmse, mae = evaluate_model(model, test_loader, device)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Train Loss:", round(train_loss, 4))
        print("Test RMSE", round(rmse, 4))
        print("Test MAE:", round(mae, 4))

if __name__ == "__main__":
    main() 







