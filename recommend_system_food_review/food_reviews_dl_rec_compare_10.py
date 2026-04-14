# Deep Learning Recommenders Ablation Comparison + Top Rec
"""
Ablation study means: keep most things the same, change one important design idea, and measure the diff.
    - There we test whether: global mean | user/item bias | bounded preds help
Compare
- Model A: plain neural recommender
    - r = f(u,i); where f(u,i) comes from user & item embedding, MLP
- Model B: neural recommender with global mean | user bias | item bias | bounded predictions
    - r = u + Bu + Bi + f(u,i)
- Compare in RMSE, MAE
Code writing thinking process
1. Keep prerocessing the same - clean | split | mapping | dataloader
2. Train both models with the same settings - same: optimizer | learning rate | batch size | # of epochs
    - Then the main diff is the model architecture.
3. Store metrics for both models - At each epoch, we record: training loss | test RMSE | test MAE
Code steps
1. Prepare one shared dataset   2. Train the plain model.  3. Train the bias-enhanced model
4. Store epoch-by-epoch results.   5. Compare the final metrics
Results interpretation
- If the bias model has lower RMSE & MAE - means the added bias structure helped rating prediction
- If similar - means the extra design did not help much on this sample
- If the plain model does better - means the bias model needs more tuning | the sample is small | the architecture choice matters
Important function: train_and_evaluate_model(...)
1. train for several epochs; 2. evaluate after each epoch; 3. store the metrics; 4. return a clean history table
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
        if i>=n: break
        rows.append(row)
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
    reco_df = reco_df.groupby(["UserId", "ProductId"], as_index=False)["Score"].mean()
    return reco_df

def filter_active_users_and_popuar_items(reco_df, min_user_ratings=5, min_item_ratings=5):
    user_counts = reco_df["UserId"].value_counts()
    active_users = user_counts[user_counts>=min_user_ratings].index
    filtered_df = reco_df[reco_df["UserId"].isin(active_users)].copy()
    item_counts = filtered_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts>=min_item_ratings].index
    filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def train_test_split_random(reco_df, test_size=0.2, random_state=42):
    test_df = reco_df.sample(frac=test_size, random_state=random_state)
    train_df = reco_df.drop(index=test_df.index)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

def build_id_mappings(train_df):
    unique_users = train_df["UserId"].unique()
    unique_items = train_df["ProductId"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    return user_to_idx, item_to_idx

def apply_id_mappings(df, user_to_idx, item_to_idx): # the df can be train_df or clean_rec_df, when calling
    mapped_df = df.copy()
    mapped_df["user_idx"] = mapped_df["UserId"].map(user_to_idx)
    mapped_df["item_idx"] = mapped_df["ProductId"].map(item_to_idx)
    mapped_df = mapped_df.dropna(subset=["user_idx", "item_idx"]).copy()
    mapped_df["user_idx"] = mapped_df["user_idx"].astype(int)
    mapped_df["item_idx"] = mapped_df["item_idx"].astype(int)
    return mapped_df

class RatingDataset(Dataset):
    def __init__(self, df): 
        self.user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long) #long for integers
        self.item_tensor = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating_tensor = torch.tensor(df["Score"].values, dtype=torch.float32)
    def __len__(self):
        return len(self.rating_tensor)
    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.rating_tensor[idx]

class PlainNeuralRec(nn.Module): # user embedding + item embedding + MLP
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64):
        super().__init__() # Initialize parent class
        self.user_embedding = nn.Embedding(num_users, embedding_dim) # User embedding table
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential( 
            nn.Linear(embedding_dim * 2, hidden_dim), #(64,64)
            nn.ReLU(), 
            nn.Linear(hidden_dim, 32), #(64,32)
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, user_idx, item_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec], dim=1) # Concatenate embeddings
        out = self.mlp(x) # pred ratings
        return out.squeeze(1)
    
class NeuralRecWithBias(nn.Module):
    # Improve with global mean, user/item biases, embeddings, MLP, bounded preds
    def __init__(self, num_users, num_items, global_mean, embedding_dim=32, hidden_dim=64, min_rating=1.0, max_rating=5.0):
        super().__init__() # Initialize parent class
        self.register_buffer("global_mean", torch.tensor(float(global_mean))) # Store global mean as buffer
        self.min_rating = float(min_rating) # Save rating bounds
        self.max_rating = float(max_rating)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1) # Scalar bias embeddings
        self.item_bias = nn.Embedding(num_items, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, user_idx, item_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        user_b = self.user_bias(user_idx).squeeze(1)
        item_b = self.item_bias(item_idx).squeeze(1)
        x = torch.cat([user_vec, item_vec], dim=1)
        interaction = self.mlp(x).squeeze(1)
        prediction = self.global_mean + user_b + item_b + interaction
        prediction = torch.clamp(prediction, min=self.min_rating, max=self.max_rating)
        return prediction
    
def create_dataloaders(train_df, test_df, batch_size=256):
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss=0.0 # track loss
    for user_idx, item_idx, ratings in dataloader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad()
        preds = model(user_idx, item_idx)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # add batch loss
    return total_loss / len(dataloader) # return avg loss

def eval_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad(): # no gradient needed in eval
        for user_idx, item_idx, ratings in dataloader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            ratings = ratings.to(device)
            preds = model(user_idx, item_idx)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = np.mean((all_preds - all_targets)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    return rmse, mae

def train_and_evaluate_model(model, train_loader, test_loader, device, num_epochs=5, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history_rows=[] # Create an empty list for metric history
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        rmse, mae = eval_model(model, test_loader, device)
        history_rows.append({"epoch": epoch+1, "train_loss":train_loss, "test_rmse":rmse, "test_mae":mae})
    history_df = pd.DataFrame(history_rows)
    return history_df

def show_model_history(model_name, history_df): # Print epoch-by-epoch history for one model. 
    print(f"\nResults for {model_name}:")
    print(history_df)
    
def compare_final_res(plain_history, bias_history): # Compare the final epoch metrics for both models
    plain_final = plain_history.iloc[-1] # Get the final row for each model
    bias_final = bias_history.iloc[-1]
    comparison_df = pd.DataFrame({
        "model": ["Plain Neural Recommender", "Neural Recommender With Bias"], 
        "final_train_loss": [plain_final["train_loss"], bias_final["train_loss"]], 
        "final_test_rmse": [plain_final["test_rmse"], bias_final["test_rmse"]],
        "final_test_mae": [plain_final["test_mae"], bias_final["test_mae"]],
    })
    return comparison_df

# Generate Top-N recommendations from the trained deep model
"""
No turn the trained deep model from a rating predictor into a real top-N recommender.
    - For one user, score unseen items, rank them, and return the best product
        - Use th trained neural rec to score all products the user has not rated yet,
            then returns the products with the highest predicted atings as recommendations.
Idea 
For a target user u:
1. Find items user u already rated in train data
2. Find candidate items user u has not rated
3. Use the trained neural model to predict: r
For every candidate item i
4. Sort perdicted scores from high to low
5. Return top N products.
How it works
The model learned embeddings for: users and items; 
So for one target user, we can pair that user with every unseen item: (u,i1), (u,i2)...
Then the model predicts a score for each pair.
Finally, we rank by predicted score. 
Note: if a user or item was never in training, the model has no embedding for it. 
"""
def get_user_seen_items(user_id, train_df): # Get products already rated by one user in the training data
    user_rows = train_df[train_df["UserId"]==user_id] # Filter rows for this user
    seen_items = set(user_rows["ProductId"]) # Convert the user's rated products into a set
    return seen_items # Return seen products

def rec_top_n_deep_model(model, user_id, train_df, user_to_idx, item_to_idx, device, top_n=10):
    # Generate top-N recs for one user using a trained deep model
    if user_id not in user_to_idx: return pd.DataFrame(columns=["ProductId", "predicted_score"])
       # If user was not seen in training, we can't rec using embeddings
    user_idx = user_to_idx[user_id] # get this user's idx
    seen_items = get_user_seen_items(user_id, train_df) # Get items already rated
    candidate_items = [item_id for item_id in item_to_idx.keys() if item_id not in seen_items]
    if len(candidate_items) == 0: return pd.DataFrame(columns=["ProductId", "predicted_score"])
    candidate_item_indices = [item_to_idx[item_id] for item_id in candidate_items] # Convert candidate raw item IDs to integer item indices
    user_tensor = torch.tensor([user_idx]*len(candidate_item_indices), dtype=torch.long).to(device)
        #Create user index tensor repeated for every candidate item
    item_tensor = torch.tensor(candidate_item_indices, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad(): # Turn off grad bcz we only predict next
        pred_scores = model(user_tensor,item_tensor) # Pred scores for all candidate items
    pred_scores = pred_scores.cpu().numpy() #move pred back to CPU np.array
    recs = pd.DataFrame({"ProductId": candidate_items, "predicted_score": pred_scores}) # Rec df
    recs = recs.sort_values(by="predicted_score", ascending=False)
    return recs.head(top_n).reset_index(drop=True)

def choose_demo_user_from_train(train_df):
    user_counts = train_df["UserId"].value_counts()
    demo_user_id = user_counts.index[0] # choose the most active user
    return demo_user_id 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)
    rows = collect_n_rows(train_ds, n=20000)
    df = rows_to_df(rows)
    reco_df = keep_rec_cols(df)
    reco_df = clean_rec_data(reco_df)
    reco_df = filter_active_users_and_popuar_items(reco_df, 5, 5)
    print("\nFiltered recommendation DataFrame shape:", reco_df.shape)
    train_df, test_df = train_test_split_random(reco_df, test_size=0.2, random_state=42)
    global_mean = train_df["Score"].mean()
    print("\nTraining global mean rating:", round(global_mean,4))
    user_to_idx, item_to_idx = build_id_mappings(train_df)
    train_df = apply_id_mappings(train_df, user_to_idx, item_to_idx)
    test_df = apply_id_mappings(test_df, user_to_idx, item_to_idx)
    print("\nMapped train shape:", train_df.shape)
    print("\nMapped test shape:", test_df.shape)
    train_loader, test_loader = create_dataloaders(train_df, test_df, 256) #batch_size=256
    plain_model = PlainNeuralRec(num_users=len(user_to_idx), num_items=len(item_to_idx), embedding_dim=32, hidden_dim=64).to(device)
    bias_model = NeuralRecWithBias(len(user_to_idx), len(item_to_idx), global_mean, 32, 64, min_rating=1.0, max_rating=5.0).to(device)
    plain_history = train_and_evaluate_model(plain_model, train_loader, test_loader, device, 5, 0.001) #num_epochs=5, learning_rate=0.001
    bias_history = train_and_evaluate_model(bias_model, train_loader, test_loader, device, 5, 0.001) 
    show_model_history("Plain Neural Recommender", plain_history)
    show_model_history("Neural Recommender With Bias", bias_history)
    comparison_df = compare_final_res(plain_history, bias_history)
    print("\nFinal comparison table:")
    print(comparison_df)

    # Recommend top-n items
    demo_user_id = choose_demo_user_from_train(train_df)
    deep_recs = rec_top_n_deep_model(model=bias_model, user_id=demo_user_id, train_df=train_df, user_to_idx=user_to_idx, item_to_idx=item_to_idx, device=device, top_n=10)
    print("\nDemo user:", demo_user_id)
    print("\nTop deep learning recommendations:")
    print(deep_recs)

if __name__ == "__main__":
    main()

