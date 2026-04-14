# Item-based collaborative filtering
"""
User-based CF: similar users -> recommend what they liked
Item-based CF: similar items -> recommend items similar to what the user already liked
For a target user
1. Look at the products they already rated
2. Find products similar to those products
3. Give stronger weight to: produccts the user rated highly, and products more similar to those products
4. Compute predicted scores for products the user has not rated yet. 
Function process:
1. Load the data
2. Keep only [[user, item, rating]]
3. Build the user-item matrix (Rows: users, Columns: items, Values: scores)
4. Compare items with items - If two items tend to be rated similarl by many users, they get higher similarity
5. Look at what the target user already liked 
6. For each new item, compare it to the items the user already rated (predict_scores_for_user_item_based(...))
    1. find the user
    2. Get items they already rated
    3. Find items they have not rated
    4. For each unrated item: 
        - compare it to rated items | keep positive sim | use the user's rating as weights for formula | compare predicted score
    5. Rank all unrated items by predicted score
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

def load_food_reviews_stream():
    ds = load_dataset("jhan21/amazon-food-reviews-dataset", streaming=True)
    return ds

def get_train_split(ds):
    train_ds = ds["train"]
    return train_ds

def collect_n_rows(train_ds, n=10000):
    data = []
    for i, row in enumerate(train_ds):
        if i >= n:
            break
        data.append(row)
    return data

def rows_to_df(rows):
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

def filter_active_users_and_popular_items(reco_df, min_user_ratings=3, min_item_ratings=3):
    user_counts = reco_df["UserId"].value_counts() # Count ratings per user
    active_users = user_counts[user_counts>=min_user_ratings].index # Keep only active users
    filtered_df = reco_df[reco_df["UserId"].isin(active_users)].copy()
    item_counts = filtered_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts>=min_item_ratings].index
    filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def build_user_item_matrix(reco_df): # user_item similarity matrix
    user_item_matrix = reco_df.pivot_table(index="UserId", columns="ProductId", values="Score", aggfunc="mean")
    return user_item_matrix

def show_matrix_info(user_item_matrix):
    print("\nUser-item matrix shape:", user_item_matrix.shape)
    print("\nUser-item matrix preview:", user_item_matrix.head())

def compute_item_similarity(user_item_matrix): # item-item simiarity matrix
    item_similarity = user_item_matrix.corr()
    return item_similarity

def get_items_rated_by_user(user_id, user_item_matrix):
    # Get all items already rated by the target user
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float) # return empty
    user_ratings = user_item_matrix.loc[user_id] # get the user's row of ratings with NaNs
    rated_items = user_ratings.dropna() # Remove NaNs
    return rated_items

def predict_scores_for_user_item_based(user_id, user_item_matrix, item_similarity):
    """
    Predict scores for items the target user has not rated yet
    Idea:
    - Look at items the user already rated
    - Find items similar to them
    - Combine those similarities with the user's known ratings
    """
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float)
    target_ratings = user_item_matrix.loc[user_id]
    rated_items = target_ratings.dropna()
    unrated_items = target_ratings[target_ratings.isna()].index # Find items the target user not rated yet
    predictions = {}
    for candidate_item in unrated_items:
        if candidate_item not in item_similarity.index: continue
        similarities_to_rated = item_similarity.loc[candidate_item, rated_items.index] # get scores between one item vs multi items
        similarities_to_rated = similarities_to_rated.dropna() # drop the ones no scores with the candidate item
        similarities_to_rated = similarities_to_rated[similarities_to_rated>0] # only keep positive sim
        if similarities_to_rated.empty: continue # if no useful similar items remain, next candidate
        valid_user_ratings = rated_items.loc[similarities_to_rated.index] #find user's ratings for these items
        denominator = np.abs(similarities_to_rated).sum() # Compute the deno as the sum of similarity weights
        if denominator == 0: continue # avoid divide by 0
        predicted_score = np.dot(similarities_to_rated, valid_user_ratings) / denominator # compute weighted avg pred
        predictions[candidate_item] = predicted_score
    predictions = pd.Series(predictions) # convert preds dict into a series
    predictions = predictions.sort_values(ascending=False) #high to low
    return predictions

def reco_products_item_based(user_id, user_item_matrix, item_similarity, top_n=5):
    """
    Recommend top products using item-based collaborative filtering
    """
    predictions = predict_scores_for_user_item_based(user_id = user_id, user_item_matrix=user_item_matrix, item_similarity=item_similarity)
    recommendations = predictions.head(top_n) #keep the top_n
    return recommendations

def choose_demo_user(reco_df): # choose a demo user with many ratings
    user_counts = reco_df["UserId"].value_counts()
    demo_user_id = user_counts.index[0]
    return demo_user_id

def show_user_history(user_id, reco_df, top_n=10):
    user_history = reco_df[reco_df["UserId"] == user_id].copy()
    user_history = user_history.sort_values(by="Score", ascending=False)
    print(f"\nRating history for user: {user_id}")
    print(user_history.head(top_n))

def show_top_similar_items(item_id, item_similarity, top_n=5):
    """
    This func is for understanding and debugging
    Show the most similar items to one product. 
    Idea 
    - Helps you understand the item-based CF idea
    - Lets you inspect item-item relationship
    """
    if item_id not in item_similarity.index: # return if the item is missing
        print(f"\nItem {item_id} not found in item similarity matrix.")
        return 
    sim_scores = item_similarity.loc[item_id]
    sim_scores = sim_scores.drop(labels=[item_id], errors="ignore") # remove the item itself
    sim_scores = sim_scores.dropna()
    sim_scores = sim_scores[sim_scores>0]
    sim_scores = sim_scores.sort_values(ascending=False)
    print(f"\nTop {top_n} items most similar to {item_id}:")
    print(sim_scores.head(top_n))
    

def main():
    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)
    rows = collect_n_rows(train_ds, n=10000)
    df = rows_to_df(rows)
    reco_df = keep_reco_cols(df)
    reco_df = clean_reco_data(reco_df)
    print(reco_df.head())
    reco_df = filter_active_users_and_popular_items(reco_df, 3, 3)
    user_item_matrix = build_user_item_matrix(reco_df)
    show_matrix_info(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    print("\nItem-item similarity shape:", item_similarity.shape)
    demo_user_id = choose_demo_user(reco_df)
    show_user_history(demo_user_id, reco_df, top_n=10)
    recommendations = reco_products_item_based(demo_user_id, user_item_matrix, item_similarity, top_n=5)
    print(f"\nTop item-based recommendations for user: {demo_user_id}")
    print(recommendations)
    
    #Optional inspect similarity for one item the user already rated
    rated_items = get_items_rated_by_user(demo_user_id, user_item_matrix)
    if not rated_items.empty:
        sample_item = rated_items.index[0]
        show_top_similar_items(sample_item, item_similarity, top_n=5)


if __name__ == "__main__":    
    main()



