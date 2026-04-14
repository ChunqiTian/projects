# User-based Collaborative Filtering
"""
Main idea:
1. find similar users
2. use similar users' raings to recommend products
ratings -> find similar user -> keep only top neighbor users (best k) -> 
      -> predict scores on unseen items -> recommend top predicted items

Thinking process:
1. Start from the interaction table (only need UserID, ProductID, Score)
   - firstly: load stream -> collect a sample -> keep only the reco cols -> cleann missing values
2. Build the user=item matrix (pivot table for Rui)
   - u=user | i=item | Rui = rating of user u on item i
3. Measure similarity between users (cosine similarity or Pearson correlation)
4. Predict a score for unseen items
    - for a target user we: find similar users -> look at products they rated -> compute a weighted score using neighbor similarity
    - A simple weighted prediction is: rui = sum(similarity(u, v) * rvi) / sum(similarity(u, v)) for all v in neighbors of u
        - where: - rui = predicted rating for user u on item i
                    - N(u) = set of neighbor users similar to user u
                    - similarity(u, v) = similarity between user u and user v
                    - rvi = rating of user v on item i
5. Add practical filters
    - minimum ratings per user
    - minimum ratings per product
    - top-k neighbors
    - ignore negative or zero similarity
    - recommend only items the target user has not rated yet

Functions:
1. load_food_reviews_stream()
2. get_train_split(ds)
3. collect_n_rows(train_ds, n=10000)
4. rows_to_dataframe(rows)
5. keep_recommendation_cols(df)
6. clean_recommendation_data(reco_df)
7. filter_active_users_and_popular_items()
    - It does not work if users have only 1 rating | items have only 1 rating
    - So, we keep users with at least m ratings | items with at least n ratings
8. build_user_item_matrix(reco_df)
    - This creates the rating matrix using pivot_table() 
    - We build Rui where each row is a user, each col is a product, and each value is the rating score
9. compute_user_similarity(user_item_matrix)
    - This is where user-based CF really starts. 
    - user_item_matrix.T.corr() - why transpose?
        - originally, rows are users and cols are items|corr() computes correlation between columns.
        - after transpose, cols become the users
10. get_top_neighbors(UserId, user_similarity, top_k=5)
    - Select the nearest neighboring users for the target user.
    - We remove: the user itself | missing similarities (NaN) | negative or zero similarities
    - Then we keep the top k
11. predict_scores_for_user(...)  * most important
    - For each item the target user has not rated, we estimate a predicted score using similar users
    - A simple weighted prediction is: rui = sum(similarity(u, v) * rvi) / sum(similarity(u, v)) for all v in neighbors of u
    - A neighbor with higher similarity should influence the prediction more strongly. 
    - So the method uses a weighted avg of neighbor ratings. 
12. recommend_products(...)
    - A wrapper func:it predicts scores for unseen items | sorts them | returns the recommendations
13. choose_demo_user(reco_df)
    - For a demo, we want a user with enough rating history. 
    - So the model can find similar neighbors | overlapping ratings | meaningful recommendations. 
14. show_user_history(...)
    - It prints the target user's rating history. eg. If the user gave many high ratings to a certain 
        kind of product, then the recom products may reflect that pattern. 
15. main() - load -> sample -> clean -> filter -> matrix -> similarity -> recommend
Limitations: Sparsity (many missing ratings) | Scalability (computational complexity) | Cold Start Problem  
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
    rows=[]
    for i, row in enumerate(train_ds):
        if i >= n: break
        rows.append(row)
    return rows

def rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    return df

def keep_recommendation_cols(df):
    reco_df = df[["UserId", "ProductId", "Score"]].copy()
    return reco_df

def clean_recommendation_data(reco_df):
    reco_df = reco_df.copy()
    reco_df.dropna(subset=["UserId", "ProductId", "Score"], inplace=True)
    reco_df = reco_df.drop_duplicates()
    return reco_df

def filter_active_users_and_popular_items(reco_df, min_user_ratings=3, min_item_ratings=3):
    # Keep only users and products with enough ratings
    user_counts = reco_df["UserId"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    item_counts = reco_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts >= min_item_ratings].index
    filtered_df = reco_df[reco_df["UserId"].isin(active_users) & reco_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def build_user_item_matrix(reco_df):
    # User-based CF needs a user-item matrix (pivot table)
    # rows = users | cols = items | values = ratings
    user_item_matrix = reco_df.pivot_table(
    index="UserId", columns="ProductId", values="Score", aggfunc="mean"
    )
    return user_item_matrix

def show_matrix_info(user_item_matrix): # print basic info about the user-item matrix
    print("User-Item Matrix Shape:", user_item_matrix.shape)
    print("Number of Users:", user_item_matrix.shape[0])
    print("Number of Items:", user_item_matrix.shape[1])
    print("\nUser-item matrix preview:", user_item_matrix.head())
    print("Sparsity: {:.2f}%".format(100 * (1 - user_item_matrix.count().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))))

def compute_user_similarity(user_item_matrix):
    # Compute user-user similarity using Pearson correlation
    user_similarity = user_item_matrix.T.corr() # corr() computes using cols; matrix's users are in rows; so use T(transpose) to make users as cols
    return user_similarity

def get_top_neighbors(user_id, user_similarity, top_k=5):
    # Get the most similar users to a target user
    # user_id = target user 
    if  user_id not in user_similarity.index: return pd.Series(dtype=float) # return an empty series if the user is not in the similarity matrix
    sim_scores = user_similarity.loc[user_id] # Get similarity scores for the target user
    sim_scores = sim_scores.drop(labels=[user_id], errors="ignore") # Remove the user itself
    sim_scores = sim_scores.dropna() # Remove missing similarities
    sim_scores = sim_scores[sim_scores > 0] # Keep only positive similarities
    neighbors = sim_scores.sort_values(ascending=False).head(top_k) # Get top beighbors
    return neighbors

def predict_scores_for_user(user_id, user_item_matrix, user_similarity, top_k=5):
    # Predict scores for items the target user has not rated yet
    # Step 1: Check whether this user exists in the matrix 
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float) # return empty predictions if the arget user is missing
    # Step 2: Get this user's row of ratings   
    target_ratings = user_item_matrix.loc[user_id] # Get the target user's ratings
    # Step 3: Find the most similar users   
    neighbors = get_top_neighbors(user_id, user_similarity, top_k) # Get top similar users
    # step 4: If find no one, stop
    if neighbors.empty: return pd.Series(dtype=float) # If no valid neighbours, return empty predictiions
    # Step 5: Create a dict to store predicted scores for unrated items
    predictions={} # Empty dict to hold predicted scores
    # Step 6: Find the products the user has not rated yet.  
    unrated_items = target_ratings[target_ratings.isna()].index # Items the target user has not rated yet
    # Step 7: Go through each unrated item  
    for item in unrated_items: # loop through each unrated item
        # Step 8: Get the neighbor's ratings for this item 
        neighbor_ratings = user_item_matrix.loc[neighbors.index, item] # Get neighbor ratings for this item
        # Step 9: Keep only neighbors who rated this item (drop NaN)
        valid_neighbor_ratings = neighbor_ratings.dropna() # Keep only neighbors who rated this item
        # Step 10: If no neighbors rated this item, we can't predict, skip to the next item
        if valid_neighbor_ratings.empty: continue # Skip if nobody rated this item
        # Step 11: Match the similarity scores to the neighbors who rated this item
        valid_similarities = neighbors.loc[valid_neighbor_ratings.index] # Align similarities with the valid neighbors
        # Step 12: Compute the total weight 
        denominator = np.abs(valid_similarities).sum() # sum of abs similarities
        # Step 13: avoid zero
        if denominator == 0: continue # Avoid division by zero
        # Step 14: Compute the predicted score using a weighted average of neighbor ratings
        predicted_score = np.dot(valid_similarities, valid_neighbor_ratings) / denominator # Weighted avg of neighbor ratings (formula for user-based CF)
        # Step 15: Save the prediction for this item
        predictions[item] = predicted_score # Store the prediction 
    predictions = pd.Series(predictions) # Convert the predictions dict to a Series
    predictions = predictions.sort_values(ascending=False) # Sort predictions by score
    return predictions

def recommend_products(user_id, user_item_matrix, user_similarity, top_k_neighbors=5, top_n_recommendations=5):
    # Generate top product recommendations for a target user
    # Predict scores for all unrated items
    predictions = predict_scores_for_user(user_id, user_item_matrix, user_similarity, top_k_neighbors) 
    # Keep only the top N recommendations
    top_recommendations = predictions.head(top_n_recommendations)
    return top_recommendations

def choose_demo_user(reco_df):
    # choose a demonstration user with relatively many ratings.
    user_counts = reco_df["UserId"].value_counts()
    demo_user_id = user_counts.index[0] # choose the user with the most ratings
    return demo_user_id

def show_user_history(user_id, reco_df, top_n=10):
    # Show the target user's rating history - see what they liked or disliked before
    user_history = reco_df[reco_df["UserId"] == user_id].copy() # Filter rows for the target user
    user_history = user_history.sort_values(by="Score", ascending=False).head(top_n)
    print(f"\nRating history for user: {user_id}")
    print(user_history)

def main():
    # 1. Load the data
    ds = load_food_reviews_stream()
    # 2. Get the train split
    train_ds = get_train_split(ds)
    # 3. Collect a sample of rows
    rows = collect_n_rows(train_ds, n=10000)
    # 4. Convert to a DataFrame
    df = rows_to_dataframe(rows)
    # 5. Keep only the recommendation columns
    reco_df = keep_recommendation_cols(df)
    # 6. Clean the data
    reco_df = clean_recommendation_data(reco_df)
    # 7. Filter sparse users and items
    reco_df = filter_active_users_and_popular_items(reco_df, min_user_ratings=3, min_item_ratings=3)
    print("\nFiltered recommendation DataFrame shape:", reco_df.shape)
    # 8. Build the user-item matrix
    user_item_matrix = build_user_item_matrix(reco_df)
    # 9. Show info about the user-item matrix
    show_matrix_info(user_item_matrix)
    # 10. Compute user similarity
    user_similarity = compute_user_similarity(user_item_matrix)
    # 11. Choose a demonstration user
    demo_user_id = choose_demo_user(reco_df)
    # 12. Show the demo user's rating history
    show_user_history(demo_user_id, reco_df, top_n=10)
    # 13. Generate recommendations for the demo user
    recommendations = recommend_products(demo_user_id, user_item_matrix, user_similarity, top_k_neighbors=5, top_n_recommendations=5)
    # 14. Show the recommendations
    print(f"\nTop recommendations for user {demo_user_id}:")
    print(recommendations)  

if __name__ == "__main__":
    main()





