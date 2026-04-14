# Comparison between user-based CF and item-based CF
"""
Goal - for the same target
- What does the user-based CF recommend?
- What does the item-based CF recommend?
- How much do the two recommendation lists overlap?
- Which method produces more usable predictions on this sample?
Code writing thinking process:
1. Reuse the same cleaned sample
2. Build one user-item matrix
    - u=user | i=item | Rui = rating
3. Compute both similarity matrics 
    - user-based CF needs user-user similarity
    - item-based CF needs item-item similarit
    - DF.corr() computes pairwise correlation of cols while excluding missing values, we use
        - user_item_matrix.T.corr() for user-user similarity
        - user_item_matrix.corr() for item-item similarity
4. Compare recommendation lists directly
    - For one target user, we'll get
        - top N user-based recommendations 
        - top N item-based recommendations
    - Then compare:
        - which items appear in both
        - which appear only in one method
        - how many predictions each method managed to produce
Steps:
1. Load one sample
2. Clean and filter the data - Keep only [[UserId, ProductId, Score]]
3. Build one shared user-item matrix - common foundation for both methods
4. Build two similarity matrics - (user-based CF & item-based CF) then corr()
5. choose one targeete user - pick a demo user with many ratings
6. Generate two reco lists - from user-based CF & item-based CF
7. Compare the lists - how many items eacch method reco | which items overlap | which items are unique to one method
Result interpretation:
Case 1: Large overlap - means both methods are seeing similar signals
Case 2: Small overlap - means the methods are using diff evidence: user-based CF relies on similar users ; item.. on item
Case 3: One method returns fewer items - often means it had less usable evidence in the sample
    - user-based CF may struggle if there are too few strong neighbors
    - item-based CF may struggle if candidate items do not have enough positive similarit to items the user already rated
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
    rows = []
    for i, row in enumerate(train_ds):
        rows.append(row)
        if i+1 >= n: break
    return rows

def rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    return df

def keep_reco_cols(df):
    reco_df=df[["UserId", "ProductId", "Score"]].copy()
    return reco_df

def clean_reco_data(reco_df):
    reco_df = reco_df.copy()
    reco_df = reco_df.dropna(subset=["UserId", "ProductId", "Score"])
    reco_df = reco_df.drop_duplicates()
    return reco_df

def filter_active_users_and_popular_tems(reco_df, min_user_ratings=3, min_item_ratings=3):
    user_counts = reco_df["UserId"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    filtered_df = reco_df[reco_df["UserId"].isin(active_users)].copy()
    item_counts = filtered_df["ProductId"].value_counts()
    popular_items = item_counts[item_counts >= min_item_ratings].index
    filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()
    return filtered_df

def build_user_item_matrix(reco_df):
    user_item_matrix = reco_df.pivot_table(index="UserId", columns="ProductId", values="Score", aggfunc="mean")
    return user_item_matrix

def compute_user_similarity(user_item_matrix):
    user_similarity = user_item_matrix.T.corr()
    return user_similarity

def compute_item_similarity(user_item_matrix):
    item_similarity = user_item_matrix.corr()
    return item_similarity

def get_top_neighbors(user_id, user_similarity, top_n):
    if user_id not in user_similarity.index: return pd.Series(dtype=float)
    sim_scores = user_similarity.loc[user_id] # note: loc[row_name]; loc[:col_name]
    sim_scores = sim_scores.drop(labels=[user_id], errors="ignore") # if user_id doesn't exist, don't raise an error
    sim_scores = sim_scores.dropna()[sim_scores>0].sort_values(ascending=False)
    return sim_scores.head(top_n)

def predict_scores_user_based(user_id, user_item_matrix, user_similarity, top_n=5): # user-based CF
    """
    Goal: predict sim scores for target user's unrated items ratings with neighbors' ratings
    - predicted_score = np.dot(user_sim, valid_neighbor_ratings) / np.abs(user_sim).sum()
    Steps:
    - check: if not valid target user, return empty
    1. Get target user's rating
    2. Get similar users (indirectly from user_similarity)
    - check: if no neighbors, return empty
    3. Create prediction dict
    4. Find unrated items for the target user
        * Loop starts
        5. loop through each unrated items
        6. Get neighbors ratings for this item (using user-item matrix)
        7. Get valid neighbor ratings (dropna)
        - check: if nobody rated this item, loop to next item
        8. Get target and these neighbors' similarities (indirectly from user_similarity)
        Formula:
        9. denominator
        - check: if dividing by 0, next item
        10. predict_score formula
        11. store scores to the prediction dict
        * Loop ends
    12. Convert dict to series
    13. Sort them
    14. return the preds
    """
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float) #returrn in Series format for consistency
    target_ratings = user_item_matrix.loc[user_id]
    neighbors = get_top_neighbors(user_id, user_similarity, top_n)
    if neighbors.empty: return pd.Series(dtype=float)
    predictions = {}
    unrated = target_ratings[target_ratings.isna()].index
    for item in unrated:
        neighbors_ratings = user_item_matrix.loc[neighbors.index, item]
        valid_neighbors_ratings = neighbors_ratings.dropna()
        if valid_neighbors_ratings.empty: continue
        valid_sim = neighbors.loc[valid_neighbors_ratings.index]
        denominator = np.abs(valid_sim).sum()
        if denominator == 0: continue
        pred = np.dot(valid_sim, valid_neighbors_ratings)/denominator
        predictions[item] = pred
    predictions = pd.Series(predictions)
    predictions = predictions.sort_values(ascending=False)
    return predictions

def get_items_rated_by_user(user_id, user_item_matrix):
    # Get the items already rated by the target user
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float)
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings.dropna()
    return rated_items

def predict_scores_item_based(user_id, user_item_matrix, item_similarity): # item-based CF
    # Predict scores for unrated items using item-based CF
    # predicted_score = np.dot(item_sim, valid_user_ratings)/denominator
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float)
    target_ratings = user_item_matrix.loc[user_id]
    rated_items = target_ratings.dropna()
    unrated_terms = target_ratings[target_ratings.isna()].index
    predictions={}
    for item in unrated_terms:
        if item not in item_similarity.index: continue
        sim_item = item_similarity.loc[item, rated_items.index] # unrated & rated
        sim_item = sim_item.dropna()[sim_item>0]
        if sim_item.empty: continue
        valid_user_ratings = rated_items.loc[sim_item.index]
        denominator = np.abs(sim_item).sum()
        if denominator==0: continue
        pred = np.dot(sim_item, valid_user_ratings) / denominator
        predictions[item] = pred
    predictions = pd.Series(predictions)
    predictions = predictions.sort_values(ascending=False)
    return predictions

def choose_demo_user(user_item_matrix):
    user_rating_counts = user_item_matrix.notna().sum(axis=1)
    demo_user_id = user_rating_counts.sort_values(ascending=False).index[0]
    return demo_user_id

def show_matrix_diagnostics(user_item_matrix, demo_user_id, user_similarity):
    total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    observed_ratings = int(user_item_matrix.count().sum())
    density = observed_ratings / total_cells if total_cells else 0
    demo_user_rating_count = int(user_item_matrix.loc[demo_user_id].notna().sum())
    positive_neighbors = len(get_top_neighbors(demo_user_id, user_similarity, top_n=5))

    print("\nUser-item matrix shape:", user_item_matrix.shape)
    print("Observed ratings in matrix:", observed_ratings)
    print("Matrix density:", round(density, 4))
    print("Demo user ratings kept in matrix:", demo_user_rating_count)
    print("Positive neighbors found for demo user:", positive_neighbors)

def compare_reco_lists(user_reco, item_reco):
    user_items = set(user_reco.index)
    item_items = set(item_reco.index)

    overlap_items = user_items & item_items
    only_user_based = user_items - item_items
    only_item_based = item_items - user_items

    comparison = {
        "user_based_count": len(user_items),
        "item_based_count": len(item_items),
        "overlap_count": len(overlap_items),
        "overlap_items": sorted(list(overlap_items)),
        "only_user_based": sorted(list(only_user_based)),
        "only_item_based": sorted(list(only_item_based)),
    }
    return comparison

def show_comparison_res(comparison):
    print("\nComparison summary:")
    print("User-based recommendation count:", comparison["user_based_count"])
    print("Item-based recommendation count:", comparison["item_based_count"])
    print("Overlap count:", comparison["overlap_count"])

    print("\nItems recommended by BOTH methods:")
    print(comparison["overlap_items"])

    print("\nItems recommended ONLY by user-based CF:")
    print(comparison["only_user_based"])

    print("\nItems recommended ONLY by item-based CF:")
    print(comparison["only_item_based"])

def main():
    sample_size = 30000
    min_user_ratings = 3
    min_item_ratings = 3
    top_n = 5

    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)     
    rows = collect_n_rows(train_ds, n=sample_size)
    df = rows_to_dataframe(rows)
    reco_df = keep_reco_cols(df)
    reco_df = clean_reco_data(reco_df)
    reco_df = filter_active_users_and_popular_tems(reco_df, min_user_ratings, min_item_ratings)
    print("\nFiltered recommendation DataFrame shape:", reco_df.shape)
    user_item_matrix = build_user_item_matrix(reco_df)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    demo_user_id = choose_demo_user(user_item_matrix)
    show_matrix_diagnostics(user_item_matrix, demo_user_id, user_similarity)
    print("\nDemo user:", demo_user_id)
    user_based_recos = predict_scores_user_based(demo_user_id, user_item_matrix, user_similarity, top_n).head(top_n)
    item_based_recos = predict_scores_item_based(demo_user_id, user_item_matrix, item_similarity).head(top_n)
    print("\nTop user-based recommendations:")
    print(user_based_recos)
    print("\nTop item-based recommendations:")
    print(item_based_recos)
    comparison = compare_reco_lists(user_based_recos, item_based_recos)
    show_comparison_res(comparison)

if __name__ == "__main__":
    main()

    







