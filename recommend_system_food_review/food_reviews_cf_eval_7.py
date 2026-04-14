# evaluation
"""
Idea:
Suppose one user rared 5 products. - 4 ratings go into train; 1 hidden in test
To answer:
1. Can user-based CF recover items hidden from the user's history?
2. Can item-based CF recover items hidden from the user's history?
3. Which one performs better on a small offline test?
Classic recommender evaluation idea:
- keep most user interactions in train
- hide a small number of interactions in test
- build recommendations from train only
- check whether the hidden test items appear in the recommendation list
Metric
- Hit Rate@K - For each evaluated user:
    - generate top K recommendations
    - check whether the hidden test item is in the top K
    - if yes: Hit=1 ; if no: Hit=0; then Hit Rate@K = # of hits / # of evaluated users
- avg_num_predictions - One average, how many items did CF manage to recommend in the top-k stage?
Code thinking process:
1. Split by user, not globally
    - why? If a user has only 3 rates and they all accidentally go into test, 
        then the model has no training history for that user. 
        So we split within each user. We hide one rate per user for testing. 
2. Use only eligible users - users with enough ratings (min_user_ratings>=4)
3. Build everything from train only - eg. uer-item matrix, similarities, generate recos from train, compare res to hidden test item
4. Compare user-based and item-based CF fairly - same train set | test set | users | K
Steps
1. Load and clean the data
2. Filter to users who have enough ratings
3. Build a leave-one-out split - eg A's rating: B,C,D,E; BCD in train; E in test
4. Build the recommender only from train
5. Generate top-k recommendations - top 10 for user-based CF and item-based CF each
6. Check whether the hidden item appears - If prod X appears in the top1 -> Hit; Hit@10∈{0,1}
7. Average across users - Hit Rate@10
Important functions:
1. train_test_split_leave_one_out(...) - create the evaluation split
    - For every user(min_ratings>4): ramdomly sample 1 into test, place the rest in train
2. evaluate_hit_rate_at_k(...) - Accuracy check. 
    - for each row in the test set: 1. read the hidden true item; 2. generate recos; 
    3. check whether the true item is inside the top-k list; 4. store the res; 5. avg
How to interpret the res?
- User-based Hit Rate@10=0.18 ; item-based Hit Rate@10=0.25
    - item-based CF recovered the hidden test item for 25% of users
    - User-based ... 18%
    - On this sample dataset and this evaluation setup, item-based CF did better. 
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

def collect_n_rows(train_ds, n=15000):
    rows=[]
    for i, row in enumerate(train_ds):
        if i>=n: break
        rows.append(row)
    return rows

def rows_to_df(rows):
    df = pd.DataFrame(rows)
    return df

def keep_reco_cols(df):
    reco_df = df[["UserId", "ProductId", "Score"]].copy()
    return reco_df

def clean_reco_data(reco_df):
    reco_df=reco_df.copy()
    reco_df=reco_df.dropna(subset=["UserId", "ProductId", "Score"])
    reco_df = (
        reco_df.groupby(["UserId", "ProductId"], as_index=False)["Score"]
        .mean()
    )
    return reco_df

def filter_active_users_and_popular_items(reco_df, min_user_ratings=4, min_item_ratings=3):
    # Re-apply both thresholds until the user-item subset is stable.
    filtered_df = reco_df.copy()
    while True:
        prev_shape = filtered_df.shape
        user_counts = filtered_df["UserId"].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        filtered_df = filtered_df[filtered_df["UserId"].isin(active_users)].copy()

        item_counts = filtered_df["ProductId"].value_counts()
        popular_items = item_counts[item_counts >= min_item_ratings].index
        filtered_df = filtered_df[filtered_df["ProductId"].isin(popular_items)].copy()

        if filtered_df.shape == prev_shape:
            break
    return filtered_df

def train_test_split_leave_one_out(reco_df, random_state=42):
    eligible_users = reco_df["UserId"].value_counts()
    eligible_users = eligible_users[eligible_users >= 3].index
    eligible_df = reco_df[reco_df["UserId"].isin(eligible_users)].copy()
    test_df=(eligible_df.groupby("UserId", group_keys=False).sample(n=1, random_state=random_state).copy())
    train_df = eligible_df.drop(index=test_df.index).copy()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

def build_user_item_matrix(reco_df):
    user_item_matrix = reco_df.pivot_table(index="UserId", columns="ProductId", values="Score", aggfunc="mean")
    return user_item_matrix

def compute_user_similarity(user_item_matrix):
    user_similarity = user_item_matrix.T.corr()
    return user_similarity

def compute_item_similarity(user_item_matrix):
    item_similarity = user_item_matrix.corr()
    return item_similarity

def get_top_neighbors(user_id, user_similarity, top_k=5):
    if user_id not in user_similarity.index: return pd.Series(dtype=float)
    sim_scores = user_similarity.loc[user_id]
    sim_scores = sim_scores.drop(labels=[user_id], errors="ignore").dropna()
    sim_scores = sim_scores[sim_scores>0].sort_values(ascending=False)
    return sim_scores.head(top_k)

def predict_scores_user_based(user_id, user_item_matrix, user_similarity, top_k=5):
    # predict scores for unrated items using user-based CF.
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float)
    target_ratings = user_item_matrix.loc[user_id]
    neighbors = get_top_neighbors(user_id, user_similarity, top_k=top_k)
    if neighbors.empty: return pd.Series(dtype=float)
    predictions={}
    unrated_items = target_ratings[target_ratings.isna()].index
    for item in unrated_items:
        neighbor_ratings = user_item_matrix.loc[neighbors.index, item]
        valid_neighbor_ratings = neighbor_ratings.dropna()
        if valid_neighbor_ratings.empty: continue
        sim_user = neighbors.loc[valid_neighbor_ratings.index]
        denominator = np.abs(sim_user).sum()
        if denominator == 0: continue
        predicted_score = np.dot(sim_user, valid_neighbor_ratings) / denominator
        predictions[item] = predicted_score
    predictions = pd.Series(predictions)
    predictions = predictions.sort_values(ascending=False)
    return predictions

def predict_scores_item_based(user_id, user_item_matrix, item_similarity):
    # Predict scores for unrated items using item-based CF.
    if user_id not in user_item_matrix.index: return pd.Series(dtype=float)
    target_ratings = user_item_matrix.loc[user_id]
    rated_items = target_ratings.dropna()
    unrated_items = target_ratings[target_ratings.isna()].index
    predictions={}
    for item in unrated_items:
        if item not in item_similarity.index: continue
        sim_item = item_similarity.loc[item, rated_items.index] # unrated vs rated
        sim_item = sim_item.dropna()[sim_item>0]
        if sim_item.empty: continue
        valid_user_ratings =rated_items.loc[sim_item.index] #target user's rating instead of neighbors
        denominator = np.abs(sim_item).sum()
        if denominator == 0: continue
        predicted_score = np.dot(sim_item, valid_user_ratings) / denominator
        predictions[item] = predicted_score
    predictions = pd.Series(predictions)
    predictions = predictions.sort_values(ascending=False)
    return predictions

def recommend_top_k_user_based(user_id, user_item_matrix, user_similarity, top_k_neighbors=5, top_n=10):
    predictions = predict_scores_user_based(user_id, user_item_matrix, user_similarity, top_k_neighbors)
    return predictions.head(top_n)

def recommend_top_k_item_based(user_id, user_item_matrix, item_similarity, top_n=10):
    predictions = predict_scores_item_based(user_id, user_item_matrix, item_similarity)
    return predictions.head(top_n)

def evaluate_hit_rate_at_k(test_df, user_item_matrix, user_similarity, item_similarity, k=10, top_k_neighbors=5):
    # Evaluate both reco using Hit Rate@K
    eval_rows = []
    for _, row in test_df.iterrows(): 
        # Extract the target user and hidden test item
        user_id = row["UserId"]
        true_item = row["ProductId"]

        # top-K recommendations
        user_recs = recommend_top_k_user_based(
            user_id,
            user_item_matrix,
            user_similarity,
            top_k_neighbors=top_k_neighbors,
            top_n=k,
        )
        item_recs = recommend_top_k_item_based(user_id, user_item_matrix, item_similarity, top_n=k)

        # Check whether each method recovered the hidden item
        user_hit = int(true_item in user_recs.index)
        item_hit = int(true_item in item_recs.index)

        # Store evaluation details
        eval_rows.append({"UserId":user_id, "TrueItem":true_item,
                          "UserBasedHit":user_hit, "ItemBasedHit":item_hit,
                          "UserBasedNumPredictions":len(user_recs), "ItemBasedNumPredictions":len(item_recs),})
        
    detail_df = pd.DataFrame(eval_rows)
    results = {
        "num_evaluated_users": len(detail_df),
        "user_based_hit_rate_at_k": detail_df["UserBasedHit"].mean() if len(detail_df) > 0 else 0.0,
        "item_based_hit_rate_at_k": detail_df["ItemBasedHit"].mean() if len(detail_df) > 0 else 0.0,
        "avg_user_based_num_predictions": detail_df["UserBasedNumPredictions"].mean() if len(detail_df) > 0 else 0.0,
        "avg_item_based_num_predictions": detail_df["ItemBasedNumPredictions"].mean() if len(detail_df) > 0 else 0.0,
    }
    return results, detail_df

def show_evaluation_summary(results):
    # Print summary title
    print("\nOffline evaluation summary:")

    # Print number of users evaluated
    print("Number of evaluated users:", results["num_evaluated_users"])

    # Print user-based Hit Rate@K
    print("User-based Hit Rate@K:", round(results["user_based_hit_rate_at_k"], 4))

    # Print item-based Hit Rate@K
    print("Item-based Hit Rate@K:", round(results["item_based_hit_rate_at_k"], 4))

    # Print average recommendation list size for user-based CF
    print("Average user-based top-K list size:", round(results["avg_user_based_num_predictions"], 2))

    # Print average recommendation list size for item-based CF
    print("Average item-based top-K list size:", round(results["avg_item_based_num_predictions"], 2))

def main():
    ds = load_food_reviews_stream()
    train_ds = get_train_split(ds)
    rows = collect_n_rows(train_ds, n=30000)
    df = rows_to_df(rows)
    reco_df = keep_reco_cols(df)
    reco_df = clean_reco_data(reco_df)
    reco_df = filter_active_users_and_popular_items(reco_df, min_user_ratings=4, min_item_ratings=3)
    print("\nFiltered receommendation DataFrame shape:", reco_df.shape)
    train_df, test_df = train_test_split_leave_one_out(reco_df, 42)
    print("\nTrain shape:", train_df.shape)
    print("\nTest shape:", test_df.shape)
    user_item_matrix = build_user_item_matrix(train_df) #train-only
    user_similarity = compute_user_similarity(user_item_matrix) #train-only
    item_similarity = compute_item_similarity(user_item_matrix) #train-only
    res, detail_df = evaluate_hit_rate_at_k(
        test_df,
        user_item_matrix,
        user_similarity,
        item_similarity,
        k=10,
        top_k_neighbors=5,
    )
    show_evaluation_summary(res)
    print("\nEvaluation detail preview:")
    print(detail_df.head())

if __name__=="__main__": 
    main()
