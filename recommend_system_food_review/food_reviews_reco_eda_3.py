# Recommendation-style exploration
"""
Code thinking process:
1. Start with a sample
2. Focus on interaction structure
   - Most important cols: user_id, product_id, score
   - user (user_id) -> interacts with (review | score) -> product (product_id)
3. Compute sparsity
   - most users interact with only a few products
   - It shows how empty the matrix is
   - U: unique users; I: unique items; N: observed interactions
   - user-item matrix: U x I; 
   - sparsity = 1 - N / (U * I)
4. We rank products in two ways
    1. By # of reviews (popularity)
    2. By average score with a minimum # of reviews (quality - best products style ranking)
5. Next steps: user-based CF | item-based CF | matrix factorization | deep learning models | evaluation metrics (RMSE, precision@k, recall@k, MAP, NDCG)

Functions:
1. load_food_reviews_stream()
2. get_train_split(ds)
3. collect_n_rows(train_ds, n=5000)
4. sample_rows_to_dataframe(rows)
5. keep_recommendation_cols(df) # UserId, ProductId, Score, Summary, Text
6. clean_recommendation_data(reco_df)
    - removes rows that are missing essential fields (user_id, product_id, score)
    - removes duplicates
7. show_interaction_overview(reco_df)
    - prints: # of unique users, # of unique products, total interactions
    - These three numbers tell you the basic size of the interation problem
8. calculate_sparsity(reco_df)
    - U = number of unique users | I = number of unique products | N = total interactions
    - possible pairs = U * I | density = N / (U * I) | sparsity = 1 - density
    - Most reco datasets are very sparse (sparsity > 0.99), meaning most users have not interacted with most products. 
    - impact on reco algorithms: 
        - User-based CF and item-based CF can struggle with sparse data, 
        - matrix factorization can help by learning latent factors, 
        - deep learning models can also help but require more data and computational resources.
9. show_user_activity_stats(reco_df)
    - It counts how many reviews each user wrote and prints summary statistics
    - It tells whether the dataset has a few very active users or many users with only a few interactions, which can impact the choice of reco algorithms and evaluation strategies.
10. show_product_popularity_stats(reco_df)
    - It counts how many reviews each product received and prints summary statistics
    - It tells whether the dataset has a few very popular products or many products with only a few reviews, which can impact the choice of reco algorithms and evaluation strategies.
11. show_most_reviewed_products(reco_df, top_n=10) 
    - It ranks products by review count
    - This is the basis for a poularit recommender. (recommend the most frequently reviewed or most popular items)
12. show_highest_rated_products(reco_df, min_reviews=5, top_n=10)
    - It calculates the each product: review count | avg score
    - then keeps only products with at least min_reviews 
        - without the threshold, a prod with only 1 perfect review may rank at the top -> misleading
    - then ranks them
13. show_score_distribution(reco_df) - It prints score counts and %. 
    - shows whether the ratings are balanced or skewed
14. build_user_item_matrix(reco_df)
    - It creates a pivot table: rows=users, cols=products, values=scores
    - Conceptually, it builds Rui (the score user u gave item i)
    - It's the classic matrix used in user-based and item-based collaborative filtering algorithms, and matrix factorization models.
15. main()
    - This function runs the full pipeine in order. 
"""

from datasets import load_dataset
import pandas as pd

def log_step(message):
    print(message, flush=True)

def load_food_reviews_stream():
    log_step("Loading dataset stream from Hugging Face...")
    try:
        ds = load_dataset("jhan21/amazon-food-reviews-dataset", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the dataset stream. Check your internet connection "
            "and Hugging Face availability."
        ) from exc
    log_step("Dataset stream loaded.")
    return ds

def get_train_split(ds):
    train_ds = ds["train"]
    return train_ds

def collect_n_rows(train_ds, n=5000):
    rows = []
    for i, row in enumerate(train_ds):
        rows.append(row)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            log_step(f"Collected {i + 1}/{n} rows...")
        if i+1 >= n: break
    return rows

def sample_rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    return df

def keep_recommendation_cols(df):
    reco_df = df[["UserId", "ProductId", "Score", "Summary", "Text"]].copy()
    return reco_df

def clean_recommendation_data(reco_df):
    reco_df = reco_df.copy() # Make a copy to avoid modifying the original df
    reco_df = reco_df.dropna(subset=["UserId", "ProductId", "Score"]) # Remove rows with missing essential fields
    reco_df = reco_df.drop_duplicates()
    return reco_df

def show_interaction_overview(reco_df):
    num_users = reco_df["UserId"].nunique() # count unique users
    num_products = reco_df["ProductId"].nunique() # count unique products
    num_interactions = len(reco_df) # total number of interactions (rows)
    print("\nInteraction overview:")
    print("Unique users:", num_users)
    print("Unique products:", num_products)
    print("Total interactions:", num_interactions)  

def calculate_sparsity(reco_df):
    num_users = reco_df["UserId"].nunique()
    num_products = reco_df["ProductId"].nunique()
    num_interactions = len(reco_df)
    possible_pairs = num_users * num_products
    if possible_pairs == 0: # Avoid division by 0
        sparsity = None
        print("\nSparsity could not be calculated because total_possible is 0.")
        return sparsity
        
    density = num_interactions / possible_pairs
    sparsity = 1 - density
    print("\nSparsity calculation:")
    print(f"Number of interactions: {num_interactions}")
    print(f"Possible user-product pairs: {possible_pairs}")
    print(f"Density: {density:.6f}")
    print(f"Sparsity: {sparsity:.6f}")
    return sparsity # Return the sparsity value

def show_user_activity_stats(reco_df):
    # Show how many reviews each user wrote
    reviews_per_user = reco_df.groupby("UserId").size()
    print("\nUser activity stats:") 
    print("Reviews per user summary:")
    print(reviews_per_user.describe()) # Summary stats: count, mean, std, min, 25%, 50%, 75%, max       

def show_product_popularity_stats(reco_df):
    # Show how many reviews each product received
    reviews_per_product = reco_df.groupby("ProductId").size()
    print("\nProduct popularity stats:")
    print("Reviews per product summary:")
    print(reviews_per_product.describe())

def show_most_reviewed_products(reco_df, top_n=10):
    # Rank products by review count
    reviews_per_product = reco_df.groupby("ProductId").size().sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} most reviewed products:")
    print(reviews_per_product)

def show_highest_rated_products(reco_df, min_reviews=5, top_n=10):
    # Show products with the highest averag score, using a minimum review threshold to avoid misleading rankings
    product_summary = reco_df.groupby("ProductId").agg(
        review_count = ("Score", "count"),
        avg_score = ("Score", "mean")
    )
    filtered = product_summary[product_summary["review_count"] >= min_reviews]
    top_products = filtered.sort_values(by=["avg_score", "review_count"], ascending=[False, False]).head(top_n)
    print(f"\nTop {top_n} highest-rated products with at least {min_reviews} reviews:")
    print(top_products)
    

def show_score_distribution(reco_df):
    # Many reco datasets are skewed towards higher ratings, which can impact the performance of reco algorithms and evaluation metrics.
    score_counts = reco_df["Score"].value_counts().sort_index()
    score_percentages = reco_df["Score"].value_counts(normalize=True).sort_index() * 100
    print("\nScore distribution:", score_counts)
    print("\nScore distribution (%):", score_percentages)

def build_user_item_matrix(reco_df):
    user_item_matrix = reco_df.pivot_table(
        index = "UserId",
        columns = "ProductId",
        values = "Score"
    )
    print("\nUser-item matrix shape:", user_item_matrix.shape)
    print("\nUser-item matrix preview:", user_item_matrix.head())
    return user_item_matrix

def main():
    try:
        # 1. Load dataset
        ds = load_food_reviews_stream()
        # 2. Get train split
        log_step("Accessing train split...")
        train_ds = get_train_split(ds)
        # 3. Collect a sample of rows
        log_step("Collecting sample rows...")
        rows = collect_n_rows(train_ds, n=5000)
        # 4. Convert to DataFrame
        log_step("Converting rows to DataFrame...")
        df = sample_rows_to_dataframe(rows)
        # 5. Keep only recommendation-relevant columns
        log_step("Selecting recommendation columns...")
        reco_df = keep_recommendation_cols(df)
        # 6. Clean the data
        log_step("Cleaning recommendation data...")
        reco_df = clean_recommendation_data(reco_df)
        # 7. Show interaction overview
        show_interaction_overview(reco_df)
        # 8. Calculate matrix sparsity
        calculate_sparsity(reco_df)
        # 9. Show user activity stats
        show_user_activity_stats(reco_df)
        # 10. Show product popularity stats
        show_product_popularity_stats(reco_df)
        # 11. Show most reviewed products
        show_most_reviewed_products(reco_df, top_n=10)
        # 12. Show highest-rated products with a minimum review threshold
        show_highest_rated_products(reco_df, min_reviews=5, top_n=10)
        # 13. Show score distribution
        show_score_distribution(reco_df)
        # 14. Build user-item matrix
        build_user_item_matrix(reco_df)
        
        log_step("Analysis complete.")
    except Exception as exc:
        log_step(f"Script failed: {exc}")
        raise

if __name__ == "__main__":
    main()



