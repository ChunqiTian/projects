"""
What this step does: load -> split -> preview -> sample -> engineer -> summarize
Functions:
1. load_food_reviews_stream()
2. get_train_split(ds)
3. preview_one_example(train_ds)
    - first_row = next(iter(train_ds))
4. preview_columns(train_ds)
    - columns = list(first_row.keys())
5. collect_n_rows(train_ds, n=1000) 
    # It loops over the stream and stores the first n rows in a list
6. sample_rows_to_dataframe(rows)
    # Once your sample is in pandas, many common operatins becomes easy:
        # head() | insa().sum() | value_counts() ...
7. add_engineered_cols(df).  * most useful 
    - review_length = len(Text) : This tells you how long each review is.
    - summary_length = len(Summary) : This tells you how long each summary is. 
    - helpfulness_ratio = helpfulnessNumerator / helpfulnessDenominator : This gives a simple normalized helpfulness measure.
    - review_year : Covert the Time (Unix timestamps format) to a calendar year using datetime tools. 
8. show_basic_info(df)
    - check did the sample load correctly / are the values readable
    - It prints: shape | columns | first 5 rows
9. show_missing_values(df)
10. show_score_distribution(df)
    - It compute the counts and % of the Score col.
    - Help you see whether your sample is balanced or skewed.
11. show_top_products(df, top_n=10)
    - It counts how often each ProductId appears.
    - A product with many reviews may be: popular, long-lived, easier to analyze
    - lead into a popularity-based recommender later
12. show_top_users(df, top_n=10)
    - It counts how often each UserId appears
    - Are most users one-time reviewers? | Are there heavy reviewers? | Is the data sparse?
13. show_numeric_summary(df)
    - It prints descriptive statistics: mean | std | min | max | quartiles
    - It gives a quick quantitative overview of the sample
14. main()
    - It runs the workflow: load -> split -> preview -> sample -> engineer -> summarize
"""

from datasets import load_dataset
import pandas as pd
from datetime import datetime

def load_food_reviews_stream():
    ds = load_dataset("jhan21/amazon-food-reviews-dataset", streaming=True)
    return ds

def get_train_split(ds):
    train_ds = ds["train"]
    return train_ds

def preview_one_example(train_ds):
    row_iterator = iter(train_ds)
    first_row = next(row_iterator)
    print("\nFirst example row:")
    print(first_row)

def preview_columns(train_ds):
    row_iterator = iter(train_ds)
    first_row = next(row_iterator)
    cols = list(first_row.keys())
    print("\nColumns:")
    print(cols)

def collect_n_rows(train_ds, n=1000):
    rows = []
    for i, row in enumerate(train_ds):
        if i >= n:
            break
        rows.append(row)
    return rows

def sample_rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    return df

def add_engineered_columns(df): 
    df = df.copy()  # Avoid modifying original df
    df["Text"] = df["Text"].fillna("")  # Handle missing text
    df["Summary"] = df["Summary"].fillna("")  # Handle missing summary
    df["review_length"] = df["Text"].apply(len)
    df["summary_length"] = df["Summary"].apply(len)
    df["helpfulness_ratio"] = df.apply(
        lambda row: row["HelpfulnessNumerator"] / row["HelpfulnessDenominator"] 
        if row["HelpfulnessDenominator"] > 0 else 0, 
        axis=1
    ) # axis = 0 row-wise
    df["review_year"] = df["Time"].apply(lambda x: datetime.fromtimestamp(x).year 
                                         if pd.notnull(x) else None)
    
    return df   

def show_basic_info(df):
    print("\nSample shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

def show_missing_values(df):
    missing_counts = df.isnull().sum()
    print("\nMissing values per column:", missing_counts)

def show_score_distribution(df):
    score_counts = df["Score"].value_counts().sort_index()
    score_percentages = df["Score"].value_counts(normalize=True).sort_index() * 100 # convert counts to %
    print("\nScore distribution:", score_percentages.round(2))

def show_top_products(df, top_n=10):
    top_products = df["ProductId"].value_counts().head(top_n)
    print(f"\nTop {top_n} products by review count:")
    print(top_products) 

"""
def show_product_rows(df, product_id, columns=None):
    product_rows = df[df["ProductId"] == product_id]
    print(f"\nRows for product {product_id}:")
    if product_rows.empty:
        print("No matching rows found.")
        return

    if columns is not None:
        print(product_rows[columns])
    else:
        print(product_rows)

def show_top_product_rows(df, top_n=10, columns=None):
    top_product_ids = df["ProductId"].value_counts().head(top_n).index
    for product_id in top_product_ids:
        show_product_rows(df, product_id, columns=columns)
"""

def show_top_users(df, top_n=10):
    top_users = df["UserId"].value_counts().head(top_n)
    print(f"\nTop {top_n} users by review count:")
    print(top_users)

def show_numeric_summary(df):
    numeric_cols = ["HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "review_length", "summary_length", "helpfulness_ratio", "review_year"]
    summary = df[numeric_cols].describe()
    print("\nNumeric summary statistics:")
    print(summary)

def main():
    ds = load_food_reviews_stream()
    print("Dataset object:", ds)
    train_ds = get_train_split(ds)
    preview_one_example(train_ds)
    preview_columns(train_ds)
    rows = collect_n_rows(train_ds, n=1000)
    df = sample_rows_to_dataframe(rows)
    df = add_engineered_columns(df)
    show_basic_info(df)
    show_missing_values(df)
    show_score_distribution(df)
    """
    show_top_products(df, top_n=10)
    show_top_product_rows(
        df,
        top_n=10,
        columns=["ProductId", "UserId", "ProfileName", "Score", "Summary", "Text"]
    )
    """
    show_top_users(df, top_n=10)
    show_numeric_summary(df)    

if __name__ == "__main__": # Run main() only when this file is executed directly
    main()  


