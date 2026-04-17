# Baseline idea: For each product family, predict future sales using its average historical sales

from datasets import load_dataset
from metrics import print_metrics
import pandas as pd



def load_sales_dataset_stream():
    ds = load_dataset("t4tiana/store-sales-time-series-forecasting", streaming=True)
    return ds

def take_sample_from_stream(ds, split_name="train", n_rows=50000, random_state=42):
    stream = ds[split_name] # Choose one split
    rows = []
    for i, row in enumerate(stream):
        rows.append(row) # Add each row dict into the list
        if i+1 >= n_rows: break
    df = pd.DataFrame(rows)
    return df

def clean_sales_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)
    return df

def add_time_features(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def train_test_split_by_date(df, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(df) * (1-test_size))
    train_df = df.iloc[:split_index] # Earlier rows become training data
    test_df = df.iloc[split_index:]
    return train_df, test_df

def make_baseline_predctions(train_df, test_df):
    family_avg_sales = train_df.groupby("family")["sales"].mean()
    overall_avg_sales = train_df["sales"].mean()
    test_df =test_df.copy()
    test_df["base_pred"] = test_df["family"].map(family_avg_sales)
        # For each family in test_df, find that family's avg sales, then put it into a new col(base_pred)
        # Means for each row, predict sales using the avg sales of that product family
    test_df["base_pred"] = test_df["base_pred"].fillna(overall_avg_sales) # If a family doesn't exist in training, use overall avg
    return test_df

def eval_baseline(test_df):
    y_true = test_df["sales"] # Actual sales
    y_pred = test_df["base_pred"] # Pred sales
    print_metrics(y_true, y_pred, label="Baseline")

def main():
    ds = load_sales_dataset_stream()
    df = take_sample_from_stream(ds, split_name="train", n_rows=5000, random_state=42)
    df = clean_sales_data(df)
    df = add_time_features(df)
    train_df, test_df = train_test_split_by_date(df, test_size=0.2)
    test_df = make_baseline_predctions(train_df, test_df)
    eval_baseline(test_df)
    print(test_df[["date", "family", "sales", "base_pred"]].head())

if __name__ == "__main__":
    main()

"""
Baseline MAE: 141.6716
Baseline RMSE: 504.4225
Baseline WMAPE: 66.18%
Baseline MAPE (non-zero actuals only): 83.25%
Baseline MAE: 141.6716. --- means the avg absolute miss is about 288 sales units
Baselne RMSE: 504.4225. --- is much larger bcz a few big misses are being penalized heabily
- large gap between MAE and RMSE usually means outlies or spikes. 
- In retail sales, it's common. 
"""




