from datasets import load_dataset
import pandas as pd


def load_sales_dataset_stream():
    ds = load_dataset("t4tiana/store-sales-time-series-forecasting", streaming=True)
    return ds

def take_sample_from_stream(ds, split_name="train", n_rows=50000):
    stream = ds[split_name] # Choose one split
    rows = []
    for i, row in enumerate(stream):
        rows.append(row) # Add each row dict into the list
        if i+1 >= n_rows: break
    df = pd.DataFrame(rows)
    return df

def inspect_sales_data(df):
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())
    
    print("\nNumeric summary:")
    print(df.describe())

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

def main():
    ds = load_sales_dataset_stream()
    df = take_sample_from_stream(ds, split_name="train", n_rows=50000)
    inspect_sales_data(df)
    df = clean_sales_data(df)
    df = add_time_features(df)
    
    print(df.head())
    print(df.dtypes)

if __name__ == "__main__":
    main()


"""
Moving Average: MAE 77528.3985, RMSE 111974.9176
Simple Exponential Smoothing: MAE 77681.6547, RMSE 112890.0788
Holt-Winters: MAE 49501.6331, RMSE 77239.8644
In the current run, the daily series is only 57 days long, and the test-set daily sales average is about 326,619. So:
- Holt-Winters MAE ≈ 49,502 is about 15% of average daily sales
- Holt-Winters RMSE ≈ 77,240 is about 24% of average daily sales
"""



