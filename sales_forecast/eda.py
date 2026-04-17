from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import random


def load_sales_dataset_stream():
    ds = load_dataset("t4tiana/store-sales-time-series-forecasting", streaming=True)
    return ds

def take_sample_from_stream(ds, split_name="train", n_rows=50000, random_state=42):
    """
    Use reservoir sampling - choose sample first, then replace with new items 
    """
    stream = ds[split_name]
    rows = []
    rng = random.Random(random_state)

    for i, row in enumerate(stream, start=1):
        if i <= n_rows:
            rows.append(row) # need to fill the sample before replacement logic can start
            continue

        replace_at = rng.randint(1, i) # generate a random # between 1 and current row #
        if replace_at <= n_rows: #Only let the current row enter the sample if the random number falls in the first n_rows positions
            rows[replace_at - 1] = row # replace the sample

    df = pd.DataFrame(rows)
    return df

def clean_sales_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def inspect_data(df):
    # Show basic shape
    print("Shape:", df.shape)

    # Show columns
    print("\nColumns:")
    print(df.columns.tolist())

    # Show data types
    print("\nData types:")
    print(df.dtypes)

    # Show missing values
    print("\nMissing values:")
    print(df.isna().sum())

    # Show numeric summary
    print("\nNumeric summary:")
    print(df.describe())

def plot_total_sales_over_time(df):
    daily_sales = df.groupby("date")["sales"].sum()
    plt.figure(figsize=(12,5))
    plt.plot(daily_sales.index, daily_sales.values)
    plt.title("Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

def plot_sales_by_family(df, top_n=10):
    family_sales=df.groupby("family")["sales"].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(2,5))
    plt.bar(family_sales.index, family_sales.values)
    plt.title(f"Top {top_n} Product Families by Sales")
    plt.xlabel("Product Family")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right") #ha=horizontal alignment
    plt.tight_layout()
    plt.show()

def plot_sales_by_store(df, top_n=10):
    store_sales = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,5))
    plt.bar(store_sales.index.astype(str), store_sales.values)
    plt.title(f"Top {top_n} Stores by Sales")
    plt.xlabel("Store Number")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

def plot_promotion_vs_sales(df):
    promo_sales = (
        df.assign(is_promoted=df["onpromotion"] > 0)
        .groupby("is_promoted")["sales"]
        .mean()
        .reindex([False, True])
        .dropna()
    )

    labels = ["Not Promoted" if not promoted else "Promoted" for promoted in promo_sales.index]

    plt.figure(figsize=(8,5))
    plt.bar(labels, promo_sales.values)
    plt.title("Average Sales by Promotion Status")
    plt.xlabel("Promotion Status")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.show()

def plot_sales_by_year(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly_sales = df.groupby("year")["sales"].sum().sort_index()
    plt.figure(figsize=(10,5))
    plt.bar(yearly_sales.index, yearly_sales.values)
    plt.title("Total Sales by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def plot_sales_by_day_of_week(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_sales = df.groupby("day_of_week")["sales"].mean().reindex(weekday_order)
    plt.figure(figsize=(10,5))
    plt.bar(weekday_sales.index, weekday_sales.values)
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Sales")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def main():
    ds = load_sales_dataset_stream()
    df = take_sample_from_stream(ds, split_name="train", n_rows=50000)
    df = clean_sales_data(df)
    inspect_data(df)
    plot_total_sales_over_time(df)
    plot_sales_by_family(df, top_n=10)
    plot_sales_by_store(df, top_n=10)
    plot_promotion_vs_sales(df)
    plot_sales_by_year(df)
    plot_sales_by_day_of_week(df)

if __name__ == "__main__":
    main()







