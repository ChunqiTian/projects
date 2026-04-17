from datasets import load_dataset
import pandas as pd

def load_sales_dataset_stream():
    ds = load_dataset("t4tiana/store-sales-time-series-forecasting", streaming=True)
    return ds

def take_sample_from_stream(ds, split_name="train", n_rows=1000):
    stream = ds[split_name] # Choose one split
    rows = []
    for i, row in enumerate(stream):
        rows.append(row) # Add each row dict into the list
        if i+1 >= n_rows: break
    df = pd.DataFrame(rows)
    return df

def main():
    ds = load_sales_dataset_stream()
    df = take_sample_from_stream(ds, split_name="train", n_rows=1000)
    print(df.head())
    print(df.info())
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()
    



