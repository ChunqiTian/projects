from datasets import load_dataset # import load_dataset so we can load datasets from Hugging Face
import pandas as pd

def load_food_reviews_stream():
    """
    Load the Amazon food reviews dataset in streaming mode.
    Why this function exists:
    - Keeps dataset loading in one place
    - Makes the code easier to reuse later
    - Makes it easy to change dataset name in the future
    Returns - ds: the streamed dataset obj
    """
    # Load the dataset directly from Hugging face
    # streaming=True means the data is read little by little as we iterate
    ds = load_dataset("jhan21/amazon-food-reviews-dataset", streaming=True)
    return ds

#d = load_food_reviews_stream()
"""
print("Dataset object:")
print(d)

train_data = d["train"]
print("\nFirst example:")
first_row = next(iter(train_data))
print(first_row)

print("\nFirst 5 examles:")
for row in islice(train_data,5):
    print(row)
"""

def inspect_dataset_object(ds):
    """
    Print basic info about the dataset object.
    Params - ds
    """
    print("Dataset obect type:", type(ds)) # Print the Python type of the dataset object
    print("\nDataset object preview:") # Usually this shows whether there are splits like train/test
    print(ds)

#inspect=inspect_dataset_object(d)
#inspect

def get_train_split(ds):
    """
    Get the train split from the dataset.
    Why this function exists:
    - Many Hugging Face datasets are stored by split
    - We usually work with the train split first
    - Keeps split selection separate from loading logic
    """
    train_ds = ds["train"]
    return train_ds

def show_one_example(train_ds):
    """
    Print one example row from the dtreamed dataset.
    """
    data_iter = iter(train_ds)
    first_example = next(data_iter)
    print("\nFirst example:")
    print(first_example)

def show_column_names(train_ds):
    """
    Print the column names from the first example. 
    """
    data_iter = iter(train_ds)
    first_example = next(data_iter)
    cols = list(first_example.keys())
    print("\nColumn names:")
    print(cols)

def collect_n_examples(train_ds, n=5):
    """
    Collect the first n examples from the streamed dataset. 
    """
    rows = []
    for i, row in enumerate(train_ds):
        rows.append(row)
        if i+1 >= n: break 
    return rows

def sample_to_dataframe(rows):
    """
    Conveert a small list of rows into a pandas DF.
    """
    df = pd.DataFrame(rows)
    return df

def main():
    """
    - Keeps the program flow organized
    - Makes the script easy to run
    - Separates "what to do" from the helper functions
    """
    ds = load_food_reviews_stream()
    inspect_dataset_object(ds) 
    train_ds = get_train_split(ds)
    show_one_example(train_ds)
    show_column_names(train_ds)
    rows = collect_n_examples(train_ds, n=5)
    df_sample = sample_to_dataframe(rows)

    # Print the data frame
    print("\nSample DataFrame:")
    print(df_sample)

    # Print the DataFrame shape
    print("\nSample DataFrame shape:")
    print(df_sample.shape)

# This line means:
# run main() only if this file is executed directly
# but do not run it automatically if this file is imported elsewhere
if __name__ == "__main__":
    main()



