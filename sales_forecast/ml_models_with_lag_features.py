import clean_data as cd
from metrics import print_metrics
import numpy as np
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def add_time_features_extend(df):
    df = cd.add_time_features(df)
    df["is_promoted"] = (df["onpromotion"] > 0).astype(int)
    return df
  
def train_test_split_by_date(df, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(df) * (1-test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def preprocessing():
    feature_cols = ["store_nbr", "is_promoted", "year", "month", "day", "day_of_week", "is_weekend", "week_of_year", "family"]
    target_col = "sales"
    cat_cols = ["family", "store_nbr", "month", "day", "day_of_week", "is_weekend", "week_of_year"]
    num_cols = ["year", "is_promoted"]
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), 
        ("num", "passthrough", num_cols), # passthrough - do not transform these cols, just pass them into model
    ])
    return feature_cols, target_col, preprocessor

# Linear regression
def linear_regression(train_df, target_col, feature_cols, preprocessor):
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[target_col])
    model.fit(X_train, y_train)
    return model, feature_cols

# XGBoost
def xgboost(train_df, target_col, feature_cols, preprocessor):
    xgb_model = XGBRegressor(
        n_estimators = 300,         # num of trees
        learning_rate = 0.05,       # how slowly the model learns
        max_depth=6,                # how complex each tree can be
        subsample=0.8,              # use 80% of rows per tree
        colsample_bytree=0.8,       # use 80% of features per tree
        random_state=42,
        objective="reg:squarederror"
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", xgb_model)])
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    model.fit(X_train, y_train)
    return model, feature_cols

# XGBoost with lag features
"""
Lag feature means - use previous sales values as input features. 
- eg.lag_1=sales from 1 day ago; eg. for Jan 4, the model can use Jan 3 sales as a feature
- For sales forecasting, this is very useful bcz today's sales are often related to recent sales.
- This dataset has many stores and many product families, we should create lag features within each store + family group 
    - eg. Store 1 + beverages has its own lag hisotry; Store 2 + Dairy has its own lag history...

Rolling avg featues
- Lag features use specific previous days; Rolling features use an avg of recent days. 
- rolling_mean_7 = previous_7_days_of_sales / 7 
- This helps the model understand the recent trend more smoothly.    
"""
def add_lag_features(df):
    df = df.copy()
    df = df.sort_values(["store_nbr","family", "date"])
    df["sales_lag_1"] = df.groupby(["store_nbr", "family"])["sales"].shift(1) # Previous 1-day sales withn the same store + family
    df["sales_lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)
    df["sales_lag_14"] = df.groupby(["store_nbr", "family"])["sales"].shift(14)
    df["sales_lag_28"] = df.groupby(["store_nbr", "family"])["sales"].shift(28)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_rolling_features(df):
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])
    df["rolling_mean_7"] = df.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.shift(7).rolling(window=7).mean())
    df["rolling_mean_14"] = df.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.shift(14).rolling(window=14).mean())
    df["rolling_mean_28"] = df.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.shift(28).rolling(window=28).mean())
    df = df.sort_values("date").reset_index(drop=True)
    return df

def remove_rows_with_missing_lag_features(df):
    df = df.copy()
    lag_feature_cols = ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28", 
                        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
    df = df.dropna(subset=lag_feature_cols)
    df = df.reset_index(drop=True)
    return df

def data_preprocessing_xgboost_with_lag_features(train_df):
    feature_cols = ["store_nbr", "is_promoted", "year", "month", "day", "day_of_week", "is_weekend", "week_of_year", "family", "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
    target_col = "sales"
    numeric_features = ["year", "is_promoted", "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
    categorical_features = ["store_nbr", "month", "day", "day_of_week", "is_weekend", "week_of_year", "family"]
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features), 
        ("num", "passthrough", numeric_features), # passthrough - do not transform these cols, just pass them into model
    ])
    return feature_cols, target_col, preprocessor


def eval_model(model, test_df, feature_cols, label="Linear Regression", pred_col="linear_regression_prediction", log_target=True):
    """
    In Python: - if you do not provide label, use "Linear Regression"
               -  if you do not provide pred_col, use "linear_regression_prediction"
               - They are just the default values
    """
    X_test = test_df[feature_cols]
    y_true = test_df["sales"]
    y_pred = model.predict(X_test)
    if log_target:
        y_pred = np.expm1(y_pred)
    y_pred = np.maximum(y_pred, 0)
    print_metrics(y_true, y_pred, label=label)
    result_df = test_df.copy()
    result_df[pred_col] = y_pred
    return result_df

def main():
    ds = cd.load_sales_dataset_stream()
    # Lag features up to 28 days plus rolling windows need substantially more than 56 days
    # of history when the loader takes the first rows of the dataset in date order.
    df = cd.take_sample_from_stream(ds, split_name="train", n_rows=150000)
    df = cd.clean_sales_data(df)
    df = add_time_features_extend(df)

    train_df, test_df = train_test_split_by_date(df, test_size=0.2)
    feature_cols, target_col, preprocessor = preprocessing()

    linear_model, feature_cols = linear_regression(train_df, target_col, feature_cols, preprocessor)
    xgb_model, _ = xgboost(train_df, target_col, feature_cols, preprocessor)

    res_df = eval_model(linear_model, test_df, feature_cols, label="Linear Regression", pred_col="linear_regression_prediction", log_target=True)
    res_df = eval_model(xgb_model, res_df, feature_cols, label="XGBoost", pred_col="xgboost_prediction", log_target=False)
    print(res_df[["date", "store_nbr", "family", "sales", "linear_regression_prediction", "xgboost_prediction"]].head())

    lag_df = add_lag_features(df)
    lag_df = add_rolling_features(lag_df)
    lag_df = remove_rows_with_missing_lag_features(lag_df)

    if lag_df.empty:
        raise ValueError(
            "Lag-feature dataframe is empty after dropping missing lag values. "
            "Increase n_rows so the dataset covers more than 56 days of history."
        )

    lag_train_df, lag_test_df = train_test_split_by_date(lag_df, test_size=0.2)
    lag_feature_cols, lag_target_col, lag_preprocessor = data_preprocessing_xgboost_with_lag_features(lag_train_df)
    xgb_lag_model, _ = xgboost(lag_train_df, lag_target_col, lag_feature_cols, lag_preprocessor)

    lag_res_df = eval_model(xgb_lag_model, lag_test_df, lag_feature_cols, label="XGBoost With Lag Features", pred_col="xgboost_lag_prediction", log_target=False)
    print(lag_res_df[["date", "store_nbr", "family", "sales", "sales_lag_1", "sales_lag_7", "rolling_mean_7", "xgboost_lag_prediction"]].head())

if __name__ == "__main__":
    main()


"""
Linear Regression MAE: 109.4653
Linear Regression RMSE: 424.7541
Linear Regression WMAPE: 58.60%
Linear Regression MAPE (non-zero actuals only): 66.92%
XGBoost MAE: 60.6763
XGBoost RMSE: 131.2024
XGBoost WMAPE: 32.48%
XGBoost MAPE (non-zero actuals only): 229.54%

WMAPE: 32.48%
total absolute error is about 32.48% of total actual sales
This is usually the more reliable business metric

MAPE: 229.54%
the average row-level percentage error is extremely large
This usually means some rows have very small actual sales, and the model misses them by a lot

The result suggests:
- on high-volume rows, XGBoost may be fairly okay
- on low-volume or near-zero rows, it is making very large relative errors
- WMAPE is more trustworthy than MAPE for this sales problem

Overall, XGBoost is doing much better than the Linear Regression.


XGBoost With Lag Features MAE: 22.2625
XGBoost With Lag Features RMSE: 91.7893
XGBoost With Lag Features WMAPE: 11.44%
XGBoost With Lag Features MAPE (non-zero actuals only): 33.60%
"""
