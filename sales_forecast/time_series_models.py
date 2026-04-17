import clean_data as cd
import matplotlib.pyplot as plt
from metrics import print_metrics
import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


def aggregate_daily_sales(df):
    daily_sales = df.groupby("date", as_index=False)["sales"].sum().sort_values("date").reset_index(drop=True)
    return daily_sales


def validate_daily_series(df, min_points=30):
    n_points = len(df)
    if n_points < min_points:
        raise ValueError(
            f"Need at least {min_points} daily points for time-series modeling, "
            f"but only found {n_points}. Increase n_rows when loading data."
        )


def train_test_split_time_series(df, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def moving_avg_forecast(train_df, test_df, window=7, update_with_actuals=False):
    history = train_df["sales"].tolist()
    if len(history) < window:
        raise ValueError(f"Need at least {window} training points for moving average.")

    predictions = []

    for actual_value in test_df["sales"]:
        recent_values = history[-window:]
        pred = float(np.mean(recent_values))
        predictions.append(pred)

        # Use actual test values only for walk-forward evaluation.
        history.append(float(actual_value) if update_with_actuals else pred)

    res_df = test_df.copy()
    res_df["moving_avg_prediction"] = predictions
    return res_df


def exponential_smoothing_forecast(
    train_df,
    test_df,
    smoothing_level=None,
    optimized=True,
    update_with_actuals=False,
):
    history = train_df["sales"].astype(float).tolist()
    predictions = []

    if update_with_actuals:
        for actual_value in test_df["sales"]:
            series = pd.Series(history)
            if smoothing_level is None:
                fitted_model = SimpleExpSmoothing(series).fit(optimized=True)
            else:
                fitted_model = SimpleExpSmoothing(series).fit(
                    smoothing_level=smoothing_level,
                    optimized=False,
                )

            pred = float(fitted_model.forecast(1).iloc[0])
            predictions.append(pred)
            history.append(float(actual_value))
    else:
        series = pd.Series(history)
        if smoothing_level is None:
            fitted_model = SimpleExpSmoothing(series).fit(optimized=optimized)
        else:
            fitted_model = SimpleExpSmoothing(series).fit(
                smoothing_level=smoothing_level,
                optimized=False,
            )
        predictions = fitted_model.forecast(len(test_df)).astype(float).tolist()

    res_df = test_df.copy()
    res_df["exp_smooth_pred"] = predictions
    return res_df


def holt_winters_forecast(train_df, test_df, seasonal_periods=7):
    """
    holt_winters_forecast is a time-series forecasting method that extends exponential smoothing to handle more than just the current level.
    It usually models 3 parts:
    - level: the overall average value
    - trend: whether the series is going up or down
    - seasonality: repeating patterns, like weekly cycles
    
    - trend="add": add a trend component if sales are generally increasing or decreasing
    - seasonal="add": add a repeating seasonal pattern
    - seasonal_periods=7: assume the pattern repeats every 7 days, so it is modeling weekly seasonality
    """
    train_sales = train_df["sales"].astype(float)
    fitted_model = ExponentialSmoothing(
        train_sales,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)

    res_df = test_df.copy()
    res_df["holt_winters_pred"] = fitted_model.forecast(len(test_df)).astype(float).tolist()
    return res_df


def evaluate_forecast(result_df, pred_col, label):
    y_true = result_df["sales"]
    y_pred = result_df[pred_col]
    print_metrics(y_true, y_pred, label=label)


def plot_forecast(result_df, pred_col, title):
    plt.figure(figsize=(12, 5))
    plt.plot(result_df["date"], result_df["sales"], label="Actual")
    plt.plot(result_df["date"], result_df[pred_col], label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ds = cd.load_sales_dataset_stream()
    df = cd.take_sample_from_stream(ds, n_rows=100000)
    df = cd.clean_sales_data(df)

    daily_sales_df = aggregate_daily_sales(df)
    validate_daily_series(daily_sales_df, min_points=30)

    train_df, test_df = train_test_split_time_series(daily_sales_df, test_size=0.2)

    moving_avg_df = moving_avg_forecast(train_df, test_df, window=7, update_with_actuals=False)
    evaluate_forecast(moving_avg_df, pred_col="moving_avg_prediction", label="Moving Average")
    print(moving_avg_df.head())
    plot_forecast(moving_avg_df, pred_col="moving_avg_prediction", title="Moving Average Forecast vs Actual")

    exp_smoothing_df = exponential_smoothing_forecast(
        train_df,
        test_df,
        smoothing_level=None,
        optimized=True,
        update_with_actuals=False,
    )
    evaluate_forecast(exp_smoothing_df, pred_col="exp_smooth_pred", label="Simple Exponential Smoothing")
    print(exp_smoothing_df.head())
    plot_forecast(
        exp_smoothing_df,
        pred_col="exp_smooth_pred",
        title="Simple Exponential Smoothing Forecast vs Actual",
    )

    holt_winters_df = holt_winters_forecast(train_df, test_df, seasonal_periods=7)
    evaluate_forecast(holt_winters_df, pred_col="holt_winters_pred", label="Holt-Winters")
    print(holt_winters_df.head())
    plot_forecast(
        holt_winters_df,
        pred_col="holt_winters_pred",
        title="Holt-Winters Forecast vs Actual",
    )


if __name__ == "__main__":
    main()


"""
Moving Average MAE: 77528.3985
Moving Average RMSE: 111974.9176
Moving Average WMAPE: 23.74%
Moving Average MAPE (non-zero actuals only): 93.62%
WMAPE = 23.74% - means total absolute error is about 23.74% of total actual sales overall
MAPE = 93.62% - means the average row-level percentage error is very large
These can be very different because MAPE treats every row equally, while WMAPE gives more weight to larger sales values.

Simple Exponential Smoothing MAE: 77681.6547
Simple Exponential Smoothing RMSE: 112890.0788
Simple Exponential Smoothing WMAPE: 23.78%
Simple Exponential Smoothing MAPE (non-zero actuals only): 92.09%

Holt-Winters MAE: 49501.6331
Holt-Winters RMSE: 77239.8644
Holt-Winters WMAPE: 15.16%
Holt-Winters MAPE (non-zero actuals only): 73.99%
"""





