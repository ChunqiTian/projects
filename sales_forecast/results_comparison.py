import pandas as pd


def build_results_table():
    results = [
        {
            "model": "Baseline",
            "family": "Baseline / heuristic",
            "task_level": "Row-level sales",
            "mae": 141.6716,
            "rmse": 504.4225,
            "wmape_pct": 66.18,
            "mape_pct": 83.25,
        },
        {
            "model": "Linear Regression",
            "family": "Machine learning",
            "task_level": "Row-level sales",
            "mae": 109.4653,
            "rmse": 424.7541,
            "wmape_pct": 58.60,
            "mape_pct": 66.92,
        },
        {
            "model": "XGBoost",
            "family": "Machine learning",
            "task_level": "Row-level sales",
            "mae": 60.6763,
            "rmse": 131.2024,
            "wmape_pct": 32.48,
            "mape_pct": 229.54,
        },
        {
            "model": "XGBoost With Lag Features",
            "family": "Machine learning",
            "task_level": "Row-level sales",
            "mae": 22.2625,
            "rmse": 91.7893,
            "wmape_pct": 11.44,
            "mape_pct": 33.60,
        },
        {
            "model": "Moving Average",
            "family": "Time series",
            "task_level": "Daily total sales",
            "mae": 77528.3985,
            "rmse": 111974.9176,
            "wmape_pct": 23.74,
            "mape_pct": 93.62,
        },
        {
            "model": "Simple Exponential Smoothing",
            "family": "Time series",
            "task_level": "Daily total sales",
            "mae": 77681.6547,
            "rmse": 112890.0788,
            "wmape_pct": 23.78,
            "mape_pct": 92.09,
        },
        {
            "model": "Holt-Winters",
            "family": "Time series",
            "task_level": "Daily total sales",
            "mae": 49501.6331,
            "rmse": 77239.8644,
            "wmape_pct": 15.16,
            "mape_pct": 73.99,
        },
    ]
    return pd.DataFrame(results)


def add_rankings(results_df):
    ranked_df = results_df.copy()
    ranked_df["wmape_rank_within_task"] = (
        ranked_df.groupby("task_level")["wmape_pct"].rank(method="dense")
    )
    ranked_df["rmse_rank_within_task"] = (
        ranked_df.groupby("task_level")["rmse"].rank(method="dense")
    )
    return ranked_df


def print_summary(results_df):
    print("\nResults Comparison")
    print(results_df.to_string(index=False))

    print("\nBest Model Within Each Task Level (lowest WMAPE)")
    best_by_task = results_df.loc[
        results_df.groupby("task_level")["wmape_pct"].idxmin(),
        ["task_level", "model", "wmape_pct", "rmse"],
    ]
    print(best_by_task.to_string(index=False))

    print("\nImportant Note")
    print(
        "Row-level sales models and daily-total time-series models are different tasks, "
        "so their MAE/RMSE values should not be compared directly."
    )


def main():
    results_df = build_results_table()
    results_df = add_rankings(results_df)
    results_df = results_df.sort_values(
        ["task_level", "wmape_rank_within_task", "rmse_rank_within_task", "model"]
    ).reset_index(drop=True)

    print_summary(results_df)


if __name__ == "__main__":
    main()


"""
Results Comparison
                       model               family        task_level        mae        rmse  wmape_pct  mape_pct  wmape_rank_within_task  rmse_rank_within_task
                Holt-Winters          Time series Daily total sales 49501.6331  77239.8644      15.16     73.99                     1.0                    1.0
              Moving Average          Time series Daily total sales 77528.3985 111974.9176      23.74     93.62                     2.0                    2.0
Simple Exponential Smoothing          Time series Daily total sales 77681.6547 112890.0788      23.78     92.09                     3.0                    3.0
   XGBoost With Lag Features     Machine learning   Row-level sales    22.2625     91.7893      11.44     33.60                     1.0                    1.0
                     XGBoost     Machine learning   Row-level sales    60.6763    131.2024      32.48    229.54                     2.0                    2.0
           Linear Regression     Machine learning   Row-level sales   109.4653    424.7541      58.60     66.92                     3.0                    3.0
                    Baseline Baseline / heuristic   Row-level sales   141.6716    504.4225      66.18     83.25                     4.0                    4.0

Best Model Within Each Task Level (lowest WMAPE)
       task_level                     model  wmape_pct       rmse
Daily total sales              Holt-Winters      15.16 77239.8644
  Row-level sales XGBoost With Lag Features      11.44    91.7893

Important Note
Row-level sales models and daily-total time-series models are different tasks, so their MAE/RMSE values should not be compared directly.
"""

