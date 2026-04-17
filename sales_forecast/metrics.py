import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def wmape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()

    if denominator == 0:
        return np.nan

    return np.abs(y_true - y_pred).sum() / denominator


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        return np.nan

    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))


def print_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape_value = wmape(y_true, y_pred)
    mape_value = safe_mape(y_true, y_pred)

    print(f"{label} MAE:", round(mae, 4))
    print(f"{label} RMSE:", round(rmse, 4))
    print(f"{label} WMAPE:", f"{wmape_value:.2%}" if not np.isnan(wmape_value) else "undefined")
    print(f"{label} MAPE (non-zero actuals only):", f"{mape_value:.2%}" if not np.isnan(mape_value) else "undefined")
