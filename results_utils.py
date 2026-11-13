from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _get_project_dir():
    file_dir = Path(__file__).resolve().parent
    if file_dir.name.lower() == "notebooks":
        return file_dir.parent
    if (file_dir / "notebooks").is_dir() or (file_dir / "data").is_dir():
        return file_dir
    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "notebooks":
        return cwd.parent
    if (cwd / "notebooks").is_dir() or (cwd / "data").is_dir():
        return cwd
    return file_dir


PROJECT_DIR = _get_project_dir()
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_RESULTS_FILE = RESULTS_DIR / "model_results.csv"
FUTURE_PREDICTIONS_FILE = RESULTS_DIR / "future_predictions.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLUMNS = [
    "Model", "Model_Family", "MAE", "RMSE", "MAPE_percent",
    "N_Evaluated", "Evaluation_Protocol"
]
FUTURE_COLUMNS = ["Model", "Date", "Predicted_Price"]

if not MODEL_RESULTS_FILE.exists():
    pd.DataFrame(columns=MODEL_COLUMNS).to_csv(MODEL_RESULTS_FILE, index=False)
if not FUTURE_PREDICTIONS_FILE.exists():
    pd.DataFrame(columns=FUTURE_COLUMNS).to_csv(FUTURE_PREDICTIONS_FILE, index=False)


def _upsert_csv(file_path, new_rows, key_columns):
    new_data = pd.DataFrame(new_rows)
    if file_path.exists() and file_path.stat().st_size > 0:
        old_data = pd.read_csv(file_path)
        for col in new_data.columns:
            if col not in old_data.columns:
                old_data[col] = np.nan
        for col in old_data.columns:
            if col not in new_data.columns:
                new_data[col] = np.nan
        new_data = new_data[old_data.columns]
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined.to_csv(file_path, index=False)


def save_model_result(
    model_name, actual, predicted, dates=None, model_family=None,
    eval_type="holdout", evaluation_protocol=None
):
    model_name = re.sub(r"\s+", " ", str(model_name)).strip()
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError(f"{model_name}: actual and predicted must have equal lengths")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        raise ValueError(f"{model_name}: no valid predictions to save")
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan
    row = {
        "Model": model_name,
        "Model_Family": model_family or model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "N_Evaluated": int(len(actual)),
        "Evaluation_Protocol": evaluation_protocol or eval_type,
    }
    _upsert_csv(MODEL_RESULTS_FILE, [row], ["Model"])
    print(f"Saved metrics for {model_name} to {MODEL_RESULTS_FILE}")
    return row


def save_future_predictions(model_name, dates, predicted_prices):
    model_name = re.sub(r"\s+", " ", str(model_name)).strip()
    dates = list(dates)
    predicted_prices = np.asarray(predicted_prices, dtype=float).reshape(-1)
    if len(dates) != len(predicted_prices):
        raise ValueError(f"{model_name}: dates and predicted prices must have equal lengths")
    rows = []
    for date, price in zip(dates, predicted_prices):
        if np.isfinite(price):
            rows.append({
                "Model": model_name,
                "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "Predicted_Price": float(price),
            })
    if not rows:
        raise ValueError(f"{model_name}: no valid future predictions to save")
    _upsert_csv(FUTURE_PREDICTIONS_FILE, rows, ["Model", "Date"])
    print(f"Saved future predictions for {model_name} to {FUTURE_PREDICTIONS_FILE}")