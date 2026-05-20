import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score


PGA_THRESHOLD = np.log10(0.25)
PGV_THRESHOLD = np.log10(0.057)


def parse_seconds_from_filename(path):
    basename = os.path.basename(path)
    match = re.search(r"(\d+)\s*sec", basename)
    return int(match.group(1)) if match else None


def get_valid_arrays(df, pred_col, ans_col):
    if pred_col not in df.columns or ans_col not in df.columns:
        missing = [c for c in [pred_col, ans_col] if c not in df.columns]
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    y_pred = df[pred_col].to_numpy()
    y_true = df[ans_col].to_numpy()
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    return y_true[mask], y_pred[mask]


def compute_regression_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan, np.nan, np.nan

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def compute_classification_metrics(y_true, y_pred, threshold):
    logic_true = (y_true > threshold).astype(int)
    logic_pred = (y_pred > threshold).astype(int)

    precision = precision_score(logic_true, logic_pred, zero_division=0)
    recall = recall_score(logic_true, logic_pred, zero_division=0)
    f1 = f1_score(logic_true, logic_pred, zero_division=0)
    return precision, recall, f1


def filter_by_actual_threshold(y_true, y_pred, threshold):
    mask = y_true > threshold
    return y_true[mask], y_pred[mask]


def format_metrics(label, count, r2, rmse, mae, precision, recall, f1, threshold):
    return (
        f"{label}: N={count}, R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, "
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
        f"threshold(log10)={threshold:.6f}"
    )


def format_subset_metrics(label, subset_label, count, r2, rmse, mae):
    return (
        f"{label} {subset_label}: N={count}, R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
    )


def main(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sec = parse_seconds_from_filename(csv_path)

    pga_true, pga_pred = get_valid_arrays(df, "predict_pga", "answer_pga")
    pgv_true, pgv_pred = get_valid_arrays(df, "predict_pgv", "answer_pgv")

    pga_r2, pga_rmse, pga_mae = compute_regression_metrics(pga_true, pga_pred)
    pgv_r2, pgv_rmse, pgv_mae = compute_regression_metrics(pgv_true, pgv_pred)

    pga_precision, pga_recall, pga_f1 = compute_classification_metrics(
        pga_true, pga_pred, PGA_THRESHOLD
    )
    pgv_precision, pgv_recall, pgv_f1 = compute_classification_metrics(
        pgv_true, pgv_pred, PGV_THRESHOLD
    )

    pga_iv_true, pga_iv_pred = filter_by_actual_threshold(pga_true, pga_pred, PGA_THRESHOLD)
    pgv_iv_true, pgv_iv_pred = filter_by_actual_threshold(pgv_true, pgv_pred, PGV_THRESHOLD)

    pga_iv_r2, pga_iv_rmse, pga_iv_mae = compute_regression_metrics(pga_iv_true, pga_iv_pred)
    pgv_iv_r2, pgv_iv_rmse, pgv_iv_mae = compute_regression_metrics(pgv_iv_true, pgv_iv_pred)

    print(f"Input: {csv_path}")
    if sec is not None:
        print(f"Seconds: {sec}")

    print(
        format_metrics(
            "PGA",
            len(pga_true),
            pga_r2,
            pga_rmse,
            pga_mae,
            pga_precision,
            pga_recall,
            pga_f1,
            PGA_THRESHOLD,
        )
    )
    print(
        format_metrics(
            "PGV",
            len(pgv_true),
            pgv_r2,
            pgv_rmse,
            pgv_mae,
            pgv_precision,
            pgv_recall,
            pgv_f1,
            PGV_THRESHOLD,
        )
    )
    print(
        format_subset_metrics(
            "PGA",
            "Actual IV+",
            len(pga_iv_true),
            pga_iv_r2,
            pga_iv_rmse,
            pga_iv_mae,
        )
    )
    print(
        format_subset_metrics(
            "PGV",
            "Actual IV+",
            len(pgv_iv_true),
            pgv_iv_r2,
            pgv_iv_rmse,
            pgv_iv_mae,
        )
    )


if __name__ == "__main__":
    # csv_path = r"..\predict_with_several_physical_feature\model_test_96\model 96 13 sec prediction_vel.csv"
    csv_path = r"..\predict_validate_without_MSFE\model_test_1\model 1 13 sec prediction_vel.csv"
    main(csv_path)
