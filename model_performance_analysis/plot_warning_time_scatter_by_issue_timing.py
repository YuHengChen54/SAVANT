from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NUM = 96
ISSUE_SECS = [3, 5, 7, 10, 13]
LABEL_TYPE = "pga"
HIGHLIGHT_EQ_ID = 24784
SAMPLE_RATE = 200
FIRST_PICK_SEC = 5
LATENCY_SEC = 4
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "1999_2019_final_catalog.csv"

if LABEL_TYPE == "pga":
    LABEL_THRESHOLD = np.log10(0.25)
elif LABEL_TYPE == "pgv":
    LABEL_THRESHOLD = np.log10(0.057)
else:
    raise ValueError(f"Unsupported LABEL_TYPE: {LABEL_TYPE}")


def build_prediction_path(base_dir: Path, sec: int) -> Path:
    return base_dir / f"{sec} sec model{MODEL_NUM} with all info_vel.csv"


def resolve_time_column(frame: pd.DataFrame) -> str:
    preferred = f"{LABEL_TYPE}_time_window"
    if preferred in frame.columns:
        return preferred

    fallback = f"{LABEL_TYPE}_time"
    if fallback in frame.columns:
        return fallback

    raise KeyError(f"Missing time column: {preferred} or {fallback}")


def normalize_prediction_frame(frame: pd.DataFrame, sec: int) -> pd.DataFrame:
    normalized = frame.copy()

    if "predict" not in normalized.columns:
        predict_column = f"predict_{LABEL_TYPE}"
        normalized["predict"] = normalized[predict_column]

    if "answer" not in normalized.columns:
        answer_column = f"answer_{LABEL_TYPE}"
        normalized["answer"] = normalized[answer_column]

    time_column = resolve_time_column(normalized)
    normalized["time_window"] = normalized[time_column]
    normalized["issue_sec"] = sec

    required_columns = ["EQ_ID", "station_name", "epdis (km)", "predict", "answer", "time_window", "issue_sec"]
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in {sec} sec file: {missing_columns}")

    return normalized[required_columns]


def load_all_issue_timings(base_dir: Path) -> pd.DataFrame:
    frames = []
    allowed_eq_ids = get_allowed_eq_ids()
    for sec in ISSUE_SECS:
        file_path = build_prediction_path(base_dir, sec)
        if not file_path.exists():
            print(f"Skip missing file: {file_path}")
            continue

        frame = pd.read_csv(file_path)
        frame = frame[frame["EQ_ID"].isin(allowed_eq_ids)].copy()
        if frame.empty:
            continue
        frames.append(normalize_prediction_frame(frame, sec))

    if not frames:
        raise FileNotFoundError(f"No prediction files were found under: {base_dir}")

    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def get_allowed_eq_ids() -> set:
    catalog = pd.read_csv(CATALOG_PATH)
    filtered = catalog[catalog["magnitude"] > 5]
    return set(filtered["EQ_ID"].astype(int).tolist())


def select_earliest_alert_rows(all_predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = all_predictions[all_predictions["predict"] > LABEL_THRESHOLD].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["EQ_ID", "station_name", "issue_sec"])
    earliest = eligible.groupby(["EQ_ID", "station_name"], as_index=False).first()
    earliest = earliest.copy()
    earliest["warning_time (sec)"] = (
        earliest["time_window"] / SAMPLE_RATE - (earliest["issue_sec"] + FIRST_PICK_SEC + LATENCY_SEC)
    )
    return earliest


def compute_regression_stats(earliest_alerts: pd.DataFrame) -> tuple:
    x_values = earliest_alerts["epdis (km)"].to_numpy(dtype=float)
    y_values = earliest_alerts["warning_time (sec)"].to_numpy(dtype=float)
    if x_values.size < 2:
        raise ValueError("At least two points are required to fit a regression line.")

    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    predicted = slope * x_values + intercept
    residual = y_values - predicted
    residual_std = np.std(residual, ddof=1)
    return slope, intercept, residual_std, predicted, residual


def plot_warning_time_scatter(
    earliest_alerts: pd.DataFrame,
    output_path: Path,
    slope: float,
    intercept: float,
    residual_std: float,
) -> None:
    if earliest_alerts.empty:
        raise ValueError("No rows exceed the threshold, so there is nothing to plot.")

    x_values = earliest_alerts["epdis (km)"].to_numpy(dtype=float)

    color_map = {
        3: "#1f77b4",
        5: "#2ca02c",
        7: "#ff7f0e",
        10: "#d62728",
        13: "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

    ax.axhspan(-10, 0, facecolor="0.7", alpha=0.25, zorder=0)
    ax.axhspan(0, 4, facecolor="red", alpha=0.15, zorder=0)

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = slope * x_grid + intercept
    # ax.fill_between(
    #     x_grid,
    #     y_grid - residual_std,
    #     y_grid + residual_std,
    #     color="0.3",
    #     alpha=0.15,
    #     zorder=1.5,
    #     label="1-sigma band",
    # )
    # ax.plot(
    #     x_grid,
    #     y_grid,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.6,
    #     zorder=4,
    #     label=f"Linear fit (y = {slope:.3f}x + {intercept:.2f})",
    # )

    for sec in ISSUE_SECS:
        subset = earliest_alerts[earliest_alerts["issue_sec"] == sec]
        if subset.empty:
            continue

        ax.scatter(
            subset["epdis (km)"],
            subset["warning_time (sec)"],
            s=28,
            alpha=0.75,
            color=color_map[sec],
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            label=f"{sec} sec (n={len(subset)})",
        )

    highlight = earliest_alerts[earliest_alerts["EQ_ID"] == HIGHLIGHT_EQ_ID]
    if not highlight.empty:
        ax.scatter(
            highlight["epdis (km)"],
            highlight["warning_time (sec)"],
            s=70,
            marker="X",
            color="gold",
            edgecolors="k",
            linewidths=0.3,
            alpha=1.0,
            zorder=6,
            label=f"Meinong earthquake",
        )

    _, ymax = ax.get_ylim()
    ax.set_ylim(-10, 34)
    ax.text(
        0.98,
        -8,
        "Blind zone",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        color="0.35",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.98,
        2.0,
        "Latency-adjusted blind zone",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        color="darkred",
        fontsize=11,
        fontweight="bold",
    )

    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.axhline(4, color="darkred", linestyle="--", linewidth=1)
    ax.set_xlabel("Epicentral distance (km)", fontsize=13)
    ax.set_ylabel("Warning time (sec)", fontsize=13)
    ax.set_title("Model 96 earliest warning time by issue timing", fontsize=14)
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "predict_with_several_physical_feature" / f"model_test_{MODEL_NUM}"
    output_dir = data_dir / f"model_{MODEL_NUM}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = load_all_issue_timings(data_dir)
    earliest_alerts = select_earliest_alert_rows(all_predictions)
    slope, intercept, residual_std, predicted, residual = compute_regression_stats(earliest_alerts)

    output_path = output_dir / f"model{MODEL_NUM}_earliest_warning_time_scatter_M5plus_eq{HIGHLIGHT_EQ_ID}_highlight.png"
    plot_warning_time_scatter(earliest_alerts, output_path, slope, intercept, residual_std)

    above_1sigma = earliest_alerts.copy()
    above_1sigma["predicted_warning_time (sec)"] = predicted
    above_1sigma["residual (sec)"] = residual
    above_1sigma["residual_minus_1sigma (sec)"] = residual - residual_std
    above_1sigma = above_1sigma[above_1sigma["residual (sec)"] > residual_std].copy()

    output_columns = [
        "EQ_ID",
        "station_name",
        "issue_sec",
        "epdis (km)",
        "warning_time (sec)",
        "predicted_warning_time (sec)",
        "residual (sec)",
        "residual_minus_1sigma (sec)",
    ]
    above_1sigma = above_1sigma.reindex(columns=output_columns)
    above_1sigma_path = output_dir / f"model{MODEL_NUM}_warning_time_above_1sigma_eqids.csv"
    above_1sigma.to_csv(above_1sigma_path, index=False)

    print(f"Saved figure to: {output_path}")
    print(f"Plotted rows: {len(earliest_alerts)}")
    print(f"Saved +1-sigma details to: {above_1sigma_path}")

    if above_1sigma.empty:
        print("No points above +1-sigma.")
    else:
        eqid_counts = above_1sigma.groupby("EQ_ID").size().sort_values(ascending=False)
        print("EQ_ID above +1-sigma (points):")
        for eq_id, count in eqid_counts.items():
            print(f"  EQ_ID {eq_id}: {count}")


if __name__ == "__main__":
    main()
