from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from shapely.geometry import Polygon


MODEL_NUM = 96
EQ_ID = 24784
LABEL_TYPE = "pga"
ISSUE_SECS = [3, 5, 7, 10, 13]
SAMPLE_RATE = 200
FIRST_PICK_SEC = 5
LATENCY_SEC = 4
CONTOUR_STEP = 5
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "1999_2019_final_catalog.csv"

if LABEL_TYPE == "pga":
    LABEL_THRESHOLD = np.log10(0.25)  # PGA intensity IV
elif LABEL_TYPE == "pgv":
    LABEL_THRESHOLD = np.log10(0.019)
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
        normalized["predict"] = normalized[f"predict_{LABEL_TYPE}"]

    if "answer" not in normalized.columns:
        normalized["answer"] = normalized[f"answer_{LABEL_TYPE}"]

    time_column = resolve_time_column(normalized)
    normalized["time_window"] = normalized[time_column]
    normalized["issue_sec"] = sec

    required_columns = [
        "EQ_ID",
        "station_name",
        "longitude",
        "latitude",
        "epdis (km)",
        "predict",
        "answer",
        "time_window",
        "issue_sec",
        "p_picks",
    ]
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in {sec} sec file: {missing_columns}")

    return normalized[required_columns]


def load_event_predictions(base_dir: Path, sec: int) -> pd.DataFrame:
    file_path = build_prediction_path(base_dir, sec)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {file_path}")

    frame = pd.read_csv(file_path)
    event_frame = frame.query(f"EQ_ID == {EQ_ID}")
    if event_frame.empty:
        raise FileNotFoundError(f"No prediction rows found for EQ_ID {EQ_ID} in {file_path}")

    return normalize_prediction_frame(event_frame, sec)


def select_alert_rows(all_predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = all_predictions[all_predictions["predict"] > LABEL_THRESHOLD].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.copy()
    eligible["warning_time (sec)"] = (
        eligible["time_window"] / SAMPLE_RATE - (eligible["issue_sec"] + FIRST_PICK_SEC + LATENCY_SEC)
    )
    return eligible


def load_all_issue_timings(base_dir: Path) -> pd.DataFrame:
    """Load predictions for all ISSUE_SECS and normalize frames (same logic as scatter script)."""
    frames = []
    for sec in ISSUE_SECS:
        file_path = build_prediction_path(base_dir, sec)
        if not file_path.exists():
            print(f"Skip missing file: {file_path}")
            continue

        frame = pd.read_csv(file_path)
        # 只保留目標地震 EQ_ID 的列
        event_frame = frame.query(f"EQ_ID == {EQ_ID}")
        if event_frame.empty:
            print(f"No rows for EQ_ID {EQ_ID} in {file_path}; skipping")
            continue

        frames.append(normalize_prediction_frame(event_frame, sec))

    if not frames:
        raise FileNotFoundError(f"No prediction files were found under: {base_dir}")

    return pd.concat(frames, ignore_index=True)


def select_earliest_alert_rows(all_predictions: pd.DataFrame) -> pd.DataFrame:
    """From multi-issue predictions, select earliest alert per (EQ_ID, station_name)."""
    eligible = all_predictions[all_predictions["predict"] > LABEL_THRESHOLD].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["EQ_ID", "station_name", "issue_sec"]) 
    earliest = eligible.groupby(["EQ_ID", "station_name"], as_index=False).first()
    earliest = earliest.copy()
    earliest["warning_time (sec)"] = (
        earliest["time_window"] / SAMPLE_RATE - (earliest["issue_sec"] + FIRST_PICK_SEC + LATENCY_SEC)
    )
    # 方案 B：裁剪異常值到合理範圍 [0, 70 秒]
    earliest["warning_time (sec)"] = earliest["warning_time (sec)"].clip(lower=0,upper=70) 
    return earliest


def build_grid(trace_info: pd.DataFrame, numcols: int = 160, numrows: int = 240, pad_factor: float = 0.15):
    """Build interpolation grid with optional padding."""
    lon_min, lon_max = trace_info["longitude"].min(), trace_info["longitude"].max()
    lat_min, lat_max = trace_info["latitude"].min(), trace_info["latitude"].max()
    
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    lon_pad = lon_range * pad_factor
    lat_pad = lat_range * pad_factor
    
    xi = np.linspace(lon_min - lon_pad, lon_max + lon_pad, numcols)
    yi = np.linspace(lat_min - lat_pad, lat_max + lat_pad, numrows)
    return np.meshgrid(xi, yi)


def make_contour_levels(warning_times: np.ndarray) -> np.ndarray:
    """根據實際資料範圍生成等高線級別"""
    data_min = np.nanmin(warning_times)
    data_max = np.nanmax(warning_times)
    
    # 向下取舍為 5 的倍數，向上取整為 5 的倍數
    level_min = int(np.floor(data_min / CONTOUR_STEP)) * CONTOUR_STEP
    level_max = int(np.ceil(data_max / CONTOUR_STEP)) * CONTOUR_STEP
    
    return np.arange(level_min, level_max + CONTOUR_STEP, CONTOUR_STEP)


def make_warning_cmap(levels: np.ndarray):
    cmap = plt.get_cmap("jet_r")  # _r 表示反轉
    norm = mcolors.Normalize(vmin=levels.min(), vmax=levels.max())
    return cmap, norm


def create_taiwan_boundary_polygon() -> Polygon:
    """建立台灣邊界多邊形（包含澎湖）"""
    # 台灣本島 + 澎湖群島的邊界頂點（逆時針）
    coords = [
        (119.25, 24), 
        (120.5, 25.5), 
        (122.1, 25.5),
        (122.1, 21.5), 
        (119.25, 21.5), 
    ]
    return Polygon(coords)


def mask_grid_outside_taiwan(xi: np.ndarray, yi: np.ndarray, grid_data: np.ndarray) -> np.ndarray:
    """將台灣邊界外的格點設為 NaN"""
    from shapely.geometry import Point
    taiwan_polygon = create_taiwan_boundary_polygon()
    masked_grid = grid_data.copy()
    
    for i in range(yi.shape[0]):
        for j in range(xi.shape[1]):
            lon, lat = xi[i, j], yi[i, j]
            if not taiwan_polygon.contains(Point(lon, lat)):
                masked_grid[i, j] = np.nan
    
    return masked_grid


def compute_label_positions_along_line(start: tuple, end: tuple, n: int) -> list:
    """Compute n positions (lon, lat) evenly spaced along the line from start to end.

    start/end are (lon, lat) tuples. Returns list of (lon, lat).
    """
    if n <= 0:
        return []
    lons = np.linspace(start[0], end[0], n)
    lats = np.linspace(start[1], end[1], n)
    return list(zip(lons, lats))


def find_nearest_contour_vertex(line, level_idx: int, target: tuple, xi=None, yi=None, grid_warning=None, taiwan_polygon=None):
    """Find nearest contour vertex for a given contour level index to the target (lon, lat).
    Preference order:
      1) nearest contour vertex that lies inside `taiwan_polygon` (if provided)
      2) nearest contour vertex anywhere
      3) nearest grid point (xi, yi) with non-NaN grid_warning inside taiwan_polygon
    Returns (x, y) or None if not found.
    """
    if target is None:
        return None
    tx, ty = float(target[0]), float(target[1])
    best = None
    best_dist = float('inf')

    verts_list = []

    # collect vertices from allsegs (preferred per-level structure)
    if hasattr(line, 'allsegs'):
        allsegs = getattr(line, 'allsegs', [])
        if level_idx < len(allsegs):
            for seg in allsegs[level_idx]:
                if getattr(seg, 'shape', (0,))[0] > 0:
                    verts_list.extend([(float(x), float(y)) for x, y in seg])

    # collect from collections
    if not verts_list and hasattr(line, 'collections'):
        try:
            coll = line.collections[level_idx]
            paths = getattr(coll, 'get_paths', lambda: [])()
            for p in paths:
                verts = getattr(p, 'vertices', None)
                if verts is None:
                    continue
                verts_list.extend([(float(x), float(y)) for x, y in verts])
        except Exception:
            pass

    # collect from geometries
    if not verts_list:
        geoms = []
        if hasattr(line, 'geoms'):
            try:
                geoms = list(line.geoms)
            except Exception:
                geoms = []
        else:
            try:
                geoms = list(line.geometries())
            except Exception:
                geoms = []

        if level_idx < len(geoms):
            g = geoms[level_idx]
            try:
                if g.geom_type in ('LineString', 'LinearRing'):
                    coords = list(g.coords)
                elif g.geom_type == 'MultiLineString':
                    parts = list(g.geoms)
                    coords = list(parts[0].coords)
                else:
                    coords = []
                verts_list.extend([(float(x), float(y)) for x, y in coords])
            except Exception:
                pass

    # If we have vertices, prefer those inside taiwan_polygon
    from shapely.geometry import Point
    if verts_list:
        inside = []
        if taiwan_polygon is not None:
            for x, y in verts_list:
                try:
                    if taiwan_polygon.contains(Point(x, y)):
                        inside.append((x, y))
                except Exception:
                    continue
        if inside:
            # pick nearest inside vertex
            for x, y in inside:
                d = (x - tx) ** 2 + (y - ty) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (x, y)
            return best

        # else pick nearest vertex overall
        for x, y in verts_list:
            d = (x - tx) ** 2 + (y - ty) ** 2
            if d < best_dist:
                best_dist = d
                best = (x, y)
        if best is not None:
            return best

    # fallback: search nearest land grid point (xi, yi, grid_warning)
    if xi is not None and yi is not None and grid_warning is not None and taiwan_polygon is not None:
        xs = xi.flatten()
        ys = yi.flatten()
        vals = grid_warning.flatten()
        # candidate indices where value is finite and inside taiwan polygon
        candidates = []
        for i, (x, y, v) in enumerate(zip(xs, ys, vals)):
            if np.isfinite(v):
                try:
                    if taiwan_polygon.contains(Point(float(x), float(y))):
                        candidates.append((float(x), float(y)))
                except Exception:
                    continue
        for x, y in candidates:
            d = (x - tx) ** 2 + (y - ty) ** 2
            if d < best_dist:
                best_dist = d
                best = (x, y)
        return best

    return None
def load_eventmeta_from_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG_PATH)
    event = catalog[catalog["EQ_ID"] == EQ_ID].copy()
    if event.empty:
        raise FileNotFoundError(f"EQ_ID {EQ_ID} not found in catalog: {CATALOG_PATH}")

    event = event.assign(
        latitude=event["lat"] + event["lat_minute"] / 60,
        longitude=event["lon"] + event["lon_minute"] / 60,
    )
    return event


def plot_warning_time_contour(trace_info: pd.DataFrame, eventmeta: pd.DataFrame, issue_sec: int, output_path: Path) -> None:
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(8, 8), dpi=500)

    ax_map.coastlines("10m", linewidth=0.8)
    ax_map.add_feature(cfeature.LAND, facecolor="#f2efe8", zorder=0)

    xi, yi = build_grid(trace_info)
    grid_warning = griddata(
        (trace_info["longitude"], trace_info["latitude"]),
        trace_info["warning_time (sec)"],
        (xi, yi),
        method="linear",
    )

    if np.isnan(grid_warning).all():
        raise ValueError("Interpolation produced only NaN values; not enough spatial coverage to draw contours.")

    nearest_grid = griddata(
        (trace_info["longitude"], trace_info["latitude"]),
        trace_info["warning_time (sec)"],
        (xi, yi),
        method="nearest",
    )
    grid_warning = np.where(np.isnan(grid_warning), nearest_grid, grid_warning)

    # 高斯滤波平滑等高線
    grid_warning = ndimage.gaussian_filter(grid_warning, sigma=1.5)
    
    # 遮罩台灣邊界外的區域
    grid_warning = mask_grid_outside_taiwan(xi, yi, grid_warning)

    # 根據實際資料範圍生成等高線
    contour_levels = make_contour_levels(grid_warning)
    cmap, norm = make_warning_cmap(contour_levels)

    filled = ax_map.contourf(
        xi,
        yi,
        grid_warning,
        levels=contour_levels,
        cmap=cmap,
        norm=norm,
        zorder=1,
        alpha=0.88,
        transform=src_crs,
    )

    ax_map.add_feature(cfeature.OCEAN, facecolor="#dceef7", zorder=2.5, edgecolor="none")

    line = ax_map.contour(
        xi,
        yi,
        grid_warning,
        levels=contour_levels,
        colors="k",
        linewidths=1.8,
        alpha=0.6,
        zorder=2,
        transform=src_crs,
    )

    # 固定標籤位置在使用者指定的連線上
    start_pos = (120.5, 23.0)
    end_pos = (121.25, 24.75)
    manual_positions = compute_label_positions_along_line(start_pos, end_pos, len(contour_levels))

    # snap each target manual position to the nearest actual contour vertex on that level
    adjusted_positions = []
    for idx in range(len(contour_levels)):
        target = manual_positions[idx] if idx < len(manual_positions) else None
        snapped = find_nearest_contour_vertex(line, idx, target)
        adjusted_positions.append(snapped if snapped is not None else target)

    # 直接把數字放在已對齊的座標上，避免 clabel 自動移到海上
    for idx, level in enumerate(contour_levels):
        level_text = int(round(float(level)))
        if level_text in (0, 70):
            continue
        pos = adjusted_positions[idx] if idx < len(adjusted_positions) else None
        if pos is None:
            continue
        ax_map.text(
            pos[0],
            pos[1],
            f"{level_text}",
            transform=src_crs,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            path_effects=[
                patheffects.withStroke(linewidth=2.2, foreground="#f2efe8"),
            ],
            zorder=12,
        )

    event_lon = float(eventmeta["longitude"].iloc[0])
    event_lat = float(eventmeta["latitude"].iloc[0])
    ax_map.scatter(
        event_lon,
        event_lat,
        color="red",
        edgecolors="k",
        linewidth=1,
        marker="*",
        s=350,
        zorder=10,
        label="Epicenter",
        transform=src_crs,
    )

    xmin, xmax = trace_info["longitude"].min(), trace_info["longitude"].max()
    ymin, ymax = trace_info["latitude"].min(), trace_info["latitude"].max()
    pad_x = (xmax - xmin) * 0.25 if xmax > xmin else 0.15
    pad_y = (ymax - ymin) * 0.25 if ymax > ymin else 0.15
    ax_map.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_map.set_ylim(ymin - pad_y, ymax + pad_y)

    xmin, xmax = ax_map.get_xlim()
    ymin, ymax = ax_map.get_ylim()
    xticks = np.arange(np.floor(xmin * 2) / 2, xmax + 0.1, 0.5)
    yticks = np.arange(np.floor(ymin * 2) / 2, ymax + 0.1, 0.5)
    ax_map.set_xticks(xticks, crs=src_crs)
    ax_map.set_yticks(yticks, crs=src_crs)

    ax_map.xaxis.set_major_formatter(LongitudeFormatter())
    ax_map.yaxis.set_major_formatter(LatitudeFormatter())
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # ax_map.set_title(
    #     f"EQ {EQ_ID} warning time contour map | issue_secs {ISSUE_SECS} | PGA threshold: III | warning time clipped to 0-70 sec | 5 sec contour interval"
    # )
    ax_map.legend(loc="upper right")

    cbar = plt.colorbar(filled, ax=ax_map, pad=0.02, shrink=0.86)
    cbar.set_label("Warning time (sec)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "predict_with_several_physical_feature" / f"model_test_{MODEL_NUM}"
    output_dir = data_dir / f"model_{MODEL_NUM}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    eventmeta = load_eventmeta_from_catalog()

    # 讀入所有 issue_sec，並取每個站最早的 alert（與 scatter 腳本保持一致）
    all_predictions = load_all_issue_timings(data_dir)
    earliest_alerts = select_earliest_alert_rows(all_predictions)

    if earliest_alerts.empty:
        raise ValueError(f"No warning_time values exceed threshold for EQ_ID {EQ_ID} in any ISSUE_SECS: {ISSUE_SECS}")

    # 使用 earliest_alerts 畫一張等高線圖
    output_path = output_dir / f"model{MODEL_NUM}_eq{EQ_ID}_earliest_warning_time_contour_map_Intensity_IV.png"
    # 傳入一個代表性的 issue_sec（最早被選到的 issue_sec），僅供圖示標題使用
    representative_issue = int(earliest_alerts['issue_sec'].min())
    plot_warning_time_contour(earliest_alerts, eventmeta, representative_issue, output_path)

    print(f"Saved figure to: {output_path}")
    print(f"Plotted stations: {len(earliest_alerts)}")


if __name__ == "__main__":
    main()