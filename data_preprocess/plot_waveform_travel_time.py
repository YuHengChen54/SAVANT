from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot waveform travel-time style figure for a single event."
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("../data/TSMIP_1999_2019_Vs30_integral.hdf5"),
        help="Path to HDF5 file (relative to this script if not absolute).",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=24784,
        help="Earthquake event ID under data/<event_id>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. If not provided, defaults to figures/waveform_example_<event_id>.png.",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="z",
        choices=["norm", "z", "ns", "ew"],
        help="Waveform component to plot.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=200.0,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration to display in seconds.",
    )
    parser.add_argument(
        "--scale-km",
        type=float,
        default=-1.0,
        help="Horizontal wiggle scale in km. Use <=0 for automatic scaling.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot waveforms for all events in the database. Output path and event-id are ignored.",
    )
    return parser.parse_args()


def resolve_from_this_file(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (THIS_DIR / path).resolve()


def decode_station_names(raw_names: np.ndarray) -> np.ndarray:
    decoded = []
    for s in raw_names:
        if isinstance(s, (bytes, np.bytes_)):
            decoded.append(s.decode("utf-8", errors="ignore").strip())
        else:
            decoded.append(str(s).strip())
    return np.asarray(decoded)


def get_waveform_component(acc_traces: np.ndarray, component: str) -> np.ndarray:
    if component == "z":
        return acc_traces[:, :, 0]
    if component == "ns":
        return acc_traces[:, :, 1]
    if component == "ew":
        return acc_traces[:, :, 2]
    norm = np.sqrt(np.sum(acc_traces**2, axis=2))
    # Demean each trace in norm mode to avoid one-sided offset before normalization.
    norm = norm - np.mean(norm, axis=1, keepdims=True)
    return norm


def compute_auto_scale_km(distances_km: np.ndarray) -> float:
    unique_sorted = np.unique(np.sort(distances_km))
    if unique_sorted.size < 2:
        return 3.0

    diffs = np.diff(unique_sorted)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return 3.0

    # Use a larger fraction of typical station spacing so waveforms are more visible.
    return 2.0 * float(np.median(positive_diffs))


def load_event_waveforms_and_distance(
    hdf5_path: Path,
    event_id: int,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        event_key = str(event_id)
        if event_key not in f["data"]:
            raise KeyError(f"Event {event_id} not found under /data.")

        event_group = f["data"][event_key]
        acc_traces = np.asarray(event_group["acc_traces"], dtype=float)
        station_names = decode_station_names(np.asarray(event_group["station_name"]))

    traces_df = pd.read_hdf(hdf5_path, key="metadata/traces_metadata")
    event_meta = traces_df.loc[traces_df["EQ_ID"] == event_id, ["station_name", "epdis (km)"]].copy()
    event_meta["station_name"] = event_meta["station_name"].astype(str).str.strip()
    event_meta = event_meta.dropna(subset=["station_name", "epdis (km)"])
    event_meta = event_meta.drop_duplicates(subset=["station_name"], keep="first")

    distance_map = dict(zip(event_meta["station_name"], event_meta["epdis (km)"]))
    distances = np.array([distance_map.get(name, np.nan) for name in station_names], dtype=float)

    valid = np.isfinite(distances)
    if not np.any(valid):
        raise ValueError(f"No valid epicentral distances found for event {event_id}.")

    waveforms = get_waveform_component(acc_traces[valid], component)
    distances = distances[valid]

    return waveforms, distances


def plot_waveform_travel_time(
    waveforms: np.ndarray,
    distances_km: np.ndarray,
    sampling_rate: float,
    duration: float,
    scale_km: float,
    event_id: int,
    component: str,
    output_path: Path,
) -> None:
    n_sta, n_samples = waveforms.shape
    max_samples = min(int(round(duration * sampling_rate)), n_samples)
    waveforms = waveforms[:, :max_samples]

    # Normalize each station to keep visual balance across traces.
    peak = np.max(np.abs(waveforms), axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    waveforms_norm = waveforms / peak

    order = np.argsort(distances_km)
    distances_km = distances_km[order]
    waveforms_norm = waveforms_norm[order]

    if scale_km <= 0:
        scale_km = compute_auto_scale_km(distances_km)

    t = np.arange(max_samples) / sampling_rate

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for i in range(n_sta):
        x = distances_km[i] + scale_km * waveforms_norm[i]
        ax.plot(x, t, color="#1a1a1a", linewidth=0.23, alpha=0.75)

    ax.set_xlim(float(np.min(distances_km) - 1.5 * scale_km), float(np.max(distances_km) + 1.5 * scale_km))
    ax.set_ylim(0.0, duration)
    ax.set_xlabel("Epicentral Distance (km)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Event {event_id} Waveform Travel-Time Plot ({component})")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hdf5_path = resolve_from_this_file(args.hdf5_path)

    if args.all:
        print("Gathering all event IDs from database...")
        with h5py.File(hdf5_path, "r") as f:
            event_ids = [int(k) for k in f["data"].keys()]
        
        # 多包一層資料夾放所有地震的圖片
        out_folder = resolve_from_this_file(Path(f"figures/all_waveforms_{args.component}"))
        out_folder.mkdir(parents=True, exist_ok=True)
        
        for eid in tqdm(event_ids, desc="Plotting ALL events"):
            try:
                waveforms, distances_km = load_event_waveforms_and_distance(
                    hdf5_path=hdf5_path,
                    event_id=eid,
                    component=args.component,
                )
                
                # 篩選測站數量 > 75 的事件才畫圖
                if len(distances_km) <= 75:
                    continue
                
                out_path = out_folder / f"waveform_example_{eid}_{args.component}.png"
                plot_waveform_travel_time(
                    waveforms=waveforms,
                    distances_km=distances_km,
                    sampling_rate=args.sampling_rate,
                    duration=args.duration,
                    scale_km=args.scale_km,
                    event_id=eid,
                    component=args.component,
                    output_path=out_path,
                )
            except Exception as e:
                # 忽略資料有缺或壞掉的地震
                continue
    else:
        if args.output is None:
            out_path = Path(f"figures/waveform_example_{args.event_id}_{args.component}.png")
        else:
            out_path = args.output
            
        output_path = resolve_from_this_file(out_path)
    
        waveforms, distances_km = load_event_waveforms_and_distance(
            hdf5_path=hdf5_path,
            event_id=args.event_id,
            component=args.component,
        )
    
        plot_waveform_travel_time(
            waveforms=waveforms,
            distances_km=distances_km,
            sampling_rate=args.sampling_rate,
            duration=args.duration,
            scale_km=args.scale_km,
            event_id=args.event_id,
            component=args.component,
            output_path=output_path,
        )
    
        print(f"Saved figure: {output_path}")
        print(f"Event: {args.event_id} | Stations plotted: {len(distances_km)}")

if __name__ == "__main__":
    main()
