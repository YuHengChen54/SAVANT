from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a single station's waveform (1-comp or 3-comp) from HDF5."
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("../data/TSMIP_1999_2019_Vs30_integral.hdf5"),
        help="Path to HDF5 file (relative to this script).",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=5932,
        help="Earthquake event ID.",
    )
    parser.add_argument(
        "--station",
        type=str,
        default="TCU071",
        help="Station name to plot.",
    )
    parser.add_argument(
        "--waveform",
        type=str,
        default="acc",
        choices=["acc", "vel", "lowfreq_vel", "pd", "cvav", "TP"],
        help="Waveform type to plot.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds (X-axis max).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=200.0,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Falls back to auto-naming if not provided.",
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


def main() -> None:
    args = parse_args()
    hdf5_path = resolve_from_this_file(args.hdf5_path)

    if args.output is None:
        out_path = Path(f"figures/single_station_{args.event_id}_{args.station}_{args.waveform}.png")
    else:
        out_path = args.output
        
    output_path = resolve_from_this_file(out_path)

    with h5py.File(hdf5_path, "r") as f:
        event_key = str(args.event_id)
        if event_key not in f["data"]:
            raise KeyError(f"Event {args.event_id} not found in HDF5.")

        event_group = f["data"][event_key]
        keys = list(event_group.keys())

        # Determine the exact key name in HDF5 (e.g. acc vs acc_traces)
        data_key = args.waveform
        if f"{args.waveform}_traces" in keys:
            data_key = f"{args.waveform}_traces"
        elif args.waveform not in keys:
            raise KeyError(f"Waveform data for '{args.waveform}' not found. Available keys: {keys}")

        dataset = event_group[data_key]
        station_names = decode_station_names(np.asarray(event_group["station_name"]))

        idx = np.where(station_names == args.station)[0]
        if len(idx) == 0:
            raise ValueError(f"Station {args.station} not found in event {args.event_id}.")
        idx = idx[0]

        data = np.asarray(dataset[idx], dtype=float)

    max_samples = int(round(args.duration * args.sampling_rate))
    
    # Trim to duration if longer
    if data.shape[0] > max_samples:
        data = data[:max_samples]

    n_samples = data.shape[0]
    t = np.arange(n_samples) / args.sampling_rate

    # Plot logic
    # Check if 3-components
    is_3_comp = (data.ndim == 2 and data.shape[1] == 3)

    if is_3_comp:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        components = ["Z", "NS", "EW"]
        for j in range(3):
            axes[j].plot(t, data[:, j], color="black", linewidth=1.0)
            axes[j].set_ylabel(f"{args.waveform.upper()} ({components[j]})")
            axes[j].grid(alpha=0.3)
            axes[j].set_xlim(0, args.duration)

        axes[2].set_xlabel("Time (s)")
        axes[0].set_title(f"Event {args.event_id} - Station {args.station} ({args.waveform.upper()})")

    else:
        # 1-component
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Flatten if it's shape (N, 1)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
            
        ax.plot(t, data, color="black", linewidth=1.0)
        ax.set_ylabel(args.waveform.upper())
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, args.duration)
        ax.set_title(f"Event {args.event_id} - Station {args.station} ({args.waveform.upper()})")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
