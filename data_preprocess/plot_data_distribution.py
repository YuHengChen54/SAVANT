from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


TEST_YEAR = 2016
TRAIN_LABEL = "Training set"
TEST_LABEL = "Test set"
TRAIN_COLOR = "#4C78A8"
TEST_COLOR = "#F58518"
INTENSITY_LABELS = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
PGA_INTENSITY_THRESHOLDS_LOG = np.log10([0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 100])
PGV_INTENSITY_THRESHOLDS_LOG = np.log10([0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 100])
HIST_LAYOUT = {"left": 0.12, "right": 0.98, "bottom": 0.14, "top": 0.84}
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot data distributions from TSMIP HDF5.")
	parser.add_argument(
		"--hdf5-path",
		type=Path,
		default=Path("../data/TSMIP_1999_2019_Vs30_integral.hdf5"),
		help="Path to TSMIP HDF5 file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("figures"),
		help="Directory to save output figures.",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=40,
		help="Number of bins for histogram plots.",
	)
	return parser.parse_args()


def resolve_from_this_file(path: Path) -> Path:
	if path.is_absolute():
		return path
	return (THIS_DIR / path).resolve()


def degree_minute_to_decimal(degree: pd.Series, minute: pd.Series) -> np.ndarray:
	sign = np.sign(degree.to_numpy())
	sign[sign == 0] = 1.0
	return degree.to_numpy() + sign * np.abs(minute.to_numpy()) / 60.0


def load_metadata(hdf5_path: Path) -> pd.DataFrame:
	event_df = pd.read_hdf(hdf5_path, key="metadata/event_metadata")
	event_df = event_df.copy()
	event_df["latitude"] = degree_minute_to_decimal(event_df["lat"], event_df["lat_minute"])
	event_df["longitude"] = degree_minute_to_decimal(event_df["lon"], event_df["lon_minute"])
	event_df["is_test"] = event_df["year"] == TEST_YEAR
	return event_df


def load_station_metadata(hdf5_path: Path) -> pd.DataFrame:
	trace_df = pd.read_hdf(hdf5_path, key="metadata/traces_metadata")
	sta_df = trace_df[["station_name", "latitude", "longitude", "Vs30"]].copy()
	sta_df = sta_df.dropna(subset=["latitude", "longitude", "Vs30"])
	sta_df = (
		sta_df.groupby("station_name", as_index=False)
		.agg({"latitude": "mean", "longitude": "mean", "Vs30": "mean"})
	)
	return sta_df


def collect_event_level_values(hdf5_path: Path, event_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	train_pga: list[np.ndarray] = []
	test_pga: list[np.ndarray] = []
	train_pgv: list[np.ndarray] = []
	test_pgv: list[np.ndarray] = []

	year_lookup = event_df.set_index("EQ_ID")["year"].to_dict()

	with h5py.File(hdf5_path, "r") as f:
		data_group = f["data"]
		for event_key in data_group.keys():
			event_id = int(event_key)
			year = year_lookup.get(event_id)
			if year is None:
				continue

			event_group = data_group[event_key]
			pga = np.asarray(event_group["pga"]).reshape(-1)
			pgv = np.asarray(event_group["pgv"]).reshape(-1)

			if year == TEST_YEAR:
				test_pga.append(pga)
				test_pgv.append(pgv)
			else:
				train_pga.append(pga)
				train_pgv.append(pgv)

	train_pga_all = np.concatenate(train_pga) if train_pga else np.array([], dtype=float)
	test_pga_all = np.concatenate(test_pga) if test_pga else np.array([], dtype=float)
	train_pgv_all = np.concatenate(train_pgv) if train_pgv else np.array([], dtype=float)
	test_pgv_all = np.concatenate(test_pgv) if test_pgv else np.array([], dtype=float)
	return train_pga_all, test_pga_all, train_pgv_all, test_pgv_all


def marker_size_from_magnitude(magnitude: np.ndarray) -> np.ndarray:
	mag_min = float(np.nanmin(magnitude))
	mag_max = float(np.nanmax(magnitude))
	if mag_max - mag_min < 1e-8:
		return np.full_like(magnitude, 40.0, dtype=float)
	return 10.0 + 350.0 * (magnitude - mag_min) / (mag_max - mag_min)


def histogram_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	bin_count = max(1, bins // 2)
	return np.linspace(np.nanmin(values), np.nanmax(values), bin_count + 1)


def apply_hist_layout(fig: plt.Figure) -> None:
	fig.subplots_adjust(**HIST_LAYOUT)


def plot_event_map(event_df: pd.DataFrame, output_dir: Path) -> None:
	fig = plt.figure(figsize=(10, 9))
	ax = plt.axes(projection=ccrs.PlateCarree())

	pad_deg = 0.5
	lon_min = float(event_df["longitude"].min()) - pad_deg
	lon_max = float(event_df["longitude"].max()) + pad_deg
	lat_min = float(event_df["latitude"].min()) - pad_deg
	lat_max = float(event_df["latitude"].max()) + pad_deg
	ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
	ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#DCEFFF", edgecolor="none", zorder=0)
	ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#E8E8E8", edgecolor="none", zorder=1)
	ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, edgecolor="#666666", zorder=2)
	ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.4, color="#999999", alpha=0.6)

	mag = event_df["magnitude"].to_numpy(dtype=float)

	train_df = event_df[~event_df["is_test"]]
	test_df = event_df[event_df["is_test"]]

	ax.scatter(
		train_df["longitude"],
		train_df["latitude"],
		c=TRAIN_COLOR,
		s=marker_size_from_magnitude(train_df["magnitude"].to_numpy(dtype=float)),
		alpha=0.5,
		edgecolors="#333333",
		linewidth=0.25,
		transform=ccrs.PlateCarree(),
		zorder=3,
	)
	ax.scatter(
		test_df["longitude"],
		test_df["latitude"],
		c=TEST_COLOR,
		s=marker_size_from_magnitude(test_df["magnitude"].to_numpy(dtype=float)),
		alpha=0.5,
		edgecolors="#333333",
		linewidth=0.25,
		transform=ccrs.PlateCarree(),
		zorder=3,
	)

	train_count = len(train_df)
	test_count = len(test_df)
	split_handles = [
		Line2D([0], [0], marker="o", color="none", markerfacecolor=TRAIN_COLOR, markeredgecolor="none", markersize=8, label=f"Training set ({train_count})"),
		Line2D([0], [0], marker="o", color="none", markerfacecolor=TEST_COLOR, markeredgecolor="none", markersize=8, label=f"Test set ({test_count})"),
	]

	mag_for_legend = np.array([4.0, 5.0, 6.0, 7.0])
	mag_sizes = marker_size_from_magnitude(mag_for_legend)
	mag_handles = [
		Line2D(
			[0],
			[0],
			marker="o",
			color="none",
			markerfacecolor="#BBBBBB",
			markeredgecolor="none",
			markersize=np.sqrt(s),
			label=f"{int(m)}",
		)
		for m, s in zip(mag_for_legend, mag_sizes)
	]

	legend1 = ax.legend(handles=split_handles, loc="upper left", frameon=True, title="Dataset split", fontsize=10)
	ax.add_artist(legend1)
	ax.legend(handles=mag_handles, loc="lower right", frameon=True, title="Magnitude scale", fontsize=11, scatterpoints=1, framealpha=0.95, labelspacing=1.2, handlelength=1.5)

	ax.set_title("Earthquake Spatial Distribution in Taiwan")
	fig.tight_layout()
	fig.savefig(output_dir / "01_event_spatial_distribution.png", dpi=600, transparent=True)
	plt.close(fig)


def plot_station_vs30_map(station_df: pd.DataFrame, output_dir: Path) -> None:
	fig = plt.figure(figsize=(10, 9))
	ax = plt.axes(projection=ccrs.PlateCarree())

	pad_deg = 0.5
	lon_min = float(station_df["longitude"].min()) - pad_deg
	lon_max = float(station_df["longitude"].max()) + pad_deg
	lat_min = float(station_df["latitude"].min()) - pad_deg
	lat_max = float(station_df["latitude"].max()) + pad_deg
	ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

	ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#DCEFFF", edgecolor="none", zorder=0)
	ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#E8E8E8", edgecolor="none", zorder=1)
	ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, edgecolor="#666666", zorder=2)
	gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.4, color="#999999", alpha=0.6)
	gl.top_labels = False
	gl.right_labels = False

	blue_cmap = LinearSegmentedColormap.from_list(
		"vs30_blue",
		["#FF5353", "#FFFF7D", "#368300"],
	)

	sta = ax.scatter(
		station_df["longitude"],
		station_df["latitude"],
		c=station_df["Vs30"],
		cmap=blue_cmap,
		s=40,
		alpha=0.9,
		edgecolors="#CFCFCF",
		linewidth=0.25,
		transform=ccrs.PlateCarree(),
		zorder=3,
	)

	cbar = fig.colorbar(sta, ax=ax, shrink=0.82, pad=0.02)
	cbar.set_label("Vs30")
	ax.set_title("Station Spatial Distribution Colored by Vs30")

	fig.tight_layout()
	fig.savefig(output_dir / "06_station_vs30_distribution.png", dpi=600, transparent=True)
	plt.close(fig)


def plot_magnitude_hist(event_df: pd.DataFrame, output_dir: Path, bins: int) -> None:
	fig, ax = plt.subplots(figsize=(6.3, 4.55))

	train_mag = event_df.loc[~event_df["is_test"], "magnitude"].to_numpy(dtype=float)
	test_mag = event_df.loc[event_df["is_test"], "magnitude"].to_numpy(dtype=float)

	all_mag = np.concatenate([train_mag, test_mag])
	hist_bins = np.arange(np.floor(np.nanmin(all_mag) * 8) / 8, np.ceil(np.nanmax(all_mag) * 8) / 8 + 0.125, 0.125)

	train_count = len(train_mag)
	test_count = len(test_mag)
	ax.hist(train_mag, bins=hist_bins, color=TRAIN_COLOR, alpha=0.7, label=f"Training set ({train_count})")
	ax.hist(test_mag, bins=hist_bins, color=TEST_COLOR, alpha=0.7, label=f"Test set ({test_count})")

	ax.set_yscale("log")
	ax.set_xlabel("Magnitude (Mw)")
	ax.set_ylabel("Count")
	ax.set_title("Magnitude Distribution", pad=14)
	ax.grid(alpha=0.25)
	ax.legend()
	apply_hist_layout(fig)
	fig.savefig(output_dir / "02_magnitude_distribution.png", dpi=450, transparent=True)
	plt.close(fig)


def plot_depth_hist(event_df: pd.DataFrame, output_dir: Path, bins: int) -> None:
	fig, ax = plt.subplots(figsize=(6.3, 4.55))

	train_depth = event_df.loc[~event_df["is_test"], "depth"].to_numpy(dtype=float)
	test_depth = event_df.loc[event_df["is_test"], "depth"].to_numpy(dtype=float)

	all_depth = np.concatenate([train_depth, test_depth])
	hist_bins = np.arange(np.floor(np.nanmin(all_depth) / 12.5) * 12.5, np.ceil(np.nanmax(all_depth) / 12.5) * 12.5 + 12.5, 12.5)

	train_count = len(train_depth)
	test_count = len(test_depth)
	ax.hist(train_depth, bins=hist_bins, color=TRAIN_COLOR, alpha=0.7, label=f"Training set ({train_count})")
	ax.hist(test_depth, bins=hist_bins, color=TEST_COLOR, alpha=0.7, label=f"Test set ({test_count})")

	ax.set_yscale("log")
	ax.set_xlabel("Depth (km)")
	ax.set_ylabel("Count")
	ax.set_title("Depth Distribution", pad=14)
	ax.grid(alpha=0.25)
	ax.legend()
	apply_hist_layout(fig)
	fig.savefig(output_dir / "03_depth_distribution.png", dpi=450, transparent=True)
	plt.close(fig)


def plot_motion_hist(
	train_values: np.ndarray,
	test_values: np.ndarray,
	output_dir: Path,
	bins: int,
	output_name: str,
	title: str,
	xlabel: str,
	intensity_thresholds: np.ndarray | None = None,
	intensity_labels: list[str] | None = None,
) -> None:
	train_values = np.asarray(train_values, dtype=float)
	test_values = np.asarray(test_values, dtype=float)
	train_values = train_values[np.isfinite(train_values)]
	test_values = test_values[np.isfinite(test_values)]
	all_values = np.concatenate([train_values, test_values])

	fig, ax = plt.subplots(figsize=(6.3, 4.55))
	hist_bins = np.arange(np.floor(np.nanmin(all_values) * 8) / 8, np.ceil(np.nanmax(all_values) * 8) / 8 + 0.125, 0.125)

	train_count = len(train_values)
	test_count = len(test_values)
	ax.hist(train_values, bins=hist_bins, color=TRAIN_COLOR, alpha=0.7, label=f"Training set ({train_count})")
	ax.hist(test_values, bins=hist_bins, color=TEST_COLOR, alpha=0.7, label=f"Test set ({test_count})")

	ax.set_yscale("log")
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Count")
	ax.set_title(title, pad=14)
	if intensity_thresholds is not None and intensity_labels is not None:
		for x in intensity_thresholds:
			if hist_bins[0] <= x <= hist_bins[-1]:
				ax.axvline(x=x, color="#6B6B6B", linestyle="--", linewidth=0.9, alpha=0.9, zorder=4)

		x_min, x_max = ax.get_xlim()
		label_count = len(intensity_labels)
		usable_thresholds = np.asarray(intensity_thresholds[: max(0, label_count - 1)], dtype=float)

		# Put labels above the top frame, centered in each intensity interval.
		for idx, label in enumerate(intensity_labels):
			if idx == 0:
				left, right = x_min, usable_thresholds[0]
			elif idx < label_count - 1:
				left, right = usable_thresholds[idx - 1], usable_thresholds[idx]
			else:
				left, right = usable_thresholds[-1], x_max

			if right <= x_min or left >= x_max:
				continue

			left = max(left, x_min)
			right = min(right, x_max)
			x_center = 0.5 * (left + right)
			ax.annotate(
				label,
				xy=(x_center, 1.0),
				xycoords=ax.get_xaxis_transform(),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=9,
				color="#444444",
				clip_on=False,
			)
	ax.grid(alpha=0.25)
	ax.legend()
	apply_hist_layout(fig)
	fig.savefig(output_dir / output_name, dpi=450, transparent=True)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	args.hdf5_path = resolve_from_this_file(args.hdf5_path)
	args.output_dir = resolve_from_this_file(args.output_dir)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	event_df = load_metadata(args.hdf5_path)
	station_df = load_station_metadata(args.hdf5_path)
	train_pga, test_pga, train_pgv, test_pgv = collect_event_level_values(args.hdf5_path, event_df)

	# plot_event_map(event_df, args.output_dir)
	# plot_magnitude_hist(event_df, args.output_dir, bins=args.bins)
	# plot_depth_hist(event_df, args.output_dir, bins=args.bins)
	# plot_motion_hist(
	# 	train_values=train_pga,
	# 	test_values=test_pga,
	# 	output_dir=args.output_dir,
	# 	bins=args.bins,
	# 	output_name="04_pga_distribution.png",
	# 	title="PGA Distribution",
	# 	xlabel="log10(m/s$^2$)",
	# 	intensity_thresholds=PGA_INTENSITY_THRESHOLDS_LOG,
	# 	intensity_labels=["", *INTENSITY_LABELS[1:]],
	# )
	# plot_motion_hist(
	# 	train_values=train_pgv,
	# 	test_values=test_pgv,
	# 	output_dir=args.output_dir,
	# 	bins=args.bins,
	# 	output_name="05_pgv_distribution.png",
	# 	title="PGV Distribution",
	# 	xlabel="log10(m/s)",
	# 	intensity_thresholds=PGV_INTENSITY_THRESHOLDS_LOG,
	# 	intensity_labels=INTENSITY_LABELS,
	# )
	plot_station_vs30_map(station_df, args.output_dir)

	print(f"Saved figures to: {args.output_dir}")
	print(f"Event count: {len(event_df)} | Train: {(~event_df['is_test']).sum()} | Test(2016): {(event_df['is_test']).sum()}")
	print(f"PGA samples | Train: {len(train_pga)} | Test: {len(test_pga)}")
	print(f"PGV samples | Train: {len(train_pgv)} | Test: {len(test_pgv)}")
	print("Note: PGA/PGV values in the HDF5 file are already stored in log10 space, so no extra log transform is applied.")


if __name__ == "__main__":
	main()
