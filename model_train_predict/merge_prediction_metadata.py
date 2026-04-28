import os
from itertools import combinations

import h5py
import pandas as pd


def build_model_feature_mapping(model_start_index=72):
    """Build model index mapping aligned with multi_station_training.py."""
    candidate_physical_features = ["cvaa_log1p", "Ia_log1p", "IV2_log1p", "TP_log1p"]
    physical_feature_combinations = []
    for r in range(1, len(candidate_physical_features) + 1):
        physical_feature_combinations.extend(combinations(candidate_physical_features, r))

    model_index = model_start_index
    model_to_features = {}
    for feature_combo in physical_feature_combinations:
        physical_feature_list = list(feature_combo)
        for chosen_intensity in ["IV"]:
            for loss_mode in ["MSFE"]:
                for batch_size in [8]:
                    for learning_rate in [5e-5]:
                        for _ in range(2):
                            model_index += 1
                            model_to_features[model_index] = {
                                "physical_feature": physical_feature_list,
                                "intensity": chosen_intensity,
                                "loss_mode": loss_mode,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate,
                            }
    return model_to_features


def build_trace_merge_catalog(catalog_path, traces_info_path):
    catalog = pd.read_csv(catalog_path)
    traces_info = pd.read_csv(traces_info_path)

    trace_merge_catalog = pd.merge(
        traces_info,
        catalog[
            [
                "EQ_ID",
                "lat",
                "lat_minute",
                "lon",
                "lon_minute",
                "depth",
                "magnitude",
                "nsta",
                "nearest_sta_dist (km)",
            ]
        ],
        on="EQ_ID",
        how="left",
    )
    trace_merge_catalog["event_lat"] = (
        trace_merge_catalog["lat"] + trace_merge_catalog["lat_minute"] / 60
    )
    trace_merge_catalog["event_lon"] = (
        trace_merge_catalog["lon"] + trace_merge_catalog["lon_minute"] / 60
    )
    trace_merge_catalog.drop(["lat", "lat_minute", "lon", "lon_minute"], axis=1, inplace=True)
    trace_merge_catalog.rename(columns={"elevation (m)": "elevation"}, inplace=True)
    return trace_merge_catalog


def decode_station_name(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def merge_metadata_for_models(
    models_to_test,
    mask_seconds,
    prediction_root="../predict_with_several_physical_feature",
    station_name_data_path="../data/TSMIP_1999_2019_Vs30_log1p.hdf5",
    catalog_path="../data/1999_2019_final_catalog.csv",
    traces_info_path="../data/1999_2019_final_traces_Vs30.csv",
):
    trace_merge_catalog = build_trace_merge_catalog(catalog_path, traces_info_path)

    with h5py.File(station_name_data_path, "r") as dataset:
        for num in models_to_test:
            output_path = f"{prediction_root}/model_test_{num}"
            print(f"Merging metadata for model {num}")

            for mask_after_sec in mask_seconds:
                prediction_file = f"{output_path}/model {num} {mask_after_sec} sec prediction_vel.csv"
                merged_output_file = f"{output_path}/{mask_after_sec} sec model{num} with all info_vel.csv"

                if not os.path.exists(prediction_file):
                    print(f"Skip missing prediction file: {prediction_file}")
                    continue

                ensemble_predict = pd.read_csv(prediction_file)
                if "station_name" not in ensemble_predict.columns:
                    ensemble_predict["station_name"] = pd.Series(
                        [None] * len(ensemble_predict), dtype="object"
                    )
                else:
                    ensemble_predict["station_name"] = ensemble_predict["station_name"].astype("object")

                for eq_id in ensemble_predict["EQ_ID"].dropna().unique():
                    eq_id = int(eq_id)
                    station_name = dataset["data"][str(eq_id)]["station_name"][:].tolist()
                    station_name = [decode_station_name(name) for name in station_name]
                    ensemble_predict.loc[
                        ensemble_predict.query(f"EQ_ID=={eq_id}").index, "station_name"
                    ] = station_name

                ensemble_predict["station_name"] = ensemble_predict["station_name"].apply(decode_station_name)

                prediction_with_info = pd.merge(
                    ensemble_predict,
                    trace_merge_catalog.drop(["latitude", "longitude", "elevation"], axis=1),
                    on=["EQ_ID", "station_name"],
                    how="left",
                    suffixes=["_window", "_file"],
                )
                prediction_with_info.to_csv(merged_output_file, index=False)
                print(f"Saved: {merged_output_file}")


if __name__ == "__main__":
    run_all_models = True
    model_num = 73
    mask_seconds = [3, 5, 7, 10, 13, 15]

    model_to_features = build_model_feature_mapping(model_start_index=40)
    if run_all_models:
        models_to_test = sorted(model_to_features.keys())
    else:
        models_to_test = [model_num]

    print(f"Models to merge: {models_to_test}")
    merge_metadata_for_models(models_to_test=models_to_test, mask_seconds=mask_seconds)
