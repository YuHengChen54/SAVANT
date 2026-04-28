import numpy as np
import pandas as pd
import os
from analysis import Intensity_Plotter,Warning_Time_Plotter
import matplotlib.pyplot as plt


model_nums = range(41, 71)  # Adjusted to match the new model indexing starting at 72
mask_after_secs = [3, 5, 7, 10, 13]
label_type = "pga"
if label_type == "pga":
    label_threshold = np.log10(0.25)
    intensity = "IV"
if label_type == "pgv":
    label_threshold = np.log10(0.057)
    intensity = "IV"

catalog = pd.read_csv(f"../data/1999_2019_final_catalog.csv")
# traces_info = pd.read_csv(
#     f"{Afile_path}/2009_2019_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
# )

# for EQ_ID in catalog.query("year==2016 & magnitude>=5.5")["EQ_ID"]:
for model_num in model_nums:
    path = f"../predict_with_several_physical_feature/model_test_{model_num}"
    output_path = f"{path}/events_analysis"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for mask_after_sec in mask_after_secs:
        prediction_file = f"{path}/{mask_after_sec} sec model{model_num} with all info_vel.csv"
        if not os.path.exists(prediction_file):
            print(f"Skip missing file: {prediction_file}")
            continue

        prediction_with_info = pd.read_csv(prediction_file)

        # for EQ_ID in catalog.query("year==2016 & magnitude>=5.5")["EQ_ID"]:
        for EQ_ID in [24784, 25900]:
            event = catalog[catalog["EQ_ID"] == EQ_ID]
            event = event.assign(
                latitude=event["lat"] + event["lat_minute"] / 60,
                longitude=event["lon"] + event["lon_minute"] / 60,
            )
            event_prediction = prediction_with_info.query(f"EQ_ID=={EQ_ID}").copy()
            if event_prediction.empty:
                print(f"Skip EQ_ID {EQ_ID}: no records in {prediction_file}")
                continue

            fig, ax = Intensity_Plotter.plot_intensity_map(
                trace_info=event_prediction,
                eventmeta=event,
                label_type=label_type,
                true_label=event_prediction[f"answer_{label_type}"],
                pred_label=event_prediction[f"predict_{label_type}"],
                sec=mask_after_sec,
                EQ_ID=EQ_ID,
                grid_method="linear",
                pad=100,
                title=f"Model {model_num} | {mask_after_sec} sec intensity Map",
            )
            fig.savefig(
                f"{output_path}/model{model_num}_{EQ_ID}_{mask_after_sec}sec intensity Map.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            fig, ax = Intensity_Plotter.plot_true_predicted(
                y_true=event_prediction[f"answer_{label_type}"],
                y_pred=event_prediction[f"predict_{label_type}"],
                agg="point",
                point_size=35,
                target=label_type,
                title=f"Model {model_num} | EQID: {EQ_ID}, mag: {event['magnitude'].values[0]}, {mask_after_sec} sec true and predict",
            )
            fig.savefig(
                f"{output_path}/model{model_num}_{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec}sec true predict plot.png", dpi=450, bbox_inches="tight"
            )
            plt.close(fig)

            try:
                # Align expected columns for warning-time utilities.
                event_prediction["predict"] = event_prediction[f"predict_{label_type}"]
                event_prediction["answer"] = event_prediction[f"answer_{label_type}"]

                fig, ax = Warning_Time_Plotter.warning_map(
                    trace_info=event_prediction,
                    eventmeta=event,
                    label_type=label_type,
                    intensity=intensity,
                    EQ_ID=EQ_ID,
                    sec=mask_after_sec,
                    label_threshold=label_threshold,
                )

                fig.savefig(f"{output_path}/model{model_num}_{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec warning map.png",
                            dpi=600)
                fig, ax = Warning_Time_Plotter.correct_warning_with_epidist(
                    event_prediction=event_prediction,
                    label_threshold=label_threshold,
                    label_type=label_type,
                    mask_after_sec=mask_after_sec,
                )
                fig.savefig(f"{output_path}/model{model_num}_{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec epidist vs time.png",
                            dpi=600)
                fig, ax = Warning_Time_Plotter.warning_time_hist(
                    event_prediction,
                    catalog,
                    EQ_ID=EQ_ID,
                    mask_after_sec=mask_after_sec,
                    warning_mag_threshold=4,
                    label_threshold=label_threshold,
                    label_type=label_type,
                    bins=14,
                )
                fig.savefig(
                    f"{output_path}/model{model_num}_{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec warning stations hist.png",
                    dpi=600,
                    bbox_inches="tight",
                )
            except Exception as e:
                print(f"Model {model_num}, sec {mask_after_sec}, EQ_ID {EQ_ID} failed: {e}")
                continue
