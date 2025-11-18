import sys
import pandas as pd

sys.path.append("..")
from analysis import Intensity_Plotter

mask_sec = 13
event_lat = 23.88
event_lon = 121.57
magnitude =7.1
answer = pd.read_csv(f"true_answer.csv")

# merge 5 7 10 sec to find maximum predicted pga
prediction_5 = pd.read_csv(f"prediction/5_sec_prediction.csv")
prediction_7 = pd.read_csv(f"prediction/7_sec_prediction.csv")
prediction_10 = pd.read_csv(f"prediction/10_sec_prediction.csv")
prediction_13 = pd.read_csv(f"prediction/13_sec_prediction.csv")

max_prediction = pd.concat(
    [
        prediction_5,
        prediction_7["predict_pga"],
        prediction_10["predict_pga"],
        prediction_13["predict_pga"],
        prediction_7["predict_pgv"],
        prediction_10["predict_pgv"],
        prediction_13["predict_pgv"],
    ],
    axis=1,
)

max_prediction.columns = [
    "5_predict_pga",
    "5_predict_pgv",
    "station_name",
    "latitude",
    "longitude",
    "elevation",
    "7_predict_pga",
    "10_predict_pga",
    "13_predict_pga",
    "7_predict_pgv",
    "10_predict_pgv",
    "13_predict_pgv",
]
max_prediction["max_predict_pga"] = max_prediction.apply(
    lambda row: max(
        row["5_predict_pga"], row["7_predict_pga"], row["10_predict_pga"], row["13_predict_pga"]
    ),
    axis=1,
)
max_prediction["max_predict_pgv"] = max_prediction.apply(
    lambda row: max(
        row["5_predict_pgv"], row["7_predict_pgv"], row["10_predict_pgv"], row["13_predict_pgv"]
    ),
    axis=1,
)

max_prediction = pd.merge(
    answer, max_prediction, how="left", left_on="location_code", right_on="station_name"
)
max_prediction.dropna(inplace=True)

eventmeta = pd.DataFrame(
    {"longitude": [event_lon], "latitude": [event_lat], "magnitude": [magnitude]}
)

fig, ax = Intensity_Plotter.plot_intensity_map(
    trace_info=max_prediction,
    eventmeta=eventmeta,
    label_type="pga",
    true_label=max_prediction["PGA"],
    pred_label=max_prediction[f"{mask_sec}_predict_pga"],
    sec=mask_sec,
    min_epdis=10.87177078,  # 0.1087度轉成km
    EQ_ID=None,
    grid_method="linear",
    pad=100,
    title=f"{mask_sec} sec intensity Map",
)
fig.savefig(f"2024_0403_pga_{mask_sec}_sec.png",dpi=300)

fig, ax_1 = Intensity_Plotter.plot_intensity_map(
    trace_info=max_prediction,
    eventmeta=eventmeta,
    label_type="pgv",
    true_label=max_prediction["PGV"],
    pred_label=max_prediction[f"{mask_sec}_predict_pgv"],
    sec=mask_sec,
    min_epdis=10.87177078,  # 0.1087度轉成km
    EQ_ID=None,
    grid_method="linear",
    pad=100,
    title=f"{mask_sec} sec intensity Map",
)
fig.savefig(f"2024_0403_pgv_{mask_sec}_sec.png",dpi=300)
