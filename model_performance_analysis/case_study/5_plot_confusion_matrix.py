import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


sys.path.append("..")
from analysis import Precision_Recall_Factory

event_lat = 23.88
event_lon = 121.57
magnitude =7.1
answer = pd.read_csv(f"true_answer.csv")

# merge 3 5 7 10 sec to find maximum predicted pga
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

#################
label_threshold_pgv = np.log10(0.057)
predict_label_pgv = np.array(max_prediction[f"max_predict_pgv"])
real_label_pgv = np.array(max_prediction["PGV"])
predict_logic_pgv = np.where(predict_label_pgv > label_threshold_pgv, 1, 0)
real_logic_pgv = np.where(real_label_pgv > label_threshold_pgv, 1, 0)
precision_pgv, recall_pgv, f1_pgv = Precision_Recall_Factory.calculate_precision_recall_f1(real_logic_pgv, predict_logic_pgv)
#################
label_threshold_pga = np.log10(0.25)
predict_label_pga = np.array(max_prediction[f"max_predict_pga"])
real_label_pga = np.array(max_prediction["PGA"])
predict_logic_pga = np.where(predict_label_pga > label_threshold_pga, 1, 0)
real_logic_pga = np.where(real_label_pga > label_threshold_pga, 1, 0)
precision_pga, recall_pga, f1_pga = Precision_Recall_Factory.calculate_precision_recall_f1(real_logic_pga, predict_logic_pga)
#################

intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
max_prediction["predicted_intensity_pga"] = max_prediction["max_predict_pga"].apply(
    Precision_Recall_Factory.pga_to_intensity
)
max_prediction["predicted_intensity_pgv"] = max_prediction["max_predict_pgv"].apply(
    Precision_Recall_Factory.pgv_to_intensity
)
max_prediction["answer_intensity_pga"] = max_prediction["PGA"].apply(Precision_Recall_Factory.pga_to_intensity)
max_prediction["answer_intensity_pgv"] = max_prediction["PGV"].apply(Precision_Recall_Factory.pgv_to_intensity)

intensity_confusion_matrix_pga = confusion_matrix(
    max_prediction["predicted_intensity_pga"],
    max_prediction["answer_intensity_pga"],
    labels=intensity,
)
intensity_confusion_matrix_pgv = confusion_matrix(
    max_prediction["predicted_intensity_pgv"],
    max_prediction["answer_intensity_pgv"],
    labels=intensity,
)

fig_pga, ax_pga = Precision_Recall_Factory.plot_intensity_confusion_matrix(
    intensity_confusion_matrix_pga,
    label="PGA",
    title=f"PGA Confusion Matrix\nPrecision: {precision_pga:.3f}, Recall: {recall_pga:.3f}, F1-score: {f1_pga:.3f}"
)
fig_pga.savefig("PGA_confusion_matrix.png", dpi=300)
fig_pgv, ax_pgv = Precision_Recall_Factory.plot_intensity_confusion_matrix(
    intensity_confusion_matrix_pgv,
    label="PGV",
    title=f"PGV Confusion Matrix\nPrecision: {precision_pgv:.3f}, Recall: {recall_pgv:.3f}, F1-score: {f1_pgv:.3f}"
)
fig_pgv.savefig("PGV_confusion_matrix.png", dpi=300)
