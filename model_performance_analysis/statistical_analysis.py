import numpy as np
import pandas as pd
import os
import re
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from analysis import Precision_Recall_Factory


model_list_MFE_III_lr5e5 = [96, 97, 98, 99, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
model_list_MFE_III_lr25e5 = [91, 92, 93, 94, 95, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
model_list_MFE_III_lr5e6 = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
model_list_MFE_IV_lr5e5 = [146, 147, 148, 149, 150]
model_list_MFE_IV_lr25e5 = [226, 227, 228, 229, 230]
model_list_MFE_IV_lr5e6 = []
model_list_MFE_Vminorlr5e5 = [156, 157, 158, 159, 160, 161, 162, 163, 164]
model_list_MFE_Vminorlr25e5 = [231, 232, 233, 234, 235]
model_list_MFE_Vminorlr5e6 = []
model_list_MFE_Vmajorlr5e5 = [165, 166, 167, 168, 169, 170]
model_list_MFE_Vmajor_lr25e5 = [236, 237, 238, 239, 240]
model_list_MFE_Vmajor_lr5e6 = []
model_list_MSFE_III_lr5e5 = [171, 172, 173, 174, 175]
model_list_MSFE_III_lr25e5 = []
model_list_MSFE_III_lr5e6 = [191, 192, 193, 194, 195]
model_list_MSFE_IV_lr5e5 = [176, 177, 178, 179, 180]
model_list_MSFE_IV_lr25e5 = [211, 212, 213, 214, 215]
model_list_MSFE_IV_lr5e6 = [196, 197, 198, 199, 200]
model_list_MSFE_Vminor_lr5e5 = [181, 182, 183, 184, 185]
model_list_MSFE_Vminor_lr25e5 = [216, 217, 218, 219, 220]
model_list_MSFE_Vminor_lr5e6 = [201, 202, 203, 204, 205]
model_list_MSFE_Vmajor_lr5e5 = [186, 187, 188, 189, 190]
model_list_MSFE_Vmajor_lr25e5 = [221, 222, 223, 224, 225]
model_list_MSFE_Vmajor_lr5e6 = [206, 207, 208, 209, 210]

MFE_III_lr5e5 = {}
MFE_III_lr25e5 = {}
MFE_III_lr5e6 = {}
MFE_IV_lr5e5 = {}
MFE_IV_lr25e5 = {}
MFE_IV_lr5e6 = {}
MFE_Vminor_lr5e5 = {}
MFE_Vminor_lr25e5 = {}
MFE_Vminor_lr5e6 = {}
MFE_Vmajor_lr5e5 = {}
MFE_Vmajor_lr25e5 = {}
MFE_Vmajor_lr5e6 = {}
MSFE_III_lr5e5 = {}
MSFE_III_lr25e5 = {}
MSFE_III_lr5e6 = {}
MSFE_IV_lr5e5 = {}
MSFE_IV_lr25e5 = {}
MSFE_IV_lr5e6 = {}
MSFE_Vminor_lr5e5 = {}
MSFE_Vminor_lr25e5 = {}
MSFE_Vminor_lr5e6 = {}
MSFE_Vmajor_lr5e5 = {}
MSFE_Vmajor_lr25e5 = {}
MSFE_Vmajor_lr5e6 = {}

model_type_list = [model_list_MFE_III_lr5e5, model_list_MFE_III_lr25e5, model_list_MFE_III_lr5e6,
                model_list_MFE_IV_lr5e5, model_list_MFE_IV_lr25e5, model_list_MFE_IV_lr5e6,
                model_list_MFE_Vminorlr5e5, model_list_MFE_Vminorlr25e5, model_list_MFE_Vminorlr5e6,
                model_list_MFE_Vmajorlr5e5, model_list_MFE_Vmajor_lr25e5, model_list_MFE_Vmajor_lr5e6,
                model_list_MSFE_III_lr5e5, model_list_MSFE_III_lr25e5, model_list_MSFE_III_lr5e6,
                model_list_MSFE_IV_lr5e5, model_list_MSFE_IV_lr25e5, model_list_MSFE_IV_lr5e6,
                model_list_MSFE_Vminor_lr5e5, model_list_MSFE_Vminor_lr25e5, model_list_MSFE_Vminor_lr5e6,
                model_list_MSFE_Vmajor_lr5e5, model_list_MSFE_Vmajor_lr25e5, model_list_MSFE_Vmajor_lr5e6]

final_score_list = [MFE_III_lr5e5, MFE_III_lr25e5, MFE_III_lr5e6,
                MFE_IV_lr5e5, MFE_IV_lr25e5, MFE_IV_lr5e6,
                MFE_Vminor_lr5e5, MFE_Vminor_lr25e5, MFE_Vminor_lr5e6,
                MFE_Vmajor_lr5e5, MFE_Vmajor_lr25e5, MFE_Vmajor_lr5e6,
                MSFE_III_lr5e5, MSFE_III_lr25e5, MSFE_III_lr5e6,
                MSFE_IV_lr5e5, MSFE_IV_lr25e5, MSFE_IV_lr5e6,
                MSFE_Vminor_lr5e5, MSFE_Vminor_lr25e5, MSFE_Vminor_lr5e6,
                MSFE_Vmajor_lr5e5, MSFE_Vmajor_lr25e5, MSFE_Vmajor_lr5e6]

time_after_p_arrival = 13
file_root_path = "../predict_pga_pgv"
for model_type, final_score in zip(model_type_list, final_score_list):
    r2_score_pga_list = []
    r2_score_pgv_list = []
    precision_pga_list = []
    recall_pga_list = []
    f1_score_pga_list = []
    precision_pgv_list = []
    recall_pgv_list = []
    f1_score_pgv_list = []
    for model_num in model_type:
        file_path = os.path.join(file_root_path, f"model_{model_num}")
        file_list = os.listdir(file_path)
        csv_files = [f for f in file_list if f.endswith('.csv')]
        # 篩選出檔名包含 " {time_after_p_arrival} " 的檔案
        file = [f for f in csv_files if f" {time_after_p_arrival} " in f][0]


        data = pd.read_csv(os.path.join(file_path, file))
        predict_pga = data["predict_pga"].values
        answer_pga = data["answer_pga"].values
        predict_pgv = data["predict_pgv"].values
        answer_pgv = data["answer_pgv"].values
        # Calculate r2 score for PGA
        r2_score_pga = metrics.r2_score(answer_pga, predict_pga)
        # Calculate r2 score for PGV
        r2_score_pgv = metrics.r2_score(answer_pgv, predict_pgv)

        predict_logic_pga = np.where(predict_pga > np.log10(0.25), 1, 0)
        predict_logic_pgv = np.where(predict_pgv > np.log10(0.057), 1, 0)
        answer_logic_pga = np.where(answer_pga > np.log10(0.25), 1, 0)
        answer_logic_pgv = np.where(answer_pgv > np.log10(0.057), 1, 0)
        precision_pga, recall_pga, f1_score_pga = Precision_Recall_Factory.calculate_precision_recall_f1(answer_logic_pga, predict_logic_pga)
        precision_pgv, recall_pgv, f1_score_pgv = Precision_Recall_Factory.calculate_precision_recall_f1(answer_logic_pgv, predict_logic_pgv)

        r2_score_pga_list.append(r2_score_pga)
        r2_score_pgv_list.append(r2_score_pgv)
        precision_pga_list.append(precision_pga)
        recall_pga_list.append(recall_pga)
        f1_score_pga_list.append(f1_score_pga)
        precision_pgv_list.append(precision_pgv)
        recall_pgv_list.append(recall_pgv)
        f1_score_pgv_list.append(f1_score_pgv)

    final_score["R2 Score PGA"] = r2_score_pga_list
    final_score["Precision PGA"] = precision_pga_list
    final_score["Recall PGA"] = recall_pga_list
    final_score["F1 Score PGA"] = f1_score_pga_list
    final_score["R2 Score PGV"] = r2_score_pgv_list
    final_score["Precision PGV"] = precision_pgv_list
    final_score["Recall PGV"] = recall_pgv_list
    final_score["F1 Score PGV"] = f1_score_pgv_list

# # # =========== find highest r2 ==============
# highest_avg_r2_pga = 0.0
# highest_avg_r2_pgv = 0.0
# highest_avg_r2_pga_model = None
# highest_avg_r2_pgv_model = None
# highest_r2_pga = 0.0
# highest_r2_pgv = 0.0
# highest_r2_pga_model = None
# highest_r2_pgv_model = None

final_score_list_name = ["MFE_III_lr5e5", "MFE_III_lr25e5", "MFE_III_lr5e6",
                "MFE_IV_lr5e5", "MFE_IV_lr25e5", "MFE_IV_lr5e6",
                "MFE_Vminor_lr5e5", "MFE_Vminor_lr25e5", "MFE_Vminor_lr5e6",
                "MFE_Vmajor_lr5e5", "MFE_Vmajor_lr25e5", "MFE_Vmajor_lr5e6",
                "MSFE_III_lr5e5", "MSFE_III_lr25e5", "MSFE_III_lr5e6",
                "MSFE_IV_lr5e5", "MSFE_IV_lr25e5", "MSFE_IV_lr5e6",
                "MSFE_Vminor_lr5e5", "MSFE_Vminor_lr25e5", "MSFE_Vminor_lr5e6",
                "MSFE_Vmajor_lr5e5", "MSFE_Vmajor_lr25e5", "MSFE_Vmajor_lr5e6"]

# for final_score, final_score_name in zip(final_score_list[0:12], final_score_list_name[0:12]):
#     avg_r2_pga = np.mean(final_score["R2 Score PGA"])
#     avg_r2_pgv = np.mean(final_score["R2 Score PGV"])
#     max_r2_pga = np.max(final_score["R2 Score PGA"]) if len(final_score["R2 Score PGA"]) > 0 else 0
#     max_r2_pgv = np.max(final_score["R2 Score PGV"]) if len(final_score["R2 Score PGV"]) > 0 else 0
#     if avg_r2_pga > highest_avg_r2_pga:
#         highest_avg_r2_pga = avg_r2_pga
#         highest_avg_r2_pga_model = final_score_name
#     if avg_r2_pgv > highest_avg_r2_pgv:
#         highest_avg_r2_pgv = avg_r2_pgv
#         highest_avg_r2_pgv_model = final_score_name
#     if max_r2_pga > highest_r2_pga:
#         highest_r2_pga = max_r2_pga
#         highest_r2_pga_model = final_score_name
#     if max_r2_pgv > highest_r2_pgv:
#         highest_r2_pgv = max_r2_pgv
#         highest_r2_pgv_model = final_score_name

#%% =========== plot ===========
plot_list = final_score_list[::3][4:]
plot_list_name = final_score_list_name[::3][4:]

# plot_list = final_score_list[::3][0:4]
# plot_list_name = final_score_list_name[::3][0:4]

pga_score_type = ["R2 Score PGA", "Precision PGA", "Recall PGA"]
pgv_score_type = ["R2 Score PGV", "Precision PGV", "Recall PGV"]
colors = ["lightskyblue", "deepskyblue", "dodgerblue"]

x = np.arange(len(plot_list_name))
bar_width = 0.2
fig, ax = plt.subplots(figsize=(9, 5), dpi=450)
for idx, (score_type, color) in enumerate(zip(pgv_score_type, colors)):
    avg_r2_scores = []
    for final_score, final_score_name in zip(plot_list, plot_list_name):
        avg_r2_score = np.mean(final_score[score_type])
        avg_r2_scores.append(avg_r2_score)
    ax.bar(x + idx * bar_width, avg_r2_scores, label=score_type, color=color, width=bar_width)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(plot_list_name)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model Type and Learning Rate")
    ax.set_ylabel("Score")
    ax.set_title("Average Performance Scores")
    ax.legend()


fig, ax1 = plt.subplots(figsize=(9, 5), dpi=450)
for idx, (score_type, color) in enumerate(zip(pgv_score_type, colors)):
    max_r2_scores = []
    for final_score, final_score_name in zip(plot_list, plot_list_name):
        max_r2_score = np.max(final_score[score_type]) if len(final_score[score_type]) > 0 else 0
        max_r2_scores.append(max_r2_score)
    ax1.bar(x + idx * bar_width, max_r2_scores, label=score_type, color=color, width=bar_width)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(plot_list_name)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Model Type and Learning Rate")
    ax1.set_ylabel("Score")
    ax1.set_title("Maximum Performance Scores")
    ax1.legend()