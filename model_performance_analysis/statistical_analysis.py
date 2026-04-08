import numpy as np
import pandas as pd
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from analysis import Precision_Recall_Factory

model_nums_msfe_iv_pa = [31, 32, 33]
model_nums_msfe_vminor_pa = [34, 35, 36]
model_nums_msfe_iv_pv = [37, 38, 39]
model_nums_msfe_vminor_pv = [40, 41, 42]
model_nums_msfe_iv_pd = [43, 44, 45]
model_nums_msfe_vminor_pd = [46, 47, 48]
model_nums_msfe_iv_cvaa = [49, 50, 51, 52, 53]
model_nums_msfe_vminor_cvaa = [54, 55, 56]
model_nums_msfe_iv_cvav = [57, 58, 59]
model_nums_msfe_vminor_cvav = [60, 61, 62]
model_nums_msfe_iv_cvad = [63, 64, 65]
model_nums_msfe_vminor_cvad = [66, 67, 68]
model_nums_msfe_iv_cav = [69, 70, 71]
model_nums_msfe_vminor_cav = [72, 73, 74]
model_nums_msfe_iv_ia = [75, 76, 77]
model_nums_msfe_vminor_ia = [78, 79, 80]
model_nums_msfe_iv_iv2 = [81, 82, 83]
model_nums_msfe_vminor_iv2 = [84, 85, 86]
model_nums_msfe_iv_tp = [87, 88, 89]
model_nums_msfe_vminor_tp = [90, 91, 92]

scores_msfe_iv_pa = {}
scores_msfe_vminor_pa = {}
scores_msfe_iv_pv = {}
scores_msfe_vminor_pv = {}
scores_msfe_iv_pd = {}
scores_msfe_vminor_pd = {}
scores_msfe_iv_cvaa = {}
scores_msfe_vminor_cvaa = {}
scores_msfe_iv_cvav = {}
scores_msfe_vminor_cvav = {}
scores_msfe_iv_cvad = {}
scores_msfe_vminor_cvad = {}
scores_msfe_iv_cav = {}
scores_msfe_vminor_cav = {}
scores_msfe_iv_ia = {}
scores_msfe_vminor_ia = {}
scores_msfe_iv_iv2 = {}
scores_msfe_vminor_iv2 = {}
scores_msfe_iv_tp = {}
scores_msfe_vminor_tp = {}

model_groups = [
    model_nums_msfe_iv_pa, model_nums_msfe_vminor_pa, 
    model_nums_msfe_iv_pv, model_nums_msfe_vminor_pv, 
    model_nums_msfe_iv_pd, model_nums_msfe_vminor_pd,
    model_nums_msfe_iv_cvaa, model_nums_msfe_vminor_cvaa,
    model_nums_msfe_iv_cvav, model_nums_msfe_vminor_cvav,
    model_nums_msfe_iv_cvad, model_nums_msfe_vminor_cvad,
    model_nums_msfe_iv_cav, model_nums_msfe_vminor_cav,
    model_nums_msfe_iv_ia, model_nums_msfe_vminor_ia,
    model_nums_msfe_iv_iv2, model_nums_msfe_vminor_iv2,
    model_nums_msfe_iv_tp, model_nums_msfe_vminor_tp
]

score_dicts = [
    scores_msfe_iv_pa, scores_msfe_vminor_pa, 
    scores_msfe_iv_pv, scores_msfe_vminor_pv,
    scores_msfe_iv_pd, scores_msfe_vminor_pd,
    scores_msfe_iv_cvaa, scores_msfe_vminor_cvaa,
    scores_msfe_iv_cvav, scores_msfe_vminor_cvav,
    scores_msfe_iv_cvad, scores_msfe_vminor_cvad,
    scores_msfe_iv_cav, scores_msfe_vminor_cav,
    scores_msfe_iv_ia, scores_msfe_vminor_ia,
    scores_msfe_iv_iv2, scores_msfe_vminor_iv2,
    scores_msfe_iv_tp, scores_msfe_vminor_tp
]

time_after_p_arrival = 13
file_root_path = "../predict_with_a_physical_feature"

for model_nums, score_dict in zip(model_groups, score_dicts):
    r2_scores_pga = []
    r2_scores_pgv = []
    precision_scores_pga = []
    recall_scores_pga = []
    f1_scores_pga = []
    precision_scores_pgv = []
    recall_scores_pgv = []
    f1_scores_pgv = []
    rmse_scores_pga = []
    rmse_scores_pgv = []
    mae_scores_pga = []
    mae_scores_pgv = []
    
    for model_num in model_nums:
        file_path = os.path.join(file_root_path, f"model_{model_num}")
        all_files = os.listdir(file_path)
        csv_files = [f for f in all_files if f.endswith('.csv')]
        target_file = [f for f in csv_files if f" {time_after_p_arrival} " in f][0]

        data = pd.read_csv(os.path.join(file_path, target_file))
        predict_pga = data["predict_pga"].values
        answer_pga = data["answer_pga"].values
        predict_pgv = data["predict_pgv"].values
        answer_pgv = data["answer_pgv"].values
        
        r2_pga = metrics.r2_score(answer_pga, predict_pga)
        r2_pgv = metrics.r2_score(answer_pgv, predict_pgv)

        logic_predict_pga = np.where(predict_pga > np.log10(0.25), 1, 0)
        logic_predict_pgv = np.where(predict_pgv > np.log10(0.057), 1, 0)
        logic_answer_pga = np.where(answer_pga > np.log10(0.25), 1, 0)
        logic_answer_pgv = np.where(answer_pgv > np.log10(0.057), 1, 0)
        
        precision_pga, recall_pga, f1_pga = Precision_Recall_Factory.calculate_precision_recall_f1(logic_answer_pga, logic_predict_pga)
        precision_pgv, recall_pgv, f1_pgv = Precision_Recall_Factory.calculate_precision_recall_f1(logic_answer_pgv, logic_predict_pgv)
        
        mse_pga = metrics.mean_squared_error(answer_pga, predict_pga)
        rmse_pga = np.sqrt(mse_pga)
        mse_pgv = metrics.mean_squared_error(answer_pgv, predict_pgv)
        rmse_pgv = np.sqrt(mse_pgv)
        
        mae_pga = metrics.mean_absolute_error(answer_pga, predict_pga)
        mae_pgv = metrics.mean_absolute_error(answer_pgv, predict_pgv)

        r2_scores_pga.append(r2_pga)
        r2_scores_pgv.append(r2_pgv)
        precision_scores_pga.append(precision_pga)
        recall_scores_pga.append(recall_pga)
        f1_scores_pga.append(f1_pga)
        precision_scores_pgv.append(precision_pgv)
        recall_scores_pgv.append(recall_pgv)
        f1_scores_pgv.append(f1_pgv)
        rmse_scores_pga.append(rmse_pga)
        rmse_scores_pgv.append(rmse_pgv)
        mae_scores_pga.append(mae_pga)
        mae_scores_pgv.append(mae_pgv)

    score_dict["R2 Score PGA"] = r2_scores_pga
    score_dict["Precision PGA"] = precision_scores_pga
    score_dict["Recall PGA"] = recall_scores_pga
    score_dict["F1 Score PGA"] = f1_scores_pga
    score_dict["RMSE PGA"] = rmse_scores_pga
    score_dict["MAE PGA"] = mae_scores_pga
    score_dict["R2 Score PGV"] = r2_scores_pgv
    score_dict["Precision PGV"] = precision_scores_pgv
    score_dict["Recall PGV"] = recall_scores_pgv
    score_dict["F1 Score PGV"] = f1_scores_pgv
    score_dict["RMSE PGV"] = rmse_scores_pgv
    score_dict["MAE PGV"] = mae_scores_pgv

model_group_names = [
    "MSFE_IV_pa", "MSFE_Vminor_pa",
    "MSFE_IV_pv", "MSFE_Vminor_pv",
    "MSFE_IV_pd", "MSFE_Vminor_pd",
    "MSFE_IV_cvaa", "MSFE_Vminor_cvaa",
    "MSFE_IV_cvav", "MSFE_Vminor_cvav",
    "MSFE_IV_cvad", "MSFE_Vminor_cvad",
    "MSFE_IV_CAV", "MSFE_Vminor_CAV",
    "MSFE_IV_Ia", "MSFE_Vminor_Ia",
    "MSFE_IV_IV2", "MSFE_Vminor_IV2",
    "MSFE_IV_TP", "MSFE_Vminor_TP"
]

def plot_performance_scores(score_containers, group_names, metric_types, palette, title, filename, use_max=False, set_ylim=False, reference_metric=None, minimize=False):
    """Plot grouped bar chart of model performance metrics and save to file."""
    x_positions = np.arange(len(group_names))
    bar_width = 0.25
    
    fig, ax = plt.subplots(figsize=(18, 6), dpi=450)
    
    for metric_index, (metric_type, bar_color) in enumerate(zip(metric_types, palette)):
        bar_heights = []
        for score_entry in score_containers:
            if use_max and reference_metric:
                best_index = np.argmin(score_entry[reference_metric]) if minimize else np.argmax(score_entry[reference_metric])
                height_value = score_entry[metric_type][best_index]
            elif use_max:
                height_value = np.max(score_entry[metric_type]) if len(score_entry[metric_type]) > 0 else 0
            else:
                height_value = np.mean(score_entry[metric_type])
            bar_heights.append(height_value)
        bars = ax.bar(x_positions + metric_index * bar_width, bar_heights, label=metric_type, color=bar_color, width=bar_width)
        ax.bar_label(bars, fmt='%.3f', padding=3)
    
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    if set_ylim:
        ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    # fig.savefig(filename)
    # plt.close()


def get_best_model_numbers_by_metric(model_groups, score_containers, reference_metric, minimize=True):
    """Return best model number per group based on a reference metric."""
    best_model_numbers = []
    best_metric_values = []

    for model_nums, score_entry in zip(model_groups, score_containers):
        metric_values = score_entry.get(reference_metric, [])
        if len(metric_values) == 0:
            best_model_numbers.append(None)
            best_metric_values.append(np.nan)
            continue

        best_index = int(np.argmin(metric_values) if minimize else np.argmax(metric_values))
        best_model_numbers.append(model_nums[best_index])
        best_metric_values.append(metric_values[best_index])

    return best_model_numbers, best_metric_values

pga_metric_types = ["R2 Score PGA", "RMSE PGA", "MAE PGA"]
pgv_metric_types = ["R2 Score PGV", "RMSE PGV", "MAE PGV"]
pga_precision_recall_types = ["Precision PGA", "Recall PGA", "F1 Score PGA"]
pgv_precision_recall_types = ["Precision PGV", "Recall PGV", "F1 Score PGV"]
metric_colors = ["deepskyblue", "lightcoral", "coral"]
precision_recall_colors = ["lightskyblue", "deepskyblue", "dodgerblue"]

best_models_pga, best_rmse_pga = get_best_model_numbers_by_metric(
    model_groups, score_dicts, "RMSE PGA", minimize=True
)
best_models_pgv, best_rmse_pgv = get_best_model_numbers_by_metric(
    model_groups, score_dicts, "RMSE PGV", minimize=True
)

print("Best model numbers by group (based on RMSE PGA):")
for group_name, model_num, rmse in zip(model_group_names, best_models_pga, best_rmse_pga):
    print(f"{group_name}: model_{model_num} (RMSE PGA={rmse:.4f})")

print("\nBest model numbers by group (based on RMSE PGV):")
for group_name, model_num, rmse in zip(model_group_names, best_models_pgv, best_rmse_pgv):
    print(f"{group_name}: model_{model_num} (RMSE PGV={rmse:.4f})")

plot_performance_scores(score_dicts, model_group_names, pga_metric_types, metric_colors,
                        "PGA - Average Performance Scores (R2, RMSE, MAE)",
                        "PGA_average_performance_scores.png", use_max=False)
plot_performance_scores(score_dicts, model_group_names, pga_metric_types, metric_colors,
                        "PGA - Maximum Performance Scores (based on RMSE PGA)",
                        "PGA_maximum_performance_scores.png", use_max=True, 
                        reference_metric="RMSE PGA", minimize=True)

plot_performance_scores(score_dicts, model_group_names, pgv_metric_types, metric_colors,
                        "PGV - Average Performance Scores (R2, RMSE, MAE)",
                        "PGV_average_performance_scores.png", use_max=False)
plot_performance_scores(score_dicts, model_group_names, pgv_metric_types, metric_colors,
                        "PGV - Maximum Performance Scores (based on RMSE PGV)",
                        "PGV_maximum_performance_scores.png", use_max=True,
                        reference_metric="RMSE PGV", minimize=True)

plot_performance_scores(score_dicts, model_group_names, pga_precision_recall_types, precision_recall_colors,
                        "PGA - Average Precision/Recall/F1 Scores",
                        "PGA_average_precision_recall_scores.png", use_max=False, set_ylim=True)
plot_performance_scores(score_dicts, model_group_names, pga_precision_recall_types, precision_recall_colors,
                        "PGA - Maximum Precision/Recall/F1 Scores (based on RMSE PGA)",
                        "PGA_maximum_precision_recall_scores.png", use_max=True, set_ylim=True,
                        reference_metric="RMSE PGA", minimize=True)

plot_performance_scores(score_dicts, model_group_names, pgv_precision_recall_types, precision_recall_colors,
                        "PGV - Average Precision/Recall/F1 Scores",
                        "PGV_average_precision_recall_scores.png", use_max=False, set_ylim=True)
plot_performance_scores(score_dicts, model_group_names, pgv_precision_recall_types, precision_recall_colors,
                        "PGV - Maximum Precision/Recall/F1 Scores (based on RMSE PGV)",
                        "PGV_maximum_precision_recall_scores.png", use_max=True, set_ylim=True,
                        reference_metric="RMSE PGV", minimize=True)