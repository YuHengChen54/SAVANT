import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from analysis import Precision_Recall_Factory

model_num = 76
path = f"../predict_pga_pgv/model_{model_num}"
output_path = f"{path}/model_{model_num}_analysis"
if not os.path.isdir(output_path):
    os.mkdir(output_path)

label = "pga"

if label == "pga":
    unit = "m/s^2"
elif label == "pgv":
    unit = "m/s"

if label == "pgv":
    #形成 warning threshold array 其中包含對應的4~5級標準
    target_value = np.log10(0.15)

    # 生成一個包含目標值的數組
    score_curve_threshold = np.linspace(np.log10(0.007), np.log10(0.3), 100)

    # 檢查最接近的值
    closest_value = min(score_curve_threshold, key=lambda x: abs(x - target_value))

    # 調整num參數以確保包含目標值
    if closest_value != target_value:
        num_adjusted = 100 + int(np.ceil(abs(target_value - closest_value) / np.diff(score_curve_threshold[:2])))
        score_curve_threshold = np.linspace(np.log10(0.002), np.log10(0.1), num_adjusted)
elif label == "pga":
    #形成 warning threshold array 其中包含對應的4~5級標準
    target_value = np.log10(0.8)

    # 生成一個包含目標值的數組
    score_curve_threshold = np.linspace(np.log10(0.025), np.log10(1.4), 100)

    # 檢查最接近的值
    closest_value = min(score_curve_threshold, key=lambda x: abs(x - target_value))

    # 調整num參數以確保包含目標值
    if closest_value != target_value:
        num_adjusted = 100 + int(np.ceil(abs(target_value - closest_value) / np.diff(score_curve_threshold[:2])))
        score_curve_threshold = np.linspace(np.log10(0.008), np.log10(0.5), num_adjusted)


intensity_score_dict = {"second": [], "intensity_score": []}
# Comprehensive curve
# compre_curve_fig, compre_curve_ax = plt.subplots(figsize=(5,5), dpi=350)
# Three score type curve
f1_curve_fig, f1_curve_ax = plt.subplots(figsize=(5,5), dpi=350)
precision_curve_fig, precision_curve_ax = plt.subplots(figsize=(5,5), dpi=350)
recall_curve_fig, recall_curve_ax = plt.subplots(figsize=(5,5), dpi=350)

for mask_after_sec in [3, 5, 7, 10]:
    data = pd.read_csv(f"{path}/{mask_after_sec} sec model{model_num} with all info_vel.csv")

    predict_label = data[f"predict_{label}"]
    real_label = data[f"answer_{label}"]

    if label == "pga":
        data["predicted_intensity"] = predict_label.apply(Precision_Recall_Factory.pga_to_intensity)
        data["answer_intensity"] = real_label.apply(Precision_Recall_Factory.pga_to_intensity)
    elif label == "pgv":
        data["predicted_intensity"] = predict_label.apply(Precision_Recall_Factory.pgv_to_intensity)
        data["answer_intensity"] = real_label.apply(Precision_Recall_Factory.pgv_to_intensity)

    # calculate intensity score
    intensity_label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    label_to_index = {label: i for i, label in enumerate(intensity_label)}
    # 將轉換後的 predicted / answer intensity label 映射成 index
    pred_index = data["predicted_intensity"].map(label_to_index)
    ans_index = data["answer_intensity"].map(label_to_index)

    # 完全正確
    strict_correct = (pred_index == ans_index).sum()

    # ±1 級內正確
    loose_correct = (abs(pred_index - ans_index) <= 1).sum()

    # 計算分數
    strict_score = strict_correct / len(data)
    loose_score = loose_correct / len(data)
    
    intensity_score_dict = {
    "second": [],
    "strict_score": [],
    "loose_score": [],
    }
    intensity_score_dict["second"].append(mask_after_sec)
    intensity_score_dict["strict_score"].append(strict_score)
    intensity_score_dict["loose_score"].append(loose_score)

    intensity_table = pd.DataFrame(intensity_score_dict)


    # plot intensity score confusion matrix
    intensity_confusion_matrix = confusion_matrix(
        data["predicted_intensity"], data["answer_intensity"], labels=intensity_label
    )
    fig,ax=Precision_Recall_Factory.plot_intensity_confusion_matrix(intensity_confusion_matrix, label, strict_score,loose_score,mask_after_sec,output_path=f"../predict_pga_pgv/model_{model_num}/model_{model_num}_analysis")

    performance_score = {
        f"{label}_threshold ({unit})": [],
        "confusion matrix": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "F1": [],
    }
    for label_threshold in score_curve_threshold:
        predict_logic = np.where(predict_label > label_threshold, 1, 0)
        real_logic = np.where(real_label > label_threshold, 1, 0)
        matrix = confusion_matrix(real_logic, predict_logic, labels=[1, 0])

        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)

        # 使用 analysis.py 的 calculate_precision_recall_f1
        precision, recall, F1_score = Precision_Recall_Factory.calculate_precision_recall_f1(real_logic, predict_logic)

        performance_score[f"{label}_threshold ({unit})"].append(np.round((10**label_threshold), 3))
        performance_score["confusion matrix"].append(matrix)
        performance_score["accuracy"].append(accuracy)
        performance_score["precision"].append(precision)
        performance_score["recall"].append(recall)
        performance_score["F1"].append(F1_score)


    # compre_curve_fig, compre_curve_ax = Precision_Recall_Factory.plot_score_curve(
    #     performance_score,
    #     compre_curve_fig,
    #     compre_curve_ax,
    #     "comprehensive",
    #     score_curve_threshold,
    #     mask_after_sec,
    #     output_path=f"../predict/model_{model_num}_analysis",
    # )
    f1_curve_fig, f1_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        f1_curve_fig,
        f1_curve_ax,
        label,
        "F1",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict_pga_pgv/model_{model_num}/model_{model_num}_analysis",
    )
    precision_curve_fig, precision_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        precision_curve_fig,
        precision_curve_ax,
        label,
        "precision",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict_pga_pgv/model_{model_num}/model_{model_num}_analysis",
    )
    recall_curve_fig, recall_curve_ax = Precision_Recall_Factory.plot_score_curve(
        performance_score,
        recall_curve_fig,
        recall_curve_ax,
        label,
        "recall",
        score_curve_threshold,
        mask_after_sec,
        output_path=f"../predict_pga_pgv/model_{model_num}/model_{model_num}_analysis",
    )

    predict_table = pd.DataFrame(performance_score)
    # predict_table.to_csv(
    #     f"{output_path}/{mask_after_sec} sec confusion matrix table.csv",
    #     index=False,
    # )
