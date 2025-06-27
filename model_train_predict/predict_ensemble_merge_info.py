import h5py
import matplotlib.pyplot as plt

# plt.subplots()
import numpy as np
import pandas as pd
import torch
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("..")
from model.CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from data.multiple_sta_dataset import multiple_station_dataset
from model_performance_analysis.analysis import Intensity_Plotter

for mask_sec in [3, 5, 7, 10, 13, 15]:
    mask_after_sec = mask_sec
    label = "pgv"
    data = multiple_station_dataset(
        "../../../TT-SAM-Velocity-PGV/code/data/TSMIP_1999_2019_Vs30_integral.hdf5",
        mode="test",
        mask_waveform_sec=mask_after_sec,
        test_year=2016,
        label_key=label,
        mag_threshold=0,
        input_type="vel",
        data_length_sec=20,
    )
    # ===========predict==============
    device = torch.device("cuda")
    for num in [21, 22, 23]:  
        path = f"../model/model{num}_vel.pt"
        # path = "../model/model19_checkpoints/epoch70_model.pt"
        emb_dim = 150
        mlp_dims = (150, 100, 50, 30, 10)
        CNN_model = CNN(downsample=3, mlp_input=7665).cuda()
        pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
        transformer_model = TransformerEncoder()
        mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
        mdn_model = MDN(input_shape=(mlp_dims[-1],)).cuda()
        full_Model = full_model(
            CNN_model,
            pos_emb_model,
            transformer_model,
            mlp_model,
            mdn_model,
            pga_targets=25,
            data_length=4000,
        ).to(device)
        full_Model.load_state_dict(torch.load(path))
        loader = DataLoader(dataset=data, batch_size=1)

        Mixture_mu = []
        Label = []
        P_picks = []
        EQ_ID = []
        Label_time = []
        Sta_name = []
        Lat = []
        Lon = []
        Elev = []
        for j, sample in tqdm(enumerate(loader)):
            picks = sample["p_picks"].flatten().numpy().tolist()
            label_time = sample[f"{label}_time"].flatten().numpy().tolist()
            lat = sample["target"][:, :, 0].flatten().tolist()
            lon = sample["target"][:, :, 1].flatten().tolist()
            elev = sample["target"][:, :, 2].flatten().tolist()
            P_picks.extend(picks)
            P_picks.extend([np.nan] * (25 - len(picks)))
            Label_time.extend(label_time)
            Label_time.extend([np.nan] * (25 - len(label_time)))
            Lat.extend(lat)
            Lon.extend(lon)
            Elev.extend(elev)

            eq_id = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
            EQ_ID.extend(eq_id)
            EQ_ID.extend([np.nan] * (25 - len(eq_id)))
            weight, sigma, mu = full_Model(sample)

            weight = weight.cpu()
            sigma = sigma.cpu()
            mu = mu.cpu()
            if j == 0:
                Mixture_mu = torch.sum(weight * mu, dim=2).cpu().detach().numpy()
                Label = sample["label"].cpu().detach().numpy()
            else:
                Mixture_mu = np.concatenate(
                    [Mixture_mu, torch.sum(weight * mu, dim=2).cpu().detach().numpy()],
                    axis=1,
                )
                Label = np.concatenate(
                    [Label, sample["label"].cpu().detach().numpy()], axis=1
                )
        Label = Label.flatten()
        Mixture_mu = Mixture_mu.flatten()

        output = {
            "EQ_ID": EQ_ID,
            "p_picks": P_picks,
            f"{label}_time": Label_time,
            "predict": Mixture_mu,
            "answer": Label,
            "latitude": Lat,
            "longitude": Lon,
            "elevation": Elev,
        }
        output_df = pd.DataFrame(output)
        output_df = output_df[output_df["answer"] != 0]
        # output_df.to_csv(
        #     f"../predict/model_{num}_analysis/model {num} {mask_after_sec} sec prediction_vel.csv", index=False
        # )

        # output_df = pd.read_csv(f"../predict/model_3_analysis(velocity)/model 3 {mask_after_sec} sec prediction_vel.csv")

        fig, ax = Intensity_Plotter.plot_true_predicted(
            y_true=output_df["answer"],
            y_pred=output_df["predict"],
            agg="point",
            point_size=12,
            target=label,
            title=f"{mask_after_sec}s True Predict Plot, 2016 data model {num}"
        )

        # 用震度4分割畫不同顏色
        # ax.scatter(
        #     output_df["answer"][output_df["answer"] >= np.log10(0.057)],
        #     output_df["predict"][output_df["answer"] >= np.log10(0.057)],
        #     c="orange",
        #     s=12
        
        # ) 

        # r2_greater4 = metrics.r2_score(output_df["answer"][output_df["answer"] >= np.log10(0.057)], output_df["predict"][output_df["answer"] >= np.log10(0.057)])
        # r2 = metrics.r2_score(output_df["answer"], output_df["predict"])

        # limits = (np.min(output_df["answer"]) - 0.5, np.max(output_df["answer"])-0.5)
        # ax.text(
        #     min(np.min(output_df["answer"]), limits[0]),
        #     max(np.max(output_df["answer"]), limits[1])-0.5,
        #     f"$R^2={r2:.2f}$",
        #     fontweight=1000, 
        #     va="top",
        #     fontsize=15,
        #     color="dodgerblue", 
            
        # )
        # ax.text(
        #     min(np.min(output_df["answer"]), limits[0])+0.7, 
        #     max(np.max(output_df["answer"]), limits[1])-0.5,
        #     f"$R^2={r2_greater4:.2f}$",
        #     fontweight=1000, 
        #     va="top",
        #     fontsize=15,
        #     color="darkorange", 
        # )
        # eq_id = 24784
        # ax.scatter(
        # output_df["answer"][output_df["EQ_ID"] == eq_id],
        # output_df["predict"][output_df["EQ_ID"] == eq_id],
        # c="r",
        # )
        
        # magnitude = data.event_metadata[data.event_metadata["EQ_ID"] == eq_id][
        #     "magnitude"
        # ].values[0]
        # ax.set_title(
        #     f"{mask_after_sec}s True Predict Plot, 2016 data model{num}",
        #     fontsize=20,
        # )

        fig.savefig(f"../predict/model {num} {mask_after_sec} sec_vel.png")

    # # ===========merge info==============
    # num = 9
    # Afile_path = "../data"
    # output_path = f"../predict/model_{num}_analysis"
    # catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
    # traces_info = pd.read_csv(f"{Afile_path}/1999_2019_final_traces_Vs30.csv")
    # ensemble_predict = pd.read_csv(
    #     f"{output_path}/model {num} {mask_after_sec} sec prediction_vel.csv"
    # )
    # trace_merge_catalog = pd.merge(
    #     traces_info,
    #     catalog[
    #         [
    #             "EQ_ID",
    #             "lat",
    #             "lat_minute",
    #             "lon",
    #             "lon_minute",
    #             "depth",
    #             "magnitude",
    #             "nsta",
    #             "nearest_sta_dist (km)",
    #         ]
    #     ],
    #     on="EQ_ID",
    #     how="left",
    # )
    # trace_merge_catalog["event_lat"] = (
    #     trace_merge_catalog["lat"] + trace_merge_catalog["lat_minute"] / 60
    # )

    # trace_merge_catalog["event_lon"] = (
    #     trace_merge_catalog["lon"] + trace_merge_catalog["lon_minute"] / 60
    # )
    # trace_merge_catalog.drop(
    #     ["lat", "lat_minute", "lon", "lon_minute"], axis=1, inplace=True
    # )
    # trace_merge_catalog.rename(columns={"elevation (m)": "elevation"}, inplace=True)


    # data_path = "../data/TSMIP_1999_2019_Vs30_integral.hdf5"
    # dataset = h5py.File(data_path, "r")
    # for eq_id in ensemble_predict["EQ_ID"].unique():
    #     eq_id = int(eq_id)
    #     station_name = dataset["data"][str(eq_id)]["station_name"][:].tolist()

    #     ensemble_predict.loc[
    #         ensemble_predict.query(f"EQ_ID=={eq_id}").index, "station_name"
    #     ] = station_name

    # ensemble_predict["station_name"] = ensemble_predict["station_name"].str.decode("utf-8")


    # prediction_with_info = pd.merge(
    #     ensemble_predict,
    #     trace_merge_catalog.drop(
    #         [
    #             "latitude",
    #             "longitude",
    #             "elevation",
    #         ],
    #         axis=1,
    #     ),
    #     on=["EQ_ID", "station_name"],
    #     how="left",
    #     suffixes=["_window", "_file"],
    # )
    # prediction_with_info.to_csv(
    #     f"{output_path}/{mask_after_sec} sec model{num} with all info_vel.csv", index=False
    # )
