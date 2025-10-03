import os
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
from model.CNN_Transformer_Mixtureoutput import (
    CNN,
    MDN_PGA,
    MDN_PGV,
    MLP_output_pga,
    MLP_output_pgv,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from data.multiple_sta_dataset import multiple_station_dataset
from model_performance_analysis.analysis import Intensity_Plotter

for mask_sec in [3, 5, 7, 10, 13, 15]:
    mask_after_sec = mask_sec
    # label = "pga"
    # dual-target: no single label variable
    data = multiple_station_dataset(
        "../data/TSMIP_1999_2019_Vs30_integral.hdf5",
        mode="test",
        mask_waveform_sec=mask_after_sec,
        test_year=2016,
        # label_key=label,
        # use default label_keys=["pga","pgv"]
        mag_threshold=0,
        input_type="acc",
        data_length_sec=20,
    )
    # ===========predict==============
    device = torch.device("cuda")
    # for num in [65]:
    for num in range(149, 158):  
        path = f"../model_pga_pgv/model{num}_pga_pgv.pt"
        emb_dim = 150
        mlp_dims = (150, 100, 50, 30, 10)
        CNN_model = CNN(downsample=3, mlp_input=7665).cuda()
        pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
        transformer_model = TransformerEncoder()
        mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
        mlp_output_pga = MLP_output_pga(input_shape=(emb_dim,), dims=mlp_dims).cuda()
        mlp_output_pgv = MLP_output_pgv(input_shape=(emb_dim,), dims=mlp_dims).cuda()
        mdn_pga_model = MDN_PGA(input_shape=(mlp_dims[-1],)).cuda()
        mdn_pgv_model = MDN_PGV(input_shape=(mlp_dims[-1],)).cuda()
        full_Model = full_model(
            CNN_model,
            pos_emb_model,
            transformer_model,
            mlp_model,
            mlp_output_pga,
            mlp_output_pgv,
            mdn_pga_model,
            mdn_pgv_model,
            pga_targets=25,
            data_length=4000,
        ).to(device)
        full_Model.load_state_dict(torch.load(path))
        loader = DataLoader(dataset=data, batch_size=1)

        Mixture_mu_pga = []
        Mixture_mu_pgv = []
        Label_pga = []
        Label_pgv = []
        P_picks = []
        EQ_ID = []
        # we can record both label times if needed
        Pga_time = []
        Pgv_time = []
        Sta_name = []
        Lat = []
        Lon = []
        Elev = []
        for j, sample in tqdm(enumerate(loader)):
            # metadata
            picks = sample["p_picks"].flatten().numpy().tolist()
            P_picks.extend(picks)
            P_picks.extend([np.nan] * (25 - len(picks)))
            lat = sample["target"][:, :, 0].flatten().tolist()
            lon = sample["target"][:, :, 1].flatten().tolist()
            elev = sample["target"][:, :, 2].flatten().tolist()
            Lat.extend(lat); Lon.extend(lon); Elev.extend(elev)
            eq_id = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
            EQ_ID.extend(eq_id); EQ_ID.extend([np.nan] * (25 - len(eq_id)))

            # model outputs: dual heads
            w_pga, s_pga, m_pga, w_pgv, s_pgv, m_pgv = full_Model(sample)
            w_pga, s_pga, m_pga = w_pga.cpu(), s_pga.cpu(), m_pga.cpu()
            w_pgv, s_pgv, m_pgv = w_pgv.cpu(), s_pgv.cpu(), m_pgv.cpu()
            # mixture means
            pga_pred = torch.sum(w_pga * m_pga, dim=2).detach().numpy()
            pgv_pred = torch.sum(w_pgv * m_pgv, dim=2).detach().numpy()
            # ground truth labelsf
            label_pga = sample['label_pga'].cpu().detach().numpy()
            label_pgv = sample['label_pgv'].cpu().detach().numpy()
            if j == 0:
                Mixture_mu_pga = pga_pred
                Mixture_mu_pgv = pgv_pred
                Label_pga = label_pga
                Label_pgv = label_pgv
            else:
                Mixture_mu_pga = np.concatenate([Mixture_mu_pga, pga_pred], axis=1)
                Mixture_mu_pgv = np.concatenate([Mixture_mu_pgv, pgv_pred], axis=1)
                Label_pga = np.concatenate([Label_pga, label_pga], axis=1)
                Label_pgv = np.concatenate([Label_pgv, label_pgv], axis=1)
        # flatten results
        Label_pga = Label_pga.flatten(); Mixture_mu_pga = Mixture_mu_pga.flatten()
        Label_pgv = Label_pgv.flatten(); Mixture_mu_pgv = Mixture_mu_pgv.flatten()

        # prepare output with dual targets
        output = {
            "EQ_ID": EQ_ID,
            "p_picks": P_picks,
            "latitude": Lat,
            "longitude": Lon,
            "elevation": Elev,
            "predict_pga": Mixture_mu_pga,
            "answer_pga": Label_pga,
            "predict_pgv": Mixture_mu_pgv,
            "answer_pgv": Label_pgv,
        }
        output_df = pd.DataFrame(output)
        # filter out zero labels
        output_df = output_df[(output_df["answer_pga"] != 0) | (output_df["answer_pgv"] != 0)]
        
        os.makedirs(f"../predict_pga_pgv/model_{num}", exist_ok=True)
        output_df.to_csv(
            f"../predict_pga_pgv/model_{num}/model {num} {mask_after_sec} sec prediction_vel.csv", index=False
        )

        # output_df = pd.read_csv(f"../predict/model_3_analysis(velocity)/model 3 {mask_after_sec} sec prediction_vel.csv")

        # plot prediction results

        # plot PGA performance
        fig_pga, ax_pga = Intensity_Plotter.plot_true_predicted(
            y_true=output_df["answer_pga"],
            y_pred=output_df["predict_pga"],
            agg="point",
            point_size=12,
            target="pga",
            title=f"{mask_after_sec}s True Predict Plot PGA, 2016 data model {num}"
        )
        fig_pga.savefig(f"../predict_pga_pgv/model_{num}/model {num} {mask_after_sec} sec_pga_acc.png")
        plt.close(fig_pga)
        
        # plot PGV performance
        fig_pgv, ax_pgv = Intensity_Plotter.plot_true_predicted(
            y_true=output_df["answer_pgv"],
            y_pred=output_df["predict_pgv"],
            agg="point",
            point_size=12,
            target="pgv",
            title=f"{mask_after_sec}s True Predict Plot PGV, 2016 data model {num}"
        )
        fig_pgv.savefig(f"../predict_pga_pgv/model_{num}/model {num} {mask_after_sec} sec_pgv_acc.png")
        plt.close(fig_pgv)

#%%
# # ===========merge info==============
num = 76
output_path = f"../predict_pga_pgv/model_{num}"
catalog = pd.read_csv(f"../../../TT-SAM/code/data/1999_2019_final_catalog.csv")
traces_info = pd.read_csv(f"../../../TT-SAM/code/data/1999_2019_final_traces_Vs30.csv")

for mask_after_sec in [3, 5, 7, 10, 13, 15]:
    ensemble_predict = pd.read_csv(
        f"{output_path}/model {num} {mask_after_sec} sec prediction_vel.csv"
    )
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
    trace_merge_catalog.drop(
        ["lat", "lat_minute", "lon", "lon_minute"], axis=1, inplace=True
    )
    trace_merge_catalog.rename(columns={"elevation (m)": "elevation"}, inplace=True)


    data_path = "../../../TT-SAM/code/data/TSMIP_1999_2019_Vs30_integral.hdf5"
    dataset = h5py.File(data_path, "r")
    for eq_id in ensemble_predict["EQ_ID"].unique():
        eq_id = int(eq_id)
        station_name = dataset["data"][str(eq_id)]["station_name"][:].tolist()

        ensemble_predict.loc[
            ensemble_predict.query(f"EQ_ID=={eq_id}").index, "station_name"
        ] = station_name

    ensemble_predict["station_name"] = ensemble_predict["station_name"].str.decode("utf-8")


    prediction_with_info = pd.merge(
        ensemble_predict,
        trace_merge_catalog.drop(
            [
                "latitude",
                "longitude",
                "elevation",
            ],
            axis=1,
        ),
        on=["EQ_ID", "station_name"],
        how="left",
        suffixes=["_window", "_file"],
    )
    prediction_with_info.to_csv(
        f"{output_path}/{mask_after_sec} sec model{num} with all info_vel.csv", index=False
    )

# %%
