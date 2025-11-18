import json
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append("../..")
from model.CNN_Transformer_Mixtureoutput import (
    CNN,
    CNN_ACC,
    MDN_PGA,
    MDN_PGV,
    MLP_output_pga,
    MLP_output_pgv,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)

num = 9
device = torch.device("cuda")
path = f"../../model_with_2_CNN/model{num}_pga.pt"
emb_dim = 150
mlp_dims = (150, 100, 50, 30, 10)
CNN_model = CNN(downsample=2, mlp_input=7665).cuda()
CNN_ACC_model = CNN_ACC(downsample=1, mlp_input=7665).cuda()
pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
transformer_model = TransformerEncoder()
mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
mlp_output_pga = MLP_output_pga(input_shape=(emb_dim,), dims=mlp_dims).cuda()
mlp_output_pgv = MLP_output_pgv(input_shape=(emb_dim,), dims=mlp_dims).cuda()
mdn_pga_model = MDN_PGA(input_shape=(mlp_dims[-1],)).cuda()
mdn_pgv_model = MDN_PGV(input_shape=(mlp_dims[-1],)).cuda()
full_Model = full_model(
    CNN_model,
    CNN_ACC_model,
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

mask_after_sec_list = [3, 5, 7, 10, 13, 15]
for mask_after_sec in mask_after_sec_list:
    Lat = []
    Lon = []
    Elev = []
    Mixture_mu_pga = []
    Mixture_mu_pgv = []
    station_name = []
    sample = {}
    for i in range(1,15):
        print(i)
        with open(f"model_input/vel_{mask_after_sec}_sec/{i}.json", "r") as json_file:
            data = json.load(json_file)

        waveform = torch.tensor(data["waveform"]).to(torch.double).unsqueeze(0)

        input_station = torch.tensor(data["sta"]).to(torch.double).unsqueeze(0)

        target_station = torch.tensor(data["target"]).to(torch.double).unsqueeze(0)
        true_target_num = torch.sum(torch.all(target_station != 0, dim=-1)).item()
        sample = {"waveform": waveform, "sta": input_station, "target": target_station}

        lat = sample["target"][:, :, 0].flatten().tolist()
        lon = sample["target"][:, :, 1].flatten().tolist()
        elev = sample["target"][:, :, 2].flatten().tolist()
        Lat.extend(lat)
        Lon.extend(lon)
        Elev.extend(elev)

        w_pga, s_pga, m_pga, w_pgv, s_pgv, m_pgv = full_Model(sample)
        w_pga, s_pga, m_pga = w_pga.cpu(), s_pga.cpu(), m_pga.cpu()
        w_pgv, s_pgv, m_pgv = w_pgv.cpu(), s_pgv.cpu(), m_pgv.cpu()
        # mixture means
        pga_pred = torch.sum(w_pga * m_pga, dim=2).detach().numpy()
        pgv_pred = torch.sum(w_pgv * m_pgv, dim=2).detach().numpy()

        Mixture_mu_pga.append(pga_pred)
        Mixture_mu_pgv.append(pgv_pred)

        station_name += data["station_name"]
    Mixture_mu_pga_flat = np.concatenate(Mixture_mu_pga).reshape(-1).tolist()
    Mixture_mu_pgv_flat = np.concatenate(Mixture_mu_pgv).reshape(-1).tolist()
    output = {
        "predict_pga": Mixture_mu_pga_flat,
        "predict_pgv": Mixture_mu_pgv_flat,
        "station_name": station_name,
        "latitude": Lat,
        "longitude": Lon,
        "elevation": Elev,
    }

    print(len(output["predict_pga"]), len(output["predict_pgv"]), len(output["station_name"]), len(output["latitude"]), len(output["longitude"]), len(output["elevation"]))

    output_df = pd.DataFrame(output)

    os.makedirs("prediction", exist_ok=True)

    output_df.to_csv(f"prediction/{mask_after_sec}_sec_prediction.csv", index=False)
