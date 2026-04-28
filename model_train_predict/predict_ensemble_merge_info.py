import os
import h5py
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations
import sys
sys.path.append("..")
from model.CNN_Transformer_Mixtureoutput import (
    CNN,
    CNN_ACC,
    CNN_Physical_features, 
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
from model_performance_analysis.analysis import MMIntensity

if torch.cuda.is_available():
    # Force classic SDP math kernel to reduce eval-time fastpath drift.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

if hasattr(torch.backends, "transformers") and hasattr(torch.backends.transformers, "nested_tensor"):
    torch.backends.transformers.nested_tensor = False

# =========== Build model_index to physical_features mapping ===========
# Keep this mapping aligned with multi_station_training.py.
def build_model_feature_mapping(model_start_index=None):
    """Build model_index -> physical_features mapping from training loop logic."""
    candidate_physical_features = ["cvaa_log1p", "Ia_log1p", "IV2_log1p", "TP_log1p"]
    physical_feature_combinations = []
    for r in range(3, len(candidate_physical_features) + 1):
        physical_feature_combinations.extend(combinations(candidate_physical_features, r))
    
    model_index = model_start_index
    model_to_features = {}
    
    # Mirror the loop structure used in training.
    for feature_combo in physical_feature_combinations:
        physical_feature_list = list(feature_combo)
        for chosen_intensity in ["IV"]:
            for loss_mode in ["MSFE"]:
                for batch_size in [8]:
                    for LR in [5e-5]:
                        for i in range(5):
                            model_index += 1
                            model_to_features[model_index] = {
                                "physical_feature": physical_feature_list,
                                "intensity": chosen_intensity,
                                "loss_mode": loss_mode,
                                "batch_size": batch_size,
                                "learning_rate": LR,
                            }
    
    return model_to_features


model_start_index = 72  # Start index used in multi_station_training.py
model_to_features = build_model_feature_mapping(model_start_index=model_start_index)

# =========== Execution mode ===========
# True: evaluate all mapped models.
# False: evaluate only one specified model_num.
run_all_models = True

if run_all_models:
    # Evaluate all mapped model indices.
    models_to_test = sorted(model_to_features.keys())
    print(f"Evaluating all mapped models: {models_to_test}")
else:
    # Evaluate only the specified model number.
    model_num = 73  # Change this to the model index you want to test.
    if model_num not in model_to_features:
        print(f"Warning: model_num {model_num} is not in the mapped training list.")
        print(f"Available model_num range: {min(model_to_features.keys())} - {max(model_to_features.keys())}")
        print(f"Available model_num values: {sorted(model_to_features.keys())}")
    models_to_test = [model_num]

# =========== Predict ===========
for num in models_to_test:
    physical_feature_list = model_to_features[num]["physical_feature"]
    print(f"\n{'='*80}")
    print(f"Start testing Model {num}")
    print(f"Physical Features: {physical_feature_list}")
    print(f"{'='*80}")
    
    for mask_sec in [3, 5, 7, 10, 13, 15]:
        mask_after_sec = mask_sec
        # Dual-target prediction: PGA and PGV.
        device = torch.device("cuda")
        data = multiple_station_dataset(
            "../data/TSMIP_1999_2019_Vs30_log1p.hdf5",
            mode="test",
            mask_waveform_sec=mask_after_sec,
            test_year=2016,
            # Use default label_keys=["pga", "pgv"].
            physical_feature=physical_feature_list,
            mag_threshold=0,
            input_type="acc",
            data_length_sec=20,
        )
        path = f"../model_with_several_physical_feature/model{num}_pga.pt"
        emb_dim = 150
        mlp_dims = (150, 100, 50, 30, 10)
        CNN_model = CNN(mlp_input=7665).cuda()
        CNN_ACC_model = CNN_ACC(mlp_input=7665).cuda()
        CNN_Physical_model = CNN_Physical_features(downsample=len(physical_feature_list), mlp_input=7665).cuda()
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
            CNN_Physical_model,
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
        full_Model.load_state_dict(torch.load(path, map_location=device))
        full_Model.eval()
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
        with torch.no_grad():
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
                pga_pred = torch.sum(w_pga * m_pga, dim=2).cpu().numpy()
                pgv_pred = torch.sum(w_pgv * m_pgv, dim=2).cpu().numpy()
                # ground truth labels
                label_pga = sample['label_pga'].cpu().numpy()
                label_pgv = sample['label_pgv'].cpu().numpy()
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
        
        os.makedirs(f"../predict_with_several_physical_feature/model_test_{num}", exist_ok=True)
        output_df.to_csv(
            f"../predict_with_several_physical_feature/model_test_{num}/model {num} {mask_after_sec} sec prediction_vel.csv", index=False
        )

        # Plot PGA performance.
        fig_pga, ax_pga = Intensity_Plotter.plot_true_predicted(
            y_true=output_df["answer_pga"],
            y_pred=output_df["predict_pga"],
            agg="point",
            point_size=12,
            target="pga",
            title=f"{mask_after_sec}s True Predict Plot PGA, 2016 data model {num}"
        )
        fig_pga.savefig(f"../predict_with_several_physical_feature/model_test_{num}/model {num} {mask_after_sec} sec_pga_acc.png")
        plt.close(fig_pga)
        
        # Plot PGV performance.
        fig_pgv, ax_pgv = Intensity_Plotter.plot_true_predicted(
            y_true=output_df["answer_pgv"],
            y_pred=output_df["predict_pgv"],
            agg="point",
            point_size=12,
            target="pgv",
            title=f"{mask_after_sec}s True Predict Plot PGV, 2016 data model {num}"
        )
        fig_pgv.savefig(f"../predict_with_several_physical_feature/model_test_{num}/model {num} {mask_after_sec} sec_pgv_acc.png")
        plt.close(fig_pgv)

# Metadata merging was moved to model_train_predict/merge_prediction_metadata.py
# to keep this script focused on inference only.
