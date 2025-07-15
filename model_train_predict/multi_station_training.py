import pickle
import os
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import log_artifact, log_metrics, log_param, log_params
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys

sys.path.append("..")
from model.CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,  # if you don't have vs30 data, please use "PositionEmbedding"
    TransformerEncoder,
    full_model,
)
from data.multiple_sta_dataset import multiple_station_dataset

"""
set up mlflow experiment:
In Terminal, you need to type
"mlflow ui"

enter to UI at local host
create an experiment, its name: "bias to close station"

connect mlflow ui :
mlflow server --host localhost --port 5000 

"""

mlflow.set_tracking_uri("http://localhost:5000")

# # 設置實驗名稱
# experiment_name = "SAVANT First Train"

# # 確認實驗是否存在，如果不存在則創建它
# experiment = mlflow.get_experiment_by_name(experiment_name)
# experiment_id = mlflow.create_experiment(experiment_name)


def train_process(
    full_Model,
    full_data,
    optimizer,
    hyper_param,
    num_of_gaussian=5,
    train_data_size=0.8,
    experiment_name=None,
    run_name=None,
):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment.experiment_id,
    ) as run:
        log_params(
            {
                "epochs": hyper_param["num_epochs"],
                "batch size": hyper_param["batch_size"],
                "learning rate": hyper_param["learning_rate"],
            }
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True

        train_set_size = int(len(full_data) * train_data_size)
        valid_set_size = len(full_data) - train_set_size
        torch.manual_seed(0)
        # for pga training
        train_dataset, val_dataset = random_split(
            full_data, [train_set_size, valid_set_size]
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )

        gaussian_loss = nn.GaussianNLLLoss(reduction="none")
        training_loss = []
        validation_loss = []
        print(f'train {hyper_param["num_epochs"]} times')
        the_last_loss = 100  # initial early stop value
        if hyper_param["learning_rate"] >= 5e-05:
            patience = 10
        elif hyper_param["learning_rate"] >= 0:
            patience = 15
        print("patience", patience)
        trigger_times = 0
        for epoch in range(hyper_param["num_epochs"]):
            print(f"Epoch:{epoch+1}")
            print("--------------------train_start--------------------")
            for sample in tqdm(train_loader):  # training
                optimizer.zero_grad()
                weight, sigma, mu = full_Model(sample)
                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)  # 不讓pga zero padding去計算loss

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                train_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
                train_loss.backward()
                optimizer.step()
            print("train_loss", train_loss)
            training_loss.append(train_loss.data)

            for sample in tqdm(valid_loader):  # validation
                weight, sigma, mu = full_Model(sample)

                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)  # 不讓pga zero padding去計算loss

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                val_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
            print("val_loss", val_loss)
            validation_loss.append(val_loss.data)
            log_metrics(
                {"train_loss": train_loss.item(), "val_loss": val_loss.item()},
                step=epoch + 1,
            )
            # checkpoint
            if train_loss.data < -1 and (epoch + 1) % 5 == 0:
                checkpoint_path = (
                    f"../model_acc/model{hyper_param['model_index']}_checkpoints"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save(
                    full_Model.state_dict(),
                    f"{checkpoint_path}/epoch{epoch+1}_model.pt",
                )
            # epoch early stop:
            current_loss = val_loss.data
            if the_last_loss < -1:  ### 測試比較少訓練的時候改這裡 原本是-1
                patience = 15  ### 測試比較少訓練的時候改這裡 原本是 15
            if current_loss > the_last_loss:  ### 測試比較少訓練的時候改這裡(註解掉這一行)
                trigger_times += 1
                print("early stop trigger times:", trigger_times)

                if trigger_times >= patience:
                    # 往前縮排測試
                    path = "../model_acc"
                    # if epoch+1 == hyper_param["num_epochs"]:
                    print(f"Early stopping! stop at epoch: {epoch+1}")
                    with open(
                        f"{path}/train loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(training_loss, fp)
                        log_artifact(f"{path}/train loss{hyper_param['model_index']}")
                    with open(
                        f"{path}/validation loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(validation_loss, fp)
                        log_artifact(
                            f"{path}/validation loss{hyper_param['model_index']}"
                        )
                    log_param("epoch early stop", epoch + 1)
                    return training_loss, validation_loss

                continue

            else:
                print("trigger 0 time")
                trigger_times = 0
                path = "../model_acc"
                model_file = f"{path}/model{hyper_param['model_index']}_acc.pt"
                torch.save(full_Model.state_dict(), model_file)
                log_artifact(model_file)

            the_last_loss = current_loss
        print(
            "Train Epoch: {}/{} Traing_Loss: {} Val_Loss: {}".format(
                epoch + 1, hyper_param["num_epochs"], train_loss.data, val_loss.data
            )
        )


if __name__ == "__main__":
    train_data_size = 0.8
    model_index = 16
    num_epochs = 300
    # batch_size=16
    for batch_size in [16]:
        for LR in [2.5e-5]:
            for i in range(3):  # 原本是3
                model_index += 1
                hyper_param = {
                    "model_index": model_index,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": LR,
                }
                print(f"learning rate: {LR}")
                print(f"batch size: {batch_size}")
                num_of_gaussian = 5
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
                )
                optimizer = torch.optim.Adam(
                    [
                        {"params": CNN_model.parameters()},
                        {"params": transformer_model.parameters()},
                        {"params": mlp_model.parameters()},
                        {"params": mdn_model.parameters()},
                    ],
                    lr=LR,
                )
                full_data = multiple_station_dataset(
                    "../../../TT-SAM/code/data/TSMIP_1999_2019_Vs30_integral.hdf5",
                    mode="train",
                    mask_waveform_sec=3,
                    weight_label=False,
                    oversample=1.25,
                    oversample_mag=4,
                    test_year=2016,
                    mask_waveform_random=True,
                    mag_threshold=0,
                    label_key="pga",
                    input_type="acc",
                    data_length_sec=20,
                    station_blind=True,
                    bias_to_closer_station=True,
                )
                training_loss, validation_loss = train_process(
                    full_Model,
                    full_data,
                    optimizer,
                    hyper_param,
                    experiment_name="SAVANT ACC Train",
                    run_name=f"6th_Train_Acc: model 17-19 | input:acc & vel | 20250714",
                )
