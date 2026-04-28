import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations

try:
    from captum.attr import IntegratedGradients
except ImportError as exc:
    raise ImportError(
        "captum is required. Install it with: pip install captum"
    ) from exc

sys.path.append("..")

from model.CNN_Transformer_Mixtureoutput import (
    CNN,
    CNN_ACC,
    CNN_Physical_features,
    MDN_PGA,
    MDN_PGV,
    MLP,
    MLP_output_pga,
    MLP_output_pgv,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from data.multiple_sta_dataset import multiple_station_dataset

WAVEFORM_BASE_CHANNEL_NAMES = [
    "acc_Z",
    "acc_N",
    "acc_E",
    "vel_Z",
    "vel_N",
    "vel_E",
    "vel_lf_Z",
    "vel_lf_N",
    "vel_lf_E",
]


def build_model_feature_mapping(model_start_index: int = 40) -> dict[int, dict]:
    """Mirror multi_station_training.py to build model_num -> feature mapping."""
    candidate_physical_features = ["cvaa_log1p", "Ia_log1p", "IV2_log1p", "TP_log1p"]
    physical_feature_combinations = []
    for r in range(3, len(candidate_physical_features) + 1):
        physical_feature_combinations.extend(combinations(candidate_physical_features, r))

    model_index = model_start_index
    model_to_features = {}

    for feature_combo in physical_feature_combinations:
        physical_feature_list = list(feature_combo)
        for chosen_intensity in ["IV"]:
            for loss_mode in ["MSFE"]:
                for batch_size in [8]:
                    for learning_rate in [5e-5]:
                        for _ in range(2):
                            model_index += 1
                            model_to_features[model_index] = {
                                "physical_feature": physical_feature_list,
                                "intensity": chosen_intensity,
                                "loss_mode": loss_mode,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate,
                            }
    return model_to_features


def get_waveform_channel_names(physical_feature_list: list[str]) -> list[str]:
    return WAVEFORM_BASE_CHANNEL_NAMES + physical_feature_list


if torch.cuda.is_available():
    # Keep Transformer numerics stable with your current inference setup.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

if hasattr(torch.backends, "transformers") and hasattr(torch.backends.transformers, "nested_tensor"):
    torch.backends.transformers.nested_tensor = False


class PGA_Wrapper(nn.Module):
    """Wrapper for Captum IntegratedGradients.

    Accepts independent tensors (waveform, sta, target), computes PGA expected
    value from MDN outputs, then reduces to (batch,) scalar target.
    """

    def __init__(self, base_model: full_model):
        super().__init__()
        self.base_model = base_model

    def _shared_backbone(
        self, waveform: torch.Tensor, sta: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # Keep gradient path from waveform to output (avoid Tensor re-wrapping).
        waveform = waveform.float().reshape(-1, self.base_model.data_length, waveform.shape[-1])
        vel_data = waveform[:, :, 3:9]
        acc_data = waveform[:, :, :3]
        physical_feature = waveform[:, :, 9:]

        cnn_output = self.base_model.model_CNN(vel_data)
        cnn_acc_output = self.base_model.model_CNN_ACC(acc_data)
        cnn_physical_output = self.base_model.model_CNN_Physical_features(physical_feature)

        cnn_combined_output = torch.cat((cnn_output, cnn_acc_output, cnn_physical_output), dim=1)
        cnn_output_reshape = torch.reshape(
            cnn_combined_output, (-1, self.base_model.max_station, self.base_model.emb_dim)
        )

        emb_output = self.base_model.model_Position(
            sta.float().reshape(-1, 1, sta.shape[2]).float().cuda()
        )
        emb_output = emb_output.reshape(-1, self.base_model.max_station, self.base_model.emb_dim)

        station_pad_mask = sta == 0
        station_pad_mask = torch.all(station_pad_mask, 2)

        pga_pos_emb_output = self.base_model.model_Position(
            target.float().reshape(-1, 1, target.shape[2]).float().cuda()
        )
        pga_pos_emb_output = pga_pos_emb_output.reshape(
            -1, self.base_model.pga_targets, self.base_model.emb_dim
        )

        # Keep your finalized design: all targets are masked as Key/Value.
        target_pad_mask = torch.ones_like(target, dtype=torch.bool)
        target_pad_mask = torch.all(target_pad_mask, 2)

        pad_mask = torch.cat((station_pad_mask, target_pad_mask), dim=1).cuda()

        add_pe_cnnoutput = torch.add(cnn_output_reshape, emb_output)
        transformer_input = torch.cat((add_pe_cnnoutput, pga_pos_emb_output), dim=1)
        transformer_output = self.base_model.model_Transformer(transformer_input, pad_mask)

        mlp_input = transformer_output[:, -self.base_model.pga_targets :, :].cuda()
        return mlp_input

    def forward(self, waveform: torch.Tensor, sta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mlp_input = self._shared_backbone(waveform, sta, target)

        mlp_output_pga_value = self.base_model.model_mlp_output_pga(mlp_input)
        weight_pga, _, mu_pga = self.base_model.model_MDN_PGA(mlp_output_pga_value)

        # Expected PGA per target station: (B, 25)
        pga_pred = torch.sum(weight_pga * mu_pga, dim=2)

        # Captum target should be scalar per sample: (B,)
        return pga_pred.sum(dim=1)


class PGV_Wrapper(PGA_Wrapper):
    """Wrapper for PGV expected value attribution with shared backbone."""

    def forward(self, waveform: torch.Tensor, sta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mlp_input = self._shared_backbone(waveform, sta, target)

        mlp_output_pgv_value = self.base_model.model_mlp_output_pgv(mlp_input)
        weight_pgv, _, mu_pgv = self.base_model.model_MDN_PGV(mlp_output_pgv_value)

        # Expected PGV per target station: (B, 25)
        pgv_pred = torch.sum(weight_pgv * mu_pgv, dim=2)

        # Captum target should be scalar per sample: (B,)
        return pgv_pred.sum(dim=1)


def build_model(
    model_path: str,
    device: torch.device,
    physical_feature_list: list[str],
) -> full_model:
    emb_dim = 150
    mlp_dims = (150, 100, 50, 30, 10)

    cnn_model = CNN(mlp_input=7665).to(device)
    cnn_acc_model = CNN_ACC(mlp_input=7665).to(device)
    cnn_physical_model = CNN_Physical_features(
        downsample=len(physical_feature_list), mlp_input=7665
    ).to(device)
    pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).to(device)
    transformer_model = TransformerEncoder()
    mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).to(device)
    mlp_output_pga = MLP_output_pga(input_shape=(emb_dim,), dims=mlp_dims).to(device)
    mlp_output_pgv = MLP_output_pgv(input_shape=(emb_dim,), dims=mlp_dims).to(device)
    mdn_pga_model = MDN_PGA(input_shape=(mlp_dims[-1],)).to(device)
    mdn_pgv_model = MDN_PGV(input_shape=(mlp_dims[-1],)).to(device)

    model = full_model(
        cnn_model,
        cnn_acc_model,
        cnn_physical_model,
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

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_importance_percent(
    wrapper_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    channel_names: list[str],
    max_batches: int | None,
    n_steps: int,
    internal_batch_size: int,
) -> tuple[np.ndarray, int]:
    ig = IntegratedGradients(wrapper_model)

    total_scores = torch.zeros(len(channel_names), device=device)
    used_batches = 0

    for batch_idx, sample in tqdm(enumerate(loader), desc="IG batches"):
        if max_batches is not None and batch_idx >= max_batches:
            break

        waveform = sample["waveform"].to(device).float().requires_grad_()
        sta = sample["sta"].to(device).float()
        target = sample["target"].to(device).float()

        baseline_waveform = torch.zeros_like(waveform)

        attributions = ig.attribute(
            inputs=waveform,
            baselines=baseline_waveform,
            additional_forward_args=(sta, target),
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )

        # Keep all input channels on the last dimension.
        # Works for both 3D (B,T,C) and 4D (B,S,T,C) inputs.
        channel_attr = attributions[..., :]

        # Aggregate all non-channel dimensions, keep final channel axis only.
        reduce_dims = tuple(range(channel_attr.ndim - 1))
        batch_scores = torch.sum(torch.abs(channel_attr), dim=reduce_dims)

        total_scores += batch_scores
        used_batches += 1

        # Release per-batch tensors early to avoid GPU accumulation in long IG runs.
        del attributions, channel_attr, batch_scores, waveform, sta, target, baseline_waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if used_batches == 0:
        raise RuntimeError("No batches were processed. Check DataLoader / max_batches.")

    scores_np = total_scores.detach().cpu().numpy()
    denom = np.sum(scores_np)
    if denom <= 0:
        importance_pct = np.zeros_like(scores_np)
    else:
        importance_pct = (scores_np / denom) * 100.0

    return importance_pct, used_batches


def save_importance_plot(
    importance_pct: np.ndarray,
    channel_names: list[str],
    out_png_path: str,
    label_name: str,
    model_path: str,
    used_batches: int,
    n_steps: int,
) -> None:

    plt.figure(figsize=(12, 6))
    bars = plt.bar(channel_names, importance_pct)
    plt.ylabel("Importance (%)")
    plt.xlabel("Waveform + Physical Features")
    plt.title(
        f"Integrated Gradients Feature Importance ({label_name})\n"
        f"model={os.path.basename(model_path)}, batches={used_batches}, n_steps={n_steps}"
    )
    plt.xticks(rotation=35)

    for bar, val in zip(bars, importance_pct):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=200)
    plt.close()


def run_ig_importance(
    model_path: str,
    data_path: str,
    physical_feature_list: list[str],
    out_png_path_pga: str,
    out_png_path_pgv: str,
    mask_after_sec: int = 3,
    max_batches: int | None = None,
    n_steps: int = 20,
    internal_batch_size: int = 1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel_names = get_waveform_channel_names(physical_feature_list)

    model = build_model(
        model_path=model_path,
        device=device,
        physical_feature_list=physical_feature_list,
    )
    # IG only needs gradients w.r.t. input waveform; disable parameter gradients to save memory.
    for param in model.parameters():
        param.requires_grad_(False)

    pga_wrapper = PGA_Wrapper(model).to(device)
    pga_wrapper.eval()
    pgv_wrapper = PGV_Wrapper(model).to(device)
    pgv_wrapper.eval()

    data = multiple_station_dataset(
        data_path,
        mode="test",
        mask_waveform_sec=mask_after_sec,
        test_year=2016,
        physical_feature=physical_feature_list,
        mag_threshold=0,
        input_type="acc",
        data_length_sec=20,
    )
    loader = DataLoader(dataset=data, batch_size=1, shuffle=False)

    pga_importance_pct, used_batches_pga = compute_importance_percent(
        wrapper_model=pga_wrapper,
        loader=loader,
        device=device,
        channel_names=channel_names,
        max_batches=max_batches,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    save_importance_plot(
        importance_pct=pga_importance_pct,
        channel_names=channel_names,
        out_png_path=out_png_path_pga,
        label_name="PGA",
        model_path=model_path,
        used_batches=used_batches_pga,
        n_steps=n_steps,
    )

    pgv_importance_pct, used_batches_pgv = compute_importance_percent(
        wrapper_model=pgv_wrapper,
        loader=loader,
        device=device,
        channel_names=channel_names,
        max_batches=max_batches,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    save_importance_plot(
        importance_pct=pgv_importance_pct,
        channel_names=channel_names,
        out_png_path=out_png_path_pgv,
        label_name="PGV",
        model_path=model_path,
        used_batches=used_batches_pgv,
        n_steps=n_steps,
    )

    print("==== Physical Feature Importance (Integrated Gradients - PGA) ====")
    for name, val in zip(channel_names, pga_importance_pct):
        print(f"{name:>5s}: {val:8.4f}%")
    print(f"Saved figure: {out_png_path_pga}")

    print("==== Physical Feature Importance (Integrated Gradients - PGV) ====")
    for name, val in zip(channel_names, pgv_importance_pct):
        print(f"{name:>5s}: {val:8.4f}%")
    print(f"Saved figure: {out_png_path_pgv}")


def run_ig_importance_batch_by_model_num(
    model_nums: list[int],
    model_to_features: dict[int, dict],
    model_dir: str,
    output_root_dir: str,
    data_path: str,
    mask_after_sec: int = 3,
    max_batches: int | None = None,
    n_steps: int = 20,
    internal_batch_size: int = 1,
) -> None:
    """Run IG importance for multiple models in one script execution.

    model_nums: model index list, e.g. [1, 2, 3].
    model checkpoint filename format: model{num}_pga.pt
    output directory format: output_root_dir/model_{num}
    """
    if len(model_nums) == 0:
        raise ValueError("model_nums is empty. Provide at least one model number.")

    total = len(model_nums)
    for idx, model_num in enumerate(model_nums, start=1):
        if model_num not in model_to_features:
            raise ValueError(f"model_num {model_num} is not in model_to_features mapping.")

        physical_feature_list = model_to_features[model_num]["physical_feature"]
        model_path = os.path.join(model_dir, f"model{model_num}_pga.pt")
        output_dir = os.path.join(output_root_dir, f"model_test_{model_num}")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        out_png_path_pga = os.path.join(
            output_dir, f"ig_physical_importance_pga_{model_name}.png"
        )
        out_png_path_pgv = os.path.join(
            output_dir, f"ig_physical_importance_pgv_{model_name}.png"
        )

        print(f"\n[{idx}/{total}] Running IG for model: {model_path}")
        print(f"Physical features: {physical_feature_list}")
        run_ig_importance(
            model_path=model_path,
            data_path=data_path,
            physical_feature_list=physical_feature_list,
            out_png_path_pga=out_png_path_pga,
            out_png_path_pgv=out_png_path_pgv,
            mask_after_sec=mask_after_sec,
            max_batches=max_batches,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )


if __name__ == "__main__":
    model_to_features = build_model_feature_mapping(model_start_index=40)

    run_all_models = True
    if run_all_models:
        model_nums = sorted(model_to_features.keys())
    else:
        model_nums = [72]  # Modify to test a single model.

    run_ig_importance_batch_by_model_num(
        model_nums=model_nums,
        model_to_features=model_to_features,
        model_dir="../model_with_several_physical_feature",
        output_root_dir="../predict_with_several_physical_feature",
        data_path="../data/TSMIP_1999_2019_Vs30_integral.hdf5",
        mask_after_sec=13,
        n_steps=20,
        internal_batch_size=1,
    )
