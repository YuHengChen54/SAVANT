import pandas as pd
import os
import obspy
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../..")
from data_preprocess.read_tsmip import get_peak_value

station_info = pd.read_csv("../../data/station_information/TSMIPstations_new.csv")
traces_info_with_vs30 = pd.read_csv(
    "../../data/1999_2019_final_traces_Vs30.csv"
)
station_trans_info = pd.read_csv("../TSMIP-Instrument-parameters-20240821.txt", sep="\t")
station_trans_info.drop(['安裝經度', '安裝緯度', '安裝高程', '啟用日期', '停用日期', 'Z軸極性', 'N軸極性', 'E軸極性', '感測器型號'], axis=1, inplace=True)
station_trans_info.columns = ["Z_trans", "N_trans", "E_trans", "station_code"]
station_trans_info = station_trans_info.loc[:675]

sample_rate = 200

path = "20240403ML7.1"
waveform_files = os.listdir(path)

stations = []
for file in waveform_files:
    station_name = file[26:30]
    if station_name not in stations:
        stations.append(station_name)

station_info = station_info[station_info["station_code"].isin(stations)]
station_info = station_info.reset_index(drop=True)

output_df = {"station_code": [], "PGV": [], "PGA": []}
for i, station in enumerate(station_info["station_code"]):
    try:
        trace_z = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLZ.D.SAC")
        trace_n = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLN.D.SAC")
        trace_e = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLE.D.SAC")
        trans_filter = station_trans_info["station_code"] == station
        z_trans = station_trans_info[trans_filter]["Z_trans"].values
        n_trans = station_trans_info[trans_filter]["N_trans"].values
        e_trans = station_trans_info[trans_filter]["E_trans"].values

        trace_z[0].data = trace_z[0].data * z_trans *100
        trace_n[0].data = trace_n[0].data * n_trans *100
        trace_e[0].data = trace_e[0].data * e_trans *100
        acc_stream = obspy.core.stream.Stream()
        vel_stream = obspy.core.stream.Stream()
        
        vel_trace_z = trace_z[0].copy()
        vel_trace_n = trace_n[0].copy()
        vel_trace_e = trace_e[0].copy()
        acc_stream.append(trace_z[0])
        acc_stream.append(trace_n[0])
        acc_stream.append(trace_e[0])
        vel_stream.append(vel_trace_z)
        vel_stream.append(vel_trace_n)
        vel_stream.append(vel_trace_e)

        acc_stream.detrend(type="demean")
        acc_stream.filter("lowpass", freq=10)

        vel_stream.detrend(type="demean")
        vel_stream.filter("lowpass", freq=10)
        vel_stream.taper(max_percentage=0.05, type="cosine")
        vel_stream.integrate()
        vel_stream.filter("bandpass", freqmin=0.075, freqmax=10)
        # plot
        # fig,ax=plt.subplots(3,1)
        # for k in range(3):
        #     ax[k].plot(stream[k].data)
        # ax[0].set_title(asc_files[i][26:30])
        # ax[2].set_xlabel("time sample (100Hz)")
        # ax[1].set_ylabel("amplitude (gal)")
        # plt.close()
        # fig.savefig(f"{data_path}/image/{asc_files[i][26:30]}.png",dpi=300)

        pga, _ = get_peak_value(acc_stream)
        pgv, _ = get_peak_value(vel_stream)
        output_df["station_code"].append(station)
        output_df["PGA"].append(pga)
        output_df["PGV"].append(pgv)
    except Exception as e:
        print(f"Error processing station {station}: {e}")

output_df = pd.DataFrame(output_df)

output_df = pd.merge(
    output_df,
    station_info[["station_code", "location_code"]],
    left_on="station_code",
    right_on="station_code",
    how="left",
)

output_df.to_csv(f"true_answer.csv", index=False)
