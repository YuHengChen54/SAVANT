#%%
import pandas as pd
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.trigger import ar_pick
from scipy.integrate import cumulative_trapezoid
import json
import sys

sys.path.append("../..")
from data_preprocess.read_tsmip import get_integrated_stream, get_integrated_stream_second

def dist(event_lat, event_lon, station_lat, station_lon):  # unit: degree
    dist = ((event_lat - station_lat) ** 2 + (event_lon - station_lon) ** 2) ** (1 / 2)
    return dist



station_info = pd.read_csv("../../data/station_information/TSMIPstations_new.csv")
traces_info_with_vs30 = pd.read_csv(
    "../../data/1999_2019_final_traces_Vs30.csv"
)
station_trans_info = pd.read_csv("../TSMIP-Instrument-parameters-20240821.txt", sep="\t")
station_trans_info.drop(['安裝經度', '安裝緯度', '安裝高程', '啟用日期', '停用日期', 'Z軸極性', 'N軸極性', 'E軸極性', '感測器型號'], axis=1, inplace=True)
station_trans_info.columns = ["Z_trans", "N_trans", "E_trans", "station_code"]
station_trans_info = station_trans_info.loc[:675]

sample_rate = 200


################################## Change parameter ##################################
path = "20240403ML7.1"
# event epicenter
event_lat = 23.88
event_lon = 121.57
######################################################################################


waveform_files = os.listdir(path)

stations = []
for file in waveform_files:
    station_name = file[26:30]
    if station_name not in stations:
        stations.append(station_name)

station_info = station_info[station_info["station_code"].isin(stations)]
station_info = station_info.reset_index(drop=True)


dist_dict = {"dist": []}
for i in range(len(station_info)):
    station_lat = station_info["latitude"][i]
    station_lon = station_info["longitude"][i]
    dist_dict["dist"].append(dist(event_lat, event_lon, station_lat, station_lon))
station_info["dist (degree)"] = dist_dict["dist"]

station_info["p_picks (sec)"] = 0
# plot and picking:
for i, station in enumerate(station_info["station_code"]):
    trace_z = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLZ.D.SAC")
    trace_n = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLN.D.SAC")
    trace_e = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLE.D.SAC")
    trace_z.resample(sample_rate, window="hann")
    trace_n.resample(sample_rate, window="hann")
    trace_e.resample(sample_rate, window="hann")

    waveforms = np.array(
        [
            trace_z[0].data,
            trace_n[0].data,
            trace_e[0].data,
        ]
    )
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(waveforms[0])
    # ax[1].plot(waveforms[1])
    # ax[2].plot(waveforms[2])
    # ax[0].set_title(
    #     f"{station}_{trace_z[0].stats.starttime}-{trace_z[0].stats.endtime}"
    # )
    try:
        p_pick, _ = ar_pick(
            waveforms[0],
            waveforms[1],
            waveforms[2],
            samp_rate=200,
            f1=1,  # Frequency of the lower bandpass window
            f2=20,  # Frequency of the upper bandpass window
            lta_p=1,  # Length of LTA for the P arrival in seconds
            sta_p=0.1,  # Length of STA for the P arrival in seconds
            lta_s=4.0,  # Length of LTA for the S arrival in seconds
            sta_s=1.0,  # Length of STA for the P arrival in seconds
            m_p=2,  # Number of AR coefficients for the P arrival
            m_s=8,  # Number of AR coefficients for the S arrival
            l_p=0.1,
            l_s=0.2,
            s_pick=False,
        )
        station_info.loc[i, "p_picks (sec)"] = p_pick
        # ax[0].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
        # ax[1].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
        # ax[2].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
    except:
        station_info.loc[i, "p_picks (sec)"] = p_pick
    # fig.savefig(f"2025_0125_waveform_image/{station}.png", dpi=300)
    # plt.close()

station_info = station_info.sort_values(by="dist (degree)")
station_info = station_info.reset_index(drop=True)

trigger_station_info = pd.merge(
    station_info,
    traces_info_with_vs30[["station_name", "Vs30"]].drop_duplicates(
        subset="station_name"
    ),
    left_on="location_code",
    right_on="station_name",
    how="left",
)
trigger_station_info = trigger_station_info.dropna(
    subset=["latitude", "longitude", "elevation (m)", "Vs30"]
)




trigger_station_info = trigger_station_info.reset_index(drop=True)
#%%

trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="F074"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="F071"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="F075"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="E053"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="F073"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="E075"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="B207"]
trigger_station_info=trigger_station_info[trigger_station_info["station_code"]!="G033"]
trigger_station_info = trigger_station_info.reset_index(drop=True)
target_station_info = trigger_station_info.copy()

P_wave_velocity = 6.5

mask_after_sec_list = [3, 5, 7, 10, 13, 15]

for mask_after_sec in mask_after_sec_list:
    stream = obspy.core.stream.Stream() 
    waveforms_window = []
    mask_station_index = []
    print(f"Processing for mask after {mask_after_sec} sec")
    for i, station in enumerate(trigger_station_info["station_code"][:25]):
        print(f"Processing station: {station}")
        trace_z = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLZ.D.SAC")
        trace_n = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLN.D.SAC")
        trace_e = obspy.read(f"{path}/2024.093.23.57.39.0000.TW.{station}.10.HLE.D.SAC")
        # bad data padding to fit time window

        #counts to m/s2
        trans_filter = station_trans_info["station_code"] == station
        z_trans = station_trans_info[trans_filter]["Z_trans"].values
        n_trans = station_trans_info[trans_filter]["N_trans"].values
        e_trans = station_trans_info[trans_filter]["E_trans"].values

        trace_z[0].data = trace_z[0].data * z_trans
        trace_n[0].data = trace_n[0].data * n_trans
        trace_e[0].data = trace_e[0].data * e_trans

        trace_z.resample(200, window="hann")
        trace_n.resample(200, window="hann")
        trace_e.resample(200, window="hann")

        vel_trace_z = trace_z.copy()
        vel_trace_n = trace_n.copy()
        vel_trace_e = trace_e.copy()
        vel_trace_z = get_integrated_stream(vel_trace_z)
        vel_trace_n = get_integrated_stream(vel_trace_n)
        vel_trace_e = get_integrated_stream(vel_trace_e)

        low_trace_z = vel_trace_z.copy()
        low_trace_n = vel_trace_n.copy()
        low_trace_e = vel_trace_e.copy()
        low_trace_z = low_trace_z.filter("lowpass", freq=0.33)
        low_trace_n = low_trace_n.filter("lowpass", freq=0.33)
        low_trace_e = low_trace_e.filter("lowpass", freq=0.33)

        dis_trace_z = vel_trace_z.copy()
        dis_trace_n = vel_trace_n.copy()
        dis_trace_e = vel_trace_e.copy()
        dis_trace_z = get_integrated_stream_second(vel_trace_z)
        dis_trace_n = get_integrated_stream_second(vel_trace_n)
        dis_trace_e = get_integrated_stream_second(vel_trace_e)


        waveforms = np.array(
            [
                vel_trace_z[0].data,
                vel_trace_n[0].data,
                vel_trace_e[0].data
            ]
        )

        if station == "F068":  # first triggered station
            p_pick, _ = ar_pick(
                waveforms[0],
                waveforms[1],
                waveforms[2],
                samp_rate=200,
                f1=1,  # Frequency of the lower bandpass window
                f2=20,  # Frequency of the upper bandpass window
                lta_p=1,  # Length of LTA for the P arrival in seconds
                sta_p=0.1,  # Length of STA for the P arrival in seconds
                lta_s=4.0,  # Length of LTA for the S arrival in seconds
                sta_s=1.0,  # Length of STA for the P arrival in seconds
                m_p=2,  # Number of AR coefficients for the P arrival
                m_s=8,  # Number of AR coefficients for the S arrival
                l_p=0.1,
                l_s=0.2,
                s_pick=False,
            )
        start_time = int((p_pick - 5) * sample_rate)
        end_time = int((p_pick + 15) * sample_rate)

        trace_z[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        trace_n[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        trace_e[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        vel_trace_z[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        vel_trace_n[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        vel_trace_e[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        low_trace_z[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        low_trace_n[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
        low_trace_e[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0

        if (
            trigger_station_info["dist (degree)"][i]
            - trigger_station_info["dist (degree)"][0]
        ) * 100 / P_wave_velocity > mask_after_sec:  # zero padding non triggered station
            mask_station_index.append(i) #for mask non trigger station information
            trace_z[0].data[:] = 0
            trace_n[0].data[:] = 0
            trace_e[0].data[:] = 0
            vel_trace_z[0].data[:] = 0
            vel_trace_n[0].data[:] = 0
            vel_trace_e[0].data[:] = 0
            low_trace_z[0].data[:] = 0
            low_trace_n[0].data[:] = 0
            low_trace_e[0].data[:] = 0
        waveforms = np.stack(
            (
                trace_z[0].data[start_time:end_time],
                trace_n[0].data[start_time:end_time],
                trace_e[0].data[start_time:end_time],
                vel_trace_z[0].data[start_time:end_time],
                vel_trace_n[0].data[start_time:end_time],
                vel_trace_e[0].data[start_time:end_time], 
                low_trace_z[0].data[start_time:end_time],
                low_trace_n[0].data[start_time:end_time],
                low_trace_e[0].data[start_time:end_time],
            ),
            axis=1,
        )
        waveforms = waveforms.reshape(4000, 9)
        waveforms_window.append(waveforms)

        fig, ax = plt.subplots(9, 1, figsize=(10, 15))
        for i in range(9):
            ax[i].plot(waveforms[:, i])
        ax[0].set_title(f"{station}")
        os.makedirs(f"model_input_waveform_image/{mask_after_sec}_sec", exist_ok=True)
        fig.savefig(
            f"model_input_waveform_image/{mask_after_sec}_sec/{mask_after_sec}_{station}.png", dpi=300
        )
        plt.close()

    waveform = np.stack(waveforms_window, axis=0).tolist()


    #mask non trigger station information
    for i in mask_station_index:
        trigger_station_info.loc[i, ["latitude", "longitude", "elevation (m)", "Vs30"]] = 0

    input_station = (
        trigger_station_info[["latitude", "longitude", "elevation (m)", "Vs30"]][:25]
        .to_numpy()
        .tolist()
    )

    os.makedirs("model_input", exist_ok=True)
    os.makedirs(f"model_input/vel_{mask_after_sec}_sec", exist_ok=True)

    for i in range(1, 16):
        print((i - 1) * 25, i * 25)
        target_station = (
            target_station_info[["latitude", "longitude", "elevation (m)", "Vs30"]][
                (i - 1) * 25 : i * 25
            ]
            .to_numpy()
            .tolist()
        )
        station_name = target_station_info["location_code"][(i - 1) * 25 : i * 25].tolist()
        output = {
            "waveform": waveform,
            "sta": input_station,
            "target": target_station,
            "station_name": station_name,
        }

        with open(f"model_input/vel_{mask_after_sec}_sec/{i}.json", "w") as json_file:
            json.dump(output, json_file)


# %%
