import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_type = "acc"
eq_id = "25396"
path = "../data"
with h5py.File(f"{path}/TSMIP_1999_2019_Vs30_integral.hdf5", "r") as h5_file:

    waveform = np.array(h5_file["data"][eq_id][f"{input_type}_traces"])
    station = np.array(h5_file["data"][eq_id]["station_name"])
    p_picks = np.array(h5_file["data"][eq_id]["p_picks"])

waveform_len = len(waveform)
times = waveform_len//5

for i in range(times):
    waveform_interest = waveform[i*5 : (i*5)+5]
    statino_interest = station[i*5 : (i*5)+5]
    p_picks_interest = p_picks[i*5 : (i*5)+5]

    fig, ax = plt.subplots(figsize=(10,8), nrows=5)
    x = np.linspace(0, 30, 6000)
    ax[0].set_title(f"eq: {eq_id}, {input_type} waveform")
    for num in range(5):
        ax[num].plot(x, waveform_interest[num][:, 0], label = statino_interest[num].decode('utf-8'))
        ax[num].vlines(p_picks_interest[num]/200, -2, 2, color='r')
        ax[num].set_ylim(waveform_interest[num][:, 0].min()-0.00001, waveform_interest[num][:, 0].max()+0.00001)
        ax[num].legend(loc='upper left')
    
    fig.savefig(f'../../analysis_image/re_integral/{eq_id}/{input_type}_waveform_{i}', dpi=400)
    plt.close(fig)