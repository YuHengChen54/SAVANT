import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

waveform = pd.read_json("2024_0816_ML6.3/model_input/vel_3_sec_data/1.json")
waveform = waveform["waveform"]
waveform = np.array(waveform[1])

fig, ax = plt.subplots(6,1, figsize=(10,12))
ax[0].plot(waveform[:, 0])
ax[1].plot(waveform[:, 1])
ax[2].plot(waveform[:, 2])
ax[3].plot(waveform[:, 3])
ax[4].plot(waveform[:, 4])
ax[5].plot(waveform[:, 5])
ax[0].set_title("json 0917 lowfreq", fontsize=20)

with h5py.File("../data/TSMIP_1999_2019_Vs30_integral.hdf5") as f:
    waveform_hdf = f["data"]["27558"]["vel_lowfreq_traces"]
    print(waveform_hdf[0:])
    print(waveform_hdf[0])
    waveform_hdf = waveform_hdf[13]

waveform_hdf[:,0][0:4000]

fig, ax1 = plt.subplots(3,1, figsize=(10,6))
ax1[0].plot(waveform_hdf[:,0][0:4000])
ax1[1].plot(waveform_hdf[:,1][0:4000])
ax1[2].plot(waveform_hdf[:,2][0:4000])
ax1[0].set_title("hdf5 2018 0206 lowfreq", fontsize=20)