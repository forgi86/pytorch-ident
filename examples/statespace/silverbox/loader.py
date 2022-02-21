import os
import numpy as np
import pandas as pd

# Column names in the dataset
COL_U = ['V1']
COL_Y = ['V2']
idx_train = 40700


def silverbox_loader(dataset, scale=True, dtype=np.float32):
    filename = "SNLS80mV.csv"
    df_data = pd.read_csv(os.path.join("SilverboxFiles", filename))
    y = np.array(df_data[COL_Y], dtype=dtype)
    u = np.array(df_data[COL_U], dtype=dtype)
    fs = 10**7/2**14
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    if scale:
        u_train = u[idx_train:]
        y_train = y[idx_train:]
        u_mean, u_std = np.mean(u_train), np.std(u_train)
        y_mean, y_std = np.mean(y_train), np.std(y_train)
        u = (u-u_mean)/u_std
        y = (y-y_mean)/y_std

    if dataset == "full":
        return t, u, y
    elif dataset == "train":
        t_train = t[idx_train:] - t[idx_train]
        u_train = u[idx_train:]
        y_train = y[idx_train:]
        return t_train, u_train, y_train
    elif dataset == "test":
        t_test = t[:idx_train]
        u_test = u[:idx_train]
        y_test = y[:idx_train]
        return t_test, u_test, y_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for dataset in ["full", "train", "test"]:
        t, u, y = silverbox_loader(dataset)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t, u)
        ax[1].plot(t, y)
        plt.suptitle(dataset)
