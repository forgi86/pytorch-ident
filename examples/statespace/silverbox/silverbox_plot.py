import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Column names in the dataset
    COL_U = ['V1']
    COL_Y = ['V2']

    # Load dataset
    df_X = pd.read_csv(os.path.join("SilverboxFiles", "SNLS80mV.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    u = u-np.mean(u)
    y = y-np.mean(y)
    fs = 10**7/2**14
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    #%% Plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y, 'k', label="$u")
    ax[0].grid()
    ax[1].plot(t, u, 'k', label="$y$")
    ax[1].grid()
    plt.legend()
    plt.show()





