import os
import numpy as np
import pandas as pd

COL_T = ['time']
COL_X = ['V_C', 'I_L']
COL_U = ['V_IN']
COL_Y = ['V_C']


def rlc_loader(dataset, dataset_type="nl", output='V_C', noise_std=0.1, dtype=np.float32, scale=True, n_data=-1):
    filename = f"RLC_data_{dataset}_{dataset_type}.csv"
    df_data = pd.read_csv(os.path.join("data", filename))
    t = np.array(df_data[['time']], dtype=dtype)
    u = np.array(df_data[['V_IN']], dtype=dtype)
    y = np.array(df_data[[output]], dtype=dtype)
    x = np.array(df_data[['V_C', 'I_L']], dtype=dtype)

    if scale:
        u = u/100
        y = y/100
        x = x/[100, 6]

    y += np.random.randn(*y.shape) * noise_std

    if n_data > 0:
        t = t[:n_data, :]
        u = u[:n_data, :]
        y = y[:n_data, :]
        x = x[:n_data, :]
    return t, u, y, x


if __name__ == "__main__":
    t, u, y, x = rlc_loader("train", "lin")
