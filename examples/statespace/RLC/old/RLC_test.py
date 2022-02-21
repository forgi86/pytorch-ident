import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchid.ss.ct.ssmodels_ct import NeuralStateSpaceModel
from torchid.ss.ct.ss_simulator_ct import ForwardEulerSimulator
import torchid.metrics as metrics

if __name__ == '__main__':

    matplotlib.rc('text', usetex=True)

    plot_input = False

    dataset_type = 'test'
    #dataset_type = 'id'

    model_type = '64step_noise'
    #model_type = 'fullsim_noise'
    #model_type = '1step_noise'
    #model_type = '1step_nonoise'
    #model_type = 'soft_noise'

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    dataset_filename = f"RLC_data_{dataset_type}.csv"
    df_X = pd.read_csv(os.path.join("data", dataset_filename))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current
    y = np.copy(x[:, [y_var_idx]])
    N = np.shape(y)[0]
    ts = time_data[1, 0] - time_data[0, 0]
    ts_integ = 1.0

    # Add measurement noise
    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Scale dataset
    #scale_vector = np.array([100.0, 10.0]).astype(np.float32)
    #x = x/scale_vector
    #x_noise = x_noise/scale_vector

    # Build validation data
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start // ts)
    idx_val_end = int(t_val_end // ts)
    u_val = u[idx_val_start:idx_val_end]
    x_meas_val = x_noise[idx_val_start:idx_val_end]
    x_true_val = x[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]

    # Setup neural model structure and load fitted model parameters
    #scale_dx =  800000.0
    #scale_dx = 100
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)#, scale_dx=scale_dx)
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts_integ)
    model_filename = f"model_SS_{model_type}.pt"
    nn_solution.f_xu.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Evaluate the model in open-loop simulation against validation data
    x_0 = x_meas_val[0, :]
    with torch.no_grad():
        x_sim_torch = nn_solution(torch.tensor(x_0), torch.tensor(u_val))
        loss = torch.mean(torch.abs(x_sim_torch - torch.tensor(x_true_val)))

    # Plot results
    x_sim = np.array(x_sim_torch)
    if not plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))
    else:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    time_val_us = time_val*1e6

    if dataset_type == 'id':
        t_plot_start = 0.0e-3#0.2e-3
    else:
        t_plot_start = 0.0e-3#1.9e-3
    t_plot_end = t_plot_start + 1.0#0.32e-3

    idx_plot_start = int(t_plot_start // ts)
    idx_plot_end = int(t_plot_end // ts)

    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_true_val[idx_plot_start:idx_plot_end,0], 'k',  label='$v_C$')
    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end,0],'r--', label='$\hat{v}^{\mathrm{sim}}_C$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time ($\mu$s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_ylim([-300, 300])

    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], np.array(x_true_val[idx_plot_start:idx_plot_end:,1]), 'k', label='$i_L$')
    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end:,1],'r--', label='$\hat i_L^{\mathrm{sim}}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time ($\mu$s)")
    ax[1].set_ylabel("Current (A)")
    ax[1].set_ylim([-25, 25])

    if plot_input:
        ax[2].plot(time_val_us[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end], 'k')
        #ax[2].legend(loc='upper right')
        ax[2].grid(True)
        ax[2].set_xlabel("Time ($\mu$s)")
        ax[2].set_ylabel("Input voltage $v_C$ (V)")
        ax[2].set_ylim([-400, 400])

    fig_name = f"RLC_SS_{dataset_type}_{model_type}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # R-squared metrics
    R_sq = metrics.r_squared(x_true_val, x_sim)
    print(f"R-squared metrics: {R_sq}")
