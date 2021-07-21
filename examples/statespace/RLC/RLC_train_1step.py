import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    t_fit = 2e-3  # fitting on t_fit ms of data
    lr = 1e-4  # learning rate
    num_iter = 40000  # gradient-based optimization steps
    test_freq = 500  # print message every test_freq iterations
    add_noise = False

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    # Compute SNR
    P_x = np.mean(x ** 2, axis=0)
    P_n = std_noise**2
    SNR = P_x/(P_n+1e-10)
    SNR_db = 10*np.log10(SNR)

    ts = time_data[1] - time_data[0]
    n_fit = int(t_fit // ts)
    ts_integ = 1.0

    # Fit data to pytorch tensors #
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    u_fit_torch = torch.from_numpy(u_fit)
    x_fit_torch = torch.from_numpy(x_fit)

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = ForwardEulerSimulator(ss_model)

    # Setup optimizer
    optimizer = optim.Adam(nn_solution.ss_model.parameters(), lr=lr)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        DX = x_fit_torch[1:, :] - x_fit_torch[0:-1, :]
        scale_error = torch.sqrt(torch.mean(DX**2, dim=0))

    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Perform one-step ahead prediction
        DX_pred = ts_integ * ss_model(x_fit_torch[0:-1, :], u_fit_torch[0:-1, :])
        DX = x_fit_torch[1:, :] - x_fit_torch[0:-1, :]

        err = DX - DX_pred
        err_scaled = err/scale_error

        # Compute fit loss
        loss = torch.mean(err_scaled**2)

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time  # 114 seconds
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = "model_SS_1step_noise.pt"
    else:
        model_filename = "model_SS_1step_nonoise.pt"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))


    # In[Plot loss]

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = "RLC_SS_loss_1step_noise.pdf"
    else:
        fig_name = "RLC_SS_loss_1step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # In[Simulate model]
    t_val = 5e-3
    n_val = int(t_val // ts)  # x.shape[0]

    u_val = u[0:n_val]
    x_val = x[0:n_val]

    x0_val = np.zeros(2, dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(u_val)
    x_true_torch_val = torch.from_numpy(x_val)

    time_start = time.time()
    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)

    x_sim = np.array(x_sim_torch_val)
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(x_val[:, 0], 'k+', label='True')
    ax[0].plot(x_sim[:, 0], 'r', label='Sim')
    ax[0].legend()
    ax[1].plot(x_val[:, 1], 'k+', label='True')
    ax[1].plot(x_sim[:, 1], 'r', label='Sim')
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)

