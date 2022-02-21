import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator


# Soft-constrained integration method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 50000  # gradient-based optimization steps
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 1e1  # fit/consistency trade-off constant
    lr = 5e-4  # learning rate
    test_freq = 100  # print message every test_freq iterations
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    #df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    # Compute SNR
    P_x = np.mean(x ** 2, axis=0)
    P_n = std_noise**2
    SNR = P_x/(P_n+1e-10)
    SNR_db = 10*np.log10(SNR)

    # Get fit data #
    Ts = t[1] - t[0]
    n_fit = int(t_fit // Ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    x_fit_nonoise = x[0:n_fit] # not used, just for reference
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    time_torch_fit = torch.from_numpy(time_fit[:, 0])
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)
    x_hidden_init = x_fit + 0*np.random.randn(*x_fit.shape)*std_noise
    x_hidden_init = x_hidden_init.astype(np.float32)
    x_hidden_fit = torch.tensor(x_hidden_init, requires_grad=True)  # hidden state is an optimization variable

    ts_integ = 1.0 # better for numerical reasons
    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64, activation='relu')
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts_integ)
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_SS_64step_noise.pkl")))

    # Setup optimizer
    params_net = list(ss_model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr)

    # Scale loss with respect to the initial one
    scale_error = torch.tensor([20.0, 1.0]).float()


    LOSS = []
    LOSS_SIM = []
    start_time = time.time()
    # Training loop

    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Compute fit loss
        x_hidden = x_hidden_fit[:]
        err_fit = x_hidden - x_meas_torch_fit
        err_fit_scaled = err_fit/scale_error
        loss_fit = torch.mean(err_fit_scaled**2)

        # Compute consistency loss
        DX = ts_integ*ss_model(x_hidden[0:-1, :], u_torch_fit[0:-1, :])
        err_consistency = x_hidden[1:, :] - x_hidden[0:-1, :] - DX
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                x0_torch_fit = x_hidden_fit[0, :]
                x_sim_torch_fit = nn_solution(x0_torch_fit[None, :], torch.tensor(u_fit)[:, None, :])
                x_sim_torch_fit = x_sim_torch_fit.squeeze(1)
                err_sim_torch_fit = x_sim_torch_fit - torch.tensor(x_fit)
                loss_sim = torch.sqrt(torch.mean(err_sim_torch_fit**2))
                LOSS_SIM.append(loss_sim.item())
                print(f'Iter {itr} | Tradeoff Loss {loss:.6f} '
                      f'Consistency Loss {loss_consistency:.6f} Fit Loss {loss_fit:.6f} Sim Loss {loss_sim:.6f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = f"model_SS_soft_noise.pt"
    else:
        model_filename = f"model_SS_soft_nonoise.pt"

    torch.save(ss_model.state_dict(), os.path.join("models", model_filename))

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    # In[Val]
    t_val = 5e-3
    n_val = int(t_val // Ts)  # x.shape[0]

    time_val = t[0:n_val]
    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = np.zeros(2, dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)
    time_torch_val = torch.from_numpy(time_val[:, 0])

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:, 0]), label='True')
    ax[0].plot(np.array(x_sim_torch_val[:, 0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:, 1]), label='True')
    ax[1].plot(np.array(x_sim_torch_val[:, 1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)

    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_fit_nonoise[:, 0], 'k', label='True')
    ax[0].plot(x_fit[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(x_fit_nonoise[:, 1], 'k', label='True')
    ax[1].plot(x_fit[:, 1], 'b', label='Measured')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)
