import os
import torch
from torchid.ss.dt.models import PolynomialStateUpdate, LinearStateUpdate, LinearOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import FlippedLSTMStateEstimator
from loader import silverbox_loader
import matplotlib.pyplot as plt
from torchid import metrics


if __name__ == '__main__':

    model_filename = "ss_poly.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    d_max = model_data["d_max"]

    # Load dataset
    t, u, y = silverbox_loader("test", scale=True)

    #%% Load models and parameters
    f_xu = PolynomialStateUpdate(n_x, n_u, d_max)
    g_x = LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    state_estimator = FlippedLSTMStateEstimator(n_u=n_u, n_y=n_y, n_x=n_x)
    model.load_state_dict(model_data["model"])
    #state_estimator.load_state_dict(model_data["estimator"])

    #%% Simulate
    with torch.no_grad():
        u_v = torch.tensor(u[:, None, :])
        # y_v = torch.tensor(y[:, None, :])
        # x0 = state_estimator(u_v, y_v)
        x0 = torch.zeros(1, n_x, dtype=torch.float32)  # initial state set to 0 for simplicity
        y_sim = model(x0, u_v).squeeze(1)  # remove batch dimension
    y_sim = y_sim.detach().numpy()

    #%% Test
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label='sim')

    #%% Metrics
    e_rms = 1000 * metrics.rmse(y, y_sim)[0]
    fit_idx = metrics.fit_index(y, y_sim)[0]
    r_sq = metrics.r_squared(y, y_sim)[0]

    print(f"RMSE: {e_rms:.1f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")



