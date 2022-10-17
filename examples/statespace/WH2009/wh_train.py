import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import wh2009_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':

    save_folder = "models"

    epochs_adam = 100
    epochs_bfgs = 5
    epochs_lin = 20
    batch_size = 1024
    seq_len = 80
    seq_est_len = 50
    est_hidden_size = 16
    hidden_size = 16
    lr = 1e-3

    no_cuda = False
    log_interval = 20

    torch.manual_seed(10)

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Constants
    n_x = 6
    n_u = 1
    n_y = 1
    n_fit = 80000

    epochs = epochs_adam + epochs_bfgs

    # %% Load dataset
    t_train, u_train, y_train = wh2009_loader("train", scale=True)
    t_fit, u_fit, y_fit = t_train[:n_fit], u_train[:n_fit], y_train[:n_fit]
    t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

    # %%  Prepare dataset, models, optimizer
    train_data = SubsequenceDataset(u_fit, y_fit, subseq_len=seq_len + seq_est_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    u_val_t = torch.tensor(u_val[:, None, :]).to(device)
    y_val_t = torch.tensor(y_val[:, None, :]).to(device)

    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=hidden_size).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    estimator = estimators.FeedForwardStateEstimator(n_u=n_u, n_y=n_y, n_x=n_x,
                                                     hidden_size=est_hidden_size,
                                                     seq_len=seq_est_len).to(device)

    # Setup optimizer
    optimizer = optim.Adam(list(model.parameters())+list(estimator.parameters()), lr=lr)

    optimizer_LBFGS = torch.optim.LBFGS(
        list(model.parameters())+list(estimator.parameters()),
        line_search_fn="strong_wolfe",
        lr=1.0
    )

    # %% Other initializations
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model_filename = "model.pt"
    model_path = os.path.join(save_folder, model_filename)

    VAL_LOSS, TRAIN_LOSS = [], []
    min_loss = np.inf  # for early stopping


    def closure():
        optimizer_LBFGS.zero_grad()

        # State is estimated on the first seq_est_len samples
        batch_u_est = batch_u[:seq_est_len]
        batch_y_est = batch_y[:seq_est_len]
        batch_x0 = estimator(batch_u_est, batch_y_est)

        # fit only after seq_est_len
        batch_u_fit = batch_u[seq_est_len:]
        batch_y_fit = batch_y[seq_est_len:]
        batch_y_sim = model(batch_x0, batch_u_fit)

        # Compute fit loss
        loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)
        loss.backward()
        return loss

    start_time = time.time()
    # %% Training loop
    itr = 0
    model.f_xu.disable_nl()
    model.f_xu.freeze_nl()
    model.g_x.disable_nl()
    model.g_x.freeze_nl()
    for epoch in range(epochs):
        train_loss = 0  # train loss for the whole epoch
        model.train()
        estimator.train()

        if epoch == epochs_lin:
            model.f_xu.enable_nl()
            model.f_xu.unfreeze_nl()
            model.f_xu.freeze_lin()
            model.g_x.enable_nl()
            model.g_x.unfreeze_nl()
            model.g_x.freeze_lin()

        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):

            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first

            if epoch < epochs_adam:
                loss = optimizer.step(closure)
            else:
                loss = optimizer_LBFGS.step(closure)

            train_loss += loss.item()

            itr += 1

        train_loss = train_loss / len(train_loader)
        TRAIN_LOSS.append(train_loss)

        # Validation loss: full simulation error
        with torch.no_grad():
            model.eval()
            estimator.eval()
            x0 = torch.zeros((1, n_x), dtype=u_val_t.dtype,
                             device=u_val_t.device)
            # x0 = state_estimator(u_val_t, y_val_t)
            y_val_sim = model(x0, u_val_t)
            val_loss = torch.nn.functional.mse_loss(y_val_t, y_val_sim)

        VAL_LOSS.append(val_loss.item())
        print(f'==== Epoch {epoch} | Train Loss {train_loss:.4f} Val (sim) Loss {val_loss:.4f} ====')

        # best model so far, save it
        if val_loss < min_loss:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "estimator": estimator.state_dict()
            },
                os.path.join(model_path)
            )
            min_loss = val_loss.item()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    if not np.isfinite(min_loss):  # model never saved as it was never giving a finite simulation loss
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "estimator": estimator.state_dict()
        },
            os.path.join(model_path)
        )
    # %% Simulate

    # Also save total training time (up to last epoch)
    model_data = torch.load(model_path)
    model_data["total_time"] = time.time() - start_time
    torch.save(model_data, model_path)

    # Reload optimal parameters (best on validation)
    model.load_state_dict(model_data["model"])
    estimator.load_state_dict(model_data["estimator"])

    t_full, u_full, y_full = wh2009_loader("full", scale=True)
    with torch.no_grad():
        model.eval()
        estimator.eval()
        u_v = torch.tensor(u_full[:, None, :]).to(device)
        y_v = torch.tensor(y_full[:, None, :]).to(device)
        x0 = torch.zeros((1, n_x), dtype=u_v.dtype, device=u_v.device)
        y_sim = model(x0, u_v).squeeze(1).to("cpu").detach().numpy()

    # %% Metrics

    from torchid import metrics
    e_rms = 1000 * metrics.rmse(y_full, y_sim)[0]
    fit_idx = metrics.fit_index(y_full, y_sim)[0]
    r_sq = metrics.r_squared(y_full, y_sim)[0]

    print(f"RMSE: {e_rms:.1f} mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

    # %% Plots

    fig, ax = plt.subplots(1, 1)
    ax.plot(TRAIN_LOSS, 'k', label='TRAIN')
    ax.plot(VAL_LOSS, 'r', label='VAL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_full[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label='sim')
