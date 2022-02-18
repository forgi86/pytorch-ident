import numpy as np
import matplotlib.pyplot as plt
import os

R_val = 3
L_val = 50e-6
C_val = 270e-9
Td_val = 1e-6


def saturation_formula(current_abs):
    sat_ratio = (1 / np.pi * np.arctan(-5 * (current_abs - 5)) + 0.5) * 0.9 + 0.1
    return sat_ratio


def fxu_ODE(t, x, u):
    A = np.array([[0.0, 1.0 / C_val],
                  [-1 / (L_val), -R_val / L_val]
                  ])
    B = np.array([[0.0], [1.0 / (L_val)]])
    dx = np.zeros(2, dtype=np.float64)
    dx[0] = A[0, 0] * x[0] + A[0, 1] * x[1] + B[0, 0] * u[0]
    dx[1] = A[1, 0] * x[0] + A[1, 1] * x[1] + B[1, 0] * u[0]
    return dx


def fxu_ODE_nl(t, x, u):
    I_abs = np.abs(x[1])
    L_val_mod = L_val * saturation_formula(I_abs)
    R_val_mod = R_val
    C_val_mod = C_val

    A = np.array([[0.0, 1.0 / C_val_mod],
                  [-1 / (L_val_mod), -R_val_mod / L_val_mod]
                  ])
    B = np.array([[0.0], [1.0 / (L_val_mod)]])

    dx = np.zeros(2, dtype=np.float64)
    dx[0] = A[0, 0] * x[0] + A[0, 1] * x[1] + B[0, 0] * u[0]
    dx[1] = A[1, 0] * x[0] + A[1, 1] * x[1] + B[1, 0] * u[0]
    return dx


A_nominal = np.array([[0.0, 1.0 / C_val],
                      [-1 / L_val, -R_val / L_val]
                      ])

B_nominal = np.array([[0.0], [1.0 / L_val]])

if __name__ == '__main__':

    x = np.zeros(2)
    u = np.zeros(1)
    dx = fxu_ODE_nl(0.0, x, u)

    I = np.arange(0., 20., 0.1)

    # Save model
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 3))
    ax.plot(I, L_val * 1e6 * saturation_formula(I), 'k')
    ax.grid(True)
    ax.set_xlabel('Inductor current $i_L$ (A)', fontsize=14)
    ax.set_ylabel('Inductance $L$ ($\mu$H)', fontsize=14)
    fig.savefig(os.path.join("fig", "RLC_characteristics.pdf"), bbox_inches='tight')
