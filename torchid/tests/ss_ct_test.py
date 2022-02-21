import torch
from torchid.ss.dt.models import NeuralStateUpdate
from torchid.ss.dt.simulator import StateSpaceSimulator


def test_batchfirst():
    """Test batch-first option of ForwardEulerSimulator"""

    L = 100  # sequence length
    N = 64  # batch size
    n_x = 2  # states
    n_u = 3

    f_xu = NeuralStateUpdate(n_x=n_x, n_u=n_u)
    model_tf = StateSpaceSimulator(f_xu)  # batch_first=False, thus time first, default
    model_bf = StateSpaceSimulator(f_xu, batch_first=True)

    x0 = torch.randn(N, n_x)
    u_tf = torch.randn(L, N, n_u)
    u_bf = torch.transpose(u_tf, 0, 1)  # transpose time/batch dimensions

    x_tf = model_tf(x0, u_tf)
    x_bf = model_bf(x0, u_bf)

    assert(torch.allclose(x_bf.transpose(0, 1), x_tf))

