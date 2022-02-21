import torch
from torchid.ss.ct.ssmodels_ct import NeuralStateSpaceModel
from torchid.ss.ct.ss_simulator_ct import ForwardEulerSimulator


def test_batchfirst():
    """Test batch-first option of ForwardEulerSimulator"""

    L = 100  # sequence length
    N = 64  # batch size
    n_x = 2  # states
    n_u = 3

    ss_model = NeuralStateSpaceModel(n_x=n_x, n_u=n_u)
    nn_solution = ForwardEulerSimulator(ss_model)  # batch_first=False, thus time_first, default
    nn_solution_bf = ForwardEulerSimulator(ss_model, batch_first=True)

    x0 = torch.randn(N, n_x)
    u_tf = torch.randn(L, N, n_u)
    u_bf = torch.transpose(u_tf, 0, 1)

    x_tf = nn_solution(x0, u_tf)
    x_bf = nn_solution_bf(x0, u_bf)

    assert(torch.allclose(x_bf.transpose(0, 1), x_tf))

