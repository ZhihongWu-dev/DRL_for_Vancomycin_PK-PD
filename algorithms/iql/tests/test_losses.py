import torch
from algorithms.iql.losses import expectile_loss


def test_expectile_loss_symmetry():
    x_pos = torch.tensor([1.0, 2.0, 3.0])
    x_neg = torch.tensor([-1.0, -2.0, -3.0])
    # expectile with tau=0.5 reduces to MSE
    l_pos = expectile_loss(x_pos, 0.5)
    l_neg = expectile_loss(x_neg, 0.5)
    assert torch.isclose(l_pos, l_neg)


def test_expectile_prefers_positive():
    diff = torch.tensor([1.0, -0.5])
    l1 = expectile_loss(diff, 0.7)
    l2 = expectile_loss(diff, 0.3)
    assert l1 != l2
