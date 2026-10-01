import pytest
import torch

from src.common.losses import LossConfig, policy_loss, reconstruction_kl


def test_policy_gradient_lowers_probability_of_worse_action():
    logits = torch.tensor([.0, .0], requires_grad=True)
    logp = logits.log_softmax(0)[0].unsqueeze(0)
    names = ('difficulty', 'skeleton', 'context', 'edit', 'validity')
    better = dict.fromkeys(names, 0.)
    worse = dict(better, difficulty=1.)
    result = policy_loss(logp, [worse], [better])
    result['total'].backward()
    assert logits.grad[0] > 0 and logits.grad[1] < 0


def test_padding_does_not_contribute_and_kl_is_finite():
    logits = torch.randn(3, 1, 8, requires_grad=True)
    output = dict(logits=logits, mu=torch.zeros(1, 4), logvar=torch.zeros(1, 4))
    target = torch.tensor([[2], [1], [0]])
    result = reconstruction_kl(output, target, 0)
    result['total'].backward()
    assert logits.grad[2].abs().sum() == 0
    assert result['kl'] == 0 and torch.isfinite(result['total'])
    with pytest.raises(ValueError):
        reconstruction_kl(output, torch.zeros_like(target), 0)
    with pytest.raises(ValueError):
        LossConfig(difficulty=float('nan')).validate()
