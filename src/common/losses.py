"""再構成＋KLと、離散指標に対する自己臨界方策勾配。"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    beta: float = .01
    free_bits: float = .0
    difficulty: float = 1.
    skeleton: float = .5
    context: float = .2
    edit: float = .1
    validity: float = 1.

    def validate(self):
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError('損失係数は有限・非負')
        return self


def reconstruction_kl(output, target, pad_id, config=LossConfig()):
    config.validate()
    logits, mu, logvar = output['logits'], output['mu'], output['logvar']
    if logits.shape[:2] != target.shape or not target.ne(pad_id).any():
        raise ValueError('再構成損失の形状または有効トークン数が不正')
    rec = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad_id)
    per_dimension = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean(dim=0)
    kl = per_dimension.clamp(min=config.free_bits).mean()
    return dict(total=rec + config.beta*kl, reconstruction=rec, kl=kl, kl_raw=per_dimension.mean())


def policy_loss(log_probabilities, sampled_costs, greedy_costs, config=LossConfig()):
    """E[cost]の勾配推定。baselineは同条件のgreedy生成から得てdetachする。

    最小化なので+(cost_sample-cost_greedy)*log p(sample)となる。
    この推定量の値そのものは負にもなり、生成品質指標として使わない。
    """
    config.validate()
    if not sampled_costs or len(sampled_costs) != len(greedy_costs) or len(sampled_costs) != len(log_probabilities):
        raise ValueError('sample・greedy・log probabilityの件数が不一致')
    parts = {}
    for name in ('difficulty', 'skeleton', 'context', 'edit', 'validity'):
        advantage = log_probabilities.new_tensor([a[name]-b[name] for a, b in zip(sampled_costs, greedy_costs)])
        if not torch.isfinite(advantage).all():
            raise ValueError('系列コストは有限値')
        parts[name] = (advantage.detach()*log_probabilities).mean()
    parts['total'] = sum(getattr(config, name)*value for name, value in parts.items())
    return parts
