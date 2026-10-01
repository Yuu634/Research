"""入力目標の解釈とは独立した、Decoderへ渡す難易度の表現。"""
import torch


def difficulty_condition(target, current, mode='relative'):
    if mode not in ('relative', 'absolute'):
        raise ValueError('difficulty_modeはrelativeまたはabsolute')
    if target.shape != current.shape or target.shape[-1] != 5:
        raise ValueError('目標と現在難易度は同形の5属性テンソル')
    if not torch.isfinite(target).all() or not torch.isfinite(current).all():
        raise ValueError('難易度は有限値')
    if ((target < 0) | (target > 1) | (current < 0) | (current > 1)).any():
        raise ValueError('難易度はクラス/7の0〜1')
    value = target - current if mode == 'relative' else target
    # 同じ差分でも出発点が異なるため、D(X_input)も明示して渡す。
    return torch.cat((value, current), dim=-1)
