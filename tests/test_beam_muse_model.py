from dataclasses import replace
import pytest
import torch

from src.beam_muse.conditioning import difficulty_condition
from src.beam_muse.features import FEATURE_SIZE
from src.beam_muse.model import BeamMuseMorphose
from src.musemorphose.conditional import ModelConfig


def tiny_model_batch(mode='in-attn'):
    torch.manual_seed(7)
    config = ModelConfig(d_model=16, d_latent=8, n_head=2, enc_layers=1, dec_layers=2,
                         d_ff=32, d_condition=8, dropout=0, max_tokens=32, cond_mode=mode)
    model = BeamMuseMorphose(12, 0, 2, config)
    batch = dict(encoder_tokens=torch.tensor([[[2, 1, 0, 0], [3, 2, 5, 1], [3, 2, 6, 1]]]),
                 context_present=torch.tensor([[0., 1.]]), target=torch.full((1, 5), .25),
                 current=torch.full((1, 5), .75), skeleton=torch.zeros(1, FEATURE_SIZE),
                 decoder_input=torch.tensor([[3], [2], [5], [6]]))
    return model, batch


@pytest.mark.parametrize('mode', ['in-attn', 'pre-attn'])
def test_condition_changes_decoder_and_backward(mode):
    model, batch = tiny_model_batch(mode)
    a = model(batch, sample=False)
    b = model(dict(batch, skeleton=torch.ones_like(batch['skeleton'])), sample=False)
    c = model(dict(batch, target=torch.ones_like(batch['target'])), sample=False)
    assert not torch.allclose(a['logits'], b['logits'])
    assert not torch.allclose(a['logits'], c['logits'])
    a['logits'].square().mean().backward()
    assert model.skeleton_embedding[0].weight.grad is not None
    assert model.difficulty_embedding[0].weight.grad.abs().sum() > 0


def test_decoder_causality_and_padding_invariance():
    model, batch = tiny_model_batch()
    model.eval()
    a = model(batch)['logits']
    tokens = batch['decoder_input'].clone()
    tokens[-1] = 9
    b = model(dict(batch, decoder_input=tokens))['logits']
    torch.testing.assert_close(a[:-1], b[:-1])
    padded = torch.nn.functional.pad(batch['encoder_tokens'], (0, 3))
    torch.testing.assert_close(a, model(dict(batch, encoder_tokens=padded))['logits'])


def test_relative_preserves_sign_and_absolute_is_target():
    target = torch.tensor([[0., .3, .5, 1., .2]])
    current = torch.tensor([[1., .1, .5, 0., .9]])
    relative = difficulty_condition(target, current, 'relative')
    absolute = difficulty_condition(target, current, 'absolute')
    torch.testing.assert_close(relative[:, :5], target-current)
    torch.testing.assert_close(absolute[:, :5], target)
    torch.testing.assert_close(relative[:, 5:], current)
    with pytest.raises(ValueError):
        difficulty_condition(target+2, current)
    with pytest.raises(ValueError):
        difficulty_condition(target, current, 'delta_ratio')


def test_strict_checkpoint_round_trip():
    model, batch = tiny_model_batch()
    restored = BeamMuseMorphose(12, 0, 2, model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(batch, sample=False)['logits'], restored(batch, sample=False)['logits'])
