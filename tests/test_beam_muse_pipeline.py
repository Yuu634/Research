from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.arrangement.beam_search import Action, SearchConfig
from src.arrangement.evaluation import EvaluationConfig
from src.arrangement.examples import make_bar
from src.arrangement.inputs import bar_to_payload, prepare_input
from src.arrangement.representation import to_tokens
from src.arrangement.scoring import ScoringModel
from src.beam_muse.data import Example, collate, source_hashes
from src.beam_muse.evaluation import context_distance, evaluate_output, sequence_costs, summarize_outputs
from src.beam_muse.experiment import build_model, expand_trials, load_checkpoint, train_trial, run_experiment
from src.beam_muse.generation import Grammar, generate, score_generation
from src.data.drum_difficulty_v2 import ATTRS
from src.musemorphose.conditional import ModelConfig


@pytest.fixture
def example():
    torch.set_num_threads(1)
    original = make_bar([(0, 'kick'), (24, 'snare'), (72, 'snare', 'Accent')]+
                        [(p, 'hh_closed') for p in range(0, 96, 12)])
    teacher = original.with_notes(n for n in original.notes if n.instrument != 'hh_closed' or n.position % 24 == 0)
    strings = ['<PAD>', '<EOS>', '<BAR>', '<TEMPO_120>'] + [f'<BEAT_{i}>' for i in range(1, 5)] + [f'<POS_{i}>' for i in range(24)]
    strings += sorted({n.token() for n in original.notes})
    vocabulary = dict(token2idx={t:i for i,t in enumerate(strings)}, idx2token=dict(enumerate(strings)), tokenization_method='standard')
    scorer = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, strings)
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=bar_to_payload(original),
                                  target=dict(mode='absolute', values=scorer.difficulty(teacher, original).normalized)), scorer)
    value = Example('case', 'song', 'train', prepared, teacher, teacher, [], .1)
    return value, vocabulary, scorer


def config():
    return ModelConfig(d_model=16, d_latent=8, n_head=2, enc_layers=1, dec_layers=1,
                       d_ff=32, d_condition=8, dropout=0., max_tokens=64)


def test_teacher_changes_only_decoder_training_input(example):
    e, vocab, scorer = example
    a = collate([e], vocab, scorer, 'beam_muse')
    b = collate([replace(e, teacher=e.prepared.original)], vocab, scorer, 'beam_muse')
    for key in ('encoder_tokens', 'target', 'current', 'skeleton', 'context_present'):
        torch.testing.assert_close(a[key], b[key])
    assert a['decoder_input'].shape != b['decoder_input'].shape
    inference = collate([replace(e, teacher=None)], vocab, scorer, 'beam_muse', training=False)
    torch.testing.assert_close(a['encoder_tokens'], inference['encoder_tokens'])
    direct = collate([e], vocab, scorer, 'musemorphose')
    assert direct['encoder_tokens'].shape != a['encoder_tokens'].shape or not torch.equal(direct['encoder_tokens'], a['encoder_tokens'])


def test_sample_log_probability_recomputation_and_gradient(example):
    e, vocab, scorer = example
    torch.manual_seed(3)
    model = build_model('beam_muse', config(), vocab).eval()
    batch = collate([e], vocab, scorer, 'beam_muse', training=False)
    with torch.no_grad():
        generated = generate(model, batch, e.prepared, vocab, sample=True)
    logp = score_generation(model, batch, generated, vocab)
    torch.testing.assert_close(logp, generated.log_probability, atol=1e-4, rtol=1e-5)
    logp.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.encoder.fc_mu.weight.grad.abs().sum() > 0
    assert model.skeleton_embedding[0].weight.grad.abs().sum() > 0


def test_grammar_forbids_duplicate_instrument_and_more_than_two_hands(example):
    e, vocab, scorer = example
    grammar = Grammar(vocab)
    ids = vocab['token2idx']
    prefix = [ids[t] for t in ['<TEMPO_120>', '<BAR>', '<BEAT_1>', '<POS_0>', 'HH_CLOSED_HIT_Normal', 'SNARE_HIT_Normal']]
    allowed = grammar.allowed(prefix)
    assert ids['KICK_HIT_Normal'] in allowed
    assert not any(i in grammar.notes and grammar.notes[i] != 'kick' for i in allowed)
    assert ids['<PAD>'] not in allowed and ids['<BAR>'] not in allowed


def test_failures_stay_in_denominator_and_context_is_masked(example):
    e, _, scorer = example
    good = evaluate_output(e.prepared, e.prepared.original, scorer, case_id='a', source_id='song')
    bad = evaluate_output(e.prepared, None, scorer, case_id='b', source_id='song', failure_reason='max_tokens')
    summary = summarize_outputs([good, bad])
    assert summary['n'] == 2 and summary['valid_rate']['value'] == .5
    assert bad['failure_adjusted_mae'] == 1
    assert context_distance(e.prepared, e.prepared.original) == 0
    assert context_distance(replace(e.prepared, previous=None), e.prepared.original) is None
    assert sequence_costs(e.prepared, None, scorer)['validity'] == 1


def test_trial_train_checkpoint_and_completed_resume(example, tmp_path):
    e, vocab, scorer = example
    experiment = json.loads(Path('config/beam_muse_candidates.json').read_text())
    trial = expand_trials(experiment)[2]
    trial['model'] = asdict(config())
    trial['training'].update(epochs=1, batch_size=1, risk_interval=1)
    identity = dict(sources=source_hashes(), smoke=True)
    completed = train_trial(trial, [e], [replace(e, split='val')], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu')
    path = tmp_path/'best.pt'
    modified = path.stat().st_mtime_ns
    model, checkpoint = load_checkpoint(path, vocab, identity)
    assert checkpoint['epoch'] == 1 and completed['validation']['n'] == 1
    assert train_trial(trial, [e], [e], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu') == completed
    assert path.stat().st_mtime_ns == modified
    changed = dict(trial, training=dict(trial['training'], seed=99))
    with pytest.raises(ValueError, match='変更'):
        train_trial(changed, [e], [e], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu')


def test_configuration_requires_both_neural_variants_and_common_contract():
    value = json.loads(Path('config/beam_muse_candidates.json').read_text())
    trials = expand_trials(value)
    assert len(trials) == 6
    assert {t['model']['difficulty_mode'] for t in trials} == {'absolute', 'relative'}
    value['trials'] = value['trials'][2:]
    with pytest.raises(ValueError):
        expand_trials(value)


def test_holdout_only_runs_after_frozen_selection(example, tmp_path, monkeypatch):
    e, vocab, scorer = example
    configuration = json.loads(Path('config/beam_muse_candidates.json').read_text())
    configuration['model'] = asdict(config())
    configuration['training'].update(epochs=1, batch_size=1, risk_interval=1)
    configuration['trials'] = [configuration['trials'][0], configuration['trials'][2]]
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'manifest.json').write_text('{}')
    manifest = dict(identity=dict(smoke_limit=None, max_tokens=64, vocabulary='v', bounds='b', beam_model='m'))
    seen = []
    def fake_cache(root, split, *args):
        seen.append(split)
        return [replace(e, split=split, case_id=split, source_id=split)], manifest
    monkeypatch.setattr('src.beam_muse.experiment.load_cache', fake_cache)
    monkeypatch.setattr('src.beam_muse.experiment.load_metadata', lambda *args: (scorer, None, None, EvaluationConfig(), None, vocab))
    output = tmp_path/'run'
    with pytest.raises(OSError):
        run_experiment(configuration, cache, 'meta', 'beam', output, 'cpu', phase='test')
    seen.clear()
    selected = run_experiment(configuration, cache, 'meta', 'beam', output, 'cpu')
    assert set(seen) == {'train', 'val'}
    weights = {p: p.stat().st_mtime_ns for p in output.glob('*/best.pt')}
    run_experiment(configuration, cache, 'meta', 'beam', output, 'cpu', phase='test')
    assert {'test', 'itest'} <= set(seen)
    assert all(p.stat().st_mtime_ns == modified for p, modified in weights.items())
    for split in ('test', 'itest'):
        summary = json.loads((output/f'{split}_comparison.json').read_text())
        assert set(summary) == {'musemorphose', 'beamsearch', 'beam_muse'}
        assert {s['n'] for s in summary.values()} == {1}
