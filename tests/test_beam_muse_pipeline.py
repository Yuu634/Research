from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.beamsearch.beam_search import Action, SearchConfig
from src.common.evaluation import EvaluationConfig
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.representation import to_tokens
from src.beamsearch.scoring import ScoringModel
from src.beam_muse.configuration import load_experiment_config
from src.beam_muse.data import Example, collate, source_hashes
from src.common.comparison import context_distance, evaluate_output, sequence_costs, summarize_outputs
from src.beam_muse.experiment import build_model, expand_trials, load_checkpoint, train_trial, run_experiment
from src.musemorphose.generation import Grammar, generate, score_generation
from src.data.drum_difficulty_v2 import ATTRS
from src.musemorphose.conditional import ModelConfig


def comparison_config():
    # 実行用configで選んだ候補に依存せず、比較経路を検証する。
    value = load_experiment_config('config/beam_muse_candidates.json')
    value['trials'] = [
        dict(name=f'{prefix}_lr{rate}', variant=variant,
             model=dict(use_skeleton=variant == 'beam_muse', difficulty_mode=mode),
             training=dict(learning_rate=rate*0.0001))
        for prefix, variant, mode in (
            ('muse_absolute', 'musemorphose', 'absolute'),
            ('hybrid_relative', 'beam_muse', 'relative'),
            ('hybrid_absolute', 'beam_muse', 'absolute'))
        for rate in (2, 1)]
    return value


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
    experiment = comparison_config()
    trial = expand_trials(experiment)[2]
    trial['model'] = asdict(config())
    trial['training'].update(epochs=1, batch_size=1, risk_interval=1)
    identity = dict(sources=source_hashes(), smoke=True)
    completed = train_trial(trial, [e], [replace(e, split='val')], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu')
    path = tmp_path/'best.pt'
    modified = path.stat().st_mtime_ns
    model, checkpoint = load_checkpoint(path, vocab, identity)
    assert checkpoint['epoch'] == 1 and completed['validation']['n'] == 1
    train_log = (tmp_path/'train_loss.log').read_text()
    entries = [json.loads(line) for line in train_log.splitlines()]
    assert [r['event'] for r in entries] == ['base_loss', 'beamsearch_score', 'auxiliary_loss']
    base, beam, auxiliary = entries
    assert base['step'] == 1 and base['case_ids'] == [e.case_id]
    assert base['musemorphose']['total'] == pytest.approx(
        base['musemorphose']['reconstruction'] + base['beta']*base['musemorphose']['kl'])
    expected = scorer.score(e.prepared.original, e.intermediate, e.prepared.target,
                            previous=e.prepared.previous, guards=e.prepared.guards, rule_count=len(e.history))
    assert beam['beamsearch']['total'] == expected.total
    assert auxiliary['auxiliary']['scaled_total'] == pytest.approx(
        auxiliary['auxiliary']['parts']['total']*auxiliary['auxiliary']['scale'])
    evaluation = [json.loads(line) for line in (tmp_path/'eval_loss.log').read_text().splitlines()]
    assert [r['event'] for r in evaluation] == ['evaluation_summary']
    assert evaluation[0]['n'] == 1
    assert evaluation[0]['evaluation'] == completed['validation']
    assert evaluation[0]['beamsearch']['total'] == expected.total
    assert all(r['phase'] == 'val' for r in evaluation)
    assert json.loads((tmp_path/'config.json').read_text())['trial'] == trial
    assert (tmp_path/'loss_curve.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert train_trial(trial, [e], [e], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu') == completed
    assert (tmp_path/'train_loss.log').read_text() == train_log
    assert path.stat().st_mtime_ns == modified
    changed = dict(trial, training=dict(trial['training'], seed=99))
    with pytest.raises(ValueError, match='変更'):
        train_trial(changed, [e], [e], vocab, scorer, EvaluationConfig(), identity, tmp_path, 'cpu')


def test_configuration_accepts_single_variant_but_requires_nonempty_valid_trials():
    value = comparison_config()
    trials = expand_trials(value)
    assert len(trials) == 6
    assert {t['model']['difficulty_mode'] for t in trials} == {'absolute', 'relative'}
    value['trials'] = value['trials'][2:]
    assert {t['variant'] for t in expand_trials(value)} == {'beam_muse'}
    value['trials'] = []
    with pytest.raises(ValueError):
        expand_trials(value)


@pytest.mark.parametrize('variants', [('musemorphose', 'beam_muse'), ('musemorphose',), ('beam_muse',)])
def test_holdout_only_runs_after_frozen_selection(example, tmp_path, monkeypatch, variants):
    e, vocab, scorer = example
    configuration = comparison_config()
    configuration['model'] = asdict(config())
    configuration['training'].update(epochs=1, batch_size=1, risk_interval=1)
    configuration['trials'] = [configuration['trials'][0], configuration['trials'][2]]
    configuration['trials'] = [t for t in configuration['trials'] if t['variant'] in variants]
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'manifest.json').write_text('{}')
    manifest = dict(identity=dict(smoke_limit=None, max_tokens=64, vocabulary='v', bounds='b', beam_model='m'))
    if variants == ('musemorphose',):
        manifest['identity']['intermediate_mode'] = 'original'
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
    # 学習後に検証済みの評価実装へ更新しても、学習時のidentityでholdoutを再評価できる。
    from src.common.provenance import validate_sources
    before = source_hashes()
    after = dict(before, **{'src/common/evaluation.py': 'reviewed-evaluation-update'})
    migration = tmp_path/'migration.json'
    migration.write_text(json.dumps(dict(schema_version=1, snapshots=[dict(before=before, after=after)])))
    monkeypatch.setattr('src.beam_muse.experiment.source_hashes', lambda: after)
    monkeypatch.setattr('src.beam_muse.experiment.validate_sources',
                        lambda old, new: validate_sources(old, new, migration_path=migration))
    run_experiment(configuration, cache, 'meta', 'beam', output, 'cpu', phase='test')
    assert {'test', 'itest'} <= set(seen)
    assert all(p.stat().st_mtime_ns == modified for p, modified in weights.items())
    for split in ('test', 'itest'):
        summary = json.loads((output/f'{split}_comparison.json').read_text())
        assert set(summary) == set(variants) | ({'beamsearch'} if 'beam_muse' in variants else set())
        assert {s['n'] for s in summary.values()} == {1}
        for result in summary.values():
            assert set(result['attribute_spearman']) == set(ATTRS[:5])
            assert set(result['attribute_label_distance']) == set(ATTRS[:5])
            assert all(key in result for key in ('label_distance', 'rhythm_similarity', 'hihat_similarity'))
        for variant in variants:
            saved_rows = json.loads((output/f'{split}_{variant}_rows.json').read_text())
            for row in saved_rows:
                if row['status'] == 'valid':
                    assert {'rhythm_similarity', 'hihat_similarity'} <= row['metrics'].keys()
    for variant, winner in selected['winners'].items():
        log_rows = [json.loads(line) for line in (output/winner['name']/'eval_loss.log').read_text().splitlines()]
        assert {r['phase'] for r in log_rows} == {'val', 'test', 'itest'}
        assert len(log_rows) == 3
        assert all(r['event'] == 'evaluation_summary' and r['n'] == 1 for r in log_rows)
        assert sum(r['beamsearch'] is not None for r in log_rows) == (3 if variant == 'beam_muse' else 0)
        for row in log_rows:
            assert 'attribute_spearman' in row['evaluation']
    after['src/common/evaluation.py'] = 'unreviewed-change'
    with pytest.raises(ValueError, match='実装'):
        run_experiment(configuration, cache, 'meta', 'beam', output, 'cpu', phase='test')


def test_log_interval_preserves_training_and_aggregates_every_evaluation(example, tmp_path):
    from src.common.losses import reconstruction_kl

    e, vocab, scorer = example
    trial = expand_trials(comparison_config())[2]
    trial['model'] = asdict(config())
    trial['training'].update(epochs=2, batch_size=1, risk_interval=1)
    examples = [e, replace(e, case_id='long', teacher=e.prepared.original)]
    identity = dict(sources=source_hashes(), smoke=True)
    weights = []
    for interval in (1, 3):
        directory = tmp_path/str(interval)
        train_trial(trial, examples, examples, vocab, scorer, EvaluationConfig(), identity, directory, 'cpu',
                    log_interval=interval)
        weights.append(torch.load(directory/'best.pt', weights_only=True)['state_dict'])
        entries = [json.loads(line) for line in (directory/'train_loss.log').read_text().splitlines()]
        assert {r['step'] for r in entries} == ({1, 2, 3, 4} if interval == 1 else {3})
        assert {r['event'] for r in entries} == {'base_loss', 'beamsearch_score', 'auxiliary_loss'}
        evaluation = [json.loads(line) for line in (directory/'eval_loss.log').read_text().splitlines()]
        assert len(evaluation) == 2
        assert [r['step'] for r in evaluation] == [2, 4]
        assert all(r['n'] == 2 and r['beamsearch'] is not None for r in evaluation)
        model, checkpoint = load_checkpoint(directory/'best.pt', vocab, identity)
        losses, tokens = [], []
        with torch.no_grad():
            for example in examples:
                batch = collate([example], vocab, scorer, 'beam_muse')
                losses.append(reconstruction_kl(model(batch, sample=False), batch['decoder_target'], model.pad_id))
                tokens.append(int(batch['decoder_target'].ne(model.pad_id).sum()))
        logged = evaluation[checkpoint['epoch']-1]['musemorphose']
        assert logged['reconstruction'] == pytest.approx(
            sum(float(loss['reconstruction'])*n for loss, n in zip(losses, tokens))/sum(tokens))
        assert logged['kl'] == pytest.approx(sum(float(loss['kl']) for loss in losses)/2)
        assert logged['total'] == pytest.approx(logged['reconstruction'] + .01*logged['kl'])
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], rtol=0, atol=0)


def test_standalone_prepares_original_without_beam_search(example, monkeypatch):
    from src.beam_muse.data import prepare_example
    e, vocabulary, scorer = example
    encode = lambda bar: [vocabulary['token2idx'][t] for t in to_tokens(bar)] if bar is not None else None
    pair = dict(pair_id='example', X_before=encode(e.prepared.original), X_after=encode(e.teacher),
                d_after=list(e.prepared.target.values()),
                context=dict(previous=encode(e.prepared.previous), following=encode(e.prepared.following),
                             section=None, after_difficulty_context='before_previous'),
                source=dict(dataset='fixture', id='song', split='train'))
    def forbidden(*args, **kwargs):
        pytest.fail('単体ではBeamSearchを実行しない')
    monkeypatch.setattr('src.beam_muse.data.beam_search', forbidden)
    prepared = prepare_example(pair, vocabulary, scorer, None, None, 512, use_beam=False)
    assert prepared.intermediate == prepared.prepared.original
    assert prepared.history == [] and prepared.beam_seconds == 0
    batch = collate([prepared], vocabulary, scorer, 'musemorphose')
    assert batch['decoder_target'].ne(vocabulary['token2idx']['<PAD>']).any()


def test_hybrid_rejects_original_only_cache(example, tmp_path, monkeypatch):
    e, vocabulary, scorer = example
    configuration = comparison_config()
    monkeypatch.setattr('src.beam_muse.experiment.load_metadata',
                        lambda *args: (scorer, None, None, EvaluationConfig(), None, vocabulary))
    monkeypatch.setattr('src.beam_muse.experiment.load_cache',
                        lambda *args: ([e], dict(identity=dict(intermediate_mode='original'))))
    with pytest.raises(ValueError, match='中間譜面付き'):
        run_experiment(configuration, tmp_path/'cache', 'meta', 'beam', tmp_path/'run', 'cpu')
