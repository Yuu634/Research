"""configの入力解決、メタデータ照合、別データのキャッシュ拒否を検証する。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from data.pairdata.io import file_hash, write_json
from src.beam_muse.configuration import load_experiment_config, resolve_inputs, validate_cache_input, validate_data_config
from src.beam_muse.experiment import expand_trials
from scripts.train import tune_beam_muse


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr('src.beam_muse.configuration.REPOSITORY_ROOT', tmp_path)
    pairdata = tmp_path / 'pairs'
    pairdata.mkdir()
    for name, value in (('index.json', '{"shards":[]}'), ('drum_vocab.pkl', 'vocabulary'), ('bounds.json', 'bounds')):
        (pairdata / name).write_text(value)
    beam = tmp_path / 'beam.json'
    write_json(beam, dict(vocabulary_sha256=file_hash(pairdata / 'drum_vocab.pkl'),
                          bounds_sha256=file_hash(pairdata / 'bounds.json')))
    config = load_experiment_config('config/beam_muse_candidates.json')
    config['trials'] = [dict(name='muse', variant='musemorphose', model=dict(use_skeleton=False)),
                        dict(name='hybrid', variant='beam_muse')]
    config['data'] = dict(pairdata='pairs', beam_model='beam.json')
    return config, pairdata, beam


def test_paths_use_repository_root_not_config_directory_or_cwd(inputs, tmp_path, monkeypatch):
    config, pairdata, beam = inputs
    original = deepcopy(config)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resolved, actual, meta, model = resolve_inputs(config)
    assert actual == meta == pairdata and model == beam
    assert resolved['data'] == dict(pairdata=str(pairdata), beam_model=str(beam))
    assert config == original


def test_cli_overrides_config_and_legacy_config_still_expands(inputs):
    config, pairdata, beam = inputs
    expected = expand_trials(config)
    config['data'] = dict(pairdata='missing', beam_model='missing.json')
    _, actual, meta, model = resolve_inputs(config, pairdata=pairdata, beam_model=beam)
    assert actual == meta == pairdata and model == beam
    del config['data']
    assert expand_trials(config) == expected
    _, actual, meta, _ = resolve_inputs(config, pairdata=pairdata, beam_model=beam)
    assert actual == meta == pairdata


@pytest.mark.parametrize('data', [None, {}, {'pairdata': 'pairs'},
    {'pairdata': 'pairs', 'beam_model': 'beam', 'meta': 'meta'},
    {'pairdata': '', 'beam_model': 'beam'}, {'pairdata': 3, 'beam_model': 'beam'}])
def test_invalid_data_rejected(data):
    with pytest.raises(ValueError, match='data'):
        validate_data_config(dict(data=data))


def test_metadata_override_must_match_pairdata(inputs, tmp_path):
    config, pairdata, _ = inputs
    meta = tmp_path / 'meta'
    meta.mkdir()
    for name in ('drum_vocab.pkl', 'bounds.json'):
        (meta / name).write_bytes((pairdata / name).read_bytes())
    assert resolve_inputs(config, meta=meta)[2] == meta
    (meta / 'bounds.json').write_text('different')
    with pytest.raises(ValueError, match='pairdata内'):
        resolve_inputs(config, meta=meta)


def test_beam_metadata_mismatch_and_missing_local_metadata_fail(inputs):
    config, pairdata, beam = inputs
    (pairdata / 'bounds.json').write_text('different')
    with pytest.raises(ValueError, match='BeamSearch'):
        resolve_inputs(config)
    config['data']['pairdata'] = 'missing'
    with pytest.raises(ValueError, match='存在しない'):
        resolve_inputs(config)


def test_wrong_dataset_cache_rejected_even_with_same_metadata(inputs, tmp_path):
    _, pairdata, _ = inputs
    cache = tmp_path / 'cache'
    write_json(cache / 'manifest.json', dict(identity=dict(pair_index=file_hash(pairdata / 'index.json'))))
    validate_cache_input(cache, pairdata)
    (pairdata / 'index.json').write_text('{"shards":[1]}')
    with pytest.raises(ValueError, match='キャッシュと指定pairdata'):
        validate_cache_input(cache, pairdata)


@pytest.mark.parametrize('standalone', [False, True])
def test_cli_prepare_select_test_use_config_paths_without_meta_option(inputs, tmp_path, monkeypatch, standalone):
    config, pairdata, beam = inputs
    if standalone:
        config['trials'] = [t for t in config['trials'] if t['variant'] == 'musemorphose']
    path = tmp_path / 'configuration.json'
    write_json(path, config)
    path.write_text('// コメント付き実験設定\n' + path.read_text())
    output = tmp_path / 'run'
    seen = []
    def prepare(actual, meta, selected, cache, **kwargs):
        assert actual == meta == pairdata and selected == beam
        assert kwargs['use_beam'] is not standalone
        write_json(cache / 'manifest.json', dict(identity=dict(pair_index=file_hash(pairdata / 'index.json'))))
        seen.append('prepare')
    def run(resolved, cache, meta, selected, directory, device, **kwargs):
        assert resolved['data'] == dict(pairdata=str(pairdata), beam_model=str(beam))
        assert meta == pairdata and selected == beam and directory == output
        seen.append(kwargs.get('phase', 'select'))
        return dict(winners={})
    monkeypatch.setattr(tune_beam_muse, 'prepare_cache', prepare)
    monkeypatch.setattr(tune_beam_muse, 'run_experiment', run)
    monkeypatch.setattr(sys, 'argv', ['tune', '--phase', 'all', '--config', str(path), '--output', str(output), '--device', 'cpu'])
    tune_beam_muse.main()
    assert seen == ['prepare', 'select', 'test']


def test_standalone_smoke_does_not_add_hybrid_trials():
    config = json.loads(Path('config/musemorphose_candidates.json').read_text())
    assert {t['variant'] for t in expand_trials(config)} == {'musemorphose'}
    smoke = tune_beam_muse.smoke_config(config)
    assert [t['variant'] for t in expand_trials(smoke)] == ['musemorphose']


def test_comments_disable_trials_and_allow_last_commented_candidate(tmp_path):
    path = tmp_path/'config.json'
    path.write_text('''{
      "trials": [
        // {"name": "first"},
        {"name": "active"},
        /* {"name": "last"} */
      ],
    }''')
    assert load_experiment_config(path) == dict(trials=[dict(name='active')])


def test_comments_preserve_strings_and_plain_json(tmp_path):
    value = dict(url='https://example.test/a//b', path='x/*literal*/y',
                 quote='escaped " // /* */', slash='\\', punctuation=',]}')
    path = tmp_path/'config.json'
    path.write_text(json.dumps(value))
    assert load_experiment_config(path) == value
    path.write_text('// 設定\n'+json.dumps(value)+'\n/* 終端 */')
    assert load_experiment_config(path) == value


@pytest.mark.parametrize('content', ['{"x": 1 /* unfinished}', '{"x": 1,,}', '{"x": 1 # comment\n}'])
def test_invalid_config_still_fails(tmp_path, content):
    path = tmp_path/'config.json'
    path.write_text(content)
    with pytest.raises(json.JSONDecodeError):
        load_experiment_config(path)


def test_default_hybrid_config_enables_only_selected_candidate():
    trials = expand_trials(load_experiment_config('config/beam_muse_candidates.json'))
    assert [t['name'] for t in trials] == ['hybrid_absolute_lr1']
    assert trials[0]['model']['difficulty_mode'] == 'absolute'
    assert trials[0]['training']['learning_rate'] == .0001
