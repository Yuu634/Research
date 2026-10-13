"""3方式のCLI接続、旧引数の保持、方式をまたぐ評価・生成の取り違えを検証する。"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from data.pairdata.io import file_hash, write_json
from scripts import tasks


@pytest.fixture
def commands(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, 'invoke', lambda args, module, *options: calls.append((module, list(map(str, options)))))
    return calls


@pytest.mark.parametrize('model,module,folder', [
    ('musemorphose', 'scripts.train.tune_beam_muse', 'musemorphose_v1'),
    ('beamsearch', 'scripts.evaluate.compare_arrangement_models', 'beamsearch_v1'),
    ('beam_muse', 'scripts.train.tune_beam_muse', 'beam-muse_v1'),
])
def test_training_model_routes_and_custom_options(commands, model, module, folder):
    tasks.main(['beam_muse', model, '--phase', 'select', '--workers', '3', '--log-interval', '100'])
    assert commands[0][0] == module
    options = commands[0][1]
    assert options[options.index('--output')+1] == 'trained_model/'+folder
    assert options[options.index('--log-interval')+1] == '100'


def test_all_training_orders_beam_selection_before_neural_cache(commands, tmp_path):
    tasks.main(['beam_muse', 'all', '--root', str(tmp_path), '--phase', 'all'])
    assert [c[0] for c in commands] == ['scripts.evaluate.compare_arrangement_models',
                                     'scripts.train.tune_beam_muse', 'scripts.train.tune_beam_muse']
    for _, options in commands[1:]:
        assert options[options.index('--beam-model')+1] == str(tmp_path/'beamsearch_v1/selected_model.json')


def test_all_training_preserves_existing_cache_beam_model(commands, tmp_path):
    manifest = tmp_path/'musemorphose_v1/cache/manifest.json'
    manifest.parent.mkdir(parents=True)
    write_json(manifest, dict(resolved_paths=dict(beam_model='original_selected.json')))
    tasks.main(['beam_muse', 'all', '--root', str(tmp_path), '--phase', 'test'])
    options = commands[1][1]
    assert options[options.index('--beam-model')+1] == 'original_selected.json'


def test_log_interval_environment_and_old_smoke_invocation(commands, monkeypatch):
    monkeypatch.setenv('LOG_INTERVAL', '27')
    tasks.main(['beam_muse', 'smoke', 'output/smoke_fixture', '--device', 'cpu'])
    options = commands[0][1]
    assert options[options.index('--phase')+1] == 'smoke'
    assert options[options.index('--output')+1] == 'output/smoke_fixture'
    assert options[options.index('--device')+1] == 'cpu'
    assert options[options.index('--log-interval')+1] == '27'


def test_evaluation_and_analysis_route_all_three_models(commands, monkeypatch):
    tasks.main(['evaluate', 'all', '--scope', 'all', '--splits', 'itest'])
    evaluations = [c for c in commands if c[0] == 'scripts.evaluate.evaluate_arrangement_models']
    assert [opts[opts.index('--model')+1] for _, opts in evaluations] == list(tasks.MODELS)
    assert all(opts[opts.index('--scope')+1] == 'all' for _, opts in evaluations)
    assert all(opts[opts.index('--splits')+1:] == ['itest'] for _, opts in evaluations)
    commands.clear()
    monkeypatch.setattr(tasks, 'model_paths', lambda *args: (Path('meta'), Path('selected.json')))
    tasks.main(['analysis', 'all'])
    assert [module for module, _ in commands if module != 'scripts.analysis.plot_loss_curves'] == [
        'scripts.analysis.analyze_arrangement_models', 'scripts.evaluate.report_arrangement_comparison',
        'scripts.analysis.analyze_arrangement_models']


def _neural_run(root, model):
    run = root/tasks.RUNS[model]
    candidate = run/('selected_'+model)
    candidate.mkdir(parents=True)
    checkpoint = candidate/'best.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    checksum = file_hash(checkpoint)
    write_json(candidate/'completed.json', dict(trial=dict(variant=model), checkpoint_sha256=checksum))
    write_json(run/'selection.json', dict(selected_on='val', winners={model:dict(name=candidate.name, checkpoint_sha256=checksum)}))
    manifest = run/'cache/manifest.json'
    manifest.parent.mkdir()
    write_json(manifest, dict(resolved_paths=dict(meta='saved meta', beam_model='saved beam.json')))
    return checkpoint


def test_generation_uses_selected_weights_and_separates_all_outputs(commands, tmp_path):
    for model in ('musemorphose','beam_muse'):
        _neural_run(tmp_path, model)
    beam = tmp_path/tasks.RUNS['beamsearch']
    (beam/'selected_beam').mkdir(parents=True)
    write_json(beam/'selected_model.json', dict(candidate=dict(name='selected_beam')))
    write_json(beam/'selected_beam/config.json', dict(meta='beam meta'))
    tasks.main(['generate', 'all', '--root', str(tmp_path), 'input request.json', 'output fixture'])
    assert len(commands) == 3
    for model, (_, opts) in zip(tasks.MODELS, commands):
        assert opts[opts.index('--output')+1] == str(Path('output fixture')/model)
        assert opts[opts.index('--input')+1] == 'input request.json'
        if model != 'beamsearch':
            assert opts[opts.index('--checkpoint')+1] == str(tmp_path/tasks.RUNS[model]/('selected_'+model)/'best.pt')
            assert opts[opts.index('--meta')+1] == 'saved meta'


def test_generation_rejects_wrong_variant_and_changed_weight_before_execution(commands, tmp_path):
    checkpoint = _neural_run(tmp_path, 'musemorphose')
    hybrid = tmp_path/tasks.RUNS['beam_muse']
    (hybrid/'cache').mkdir(parents=True)
    write_json(hybrid/'cache/manifest.json', dict(resolved_paths=dict(meta='meta', beam_model='beam.json')))
    with pytest.raises(SystemExit):
        tasks.main(['generate', 'beam_muse', '--root', str(tmp_path), '--checkpoint', str(checkpoint), 'in.json', 'out'])
    assert commands == []
    checkpoint.write_bytes(b'changed')
    with pytest.raises(SystemExit):
        tasks.main(['generate', 'musemorphose', '--root', str(tmp_path), 'in.json', 'out'])
    assert commands == []


def test_shared_data_creation_runs_once_and_keeps_test_out_of_reconstruction(commands):
    tasks.main(['data_creator', 'all', '--stage', 'all'])
    modules = [c[0] for c in commands]
    assert modules == ['scripts.preprocess.build_drum_collection', 'scripts.preprocess.validate_drum_collection',
                       'scripts.preprocess.label_drum_difficulty_v2', 'data.pairdata.build',
                       'scripts.preprocess.add_pairdata_reconstruction']
    assert commands[-1][1][-2:] == ['--reconstruction-splits', 'train']


@pytest.mark.parametrize('action,args,module', [
    ('generate', ['input.mid','output.mid','--device','cpu','--difficulty_shift','0','1','2','3','4'],
     'scripts.generate.generate_drum_from_input'),
    ('evaluate', ['--model_path','custom.pt','--device','cpu'], 'scripts.evaluate.evaluate'),
    ('analysis', ['old model dir'], 'scripts.analysis.inspect_alpha_limb'),
    ('data_creator', ['MIDI directory'], 'scripts.preprocess.prepare_drum_dataset'),
])
def test_legacy_invocation_keeps_paths_and_known_cli_flags(commands, action, args, module):
    tasks.main([action, *args])
    assert commands[0][0] == module
    if '--device' in args:
        assert '--device' in commands[0][1]
    if action == 'generate':
        assert commands[0][1][-len(args[2:]):] == args[2:]


def test_dry_run_does_not_start_subprocess_or_create_data(monkeypatch, tmp_path):
    monkeypatch.setattr(tasks.subprocess, 'run', lambda *a, **k: pytest.fail('dry-runで実行した'))
    tasks.main(['beam_muse', 'all', '--root', str(tmp_path), '--dry-run'])
    tasks.main(['data_creator', 'all', '--pairdata', str(tmp_path/'pairdata'), '--dry-run'])
    assert list(tmp_path.iterdir()) == []


def test_inapplicable_options_fail_before_starting_commands(commands):
    for argv in (['evaluate', 'all', '--device', 'cuda'], ['analysis', 'beam_muse', '--checkpoint', 'wrong.pt'],
                 ['data_creator', 'all', '--run', 'wrong'], ['generate', 'beamsearch', '--checkpoint', 'wrong.pt']):
        with pytest.raises(SystemExit):
            tasks.main(argv)
    assert commands == []


@pytest.mark.parametrize('model', ['musemorphose', 'beamsearch', 'beam_muse'])
def test_evaluation_entrypoint_scopes_run_and_variant(tmp_path, monkeypatch, model):
    from scripts.evaluate import evaluate_arrangement_models as entrypoint
    from scripts.evaluate import refresh_test_metrics as refresh
    root = tmp_path/'trained_model'
    root.mkdir()
    run = root/tasks.RUNS[model]
    run.mkdir()
    if model == 'beamsearch':
        write_json(run/'run_manifest.json', {})
    else:
        candidate = run/'selected'
        candidate.mkdir()
        (candidate/'best.pt').write_bytes(b'fixture')
        checksum = file_hash(candidate/'best.pt')
        write_json(candidate/'completed.json', dict(trial=dict(variant=model),checkpoint_sha256=checksum))
        write_json(run/'selection.json', dict(selected_on='val',winners={model:dict(name='selected',checkpoint_sha256=checksum)}))
    called = []
    def evaluate(run_path, args, status):
        called.append((run_path,args.variant,args.scope,args.splits))
        status[str(run_path)] = dict(status='complete')
    monkeypatch.setattr(refresh,'refresh_beam',evaluate)
    monkeypatch.setattr(refresh,'refresh_neural',evaluate)
    monkeypatch.setattr('sys.argv',['evaluate_arrangement_models','--model',model,'--run',str(run),'--splits','itest'])
    entrypoint.main()
    assert called == [(run,None if model == 'beamsearch' else model,'selected',['itest'])]
    assert read_status(root) == {str(run):dict(status='complete')}


def read_status(root):
    return json.loads((root/'test_evaluation_status.json').read_text())


@pytest.mark.parametrize('task', ['beam_muse','generate','evaluate','analysis','data_creator','plot_loss'])
def test_shell_help_works_outside_repository(task, tmp_path):
    script = tasks.ROOT/'tasks'/f'{task}.sh'
    result = subprocess.run(['bash', str(script), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--model' in result.stdout and '--dry-run' in result.stdout


def test_train_wrapper_remains_baseline():
    assert (tasks.ROOT/'tasks/train.sh').read_text() == ('#!/usr/bin/env bash\nset -euo pipefail\n'
        'cd "$(dirname "$0")/.."\npython -m scripts.train.train_drum \\\n'
        '    --config config/drum_config_standard-lakh1-kl14.yaml "$@"\n')
