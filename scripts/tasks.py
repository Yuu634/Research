"""tasks/*.shのモデル選択と実行先を統一する。旧基準モデルはlegacyで利用する。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('musemorphose', 'beamsearch', 'beam_muse')
RUNS = dict(musemorphose='musemorphose_v1', beamsearch='beamsearch_v1', beam_muse='beam-muse_v1')
CONFIGS = dict(musemorphose='config/musemorphose_candidates.json',
               beamsearch='config/beamsearch_candidates.json', beam_muse='config/beam_muse_candidates.json')
PAIRDATA = Path('data/processed/drum_pairdata_v1+reconst')
LEGACY_MODEL = 'trained_model/checkpoints_drum_standard-lakh1-kl14/params/best_params.pt'
LEGACY_BOUNDS = 'data/metadata_new/difficulty_bounds_standard-lakh.pkl'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def invoke(args, module, *options):
    command = [sys.executable, '-m', module, *map(str, options)]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_path(args, model):
    return args.run if args.run is not None else args.root / RUNS[model]


def model_paths(args, model, run):
    """推論・解析は保存済み実験のメタデータを優先する。"""
    if model == 'beamsearch':
        selected_path = args.beam_model or run / 'selected_model.json'
        selected = read_json(selected_path)
        settings = read_json(run / selected['candidate']['name'] / 'config.json')
        return args.meta or Path(settings['meta']), selected_path
    manifest = read_json(run / 'cache/manifest.json')
    paths = manifest['resolved_paths']
    return args.meta or Path(paths['meta']), args.beam_model or Path(paths['beam_model'])


def selected_checkpoint(args, model, run):
    if args.checkpoint is not None:
        path = args.checkpoint
    else:
        selected = read_json(run / 'selection.json')
        if selected['selected_on'] != 'val' or model not in selected['winners']:
            raise ValueError('指定方式のval選定記録がありません: ' + model)
        winner = selected['winners'][model]
        path = run / winner['name'] / 'best.pt'
    completed = read_json(path.parent / 'completed.json')
    if completed['trial']['variant'] != model:
        raise ValueError('指定方式とチェックポイントのモデルが異なります')
    from data.pairdata.io import file_hash
    expected = completed['checkpoint_sha256']
    if args.checkpoint is None and expected != winner['checkpoint_sha256']:
        raise ValueError('選定記録と学習完了記録の重みが異なります')
    if file_hash(path) != expected:
        raise ValueError('チェックポイントのハッシュが学習完了記録と異なります')
    return path


def train(args, models, extra):
    phase = args.phase or 'all'
    for model in sorted(models, key=lambda value: value != 'beamsearch'):
        run = run_path(args, model)
        config = args.config or Path(CONFIGS[model])
        if model == 'beamsearch':
            if phase == 'prepare':
                if len(models) == 1:
                    raise ValueError('BeamSearchはprepare不要です。select/test/all/smokeを指定してください')
                continue
            pairdata = args.pairdata or PAIRDATA
            invoke(args, 'scripts.evaluate.compare_arrangement_models', '--phase', phase,
                   '--config', config, '--output', run, '--pairdata', pairdata,
                   '--meta', args.meta or pairdata, '--workers', args.workers,
                   '--log-interval', args.log_interval, *extra)
        else:
            options = []
            for flag, value in (('--pairdata', args.pairdata), ('--meta', args.meta), ('--device', args.device)):
                if value is not None:
                    options.extend((flag, value))
            beam_model = args.beam_model
            if beam_model is None and 'beamsearch' in models and phase in ('select', 'all', 'test'):
                manifest = run / 'cache/manifest.json'
                beam_model = (Path(read_json(manifest)['resolved_paths']['beam_model']) if manifest.exists()
                              else run_path(args, 'beamsearch') / 'selected_model.json')
            if beam_model is not None:
                options.extend(('--beam-model', beam_model))
            invoke(args, 'scripts.train.tune_beam_muse', '--phase', phase, '--config', config,
                   '--output', run, '--log-interval', args.log_interval, *options, *extra)


def generate(args, models, extra):
    # JSON入力・出力はフラグ、または従来同様に先頭の2位置引数で指定できる。
    input_path, output = args.input, args.output
    remaining = list(extra)
    if input_path is None and remaining and not remaining[0].startswith('-'):
        input_path = Path(remaining.pop(0))
    if output is None and remaining and not remaining[0].startswith('-'):
        output = Path(remaining.pop(0))
    if input_path is None or output is None:
        raise ValueError('生成には入力JSONと出力ディレクトリが必要です（--input / --output）')
    # allでも途中まで別方式だけを生成しないよう、全方式の入力先を先に照合する。
    prepared = []
    for model in models:
        run = run_path(args, model)
        meta, beam_model = model_paths(args, model, run)
        checkpoint = None if model == 'beamsearch' else selected_checkpoint(args, model, run)
        prepared.append((model, meta, beam_model, checkpoint))
    for model, meta, beam_model, checkpoint in prepared:
        directory = output / model if len(models) > 1 else output
        if model == 'beamsearch':
            invoke(args, 'scripts.generate.arrange_beam_search', '--input', input_path,
                   '--output', directory, '--meta', meta, '--model', beam_model, *remaining)
        else:
            invoke(args, 'scripts.generate.arrange_beam_muse', '--input', input_path, '--output', directory,
                   '--meta', meta, '--beam-model', beam_model, '--checkpoint', checkpoint,
                   '--device', args.device or 'cpu', *remaining)


def evaluate(args, models, extra):
    for model in models:
        run = run_path(args, model)
        invoke(args, 'scripts.evaluate.evaluate_arrangement_models', '--model', model, '--run', run,
               '--scope', args.scope, '--workers', args.workers, '--batch-size', args.batch_size,
               '--splits', *args.splits, *extra)
        invoke(args, 'scripts.analysis.plot_loss_curves', '--trained-model-dir', run)


def analysis(args, models, extra):
    for model in models:
        run = run_path(args, model)
        output = (args.output or Path('output/model_analysis')) / model
        if model == 'beamsearch':
            meta, _ = model_paths(args, model, run)
            invoke(args, 'scripts.evaluate.report_arrangement_comparison', '--run', run,
                   '--meta', meta, '--output', output, *extra)
        else:
            invoke(args, 'scripts.analysis.analyze_arrangement_models', '--model', model,
                   '--run', run, '--output', output, '--scope', args.scope, *extra)
        invoke(args, 'scripts.analysis.plot_loss_curves', '--trained-model-dir', run)


def data_creator(args, models, extra):
    # 3方式は同じpairdataを共用するため、allでもデータ作成は1回だけ。
    pairdata = args.pairdata or PAIRDATA
    base = args.pair_base
    labeled = args.labeled_data
    meta = args.labeled_meta
    if extra and args.stage not in ('pairs', 'all'):
        raise ValueError('追加のpairdata作成オプションは--stage pairs/allで指定してください')
    if args.stage in ('collection', 'all'):
        invoke(args, 'scripts.preprocess.build_drum_collection', '--data-root', args.data_root,
               '--collection', args.collection, '--workers', args.workers)
        invoke(args, 'scripts.preprocess.validate_drum_collection', '--data-root', args.data_root,
               '--collection', args.collection, '--workers', args.workers)
    if args.stage in ('label', 'all'):
        invoke(args, 'scripts.preprocess.label_drum_difficulty_v2', '--input', args.data_root / 'tokens' / args.collection,
               '--source-meta', args.data_root / 'metadata' / args.collection,
               '--output', labeled, '--meta', meta, '--workers', args.workers)
    if args.stage in ('pairs', 'all'):
        options = ['--config', args.pair_config] if args.pair_config else []
        invoke(args, 'data.pairdata.build', '--input', labeled, '--source-meta', meta,
               '--output', base, '--work', args.work, '--workers', args.workers, *options, *extra)
    if args.stage in ('pairs', 'reconstruction', 'all'):
        invoke(args, 'scripts.preprocess.add_pairdata_reconstruction', '--input', base, '--output', pairdata,
               '--reconstruction-splits', 'train')
    if args.stage == 'validate':
        invoke(args, 'data.pairdata.validate', '--output', base, '--workers', args.workers)
        invoke(args, 'scripts.preprocess.add_pairdata_reconstruction', '--input', base,
               '--output', pairdata, '--validate-only')


def legacy(args, extra):
    """以前の位置引数・CLI引数と、基準YAMLの重み形式を保持する。"""
    if args.action == 'generate':
        if len(extra) < 2:
            raise ValueError('legacy生成: 入力.mid 出力.mid [生成オプション...] を指定してください')
        invoke(args, 'scripts.generate.generate_drum_from_input', '--model_path', LEGACY_MODEL,
               '--difficulty_bounds', LEGACY_BOUNDS, '--input_midi', extra[0], '--output_midi', extra[1],
               '--difficulty_mode', 'global_shift', '--difficulty_shift', 0, 0, 0, 0, 0, *extra[2:])
    elif args.action == 'evaluate':
        invoke(args, 'scripts.evaluate.evaluate', '--input_midi_dir', 'data/test_dataset',
               '--model_path', LEGACY_MODEL, '--difficulty_bounds', LEGACY_BOUNDS,
               '--difficulty_labels_file', 'config/evaluation_difficulty_labels.json',
               '--output_dir', 'output/eval/checkpoints_drum_standard-lakh1-kl14', '--use_input_bpm', *extra)
    elif args.action == 'analysis':
        if not extra:
            raise ValueError('legacy解析: MODEL_DIR [MODEL_DIR ...] を指定してください')
        for value in extra:
            directory = Path(value)
            for module, category in (('inspect_alpha_limb', 'alpha_limb'), ('inspect_position_bias', 'position_bias'),
                                     ('visualize_attention_maps', 'attention_maps'), ('qk_limb_bias_ratio', 'qk_limb_ratio')):
                invoke(args, 'scripts.analysis.' + module, '--model_dir', directory,
                       '--output_dir', Path('output') / category / directory.name)
    elif args.action == 'data_creator':
        inputs = extra or ['data/raw/e-gmd', 'data/filtered/lakhmidi']
        dataset = 'data/processed_new/drum_dataset_standard-lakh'
        vocab = 'data/metadata_new/drum_vocab_standard-lakh.pkl'
        if not args.dry_run:
            (ROOT / 'data/metadata_new').mkdir(parents=True, exist_ok=True)
        invoke(args, 'scripts.preprocess.prepare_drum_dataset', '--midi_dir', *inputs,
               '--output_dir', dataset, '--vocab_path', vocab, '--tokenization_method', 'standard',
               '--file_extension', '.midi', '.mid', '--train_ratio', .9, '--seed', 42)
        invoke(args, 'src.utils.compute_difficulty_bounds', dataset, vocab, LEGACY_BOUNDS)
        invoke(args, 'scripts.preprocess.prepare_drum_dataset_with_difficulty', dataset, vocab,
               LEGACY_BOUNDS, dataset + '_with_difficulty', '--train_ratio', .9, '--seed', 42)
    elif args.action == 'plot_loss':
        invoke(args, 'src.utils.plot_loss_curves', '--trained_model_dir', args.root, *extra)
    elif args.action == 'beam_muse':
        invoke(args, 'scripts.train.train_drum', '--config', 'config/drum_config_standard-lakh1-kl14.yaml', *extra)
    else:
        raise ValueError('このタスクのlegacyモードはありません')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # bash tasks/generate.sh musemorphose ... と --model musemorphose の両方を許容する。
    if len(argv) > 1 and argv[1] in (*MODELS, 'all', 'legacy'):
        argv[1:2] = ['--model', argv[1]]
    elif len(argv) > 1 and argv[0] == 'beam_muse' and argv[1] in ('prepare', 'select', 'test', 'smoke'):
        argv[1:2] = ['--phase', argv[1]]
        if len(argv) > 3 and not argv[3].startswith('-'):
            argv[3:4] = ['--run', argv[3]]
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        '例: bash tasks/generate.sh musemorphose 入力.json 出力先 / '
        'bash tasks/evaluate.sh all --scope selected / bash tasks/beam_muse.sh beamsearch --phase select')
    parser.add_argument('action', choices=('beam_muse', 'generate', 'evaluate', 'analysis', 'data_creator', 'plot_loss'))
    parser.add_argument('--model', choices=(*MODELS, 'all', 'legacy'), help='方式名。省略時は旧タスクの動作を保持')
    parser.add_argument('--root', type=Path, default=Path('trained_model'))
    parser.add_argument('--run', type=Path, help='単一方式の実験ディレクトリを上書き')
    parser.add_argument('--input', type=Path, help='新モデルの1小節JSON入力')
    parser.add_argument('--output', type=Path, help='生成・解析の出力先。学習では--runを使用')
    parser.add_argument('--config', type=Path, help='単一方式の学習・探索設定')
    parser.add_argument('--checkpoint', type=Path, help='新ニューラルモデルのbest.pt（既定はval選定重み）')
    parser.add_argument('--meta', type=Path)
    parser.add_argument('--beam-model', type=Path)
    parser.add_argument('--device', help='ニューラル学習・生成のデバイス。生成既定cpu、学習はCLI既定')
    parser.add_argument('--phase', choices=('prepare', 'select', 'test', 'all', 'smoke'))
    parser.add_argument('--scope', choices=('selected', 'all'), default='selected')
    parser.add_argument('--splits', nargs='+', choices=('test', 'itest'), default=['test', 'itest'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16, help='再評価の保存バッチ単位')
    parser.add_argument('--log-interval', type=int, default=os.environ.get('LOG_INTERVAL', '500'))
    parser.add_argument('--stage', choices=('collection', 'label', 'pairs', 'reconstruction', 'validate', 'all'), default='pairs',
                        help='新データ準備。pairsはv2ラベル付き譜面から開始、allは収集済み原データから開始')
    parser.add_argument('--pairdata', type=Path)
    parser.add_argument('--pair-base', type=Path, default=Path('data/processed/drum_pairdata_v1'))
    parser.add_argument('--pair-config', type=Path)
    parser.add_argument('--work', type=Path, default=Path('data/work/drum_pairdata_v1'))
    parser.add_argument('--labeled-data', type=Path, default=Path('data/processed/drum_collection_difficulty_v2'))
    parser.add_argument('--labeled-meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--collection', default='drum_collection_v1')
    parser.add_argument('--dry-run', action='store_true', help='実行コマンドだけを表示（学習・生成・保存をしない）')
    args, extra = parser.parse_known_args(argv)
    try:
        if min(args.workers, args.batch_size, args.log_interval) < 1:
            raise ValueError('workers・batch-size・log-intervalは正整数です')
        model = args.model or ('beam_muse' if args.action == 'beam_muse' else 'all' if args.action == 'plot_loss' else 'legacy')
        if model == 'legacy':
            # 共通parserが認識した--deviceなども、旧CLIへ欠落なく引き渡す。
            raw, index = [], 1
            while index < len(argv):
                if argv[index] == '--model':
                    index += 2
                    continue
                if argv[index] == '--dry-run' or argv[index].startswith('--model='):
                    index += 1
                    continue
                raw.append(argv[index])
                index += 1
            legacy(args, raw)
            return
        allowed = {
            'beam_muse': {'--root','--run','--output','--config','--phase','--device','--meta','--pairdata',
                          '--beam-model','--workers','--log-interval'},
            'generate': {'--root','--run','--input','--output','--checkpoint','--meta','--beam-model','--device'},
            'evaluate': {'--root','--run','--scope','--splits','--workers','--batch-size'},
            'analysis': {'--root','--run','--output','--scope','--meta'},
            'data_creator': {'--stage','--pairdata','--pair-base','--pair-config','--labeled-data','--labeled-meta',
                             '--data-root','--collection','--work','--workers'},
            'plot_loss': {'--root','--run'},
        }[args.action] | {'--model','--dry-run'}
        for token in argv[1:]:
            flag = token.split('=', 1)[0]
            if flag in parser._option_string_actions and flag not in allowed:
                raise ValueError(f'{args.action}では{flag}を使用しません。--helpを確認してください')
        models = MODELS if model == 'all' else (model,)
        if model == 'beamsearch' and args.checkpoint is not None:
            raise ValueError('BeamSearchは重みを使いません。--beam-modelで選定設定を指定してください')
        if len(models) > 1 and any(value is not None for value in (args.run, args.config, args.checkpoint)):
            raise ValueError('allでは--run・--config・--checkpointを共用できません。単一方式を指定してください')
        if args.action == 'beam_muse' and args.output is not None:
            if args.run is not None or len(models) > 1:
                raise ValueError('学習出力の指定は単一方式の--runまたは--outputを使用してください')
            args.run = args.output
        if args.action == 'plot_loss':
            if args.model is None or model == 'all':
                invoke(args, 'scripts.analysis.plot_loss_curves', '--trained-model-dir', args.root, *extra)
            else:
                invoke(args, 'scripts.analysis.plot_loss_curves', '--trained-model-dir', run_path(args, model), *extra)
        else:
            handlers = dict(beam_muse=train, generate=generate, evaluate=evaluate, analysis=analysis,
                            data_creator=data_creator)
            handlers[args.action](args, models, extra)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
