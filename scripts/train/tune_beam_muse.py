"""単体／ハイブリッドのペア準備、学習・候補選定、固定後の評価。"""
import argparse
import copy
import json
from pathlib import Path
import torch

from src.beam_muse.data import prepare_cache
from src.beam_muse.experiment import expand_trials, run_experiment
from src.beam_muse.configuration import load_experiment_config, resolve_inputs, validate_cache_input


def smoke_config(config):
    """接続検証専用。研究用設定や選定済みモデルとして扱わない。"""
    result = copy.deepcopy(config)
    result['model'].update(d_model=32, d_latent=16, n_head=4, enc_layers=1, dec_layers=1,
                           d_ff=64, d_condition=16, dropout=0., max_tokens=128)
    result['training'].update(epochs=1, batch_size=2, risk_interval=1, risk_examples=1)
    # relative/absoluteと単体の3候補だけで経路を確認する。
    variants = {t['variant'] for t in expand_trials(config)}
    candidates = [dict(name='smoke_muse', variant='musemorphose', model=dict(use_skeleton=False, difficulty_mode='absolute')),
                  dict(name='smoke_relative', variant='beam_muse', model=dict(use_skeleton=True, difficulty_mode='relative')),
                  dict(name='smoke_absolute', variant='beam_muse', model=dict(use_skeleton=True, difficulty_mode='absolute'))]
    result['trials'] = [t for t in candidates if t['variant'] in variants]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'select', 'test', 'all', 'smoke'), required=True)
    parser.add_argument('--config', type=Path, default=Path('config/beam_muse_candidates.json'),
                        help='JSON設定（//・/* */コメント、末尾カンマも使用可）')
    parser.add_argument('--pairdata', type=Path, help='configのdata.pairdataを上書き（相対パスはリポジトリ直下基準）')
    parser.add_argument('--meta', type=Path, help='互換用。省略時はpairdata内の語彙・境界を使用。指定時も内容一致が必要')
    parser.add_argument('--beam-model', type=Path, help='configのdata.beam_modelを上書き')
    parser.add_argument('--cache', type=Path, help='既定は出力先/cache。smokeと本実験を混在させない')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=500,
                        help='学習損失ログの保存間隔（Step数、正整数、既定500）。評価は毎回全件を1行に集約')
    parser.add_argument('--smoke-limit', type=int, default=4, help='smoke時のみtrain/valから読む件数')
    args = parser.parse_args()
    try:
        if args.threads < 1 or args.smoke_limit < 1:
            raise ValueError('threadsとsmoke-limitは正整数')
        if args.log_interval < 1:
            raise ValueError('log-intervalは正整数')
        torch.set_num_threads(args.threads)
        config = load_experiment_config(args.config)
        config, pairdata, meta, beam_model = resolve_inputs(
            config, pairdata=args.pairdata, beam_model=args.beam_model, meta=args.meta)
        print(f'入力: pairdata={pairdata}, meta={meta}, beam_model={beam_model}', flush=True)
        if args.phase == 'smoke':
            config = smoke_config(config)
        trials = expand_trials(config)
        use_beam = any(t['variant'] == 'beam_muse' for t in trials)
        print('準備方式: ' + ('BeamSearch中間譜面を作成' if use_beam else 'MuseMorphose単体（原譜を入力、BeamSearch探索なし）'), flush=True)
        lengths = {t['model']['max_tokens'] for t in trials}
        if len(lengths) != 1:
            raise ValueError('候補間でトークン上限を共通にする')
        cache = args.cache or args.output/'cache'
        if args.phase in ('prepare', 'all', 'smoke'):
            prepare_cache(pairdata, meta, beam_model, cache,
                          max_tokens=lengths.pop(), smoke_limit=args.smoke_limit if args.phase == 'smoke' else None,
                          use_beam=use_beam)
        if args.phase in ('select', 'all', 'smoke'):
            validate_cache_input(cache, pairdata)
            selected = run_experiment(config, cache, meta, beam_model, args.output, args.device,
                                      smoke=args.phase == 'smoke', log_interval=args.log_interval)
            print('val選定: '+json.dumps({k:v['name'] for k,v in selected['winners'].items()}, ensure_ascii=False), flush=True)
        if args.phase in ('test', 'all'):
            validate_cache_input(cache, pairdata)
            run_experiment(config, cache, meta, beam_model, args.output, args.device, phase='test',
                           log_interval=args.log_interval)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
