"""3方式比較用のペア準備、学習・ハイパーパラメータ選定、固定後の評価。"""
import argparse
import copy
import json
from pathlib import Path
import torch

from src.beam_muse.data import prepare_cache
from src.beam_muse.experiment import expand_trials, run_experiment


def smoke_config(config):
    """接続検証専用。研究用設定や選定済みモデルとして扱わない。"""
    result = copy.deepcopy(config)
    result['model'].update(d_model=32, d_latent=16, n_head=4, enc_layers=1, dec_layers=1,
                           d_ff=64, d_condition=16, dropout=0., max_tokens=128)
    result['training'].update(epochs=1, batch_size=2, risk_interval=1, risk_examples=1)
    # relative/absoluteと単体の3候補だけで経路を確認する。
    result['trials'] = [dict(name='smoke_muse', variant='musemorphose', model=dict(use_skeleton=False, difficulty_mode='absolute')),
                        dict(name='smoke_relative', variant='beam_muse'),
                        dict(name='smoke_absolute', variant='beam_muse', model=dict(difficulty_mode='absolute'))]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'select', 'test', 'all', 'smoke'), required=True)
    parser.add_argument('--config', type=Path, default=Path('config/beam_muse_candidates.json'))
    parser.add_argument('--pairdata', type=Path, default=Path('data/processed/drum_pairdata_v1'))
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--beam-model', type=Path, default=Path('config/beamsearch_selected.json'))
    parser.add_argument('--cache', type=Path, help='既定は出力先/cache。smokeと本実験を混在させない')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke-limit', type=int, default=4, help='smoke時のみtrain/valから読む件数')
    args = parser.parse_args()
    try:
        if args.threads < 1 or args.smoke_limit < 1:
            raise ValueError('threadsとsmoke-limitは正整数')
        torch.set_num_threads(args.threads)
        config = json.loads(args.config.read_text())
        if args.phase == 'smoke':
            config = smoke_config(config)
        trials = expand_trials(config)
        lengths = {t['model']['max_tokens'] for t in trials}
        if len(lengths) != 1:
            raise ValueError('候補間でトークン上限を共通にする')
        cache = args.cache or args.output/'cache'
        if args.phase in ('prepare', 'all', 'smoke'):
            prepare_cache(args.pairdata, args.meta, args.beam_model, cache,
                          max_tokens=lengths.pop(), smoke_limit=args.smoke_limit if args.phase == 'smoke' else None)
        if args.phase in ('select', 'all', 'smoke'):
            selected = run_experiment(config, cache, args.meta, args.beam_model, args.output, args.device,
                                      smoke=args.phase == 'smoke')
            print('val選定: '+json.dumps({k:v['name'] for k,v in selected['winners'].items()}, ensure_ascii=False), flush=True)
        if args.phase in ('test', 'all'):
            run_experiment(config, cache, args.meta, args.beam_model, args.output, args.device, phase='test')
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
