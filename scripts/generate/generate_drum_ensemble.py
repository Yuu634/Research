"""
ドラム譜アンサンブルデコーディング生成スクリプト

複数の学習済みチェックポイント（同一アーキテクチャ・同一vocab、tokenization_method
in {standard, remi} のみ対応）を用い、生成ステップ毎に全モデルの次トークン確率分布を
平均してから1トークンをサンプルし、その同一トークンを全モデルにフィードバックする
「アンサンブルデコーディング」を行う。

各モデルのforwardはKVキャッシュを持たない（本リポジトリ共通の制約）ため、単一GPU上で
torch.cuda.Stream + Pythonスレッドを用いてモデルごとのforward呼び出しを同時発行し、
ベストエフォートでオーバーラップさせる（真のN分の1速度は保証しない。deviceが非CUDAの
場合は逐次実行にフォールバックする）。

依存関係の向き: 本ファイルは scripts/evaluate/evaluate.py の _resolve_bias_params /
_build_model をimportする。evaluate.py は既存の generate_drum_from_input.py から
importしており、本ファイルとevaluate.pyの間に循環importは無い
（evaluate.pyは本ファイルをimportしない）が、依存の向きが従来と逆になる点に注意。

使用方法:
    python scripts/generate/generate_drum_ensemble.py \\
        --model_paths ckpt1/params/best_params.pt ckpt2/params/best_params.pt \\
        --output output_ensemble --n_pieces 5 --n_samples_per_piece 3
"""

import sys
import os
import time
import random
import pickle
import argparse
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.drum_dataloader import DrumTransformerDataset
from src.utils.utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from src.utils.drum_to_midi import tokens_to_midi
from src.utils.tokenization_mode import validate_tokenization_method

from scripts.generate.generate_drum_from_input import (
    adapt_checkpoint_to_model,
    resolve_mode_adapter,
    validate_vocabulary_compatibility,
    extract_bar_latents_standard_or_remi,
    temperatured_softmax,
    nucleus,
)
from scripts.generate.generate_drum_difficulty import (
    get_latent_embedding_fast,
    random_shift_difficulty_cls,
)
# evaluate.py 側の bias パラメータ解決・モデル構築ロジックを再利用する
# （チェックポイントごとに use_limb_bias_*/use_position_bias_* 等が異なるため、
#  各チェックポイント自身のconfigから個別にモデルを構築する必要がある）
from scripts.evaluate.evaluate import _resolve_bias_params, _build_model


# ---------------------------------------------------------------------------
# チェックポイント読み込み
# ---------------------------------------------------------------------------

def _peek_device(model_path: Path) -> str:
    """torch.loadを行わずconfig.yamlのみからtraining.deviceを読む（アンサンブル全体の
    共有deviceを、全モデルロード前に軽量に解決するため）。"""
    config_path = Path(model_path).parent.parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml が見つかりません: {config_path}')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    return config['training']['device']


def load_ensemble_member(model_path: Path, device: str) -> Dict[str, Any]:
    """
    1チェックポイントを、そのチェックポイント自身のconfig.yamlから構築する。
    biaslossチェックポイント群はそれぞれ独立にuse_limb_bias_*/use_position_bias_*/
    cond_mode等を持つため、共有configは使わずチェックポイントごとにフレッシュに
    yaml.loadし直す（evaluate.pyの単一モデルロード手順と同一ロジックを1個ずつ適用）。

    tokenization_methodが 'cp_limb_v1' の場合はValueErrorで即エラーにする
    （本スクリプトはstandard/remiのみ対応）。
    """
    model_path = Path(model_path)
    config_path = model_path.parent.parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml が見つかりません: {config_path} (checkpoint={model_path})')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard')
    )
    if tokenization_method not in ('standard', 'remi'):
        raise ValueError(
            f'generate_drum_ensemble.py は tokenization_method in (standard, remi) のみ対応です。 '
            f'checkpoint={model_path} は tokenization_method={tokenization_method!r} (config={config_path})'
        )

    with open(config['data']['vocab_path'], 'rb') as f:
        vocab_data = pickle.load(f)
    adapter = resolve_mode_adapter(tokenization_method, vocab_data)
    tokenizer = adapter['tokenizer']

    mconf = config['model']
    checkpoint = torch.load(model_path, map_location=device)
    ckpt_sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # チェックポイントの実際のembedding形状からn_s_*_clsを上書き（evaluate.pyと同一ロジック）
    for attr in ('tech', 'indep', 'hand', 'foot', 'move'):
        key = f's_{attr}_attr_emb.emb_lookup.weight'
        if key in ckpt_sd:
            mconf[f'n_s_{attr}_cls'] = int(ckpt_sd[key].size(0))

    bias_params = _resolve_bias_params(mconf, tokenizer)
    model = _build_model(mconf, tokenizer, tokenization_method, device, None, bias_params)

    validate_vocabulary_compatibility(tokenization_method, tokenizer, ckpt_sd)
    adapted_sd = adapt_checkpoint_to_model(
        ckpt_sd, tokenizer.vocab_size,
        ckpt_sd['token_emb.emb_lookup.weight'].size(0),
    )
    try:
        model.load_state_dict(adapted_sd)
        print(f'[info] {model_path}: strict=True でロード成功')
    except RuntimeError as e:
        print(f'[warning] {model_path}: strict=True 失敗 ({e})。strict=False で再試行します。')
        model.load_state_dict(adapted_sd, strict=False)
    model.eval()

    return dict(
        model=model,
        config=config,
        tokenization_method=tokenization_method,
        tokenizer=tokenizer,
        vocab_path=str(config['data']['vocab_path']),
        model_path=model_path,
    )


def validate_ensemble_vocabulary_compatibility(members: List[Dict[str, Any]]) -> None:
    """
    全メンバーが同一vocab（token2idx完全一致・vocab_size一致）・同一tokenization_method
    であることを検証する。確率分布の平均化はインデックス空間が同一でなければ無意味なため、
    無条件必須のチェックとしスキップ不可にする。
    """
    if len(members) < 2:
        raise ValueError('アンサンブルには2個以上のモデルが必要です。')

    modes = {m['tokenization_method'] for m in members}
    if len(modes) > 1:
        detail = ', '.join(f'{m["model_path"]}={m["tokenization_method"]}' for m in members)
        raise ValueError(f'アンサンブルメンバー間でtokenization_methodが一致しません: {detail}')

    ref = members[0]
    ref_tokenizer = ref['tokenizer']
    for m in members[1:]:
        if m['tokenizer'].vocab_size != ref_tokenizer.vocab_size:
            raise ValueError(
                f'vocab_sizeが一致しません: {ref["model_path"]}={ref_tokenizer.vocab_size}, '
                f'{m["model_path"]}={m["tokenizer"].vocab_size}'
            )
        if m['tokenizer'].token2idx != ref_tokenizer.token2idx:
            raise ValueError(
                f'token2idxが一致しません: {ref["model_path"]} と {m["model_path"]} は '
                f'同一vocab_pathである必要があります。'
            )

    print(f'[info] EnsembleVocabularyGuard: OK ({len(members)}モデル, '
          f'vocab_size={ref_tokenizer.vocab_size}, mode={ref["tokenization_method"]})')


# ---------------------------------------------------------------------------
# 並列ステップ実行（CUDA Stream + スレッド）
# ---------------------------------------------------------------------------

class EnsembleStepRunner:
    """
    N個のモデル + N個のtorch.cuda.Streamを保持し、1生成ステップ分のforwardを
    Pythonスレッドで同時発行する（カーネル起動はGILを解放するため、複数スレッドから
    同時にforwardを起動することでGPU側のカーネル実行が重なる余地が生まれる）。
    device が非CUDAの場合は単純な逐次呼び出しにフォールバックする
    （torch.cuda.StreamはCUDA専用のため）。

    スレッド内で例外が発生してもjoin()自体は失敗しないため、各スレッドで例外を捕捉して
    メインスレッドに戻ってから再送出する。
    """

    def __init__(self, models: List[Any], device: str):
        self.models = models
        self.device = device
        self.is_cuda = torch.device(device).type == 'cuda'
        self.streams = (
            [torch.cuda.Stream(device=device) for _ in models] if self.is_cuda else None
        )

    def step(self, step_fns: List[Callable[[], torch.Tensor]]) -> List[torch.Tensor]:
        n = len(self.models)
        results: List[Optional[torch.Tensor]] = [None] * n
        errors: List[Optional[BaseException]] = [None] * n

        if not self.is_cuda:
            for i, fn in enumerate(step_fns):
                try:
                    results[i] = fn()
                except BaseException as e:  # noqa: BLE001
                    errors[i] = e
        else:
            def _run(i):
                try:
                    with torch.cuda.stream(self.streams[i]):
                        results[i] = step_fns[i]()
                except BaseException as e:  # noqa: BLE001
                    errors[i] = e

            threads = [threading.Thread(target=_run, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # thread.join()はカーネル起動のみ待つ（CUDA呼び出しは非同期のため）。
            # 結果を読む前に必ず全streamの完了を待つ。
            for s in self.streams:
                s.synchronize()

        for i, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f'アンサンブルメンバー{i}の生成ステップでエラーが発生しました') from e

        return results  # type: ignore[return-value]


def combine_logits_to_probs(logits_list: List[np.ndarray], temperature: float) -> np.ndarray:
    """
    各モデルのlogitsに temperatured_softmax を適用してから算術平均する
    （logitsそのものは平均しない。理由: 各チェックポイントは学習時の正則化強度が
    異なりlogitスケールに互換性がないため、logit平均だとスケールの大きいモデルが
    不当に支配する。post-softmax平均はスケール不変であり、意見が割れた場合も
    全ゼロのような退化を起こさない）。
    """
    per_model_probs = [temperatured_softmax(lg, temperature) for lg in logits_list]
    avg_probs = np.mean(np.stack(per_model_probs, axis=0), axis=0)
    avg_probs = avg_probs / avg_probs.sum()
    return avg_probs


# ---------------------------------------------------------------------------
# 潜在表現抽出（メンバーごと。encoder重みが異なるため共有不可）
# ---------------------------------------------------------------------------

def extract_bar_latents_ensemble(
    members: List[Dict[str, Any]],
    window_tokens: List[str],
    window_bar_positions: List[int],
    enc_seqlen: int,
    device: str,
    use_sampling: bool,
    sampling_var: float,
) -> List[torch.Tensor]:
    """evaluate.py 側 (_generate_one_ensemble) から呼ばれる、window_tokens ベースの
    latent抽出。各メンバーのencoderは重みが異なるためlatentは個別抽出する。
    生成ループ前に1回だけ呼ばれる処理であり、並列化は不要（逐次ループで十分）。"""
    return [
        extract_bar_latents_standard_or_remi(
            m['model'], m['tokenizer'], window_tokens, window_bar_positions,
            enc_seqlen, device, use_sampling, sampling_var,
        )
        for m in members
    ]


# ---------------------------------------------------------------------------
# アンサンブル生成ループ
# ---------------------------------------------------------------------------

def generate_with_difficulty_control_ensemble(
        models: List[Any],
        latents_per_model: List[torch.Tensor],
        s_tech_cls, s_indep_cls, s_hand_cls, s_foot_cls, s_move_cls,
        tokenizer,
        device: str = 'cuda',
        max_events: int = 12800,
        primer: Optional[List[str]] = None,
        max_input_len: int = 1280,
        truncate_len: int = 512,
        nucleus_p: float = 0.9,
        temperature: float = 1.2,
) -> Tuple[List[int], float, np.ndarray]:
    """
    generate_with_difficulty_control (generate_drum_from_input.py:1102-1300) のアンサンブル版。
    制御フロー（<BAR>/<PAD>/<EOS>処理、EOS dead-loop対策、コンテキストトランケート）は
    単一モデル版と完全に同一。差分は:
      - dec_input（トークン列）は全モデル共有の単一テンソル
      - latent_placeholder / s_*_placeholder はモデルごとに独立して保持
        （トランケート時も各モデル分をそれぞれシフトする）
      - サンプリング部分のみ EnsembleStepRunner.step() で全モデルのforwardを
        同時発行 → combine_logits_to_probs() で確率分布を平均 → nucleus() で
        サンプル → 得られた同一トークンindexを全モデルのplaceholderに書き戻す
    戻り値の形は単一モデル版と同一 (generated_final[:-1], time_elapsed, entropies)。
    """
    n_models = len(models)
    assert n_models == len(latents_per_model)
    if n_models < 2:
        raise ValueError('アンサンブル生成には2個以上のモデルが必要です。')

    s_tech_cls_t = numpy_to_tensor(s_tech_cls, device=device).long()
    s_indep_cls_t = numpy_to_tensor(s_indep_cls, device=device).long()
    s_hand_cls_t = numpy_to_tensor(s_hand_cls, device=device).long()
    s_foot_cls_t = numpy_to_tensor(s_foot_cls, device=device).long()
    s_move_cls_t = numpy_to_tensor(s_move_cls, device=device).long()

    states = []
    for latents in latents_per_model:
        states.append(dict(
            latent_placeholder=torch.zeros(max_events, 1, latents.size(-1)).to(device),
            s_tech_placeholder=torch.zeros(max_events, 1, dtype=torch.long).to(device),
            s_indep_placeholder=torch.zeros(max_events, 1, dtype=torch.long).to(device),
            s_hand_placeholder=torch.zeros(max_events, 1, dtype=torch.long).to(device),
            s_foot_placeholder=torch.zeros(max_events, 1, dtype=torch.long).to(device),
            s_move_placeholder=torch.zeros(max_events, 1, dtype=torch.long).to(device),
        ))

    print('[info] Difficulty classes per bar:')
    print(f'  S_tech:  {tensor_to_numpy(s_tech_cls_t)}')
    print(f'  S_indep: {tensor_to_numpy(s_indep_cls_t)}')
    print(f'  S_hand:  {tensor_to_numpy(s_hand_cls_t)}')
    print(f'  S_foot:  {tensor_to_numpy(s_foot_cls_t)}')
    print(f'  S_move:  {tensor_to_numpy(s_move_cls_t)}')

    # 初期化（共有トークン列）
    if primer is None:
        generated = [tokenizer.token2idx['<BAR>']]
    else:
        generated = [tokenizer.token2idx[e] for e in primer]
        for i, latents in enumerate(latents_per_model):
            st = states[i]
            st['latent_placeholder'][:len(generated), 0, :] = latents[0].squeeze(0)
            st['s_tech_placeholder'][:len(generated), 0] = s_tech_cls_t[0]
            st['s_indep_placeholder'][:len(generated), 0] = s_indep_cls_t[0]
            st['s_hand_placeholder'][:len(generated), 0] = s_hand_cls_t[0]
            st['s_foot_placeholder'][:len(generated), 0] = s_foot_cls_t[0]
            st['s_move_placeholder'][:len(generated), 0] = s_move_cls_t[0]

    target_bars = latents_per_model[0].size(0)
    generated_bars = 0

    runner = EnsembleStepRunner(models, device)

    time_st = time.time()
    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []
    early_eos_streak = 0

    while generated_bars < target_bars:
        if len(generated) == 1:
            dec_input = numpy_to_tensor([generated], device=device).long()
        else:
            dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

        closures = []
        for i in range(n_models):
            st = states[i]
            latents = latents_per_model[i]
            st['latent_placeholder'][len(generated) - 1, 0, :] = latents[generated_bars]
            st['s_tech_placeholder'][len(generated) - 1, 0] = s_tech_cls_t[generated_bars]
            st['s_indep_placeholder'][len(generated) - 1, 0] = s_indep_cls_t[generated_bars]
            st['s_hand_placeholder'][len(generated) - 1, 0] = s_hand_cls_t[generated_bars]
            st['s_foot_placeholder'][len(generated) - 1, 0] = s_foot_cls_t[generated_bars]
            st['s_move_placeholder'][len(generated) - 1, 0] = s_move_cls_t[generated_bars]

            dec_seg_emb = st['latent_placeholder'][:len(generated), :]
            dec_s_tech = st['s_tech_placeholder'][:len(generated), :]
            dec_s_indep = st['s_indep_placeholder'][:len(generated), :]
            dec_s_hand = st['s_hand_placeholder'][:len(generated), :]
            dec_s_foot = st['s_foot_placeholder'][:len(generated), :]
            dec_s_move = st['s_move_placeholder'][:len(generated), :]

            def make_closure(model=models[i], dec_seg_emb=dec_seg_emb, dec_s_tech=dec_s_tech,
                              dec_s_indep=dec_s_indep, dec_s_hand=dec_s_hand,
                              dec_s_foot=dec_s_foot, dec_s_move=dec_s_move):
                def _call():
                    with torch.no_grad():
                        return model.generate(
                            dec_input, dec_seg_emb,
                            s_tech_cls=dec_s_tech, s_indep_cls=dec_s_indep,
                            s_hand_cls=dec_s_hand, s_foot_cls=dec_s_foot,
                            s_move_cls=dec_s_move,
                        )
                return _call

            closures.append(make_closure())

        logits_list = runner.step(closures)
        logits_np = [tensor_to_numpy(lg[0]) for lg in logits_list]
        avg_probs = combine_logits_to_probs(logits_np, temperature)
        word = nucleus(avg_probs, nucleus_p)
        word_token = tokenizer.idx2token[word]

        # <PAD>はスキップ
        if word_token == '<PAD>':
            continue

        # target_bars未達のEOSは破棄して継続（早期終了ガード）
        if word_token == '<EOS>':
            if generated_bars < target_bars:
                early_eos_streak += 1
                print(
                    '[warning] Early EOS detected. '
                    f'generated_bars={generated_bars}, target_bars={target_bars}, '
                    f'streak={early_eos_streak}. Continue generation.'
                )
                if early_eos_streak >= 32:
                    forced_bar_idx = tokenizer.token2idx['<BAR>']
                    generated.append(forced_bar_idx)
                    generated_final.append(forced_bar_idx)
                    entropies.append(entropy(avg_probs))
                    generated_bars += 1
                    cur_input_len += 1
                    early_eos_streak = 0
                    print(
                        '[warning] DecoderTerminationGuard injected <BAR> to avoid EOS dead-loop. '
                        f'Generated {generated_bars}/{target_bars} bars.'
                    )
                continue
            print('[info] Generation completed with EOS')
            break

        early_eos_streak = 0

        if word_token == '<BAR>':
            generated_bars += 1
            print(f'[info] Generated {generated_bars}/{target_bars} bars, #events = {len(generated_final)}')

        if len(generated) >= max_events:
            raise RuntimeError(
                'DecoderTerminationGuard: reached max_events before completing target bars '
                f'(generated_bars={generated_bars}, target_bars={target_bars}, max_events={max_events})'
            )

        generated.append(word)
        generated_final.append(word)
        entropies.append(entropy(avg_probs))

        cur_input_len += 1

        # コンテキストウィンドウのトランケート（全メンバー分をシフト）
        assert cur_input_len == len(generated)
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]
            for st in states:
                st['latent_placeholder'][:len(generated) - 1, 0, :] = st['latent_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0, :
                ]
                st['s_tech_placeholder'][:len(generated) - 1, 0] = st['s_tech_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0
                ]
                st['s_indep_placeholder'][:len(generated) - 1, 0] = st['s_indep_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0
                ]
                st['s_hand_placeholder'][:len(generated) - 1, 0] = st['s_hand_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0
                ]
                st['s_foot_placeholder'][:len(generated) - 1, 0] = st['s_foot_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0
                ]
                st['s_move_placeholder'][:len(generated) - 1, 0] = st['s_move_placeholder'][
                    cur_input_len - truncate_len:cur_input_len - 1, 0
                ]
            print(f'[info] Context truncated: accumulated_len={len(generated_final)}')
            cur_input_len = len(generated)

    if generated_bars != target_bars:
        raise RuntimeError(
            'DecoderTerminationGuard: generated bar count mismatch at decode end '
            f'(generated_bars={generated_bars}, target_bars={target_bars})'
        )
    print(f'-- Generated {len(generated_final)} events (ensemble of {n_models} models)')
    print(f'-- Time elapsed: {time.time() - time_st:.2f} secs')

    return generated_final[:-1], time.time() - time_st, np.array(entropies)


# ---------------------------------------------------------------------------
# 単体実行用CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='MuseMorphose アンサンブルデコーディング（複数モデルの確率分布平均化による生成）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/generate/generate_drum_ensemble.py \\
    --model_paths trained_model/checkpoints_drum_standard-lakhaug1-biasloss1/params/best_params.pt \\
                  trained_model/checkpoints_drum_standard-lakhaug1-biasloss5/params/best_params.pt \\
    --output output_ensemble --n_pieces 5 --n_samples_per_piece 3
        """,
    )
    parser.add_argument('--model_paths', nargs='+', required=True,
                         help='2個以上のチェックポイント.ptパス。config.yamlは'
                              '<ckpt>.parent.parent/config.yaml から自動解決する')
    parser.add_argument('--output', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--n_pieces', type=int, default=5)
    parser.add_argument('--n_samples_per_piece', type=int, default=3)
    parser.add_argument('--nucleus_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--device', type=str, default=None,
                         help='全メンバー共通のdeviceを上書き指定。未指定時は先頭チェックポイントの'
                              'config.yaml training.device を使用する')
    parser.add_argument('--seed', type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if len(args.model_paths) < 2:
        raise ValueError(
            'generate_drum_ensemble.py には2個以上の --model_paths が必要です。'
            '単一モデルの生成には generate_drum_difficulty.py を使用してください。'
        )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model_paths = [Path(p) for p in args.model_paths]
    device = args.device or _peek_device(model_paths[0])
    print(f'[info] device={device}')

    print('[info] Loading ensemble members...')
    members = [load_ensemble_member(p, device) for p in model_paths]
    validate_ensemble_vocabulary_compatibility(members)

    ref_config = members[0]['config']
    tokenizer = members[0]['tokenizer']
    data_cfg = ref_config['data']
    generate_cfg = ref_config.get('generate', {})

    total_param_bytes = sum(
        p.numel() * p.element_size() for m in members for p in m['model'].parameters()
    )
    print(f'[info] Ensemble total parameter memory: {total_param_bytes / 1e6:.1f} MB '
          f'({len(members)} models)')

    pieces_path = data_cfg.get('test_split') or data_cfg.get('val_split') or data_cfg.get('train_split')
    if pieces_path is None:
        raise ValueError("config['data'] に test_split/val_split/train_split のいずれも見つかりません。")

    print(f'[info] Loading dataset (pieces={pieces_path})...')
    dset = DrumTransformerDataset(
        data_cfg['data_dir'],
        data_cfg['vocab_path'],
        model_enc_seqlen=data_cfg['enc_seqlen'],
        model_dec_seqlen=data_cfg.get('dec_seqlen', 1280),
        model_max_bars=data_cfg.get('max_bars', 16),
        pieces=pickle_load(pieces_path),
        pad_to_same=False,
        use_difficulty=True,
        tokenization_method=members[0]['tokenization_method'],
    )
    print(f'[info] Dataset size: {len(dset)} pieces')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces = random.sample(range(len(dset)), min(args.n_pieces, len(dset)))
    print(f'[info] Sampled pieces: {pieces}')

    max_input_dec_seqlen = int(generate_cfg.get('max_input_dec_seqlen', 1280))
    truncate_len = min(512, max(64, max_input_dec_seqlen - 32))

    times = []
    all_entropies = []

    for p_idx, p in enumerate(pieces):
        print(f'\n{"=" * 60}')
        print(f'Processing piece {p_idx + 1}/{len(pieces)} (dataset index: {p})')
        print(f'{"=" * 60}')

        p_data = dset[p]
        p_id = p_data['piece_id']
        p_bar_id = p_data['st_bar_id']

        for k in list(p_data.keys()):
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        original_tokens = [
            dset.idx2token[idx] for idx in tensor_to_numpy(p_data['dec_input'][:int(p_data['length'])])
        ]
        original_file = out_dir / f'id{p}_bar{p_bar_id}_original.mid'
        tokens_to_midi(original_tokens, str(original_file), bpm=120)
        print(f'[info] Saved original MIDI: {original_file}')

        print('[info] Extracting latent embeddings for each ensemble member...')
        latents_per_member = [
            get_latent_embedding_fast(
                m['model'], p_data,
                use_sampling=generate_cfg.get('use_latent_sampling', True),
                sampling_var=generate_cfg.get('latent_sampling_var', 0.1),
                device=device,
            )
            for m in members
        ]

        # 難易度シフト演算はCPU/numpy側で行う（device tensor + numpy array の混在演算を避けるため）
        original_s_tech = tensor_to_numpy(p_data['s_tech_cls'])
        original_s_indep = tensor_to_numpy(p_data['s_indep_cls'])
        original_s_hand = tensor_to_numpy(p_data['s_hand_cls'])
        original_s_foot = tensor_to_numpy(p_data['s_foot_cls'])
        original_s_move = tensor_to_numpy(p_data['s_move_cls'])
        # 'enc_n_bars' は元譜面全体の小節数であり、このウィンドウのs_tech_cls等の長さとは
        # 一致しない場合があるため、difficultyシフトの形状は実際の配列長から決める。
        n_bars = len(original_s_tech)

        diff_shifts = random_shift_difficulty_cls(args.n_samples_per_piece, n_bars, upper=4, lower=-3)

        for samp in range(args.n_samples_per_piece):
            print(f'\n[info] Generating ensemble sample {samp + 1}/{args.n_samples_per_piece}')

            s_tech_cls = np.clip(original_s_tech + diff_shifts[samp, 0], 0, 7).astype(np.int64)
            s_indep_cls = np.clip(original_s_indep + diff_shifts[samp, 1], 0, 7).astype(np.int64)
            s_hand_cls = np.clip(original_s_hand + diff_shifts[samp, 2], 0, 7).astype(np.int64)
            s_foot_cls = np.clip(original_s_foot + diff_shifts[samp, 3], 0, 7).astype(np.int64)
            s_move_cls = np.clip(original_s_move + diff_shifts[samp, 4], 0, 7).astype(np.int64)

            out_file = out_dir / (
                f'id{p}_bar{p_bar_id}_ensemble_sample{samp + 1:02d}_'
                f'tech{int(diff_shifts[samp, 0, 0]):+d}_indep{int(diff_shifts[samp, 1, 0]):+d}_'
                f'hand{int(diff_shifts[samp, 2, 0]):+d}_foot{int(diff_shifts[samp, 3, 0]):+d}_'
                f'move{int(diff_shifts[samp, 4, 0]):+d}'
            )

            song_indices, t_sec, ent = generate_with_difficulty_control_ensemble(
                [m['model'] for m in members], latents_per_member,
                s_tech_cls, s_indep_cls, s_hand_cls, s_foot_cls, s_move_cls,
                tokenizer,
                device=device,
                max_input_len=max_input_dec_seqlen,
                truncate_len=truncate_len,
                nucleus_p=args.nucleus_p,
                temperature=args.temperature,
            )
            times.append(t_sec)
            all_entropies.append(ent)

            song_tokens = [tokenizer.idx2token[idx] for idx in song_indices]

            tokens_to_midi(song_tokens, str(out_file) + '.mid', bpm=120)
            print(f'[info] Saved MIDI: {out_file}.mid')

            with open(str(out_file) + '.txt', 'w') as f:
                for token in song_tokens:
                    f.write(token + '\n')

            np.save(str(out_file) + '_S_TECH_CLS.npy', s_tech_cls)
            np.save(str(out_file) + '_S_INDEP_CLS.npy', s_indep_cls)
            np.save(str(out_file) + '_S_HAND_CLS.npy', s_hand_cls)
            np.save(str(out_file) + '_S_FOOT_CLS.npy', s_foot_cls)
            np.save(str(out_file) + '_S_MOVE_CLS.npy', s_move_cls)
            np.save(str(out_file) + '_entropies.npy', ent)

    print(f'\n{"=" * 60}')
    print('Ensemble generation completed!')
    print(f'{"=" * 60}')
    print(f'Total pieces: {len(pieces)}')
    print(f'Samples per piece: {args.n_samples_per_piece}')
    print(f'Total samples: {len(pieces) * args.n_samples_per_piece}')
    print(f'Ensemble size: {len(members)} models')
    if times:
        print(f'Average generation time: {np.mean(times):.2f} +/- {np.std(times):.2f} secs')
    if all_entropies:
        print(f'Average entropy: {np.mean([e.mean() for e in all_entropies]):.3f}')
    print(f'Output directory: {out_dir}')


if __name__ == '__main__':
    main()
