"""
ドラム譜評価スクリプト

処理フロー:
1. 入力MIDIディレクトリ × 指定難易度ラベルのリスト　の組み合わせ分だけ生成し
   manifest.json（譜面と難易度ラベルの対応）を保存する。
2. 入力譜面・生成譜面を standard トークンに変換し各評価指標を算出、
   results.json に保存する。

使用方法:
    python scripts/evaluate/evaluate.py \
        --input_midi_dir <dir> \
        --model_path <ckpt.pt> \
        --difficulty_labels_file <labels.json> \
        --output_dir <out_dir>

難易度ラベルファイル形式 (JSON):
    [
      {"s_tech_cls": 2, "s_indep_cls": 2, "s_hand_cls": 2, "s_foot_cls": 2, "s_move_cls": 2},
      {"s_tech_cls": 5, "s_indep_cls": 5, "s_hand_cls": 5, "s_foot_cls": 5, "s_move_cls": 5}
    ]
    または [[2,2,2,2,2], [5,5,5,5,5]] 形式も可。
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.musemorphose import MuseMorphose
from src.models.limb_bias import build_token_to_limb_map
from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.utils.drum_to_midi import tokens_to_midi, cp_data_to_midi
from src.utils.tokenization_mode import validate_tokenization_method

from scripts.generate.generate_drum_from_input import (
    DIFFICULTY_KEYS,
    adapt_checkpoint_to_model,
    build_window_plan,
    compute_difficulty_levels,
    count_cp_bars,
    count_token_bars,
    extract_bar_latents_cp,
    extract_bar_latents_standard_or_remi,
    generate_with_difficulty_control,
    merge_window_cp_chunks,
    merge_window_token_chunks,
    resolve_mode_adapter,
    split_cp_data_into_bar_chunks,
    split_tokens_into_bar_chunks,
    validate_vocabulary_compatibility,
    verify_bar_count,
    _resolve_difficulty_upper_bounds,
    _resolve_runtime_model_options_cp,
    _slice_cp_data_by_bar_range,
    _slice_tokens_by_bar_range,
)
from scripts.generate.generate_drum_cp import generate_cp_sequence

from scripts.evaluate.metrics.groove_similarity import (
    groove_similarity,
    hihat_similarity,
    rhythm_similarity,
)
from scripts.evaluate.metrics.token_iou import token_iou
from scripts.evaluate.metrics.spearman import compute_spearman_metrics


# ---------------------------------------------------------------------------
# 難易度ラベル読み込み
# ---------------------------------------------------------------------------

def load_difficulty_labels(labels_file: str) -> List[Dict[str, Any]]:
    with open(labels_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    labels: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            for k in DIFFICULTY_KEYS:
                if k not in item:
                    raise ValueError(f'難易度ラベルにキーが不足しています: {k}')
            mode = item.get('difficulty_mode', 'full_assign')
            labels.append({k: int(item[k]) for k in DIFFICULTY_KEYS} | {'difficulty_mode': mode})
        elif isinstance(item, list) and len(item) == len(DIFFICULTY_KEYS):
            labels.append({k: int(v) for k, v in zip(DIFFICULTY_KEYS, item)} | {'difficulty_mode': 'full_assign'})
        else:
            raise ValueError(f'不正な難易度ラベル形式: {item}')
    return labels


# ---------------------------------------------------------------------------
# Limb/Position Bias パラメータ解決
# ---------------------------------------------------------------------------

def _resolve_bias_params(mconf: dict, tokenizer) -> dict:
    """
    config と tokenizer から limb/position bias 関連パラメータを解決する。
    学習スクリプト (train_drum.py) と同一ロジック。
    """
    use_limb_bias_encoder = bool(mconf.get('use_limb_bias_encoder', False))
    use_limb_bias_decoder = bool(mconf.get('use_limb_bias_decoder', False))
    alpha_limb_fixed = mconf.get('limb_bias_alpha', None)
    if alpha_limb_fixed is not None:
        alpha_limb_fixed = float(alpha_limb_fixed)

    token_to_limb = None
    if use_limb_bias_encoder or use_limb_bias_decoder:
        idx2token = getattr(tokenizer, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(tokenizer.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            print('[warning] idx2token not found in tokenizer; limb bias disabled.')
            use_limb_bias_encoder = False
            use_limb_bias_decoder = False

    use_position_bias_encoder = bool(mconf.get('use_position_bias_encoder', False))
    use_position_bias_decoder = bool(mconf.get('use_position_bias_decoder', False))
    position_bias_mode = mconf.get('position_bias_mode', 'uniform')
    alpha_pos_fixed = mconf.get('position_bias_alpha', None)
    if alpha_pos_fixed is not None:
        alpha_pos_fixed = float(alpha_pos_fixed)
    position_bias_per_layer = bool(mconf.get('position_bias_per_layer', True))

    pos_token_ids = None
    if use_position_bias_encoder or use_position_bias_decoder:
        t2i = getattr(tokenizer, 'token2idx', None)
        if t2i is not None:
            pos_keys  = sorted([k for k in t2i if k.startswith('<POS_')],  key=lambda k: t2i[k])
            beat_keys = sorted([k for k in t2i if k.startswith('<BEAT_')], key=lambda k: t2i[k])
            if pos_keys and beat_keys:
                pos_token_ids = {
                    'pos_start':  t2i[pos_keys[0]],
                    'pos_end':    t2i[pos_keys[-1]],
                    'beat_start': t2i[beat_keys[0]],
                    'beat_end':   t2i[beat_keys[-1]],
                    'pad':        t2i.get('<PAD>', 0),
                    'eos':        t2i.get('<EOS>', 1),
                }
                print(f'[info] position bias token IDs: {pos_token_ids}')
            else:
                print('[warning] POS/BEAT tokens not found in tokenizer; position bias disabled.')
                use_position_bias_encoder = False
                use_position_bias_decoder = False
        else:
            print('[warning] token2idx not found in tokenizer; position bias disabled.')
            use_position_bias_encoder = False
            use_position_bias_decoder = False

    # position bias は token_to_limb_table も必要（musemorphose.py の _pos_ready チェック用）
    if (use_position_bias_encoder or use_position_bias_decoder) and token_to_limb is None:
        idx2token = getattr(tokenizer, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(tokenizer.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            print('[warning] idx2token not found in tokenizer; position bias disabled.')
            use_position_bias_encoder = False
            use_position_bias_decoder = False
            pos_token_ids = None

    return dict(
        use_limb_bias_encoder=use_limb_bias_encoder,
        use_limb_bias_decoder=use_limb_bias_decoder,
        token_to_limb=token_to_limb,
        alpha_limb_fixed=alpha_limb_fixed,
        use_position_bias_encoder=use_position_bias_encoder,
        use_position_bias_decoder=use_position_bias_decoder,
        position_bias_mode=position_bias_mode,
        alpha_pos_fixed=alpha_pos_fixed,
        position_bias_per_layer=position_bias_per_layer,
        pos_token_ids=pos_token_ids,
        bias_schedule_encoder=mconf.get('bias_schedule_encoder', 'both'),
        bias_schedule_decoder=mconf.get('bias_schedule_decoder', 'both'),
        bias_schedule_encoder_custom=mconf.get('bias_schedule_encoder_custom', None),
        bias_schedule_decoder_custom=mconf.get('bias_schedule_decoder_custom', None),
    )


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

def _build_model(
    mconf: dict,
    tokenizer,
    tokenization_method: str,
    device: str,
    runtime_opts_cp: Optional[dict] = None,
    bias_params: Optional[dict] = None,
) -> MuseMorphose:
    common_kwargs = dict(
        d_s_tech_emb=mconf.get('d_s_tech_emb', 32),
        d_s_indep_emb=mconf.get('d_s_indep_emb', 32),
        d_s_hand_emb=mconf.get('d_s_hand_emb', 32),
        d_s_foot_emb=mconf.get('d_s_foot_emb', 32),
        d_s_move_emb=mconf.get('d_s_move_emb', 32),
        n_s_tech_cls=mconf.get('n_s_tech_cls', 8),
        n_s_indep_cls=mconf.get('n_s_indep_cls', 8),
        n_s_hand_cls=mconf.get('n_s_hand_cls', 8),
        n_s_foot_cls=mconf.get('n_s_foot_cls', 8),
        n_s_move_cls=mconf.get('n_s_move_cls', 8),
        use_difficulty=True,
        use_attr_cls=True,
        tokenization_method=tokenization_method,
        **(bias_params or {}),
    )

    if tokenization_method == 'cp_limb_v1':
        model = MuseMorphose(
            mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
            mconf['d_latent'], mconf['d_embed'], tokenizer.vocab_size,
            cp_event_type_vocab_size=len(tokenizer.event_type2idx),
            cp_type_vocab_size=len(tokenizer.type_token2idx),
            cp_struct_vocab_size=len(tokenizer.struct_token2idx),
            cp_pos_vocab_size=tokenizer.pos_pad_value + 1,
            cp_limb_vocab_size=len(tokenizer.limb_token2idx),
            cp_event_pad_idx=tokenizer.event_type2idx['<PAD_EVENT>'],
            cp_struct_pad_idx=tokenizer.struct_token2idx['<PAD>'],
            cp_pos_pad_idx=tokenizer.pos_pad_value,
            cp_limb_pad_idx=tokenizer.limb_token2idx['<PAD>'],
            d_cp_type_emb=runtime_opts_cp['d_cp_type_emb'],
            d_cp_pos_emb=mconf.get('d_cp_pos_emb', 64),
            d_cp_limb_emb=mconf.get('d_cp_limb_emb', 64),
            cond_mode=runtime_opts_cp['cond_mode'],
            bar_anchor_mode=runtime_opts_cp['bar_anchor_mode'],
            bar_token_idx=runtime_opts_cp['bar_token_idx'],
            use_hierarchical_pe=runtime_opts_cp['use_hierarchical_pe'],
            hier_pe_bar_vocab_size=runtime_opts_cp['hier_pe_bar_vocab_size'],
            hier_pe_beat_vocab_size=runtime_opts_cp['hier_pe_beat_vocab_size'],
            hier_pe_pos_vocab_size=runtime_opts_cp['hier_pe_pos_vocab_size'],
            hier_pe_use_struct_bias=runtime_opts_cp['hier_pe_use_struct_bias'],
            **common_kwargs,
        ).to(device)
    else:
        cond_mode = mconf.get('cond_mode', 'in-attn')
        if cond_mode not in ('in-attn', 'pre-attn'):
            cond_mode = 'in-attn'
        bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
        if bar_anchor_mode not in ('first_token', 'bar_token'):
            bar_anchor_mode = 'first_token'
        model = MuseMorphose(
            mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
            mconf['d_latent'], mconf['d_embed'], tokenizer.vocab_size,
            cond_mode=cond_mode,
            bar_anchor_mode=bar_anchor_mode,
            bar_token_idx=tokenizer.token2idx.get('<BAR>', None),
            use_hierarchical_pe=bool(mconf.get('use_hierarchical_pe', False)),
            hier_pe_bar_vocab_size=max(1, int(mconf.get('hier_pe_bar_vocab_size', 512))),
            hier_pe_beat_vocab_size=max(1, int(mconf.get('hier_pe_beat_vocab_size', 8))),
            hier_pe_pos_vocab_size=max(1, int(mconf.get('hier_pe_pos_vocab_size', 32))),
            hier_pe_use_struct_bias=bool(mconf.get('hier_pe_use_struct_bias', True)),
            **common_kwargs,
        ).to(device)

    return model


# ---------------------------------------------------------------------------
# 1サンプル生成
# ---------------------------------------------------------------------------

def _generate_one(
    model,
    tokenizer,
    tokenize_fn,
    tokenization_method: str,
    midi_path: Path,
    target_label: Dict[str, Any],
    config: dict,
    device: str,
    bpm: int,
    nucleus_p: float,
    temperature: float,
    runtime_opts_cp: Optional[dict] = None,
    difficulty_bounds: Optional[dict] = None,
) -> Tuple[Any, str, Dict[str, float]]:
    """生成済みデータと種別 ('tokens' or 'cp_data') および実効指定難易度を返す。"""
    mode_payload, mode_bar_positions = tokenize_fn(str(midi_path))
    n_bars = len(mode_bar_positions)

    generate_cfg = config.get('generate', {})
    max_bars = int(generate_cfg.get('max_bars', n_bars))
    enc_seqlen = int(config['data']['enc_seqlen'])
    max_dec_len = int(generate_cfg.get('max_input_dec_seqlen', 1280))
    truncate_len = min(512, max(64, max_dec_len - 32))

    window_plan = build_window_plan(n_bars, max_bars, max_bars)

    bounds = _resolve_difficulty_upper_bounds(config['model'])
    diff_mode = target_label.get('difficulty_mode', 'full_assign')

    if diff_mode in ('global_shift', 'upper_cap'):
        std_tok = DrumTokenizer()
        std_tokens, std_bar_pos = std_tok.midi_to_tokens(str(midi_path))
        base = compute_difficulty_levels(std_tokens, std_bar_pos, difficulty_bounds, bpm)
        base = {k: np.resize(base[k], n_bars) for k in DIFFICULTY_KEYS}
        if diff_mode == 'global_shift':
            target_per_bar = {
                k: np.clip(base[k] + target_label[k], 0, bounds[k]).astype(np.int64)
                for k in DIFFICULTY_KEYS
            }
        else:  # upper_cap
            target_per_bar = {
                k: np.clip(np.minimum(base[k], target_label[k]), 0, bounds[k]).astype(np.int64)
                for k in DIFFICULTY_KEYS
            }
    else:  # full_assign
        target_per_bar = {
            k: np.clip(np.full(n_bars, target_label[k], dtype=np.int64), 0, bounds[k])
            for k in DIFFICULTY_KEYS
        }

    window_results = []

    for st_bar, ed_bar in window_plan:
        win_diff = {k: target_per_bar[k][st_bar:ed_bar] for k in DIFFICULTY_KEYS}

        if tokenization_method in ('standard', 'remi'):
            win_tokens = _slice_tokens_by_bar_range(mode_payload, mode_bar_positions, st_bar, ed_bar)
            local_bar_pos = [p - mode_bar_positions[st_bar] for p in mode_bar_positions[st_bar:ed_bar]]

            latents = extract_bar_latents_standard_or_remi(
                model, tokenizer, win_tokens, local_bar_pos, enc_seqlen, device,
                use_sampling=False, sampling_var=0.,
            )
            gen_indices, _, _ = generate_with_difficulty_control(
                model, latents,
                win_diff['s_tech_cls'], win_diff['s_indep_cls'],
                win_diff['s_hand_cls'], win_diff['s_foot_cls'], win_diff['s_move_cls'],
                tokenizer, device=device,
                max_input_len=max_dec_len, truncate_len=truncate_len,
                nucleus_p=nucleus_p, temperature=temperature,
            )
            gen_tokens = [tokenizer.idx2token[idx] for idx in gen_indices]
            chunks = split_tokens_into_bar_chunks(gen_tokens)
        else:
            win_cp, local_bar_pos = _slice_cp_data_by_bar_range(
                mode_payload, mode_bar_positions, st_bar, ed_bar
            )
            latents = extract_bar_latents_cp(
                model, win_cp, local_bar_pos, enc_seqlen, tokenizer, device,
                use_sampling=False, sampling_var=0.,
            )
            cp_gen, _, _ = generate_cp_sequence(
                model, latents,
                tokenizer.struct_token2idx, tokenizer.idx2struct_token,
                tokenizer.idx2limb_token, tokenizer.pos_pad_value,
                tokenizer.limb_token2idx['<PAD>'],
                tokenizer.event_type2idx['<PAD_EVENT>'],
                device=device,
                max_events=int(generate_cfg.get('dec_seqlen', 12800)),
                max_input_len=max_dec_len, truncate_len=truncate_len,
                nucleus_p=nucleus_p, temperature=temperature,
                use_difficulty=True,
                s_tech_cls=win_diff['s_tech_cls'],
                s_indep_cls=win_diff['s_indep_cls'],
                s_hand_cls=win_diff['s_hand_cls'],
                s_foot_cls=win_diff['s_foot_cls'],
                s_move_cls=win_diff['s_move_cls'],
                use_hierarchical_pe=bool(runtime_opts_cp and runtime_opts_cp['use_hierarchical_pe']),
            )
            chunks = split_cp_data_into_bar_chunks(cp_gen, tokenizer.idx2type_token)

        verify_bar_count(ed_bar - st_bar, len(chunks), context=f'bars [{st_bar},{ed_bar})')
        window_results.append((st_bar, ed_bar, chunks))

    specified_diff = {k: float(np.mean(target_per_bar[k])) for k in DIFFICULTY_KEYS}
    if tokenization_method in ('standard', 'remi'):
        merged = merge_window_token_chunks(n_bars, window_results)
        return merged, 'tokens', specified_diff
    else:
        merged = merge_window_cp_chunks(n_bars, window_results, tokenizer)
        return merged, 'cp_data', specified_diff


# ---------------------------------------------------------------------------
# フェーズ1：生成
# ---------------------------------------------------------------------------

def phase1_generate(
    args,
    model,
    tokenizer,
    tokenize_fn,
    tokenization_method: str,
    config: dict,
    device: str,
    difficulty_labels: List[Dict[str, Any]],
    output_dir: Path,
    runtime_opts_cp: Optional[dict] = None,
    difficulty_bounds: Optional[dict] = None,
) -> List[Dict]:
    midi_files = sorted(
        list(Path(args.input_midi_dir).glob('*.mid'))
        + list(Path(args.input_midi_dir).glob('*.midi'))
    )
    if not midi_files:
        print(f'[ERROR] MIDIファイルが見つかりません: {args.input_midi_dir}')
        sys.exit(1)

    gen_dir = output_dir / 'generated_midi'
    gen_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    total = len(midi_files) * len(difficulty_labels)
    done = 0

    for midi_path in midi_files:
        for label_idx, label in enumerate(difficulty_labels):
            out_name = f'{midi_path.stem}_label{label_idx:03d}.mid'
            out_path = gen_dir / out_name
            done += 1
            print(f'[{done}/{total}] {midi_path.name}  label={label_idx} {label}')

            try:
                gen_data, data_type, specified_diff = _generate_one(
                    model, tokenizer, tokenize_fn, tokenization_method,
                    midi_path, label, config, device, args.bpm,
                    args.nucleus_p, args.temperature,
                    runtime_opts_cp=runtime_opts_cp,
                    difficulty_bounds=difficulty_bounds,
                )
                if data_type == 'tokens':
                    tokens_to_midi(gen_data, str(out_path), bpm=args.bpm)
                else:
                    cp_data_to_midi(
                        gen_data, tokenizer.idx2type_token, tokenizer.idx2limb_token,
                        str(out_path), bpm=args.bpm,
                    )
                manifest.append({
                    'input_midi': str(midi_path),
                    'generated_midi': str(out_path),
                    'specified_difficulty': specified_diff,
                    'label_idx': label_idx,
                })
                print(f'  → {out_path}')
            except Exception as e:
                print(f'  [WARNING] 生成失敗: {e}')

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'\nManifest → {manifest_path}  ({len(manifest)} 件)')
    return manifest


# ---------------------------------------------------------------------------
# フェーズ2：評価
# ---------------------------------------------------------------------------

def phase2_evaluate(
    manifest: List[Dict],
    args,
    difficulty_bounds: dict,
    output_dir: Path,
) -> Dict:
    std_tokenizer = DrumTokenizer()
    sample_results: List[Dict] = []

    print(f'\n評価: {len(manifest)} サンプル')
    for i, entry in enumerate(manifest):
        try:
            ref_tokens, _ = std_tokenizer.midi_to_tokens(entry['input_midi'])
            gen_tokens, _ = std_tokenizer.midi_to_tokens(entry['generated_midi'])
        except Exception as e:
            print(f'  [WARNING] トークン化失敗 (sample {i}): {e}')
            continue

        hs = hihat_similarity(ref_tokens, gen_tokens)
        rs = rhythm_similarity(ref_tokens, gen_tokens)
        gs = groove_similarity(ref_tokens, gen_tokens, args.alpha, args.beta)
        iou = token_iou(ref_tokens, gen_tokens)

        sample_results.append({
            'input_midi': entry['input_midi'],
            'generated_midi': entry['generated_midi'],
            'specified_difficulty': entry['specified_difficulty'],
            'label_idx': entry['label_idx'],
            'hihat_similarity': hs,
            'rhythm_similarity': rs,
            'groove_similarity': gs,
            'token_iou': iou,
        })

    def _mean(key):
        return float(np.mean([s[key] for s in sample_results])) if sample_results else float('nan')

    results: Dict = {
        'n_samples': len(sample_results),
        'alpha': args.alpha,
        'beta': args.beta,
        'mean_hihat_similarity': _mean('hihat_similarity'),
        'mean_rhythm_similarity': _mean('rhythm_similarity'),
        'mean_groove_similarity': _mean('groove_similarity'),
        'mean_token_iou': _mean('token_iou'),
    }

    print('Spearman 相関を計算中...')
    results['spearman'] = compute_spearman_metrics(manifest, difficulty_bounds, bpm=args.bpm)

    results['samples'] = sample_results

    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'Results → {results_path}')
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Drum generation evaluation')
    parser.add_argument('--input_midi_dir', required=True, help='入力MIDIディレクトリ')
    parser.add_argument('--model_path', required=True, help='モデルチェックポイントパス')
    parser.add_argument('--difficulty_bounds', required=True, help='難易度境界値pklファイル')
    parser.add_argument('--difficulty_labels_file', required=True,
                        help='難易度ラベルリストJSONファイル')
    parser.add_argument('--output_dir', required=True, help='出力ディレクトリ')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Groove Similarity における Rhythm Similarity の重み (default: 0.7)')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Groove Similarity における Hihat Similarity の重み (default: 0.3)')
    parser.add_argument('--bpm', type=int, default=120)
    parser.add_argument('--nucleus_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--no_limb_bias_encoder', action='store_true',
                        help='use_limb_bias_encoder を強制的に False にする')
    parser.add_argument('--no_limb_bias_decoder', action='store_true',
                        help='use_limb_bias_decoder を強制的に False にする')
    parser.add_argument('--no_position_bias_encoder', action='store_true',
                        help='use_position_bias_encoder を強制的に False にする')
    parser.add_argument('--no_position_bias_decoder', action='store_true',
                        help='use_position_bias_decoder を強制的に False にする')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 難易度ラベル・境界値ロード
    difficulty_labels = load_difficulty_labels(args.difficulty_labels_file)
    print(f'難易度ラベル: {len(difficulty_labels)} 件')

    with open(args.difficulty_bounds, 'rb') as f:
        difficulty_bounds = pickle.load(f)

    # config・vocab・モデルロード
    model_path = Path(args.model_path)
    config_path = model_path.parent.parent / 'config.yaml'
    if not config_path.exists():
        print(f'[ERROR] config.yaml が見つかりません: {config_path}')
        sys.exit(1)

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard')
    )
    device = config['training']['device']
    print(f'tokenization_method={tokenization_method}  device={device}')

    with open(config['data']['vocab_path'], 'rb') as f:
        vocab_data = pickle.load(f)

    adapter = resolve_mode_adapter(tokenization_method, vocab_data)
    tokenizer = adapter['tokenizer']
    tokenize_fn = adapter['tokenize_fn']

    mconf = config['model']
    runtime_opts_cp = None
    if tokenization_method == 'cp_limb_v1':
        runtime_opts_cp = _resolve_runtime_model_options_cp(mconf, tokenizer)

    checkpoint = torch.load(model_path, map_location=device)
    ckpt_sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # チェックポイントの実際のembedding形状からn_s_*_cls を上書き
    # (学習時はdifficulty_num_classes = max_label+2 でbuildされるため configの値と異なる場合がある)
    for attr in ('tech', 'indep', 'hand', 'foot', 'move'):
        key = f's_{attr}_attr_emb.emb_lookup.weight'
        if key in ckpt_sd:
            mconf[f'n_s_{attr}_cls'] = int(ckpt_sd[key].size(0))

    # limb/position biasを強制的に無効化
    if args.no_limb_bias_encoder:
        mconf['use_limb_bias_encoder'] = False
    if args.no_limb_bias_decoder:
        mconf['use_limb_bias_decoder'] = False
    if args.no_position_bias_encoder:
        mconf['use_position_bias_encoder'] = False
    if args.no_position_bias_decoder:
        mconf['use_position_bias_decoder'] = False

    bias_params = _resolve_bias_params(mconf, tokenizer)
    model = _build_model(mconf, tokenizer, tokenization_method, device, runtime_opts_cp, bias_params)

    validate_vocabulary_compatibility(tokenization_method, tokenizer, ckpt_sd)
    adapted_sd = adapt_checkpoint_to_model(
        ckpt_sd, tokenizer.vocab_size,
        ckpt_sd['token_emb.emb_lookup.weight'].size(0),
    )
    model.load_state_dict(adapted_sd, strict=False)
    model.eval()
    print(f'モデルロード完了: {model_path}')

    # フェーズ1: 生成
    manifest = phase1_generate(
        args, model, tokenizer, tokenize_fn, tokenization_method,
        config, device, difficulty_labels, output_dir, runtime_opts_cp,
        difficulty_bounds=difficulty_bounds,
    )

    # フェーズ2: 評価
    results = phase2_evaluate(manifest, args, difficulty_bounds, output_dir)

    print('\n' + '=' * 60)
    print('評価完了')
    print(f'  サンプル数:           {results["n_samples"]}')
    print(f'  Hihat Similarity:     {results["mean_hihat_similarity"]:.4f}')
    print(f'  Rhythm Similarity:    {results["mean_rhythm_similarity"]:.4f}')
    print(f'  Groove Similarity:    {results["mean_groove_similarity"]:.4f}  '
          f'(α={args.alpha}, β={args.beta})')
    print(f'  Token IoU:            {results["mean_token_iou"]:.4f}')
    print(f'  Results:              {output_dir / "results.json"}')
    print('=' * 60)


if __name__ == '__main__':
    main()
