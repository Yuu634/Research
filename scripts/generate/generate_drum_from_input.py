"""
ドラム譜面生成スクリプト（入力MIDIから）

入力されたドラム譜MIDIファイルから学習済みモデルを用いて、
指定した演奏難易度レベルに沿った新しいドラム譜を生成します。

処理フロー:
1. 学習済みモデルのロード（トークン設計手法はconfig.yamlから取得）
2. 入力譜面データ（ドラム譜MIDI）の読み込み
3. 入力譜面データから各小節の演奏難易度レベル算出
4. 入力譜面データ&各小節の演奏難易度レベルをモデルに入力し、
   指定した演奏難易度レベルに沿ったドラム譜を生成
5. 指定した保存先パスにMIDIファイルを保存

使用方法:
    python generate_drum_from_input.py --model_path <model_path> \\
        --input_midi <input_midi> --output_midi <output_midi>
"""

import sys, os, time
import json
from copy import deepcopy
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
import numpy as np
from scipy.stats import entropy
import miditoolkit

# パス設定
sys.path.append('./model')

from src.models.musemorphose import MuseMorphose
from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
from src.data.drum_difficulty_calculator import (
    compute_all_difficulty_scores,
    discretize_difficulty_score
)
from src.utils.utils import numpy_to_tensor, tensor_to_numpy
from src.utils.drum_to_midi import tokens_to_midi
from src.utils.tokenization_mode import (
    ensure_runtime_mode_compatibility,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
)
from src.utils.tokenization_policy import POSITIONS_PER_BEAT
from src.utils.drum_to_midi import cp_data_to_midi

try:
    from scripts.preprocess.prepare_drum_dataset import RemiDrumTokenizer
except ModuleNotFoundError:
    RemiDrumTokenizer = None

try:
    from generate_drum_cp import generate_cp_sequence
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from scripts.generate.generate_drum_cp import generate_cp_sequence


DIFFICULTY_KEYS = ('s_tech_cls', 's_indep_cls', 's_hand_cls', 's_foot_cls', 's_move_cls')
SUPPORTED_MODES = ('standard', 'remi', 'cp_limb_v1')



###########################################
# Checkpoint互換性処理
###########################################
def adapt_checkpoint_to_model(state_dict, model_vocab_size, old_vocab_size):
    """
    古いcheckpointを新しいモデル構造に適応
    
    Args:
        state_dict: 読み込まれたcheckpointの状態辞書
        model_vocab_size: 現在のモデルのトークン数
        old_vocab_size: checkpointの古いトークン数
        
    Returns:
        adjusted_state_dict: 適応されたstate_dict
    """
    adjusted_state_dict = {}
    
    # 削除すべき古い属性埋め込みキーのプレフィックス
    old_attr_prefixes = ['rfreq_attr_emb', 'polyph_attr_emb']
    
    # 新しい難易度属性キー（checkpointに存在しないはず）
    new_difficulty_prefixes = ['s_tech_attr_emb', 's_indep_attr_emb', 
                               's_hand_attr_emb', 's_foot_attr_emb', 's_move_attr_emb']
    
    for key, param in state_dict.items():
        # 古い属性埋め込みは削除
        skip = False
        for prefix in old_attr_prefixes:
            if key.startswith(prefix):
                print(f'[info] Removing outdated key: {key}')
                skip = True
                break
        
        if skip:
            continue
        
        # tokenサイズが異なる場合の処理
        if key in ['token_emb.emb_lookup.weight', 'dec_out_proj.weight', 'dec_out_proj.bias']:
            if key == 'dec_out_proj.bias':
                expected_size = model_vocab_size
            else:
                expected_size = model_vocab_size
            
            if param.size(0) != expected_size:
                old_size = param.size(0)
                print(f'[warning] Size mismatch for {key}: checkpoint={old_size}, model={expected_size}')
                
                if key in ['token_emb.emb_lookup.weight', 'dec_out_proj.weight']:
                    # 重みを切り詰めまたはパディング
                    if old_size > expected_size:
                        # 古いvocabサイズが大きい場合は切り詰め
                        adjusted_state_dict[key] = param[:expected_size, :]
                        print(f'[info] Truncated {key} from {old_size} to {expected_size}')
                    else:
                        # 新しいvocabサイズが大きい場合はパディング
                        padding = torch.zeros(expected_size - old_size, param.size(1), 
                                             dtype=param.dtype, device=param.device)
                        adjusted_state_dict[key] = torch.cat([param, padding], dim=0)
                        print(f'[info] Padded {key} from {old_size} to {expected_size}')
                
                elif key == 'dec_out_proj.bias':
                    # バイアスの処理
                    if old_size > expected_size:
                        adjusted_state_dict[key] = param[:expected_size]
                        print(f'[info] Truncated {key} from {old_size} to {expected_size}')
                    else:
                        padding = torch.zeros(expected_size - old_size, 
                                             dtype=param.dtype, device=param.device)
                        adjusted_state_dict[key] = torch.cat([param, padding], dim=0)
                        print(f'[info] Padded {key} from {old_size} to {expected_size}')
            else:
                adjusted_state_dict[key] = param
        else:
            adjusted_state_dict[key] = param
    
    return adjusted_state_dict


###########################################
# MIDI読み込みとトークン化
###########################################
def load_midi_and_tokenize(midi_path, tokenizer):
    """
    MIDIファイルを読み込み、トークン列に変換
    
    Args:
        midi_path: MIDIファイルパス
        tokenizer: DrumTokenizer または DrumCPTokenizer
        
    Returns:
        tokens: トークン列のリスト
        bar_positions: 各小節の開始位置インデックス
    """
    # midi_to_tokens は既に bar_positions を計算して返す
    tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
    
    return tokens, bar_positions


###########################################
# 難易度レベル計算
###########################################
def compute_difficulty_levels(tokens, bar_positions, difficulty_bounds, bpm=120.0):
    """
    トークン列から各小節の難易度レベルを計算
    
    Args:
        tokens: トークン列
        bar_positions: 小節の開始位置リスト
        difficulty_bounds: 難易度境界値辞書（pkl）
        bpm: テンポ
        
    Returns:
        difficulty_levels: dict of numpy arrays
            's_tech_cls': (n_bars,)
            's_indep_cls': (n_bars,)
            's_hand_cls': (n_bars,)
            's_foot_cls': (n_bars,)
            's_move_cls': (n_bars,)
    """
    n_bars = len(bar_positions)
    bar_edges = list(bar_positions) + [len(tokens)]
    
    s_tech_levels = []
    s_indep_levels = []
    s_hand_levels = []
    s_foot_levels = []
    s_move_levels = []
    
    for i in range(n_bars):
        start_pos = bar_edges[i]
        end_pos = bar_edges[i + 1]
        bar_tokens = tokens[start_pos:end_pos]
        
        # 難易度スコア計算
        scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)
        
        # スコアをクラスに離散化
        s_tech_cls = discretize_difficulty_score(scores['s_tech'], difficulty_bounds['s_tech'])
        s_indep_cls = discretize_difficulty_score(scores['s_indep'], difficulty_bounds['s_indep'])
        s_hand_cls = discretize_difficulty_score(scores['s_hand'], difficulty_bounds['s_hand'])
        s_foot_cls = discretize_difficulty_score(scores['s_foot'], difficulty_bounds['s_foot'])
        s_move_cls = discretize_difficulty_score(scores['s_move'], difficulty_bounds['s_move'])
        
        s_tech_levels.append(s_tech_cls)
        s_indep_levels.append(s_indep_cls)
        s_hand_levels.append(s_hand_cls)
        s_foot_levels.append(s_foot_cls)
        s_move_levels.append(s_move_cls)
    
    return {
        's_tech_cls': np.array(s_tech_levels, dtype=np.int64),
        's_indep_cls': np.array(s_indep_levels, dtype=np.int64),
        's_hand_cls': np.array(s_hand_levels, dtype=np.int64),
        's_foot_cls': np.array(s_foot_levels, dtype=np.int64),
        's_move_cls': np.array(s_move_levels, dtype=np.int64),
    }


def _resolve_difficulty_upper_bounds(model_config):
    return {
        's_tech_cls': max(0, int(model_config.get('n_s_tech_cls', 8)) - 1),
        's_indep_cls': max(0, int(model_config.get('n_s_indep_cls', 8)) - 1),
        's_hand_cls': max(0, int(model_config.get('n_s_hand_cls', 8)) - 1),
        's_foot_cls': max(0, int(model_config.get('n_s_foot_cls', 8)) - 1),
        's_move_cls': max(0, int(model_config.get('n_s_move_cls', 8)) - 1),
    }


def _parse_per_bar_difficulty_labels(path, expected_bars):
    """
    full_assign 用の難易度ラベルファイル(JSON)をパースする。

    対応形式:
    1) dict of arrays:
       {"s_tech_cls":[...], "s_indep_cls":[...], ...}
    2) list of dicts:
       [{"s_tech_cls":1, "s_indep_cls":2, ...}, ...]
    3) list of lists:
       [[s_tech, s_indep, s_hand, s_foot, s_move], ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    parsed = {k: None for k in DIFFICULTY_KEYS}

    if isinstance(payload, dict):
        for key in DIFFICULTY_KEYS:
            if key not in payload:
                raise ValueError(f'full_assign labels missing key: {key}')
            values = payload[key]
            if not isinstance(values, list):
                raise ValueError(f'full_assign labels for {key} must be a list')
            parsed[key] = np.array(values, dtype=np.int64)

    elif isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError('full_assign labels cannot be empty')

        if isinstance(payload[0], dict):
            rows = payload
            for key in DIFFICULTY_KEYS:
                try:
                    parsed[key] = np.array([int(row[key]) for row in rows], dtype=np.int64)
                except KeyError:
                    raise ValueError(f'full_assign row dict missing key: {key}')

        elif isinstance(payload[0], list):
            rows = payload
            for i, row in enumerate(rows):
                if len(row) != len(DIFFICULTY_KEYS):
                    raise ValueError(
                        f'full_assign row length must be {len(DIFFICULTY_KEYS)} at row {i}, got {len(row)}'
                    )
            arr = np.array(rows, dtype=np.int64)
            for idx, key in enumerate(DIFFICULTY_KEYS):
                parsed[key] = arr[:, idx]
        else:
            raise ValueError('full_assign list format must be list[dict] or list[list]')
    else:
        raise ValueError('full_assign labels file must be a dict or list JSON payload')

    for key in DIFFICULTY_KEYS:
        if len(parsed[key]) != expected_bars:
            raise ValueError(
                f'full_assign labels length mismatch for {key}: expected {expected_bars}, got {len(parsed[key])}'
            )

    return parsed


def resolve_target_difficulty_levels(
    base_levels,
    difficulty_mode,
    bounds,
    difficulty_shift=None,
    difficulty_cap=None,
    labels_file=None,
):
    """難易度指定方式（full_assign / global_shift / upper_cap）を解決する。"""
    if difficulty_mode == 'global_shift':
        if difficulty_shift is None:
            raise ValueError('global_shift mode requires --difficulty_shift (5 integers)')

        shift_map = {
            's_tech_cls': int(difficulty_shift[0]),
            's_indep_cls': int(difficulty_shift[1]),
            's_hand_cls': int(difficulty_shift[2]),
            's_foot_cls': int(difficulty_shift[3]),
            's_move_cls': int(difficulty_shift[4]),
        }

        return {
            key: np.clip(base_levels[key] + shift_map[key], 0, bounds[key]).astype(np.int64)
            for key in DIFFICULTY_KEYS
        }

    if difficulty_mode == 'upper_cap':
        if difficulty_cap is None:
            raise ValueError('upper_cap mode requires --difficulty_cap (5 integers)')

        cap_map = {
            's_tech_cls': int(difficulty_cap[0]),
            's_indep_cls': int(difficulty_cap[1]),
            's_hand_cls': int(difficulty_cap[2]),
            's_foot_cls': int(difficulty_cap[3]),
            's_move_cls': int(difficulty_cap[4]),
        }

        return {
            key: np.clip(np.minimum(base_levels[key], cap_map[key]), 0, bounds[key]).astype(np.int64)
            for key in DIFFICULTY_KEYS
        }

    if difficulty_mode == 'full_assign':
        if not labels_file:
            raise ValueError('full_assign mode requires --difficulty_labels_file')

        assigned = _parse_per_bar_difficulty_labels(labels_file, len(base_levels['s_tech_cls']))
        return {
            key: np.clip(assigned[key], 0, bounds[key]).astype(np.int64)
            for key in DIFFICULTY_KEYS
        }

    raise ValueError(
        f'Unsupported difficulty_mode={difficulty_mode}. '
        'Available: full_assign, global_shift, upper_cap'
    )


def validate_difficulty_cli_args(args):
    """タスク1.1向け: 難易度指定引数の整合性を事前検証する。"""
    if args.difficulty_mode == 'global_shift':
        if args.difficulty_shift is None or len(args.difficulty_shift) != 5:
            raise ValueError('global_shift mode requires --difficulty_shift with 5 integers')

    if args.difficulty_mode == 'upper_cap':
        if args.difficulty_cap is None or len(args.difficulty_cap) != 5:
            raise ValueError('upper_cap mode requires --difficulty_cap with 5 integers')

    if args.difficulty_mode == 'full_assign':
        if not args.difficulty_labels_file:
            raise ValueError('full_assign mode requires --difficulty_labels_file')

        labels_path = Path(args.difficulty_labels_file)
        if not labels_path.exists():
            raise ValueError(f'full_assign labels file not found: {labels_path}')


def _safe_positive_int(value, default_value, field_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        print(f'[warning] Invalid {field_name}={value}. Falling back to {default_value}.')
        return default_value

    if parsed <= 0:
        print(f'[warning] Non-positive {field_name}={parsed}. Falling back to {default_value}.')
        return default_value

    return parsed


def _resolve_runtime_model_options_cp(mconf, tokenizer):
    cond_mode = mconf.get('cond_mode', 'in-attn')
    if cond_mode not in ('in-attn', 'pre-attn'):
        print(f"[warning] Unsupported cond_mode={cond_mode}. Falling back to 'in-attn'.")
        cond_mode = 'in-attn'

    bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
    if bar_anchor_mode not in ('first_token', 'bar_token'):
        print(f"[warning] Unsupported bar_anchor_mode={bar_anchor_mode}. Falling back to 'first_token'.")
        bar_anchor_mode = 'first_token'

    bar_token_idx = getattr(tokenizer, 'type_token2idx', {}).get('<BAR>', None)
    if bar_anchor_mode == 'bar_token' and bar_token_idx is None:
        print('[warning] bar_anchor_mode=bar_token but bar token is unavailable. Falling back to first_token.')
        bar_anchor_mode = 'first_token'

    return {
        'cond_mode': cond_mode,
        'bar_anchor_mode': bar_anchor_mode,
        'bar_token_idx': bar_token_idx,
        'use_hierarchical_pe': bool(mconf.get('use_hierarchical_pe', False)),
        'hier_pe_bar_vocab_size': _safe_positive_int(mconf.get('hier_pe_bar_vocab_size', 512), 512, 'model.hier_pe_bar_vocab_size'),
        'hier_pe_beat_vocab_size': _safe_positive_int(mconf.get('hier_pe_beat_vocab_size', 8), 8, 'model.hier_pe_beat_vocab_size'),
        'hier_pe_pos_vocab_size': _safe_positive_int(mconf.get('hier_pe_pos_vocab_size', 32), 32, 'model.hier_pe_pos_vocab_size'),
        'hier_pe_use_struct_bias': bool(mconf.get('hier_pe_use_struct_bias', True)),
        'd_cp_type_emb': _safe_positive_int(mconf.get('d_cp_type_emb', 64), 64, 'model.d_cp_type_emb'),
    }


def resolve_mode_adapter(tokenization_method, vocab_data):
    """tokenization_method に応じて tokenizer adapter を返す。"""
    if tokenization_method == 'standard':
        if not isinstance(vocab_data, tuple) or len(vocab_data) < 2:
            raise ValueError('standard mode expects tuple vocab payload (token2idx, idx2token).')
        tokenizer = DrumTokenizer()
        tokenizer.token2idx, tokenizer.idx2token = vocab_data
        tokenizer.vocab_size = len(tokenizer.token2idx)
        return {
            'mode': 'standard',
            'tokenizer': tokenizer,
            'tokenize_fn': lambda midi_path: tokenizer.midi_to_tokens(midi_path),
        }

    if tokenization_method == 'remi':
        if RemiDrumTokenizer is None:
            raise ValueError('RemiDrumTokenizer import failed. Check scripts/preprocess/prepare_drum_dataset.py')
        tokenizer = RemiDrumTokenizer()
        if isinstance(vocab_data, dict):
            tokenizer.token2idx = vocab_data['token2idx']
            tokenizer.idx2token = vocab_data['idx2token']
            tokenizer.vocab_size = len(tokenizer.token2idx)
        else:
            raise ValueError('remi mode expects dict vocab payload with token2idx/idx2token.')
        return {
            'mode': 'remi',
            'tokenizer': tokenizer,
            'tokenize_fn': lambda midi_path: tokenizer.midi_to_tokens(midi_path),
        }

    if tokenization_method == 'cp_limb_v1':
        tokenizer = DrumCPTokenizer()
        if not isinstance(vocab_data, dict):
            raise ValueError('cp_limb_v1 mode expects dict vocab payload.')

        tokenizer.event_type2idx = vocab_data.get('event_type2idx', tokenizer.event_type2idx)
        tokenizer.idx2event_type = vocab_data.get('idx2event_type', {v: k for k, v in tokenizer.event_type2idx.items()})
        tokenizer.type_token2idx = vocab_data.get('type_token2idx', vocab_data.get('struct_token2idx', tokenizer.type_token2idx))
        tokenizer.idx2type_token = vocab_data.get('idx2type_token', vocab_data.get('idx2struct_token', tokenizer.idx2type_token))
        tokenizer.struct_token2idx = tokenizer.type_token2idx
        tokenizer.idx2struct_token = tokenizer.idx2type_token
        tokenizer.limb_token2idx = vocab_data.get('limb_token2idx', tokenizer.limb_token2idx)
        tokenizer.idx2limb_token = vocab_data.get('idx2limb_token', tokenizer.idx2limb_token)
        tokenizer.pos_pad_value = int(vocab_data.get('pos_pad_value', tokenizer.pos_pad_value))
        tokenizer.vocab_size = len(tokenizer.type_token2idx)

        return {
            'mode': 'cp_limb_v1',
            'tokenizer': tokenizer,
            'tokenize_fn': lambda midi_path: tokenizer.midi_to_cp_data(midi_path),
        }

    raise ValueError(
        f'Unsupported tokenization_method={tokenization_method}. '
        f'Supported: {", ".join(SUPPORTED_MODES)}'
    )


def _lookup_idx2token(idx2token, idx):
    if isinstance(idx2token, dict):
        if idx in idx2token:
            return idx2token[idx]
        return idx2token.get(str(idx), None)
    if isinstance(idx2token, list):
        if 0 <= idx < len(idx2token):
            return idx2token[idx]
    return None


def _validate_special_token_contract(token2idx, idx2token, required_tokens, label):
    missing_tokens = [tok for tok in required_tokens if tok not in token2idx]
    if missing_tokens:
        raise ValueError(
            f'{label}: missing required special tokens: {missing_tokens}'
        )

    broken_roundtrip = []
    for tok in required_tokens:
        idx = int(token2idx[tok])
        restored = _lookup_idx2token(idx2token, idx)
        if restored != tok:
            broken_roundtrip.append(
                f'{tok}(idx={idx}, idx2token={restored})'
            )

    if broken_roundtrip:
        raise ValueError(
            f'{label}: special token round-trip mismatch: {broken_roundtrip}'
        )


def _validate_checkpoint_rows(checkpoint_state_dict, key, expected_rows, label):
    tensor = checkpoint_state_dict.get(key, None)
    if tensor is None:
        raise ValueError(f'{label}: checkpoint key not found: {key}')

    actual_rows = int(tensor.size(0))
    if actual_rows != expected_rows:
        raise ValueError(
            f'{label}: vocabulary mismatch at {key} '
            f'(expected={expected_rows}, actual={actual_rows})'
        )


def validate_vocabulary_compatibility(tokenization_method, tokenizer, checkpoint_state_dict):
    """生成前に語彙整合を検証し、不一致時は fail-fast する。"""
    print(f'[info] Running vocabulary compatibility guard for mode={tokenization_method}')

    expected_token_vocab = int(tokenizer.vocab_size)
    _validate_checkpoint_rows(
        checkpoint_state_dict,
        'token_emb.emb_lookup.weight',
        expected_token_vocab,
        'VocabularyCompatibilityGuard',
    )
    _validate_checkpoint_rows(
        checkpoint_state_dict,
        'dec_out_proj.weight',
        expected_token_vocab,
        'VocabularyCompatibilityGuard',
    )
    _validate_checkpoint_rows(
        checkpoint_state_dict,
        'dec_out_proj.bias',
        expected_token_vocab,
        'VocabularyCompatibilityGuard',
    )

    if tokenization_method in ('standard', 'remi'):
        _validate_special_token_contract(
            tokenizer.token2idx,
            tokenizer.idx2token,
            required_tokens=['<PAD>', '<BAR>', '<EOS>'],
            label='VocabularyCompatibilityGuard',
        )
        print(
            '[info] VocabularyCompatibilityGuard passed: '
            f'mode={tokenization_method}, vocab_size={expected_token_vocab}, '
            'special_tokens=[<PAD>, <BAR>, <EOS>]'
        )
        return

    if tokenization_method == 'cp_limb_v1':
        _validate_special_token_contract(
            tokenizer.type_token2idx,
            tokenizer.idx2type_token,
            required_tokens=['<PAD>', '<BAR>', '<EOS>'],
            label='VocabularyCompatibilityGuard(type)',
        )
        _validate_special_token_contract(
            tokenizer.limb_token2idx,
            tokenizer.idx2limb_token,
            required_tokens=['<PAD>', '<NA>'],
            label='VocabularyCompatibilityGuard(limb)',
        )

        if '<PAD_EVENT>' not in tokenizer.event_type2idx:
            raise ValueError(
                'VocabularyCompatibilityGuard(event): missing required special token <PAD_EVENT>'
            )
        pad_event_idx = int(tokenizer.event_type2idx['<PAD_EVENT>'])
        pad_event_tok = _lookup_idx2token(tokenizer.idx2event_type, pad_event_idx)
        if pad_event_tok != '<PAD_EVENT>':
            raise ValueError(
                'VocabularyCompatibilityGuard(event): round-trip mismatch for <PAD_EVENT> '
                f'(idx={pad_event_idx}, idx2event_type={pad_event_tok})'
            )

        cp_pos_vocab = int(tokenizer.pos_pad_value) + 1
        if cp_pos_vocab <= 0:
            raise ValueError(
                f'VocabularyCompatibilityGuard(cp_pos): invalid pos vocabulary size={cp_pos_vocab}'
            )

        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_composite_emb.type_emb.weight',
            len(tokenizer.type_token2idx),
            'VocabularyCompatibilityGuard(cp_type)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_composite_emb.pos_emb.weight',
            cp_pos_vocab,
            'VocabularyCompatibilityGuard(cp_pos)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_composite_emb.hand1_emb.weight',
            len(tokenizer.limb_token2idx),
            'VocabularyCompatibilityGuard(cp_limb)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_event_type_head.weight',
            len(tokenizer.event_type2idx),
            'VocabularyCompatibilityGuard(cp_event_head)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_struct_head.weight',
            len(tokenizer.struct_token2idx),
            'VocabularyCompatibilityGuard(cp_struct_head)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_pos_head.weight',
            cp_pos_vocab,
            'VocabularyCompatibilityGuard(cp_pos_head)',
        )
        _validate_checkpoint_rows(
            checkpoint_state_dict,
            'cp_hand1_head.weight',
            len(tokenizer.limb_token2idx),
            'VocabularyCompatibilityGuard(cp_limb_head)',
        )

        print(
            '[info] VocabularyCompatibilityGuard passed: '
            f'mode={tokenization_method}, '
            f'token_vocab={expected_token_vocab}, '
            f'event_vocab={len(tokenizer.event_type2idx)}, '
            f'type_vocab={len(tokenizer.type_token2idx)}, '
            f'pos_vocab={cp_pos_vocab}, '
            f'limb_vocab={len(tokenizer.limb_token2idx)}'
        )
        return

    raise ValueError(
        f'VocabularyCompatibilityGuard: unsupported mode={tokenization_method}'
    )


def verify_bar_count(expected_bars, actual_bars, context):
    if int(actual_bars) != int(expected_bars):
        raise ValueError(
            f'GenerationResultVerifier failed at {context}: '
            f'expected_bars={expected_bars}, actual_bars={actual_bars}'
        )


def count_token_bars(tokens):
    count = 0
    for tok in tokens:
        if tok == '<EOS>':
            break
        if tok == '<BAR>':
            count += 1
    return count


def count_cp_bars(cp_data, idx2type_token):
    count = 0
    for cp_type in cp_data['cp_type']:
        tok = idx2type_token.get(int(cp_type), '<UNK>')
        if tok == '<EOS>':
            break
        if tok == '<BAR>':
            count += 1
    return count


def _resolve_debug_txt_path(output_midi_path: Path) -> Path:
    if output_midi_path.suffix:
        return output_midi_path.with_suffix('.txt')
    return Path(str(output_midi_path) + '.txt')


def ensure_output_parent_dir(output_midi_path: Path):
    try:
        output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(
            f'Output Writer Service: failed to create output directory '
            f'({output_midi_path.parent}) due to: {e}'
        )


def write_debug_tokens(output_midi_path: Path, tokens: List[str]):
    debug_txt_path = _resolve_debug_txt_path(output_midi_path)
    try:
        with open(debug_txt_path, 'w', encoding='utf-8') as f:
            for tok in tokens:
                f.write(f'{tok}\n')
    except OSError as e:
        raise ValueError(
            f'Output Writer Service: failed to write debug text ({debug_txt_path}) due to: {e}'
        )
    return debug_txt_path


def write_debug_cp_data(output_midi_path: Path, cp_data, idx2type_token, idx2limb_token):
    debug_txt_path = _resolve_debug_txt_path(output_midi_path)
    try:
        with open(debug_txt_path, 'w', encoding='utf-8') as f:
            for cp_type, pos, h1, h2, rf, lf in zip(
                cp_data['cp_type'],
                cp_data['cp_pos'],
                cp_data['cp_hand1'],
                cp_data['cp_hand2'],
                cp_data['cp_right_foot'],
                cp_data['cp_left_foot'],
            ):
                cp_type_tok = idx2type_token.get(int(cp_type), '<UNK>')
                if cp_type_tok != '<NOTE>':
                    f.write(f'STRUCT {cp_type_tok}\n')
                else:
                    f.write(
                        'CP '
                        f'POS={int(pos)} '
                        f"H1={idx2limb_token.get(int(h1), '<UNK>')} "
                        f"H2={idx2limb_token.get(int(h2), '<UNK>')} "
                        f"RF={idx2limb_token.get(int(rf), '<UNK>')} "
                        f"LF={idx2limb_token.get(int(lf), '<UNK>')}\n"
                    )
    except OSError as e:
        raise ValueError(
            f'Output Writer Service: failed to write debug text ({debug_txt_path}) due to: {e}'
        )
    return debug_txt_path


def save_standard_or_remi_outputs(output_midi_path: Path, tokens: List[str], bpm: int):
    try:
        tokens_to_midi(tokens, str(output_midi_path), bpm=bpm)
    except Exception as e:
        raise ValueError(
            f'Output Writer Service: failed to write MIDI ({output_midi_path}) due to: {e}'
        )

    debug_txt_path = write_debug_tokens(output_midi_path, tokens)
    return debug_txt_path


def save_cp_outputs(output_midi_path: Path, cp_data, idx2type_token, idx2limb_token, bpm: int):
    try:
        cp_data_to_midi(
            cp_data,
            idx2type_token,
            idx2limb_token,
            str(output_midi_path),
            bpm=bpm,
        )
    except Exception as e:
        raise ValueError(
            f'Output Writer Service: failed to write MIDI ({output_midi_path}) due to: {e}'
        )

    debug_txt_path = write_debug_cp_data(output_midi_path, cp_data, idx2type_token, idx2limb_token)
    return debug_txt_path


def build_window_plan(n_bars: int, max_bars: int, window_stride_bars: int) -> List[Tuple[int, int]]:
    if n_bars <= 0:
        return []
    if max_bars <= 0:
        raise ValueError(f'Invalid max_bars={max_bars}. It must be > 0.')
    if window_stride_bars <= 0:
        raise ValueError(f'Invalid window_stride_bars={window_stride_bars}. It must be > 0.')
    if window_stride_bars > max_bars:
        raise ValueError(
            f'Invalid window settings: window_stride_bars({window_stride_bars}) > max_bars({max_bars}).'
        )

    plan = []
    start = 0
    while start < n_bars:
        end = min(start + max_bars, n_bars)
        plan.append((start, end))
        if end >= n_bars:
            break
        start += window_stride_bars
    return plan


def _bar_edges(bar_positions, seq_len):
    return list(bar_positions) + [int(seq_len)]


def _slice_tokens_by_bar_range(tokens, bar_positions, st_bar, ed_bar):
    edges = _bar_edges(bar_positions, len(tokens))
    st = edges[st_bar]
    ed = edges[ed_bar]
    return tokens[st:ed]


def _slice_cp_data_by_bar_range(cp_data, bar_positions, st_bar, ed_bar):
    edges = _bar_edges(bar_positions, len(cp_data['cp_type']))
    st = edges[st_bar]
    ed = edges[ed_bar]

    sliced = {
        'cp_type': cp_data['cp_type'][st:ed],
        'cp_pos': cp_data['cp_pos'][st:ed],
        'cp_hand1': cp_data['cp_hand1'][st:ed],
        'cp_hand2': cp_data['cp_hand2'][st:ed],
        'cp_right_foot': cp_data['cp_right_foot'][st:ed],
        'cp_left_foot': cp_data['cp_left_foot'][st:ed],
    }
    local_bar_positions = [pos - st for pos in bar_positions[st_bar:ed_bar]]
    return sliced, local_bar_positions


def split_tokens_into_bar_chunks(tokens):
    chunks = []
    pre_bar = []
    current = None
    for tok in tokens:
        if tok == '<EOS>':
            break

        if tok == '<BAR>':
            if current is not None:
                chunks.append(current)
            current = pre_bar + ['<BAR>']
            pre_bar = []
        else:
            if current is None:
                pre_bar.append(tok)
            else:
                current.append(tok)

    if current is not None:
        chunks.append(current)
    return chunks


def split_cp_data_into_bar_chunks(cp_data, idx2type_token):
    chunks = []
    pre_bar = {
        'cp_type': [], 'cp_pos': [], 'cp_hand1': [], 'cp_hand2': [],
        'cp_right_foot': [], 'cp_left_foot': []
    }
    cur = None

    for i, cp_type in enumerate(cp_data['cp_type']):
        tok = idx2type_token.get(int(cp_type), '<PAD>')
        if tok == '<EOS>':
            break
        if tok == '<BAR>':
            if cur is not None and len(cur['cp_type']) > 0:
                chunks.append(cur)
            cur = {
                'cp_type': [], 'cp_pos': [], 'cp_hand1': [], 'cp_hand2': [],
                'cp_right_foot': [], 'cp_left_foot': []
            }

            for k in pre_bar.keys():
                cur[k].extend(pre_bar[k])
                pre_bar[k].clear()

        if cur is None:
            target = pre_bar
        else:
            target = cur

        target['cp_type'].append(cp_data['cp_type'][i])
        target['cp_pos'].append(cp_data['cp_pos'][i])
        target['cp_hand1'].append(cp_data['cp_hand1'][i])
        target['cp_hand2'].append(cp_data['cp_hand2'][i])
        target['cp_right_foot'].append(cp_data['cp_right_foot'][i])
        target['cp_left_foot'].append(cp_data['cp_left_foot'][i])

    if cur is not None and len(cur['cp_type']) > 0:
        chunks.append(cur)

    return chunks


def merge_window_token_chunks(total_bars, window_results, merge_policy='keep_first'):
    merged = {}
    for st_bar, _, chunks in window_results:
        for local_idx, chunk in enumerate(chunks):
            gidx = st_bar + local_idx
            if gidx >= total_bars:
                continue
            if merge_policy == 'keep_last' or gidx not in merged:
                merged[gidx] = chunk

    ordered = []
    for b in range(total_bars):
        ordered.append(merged.get(b, ['<BAR>']))
    out_tokens = []
    for chunk in ordered:
        out_tokens.extend(chunk)
    out_tokens.append('<EOS>')
    return out_tokens


def merge_window_cp_chunks(total_bars, window_results, tokenizer, merge_policy='keep_first'):
    merged = {}
    for st_bar, _, chunks in window_results:
        for local_idx, chunk in enumerate(chunks):
            gidx = st_bar + local_idx
            if gidx >= total_bars:
                continue
            if merge_policy == 'keep_last' or gidx not in merged:
                merged[gidx] = chunk

    eos_idx = tokenizer.type_token2idx['<EOS>']
    pos_pad = tokenizer.pos_pad_value
    limb_na = tokenizer.limb_token2idx['<NA>']

    out = {
        'cp_type': [], 'cp_pos': [], 'cp_hand1': [], 'cp_hand2': [],
        'cp_right_foot': [], 'cp_left_foot': []
    }

    for b in range(total_bars):
        chunk = merged.get(b)
        if chunk is None:
            chunk = {
                'cp_type': [tokenizer.type_token2idx['<BAR>']],
                'cp_pos': [pos_pad],
                'cp_hand1': [limb_na],
                'cp_hand2': [limb_na],
                'cp_right_foot': [limb_na],
                'cp_left_foot': [limb_na],
            }
        for k in out.keys():
            out[k].extend(chunk[k])

    out['cp_type'].append(eos_idx)
    out['cp_pos'].append(pos_pad)
    out['cp_hand1'].append(limb_na)
    out['cp_hand2'].append(limb_na)
    out['cp_right_foot'].append(limb_na)
    out['cp_left_foot'].append(limb_na)
    return out


def extract_bar_latents_standard_or_remi(model, tokenizer, window_tokens, window_bar_positions, enc_seqlen, device, use_sampling, sampling_var):
    latents = []
    edges = _bar_edges(window_bar_positions, len(window_tokens))
    for bi in range(len(window_bar_positions)):
        st = edges[bi]
        ed = edges[bi + 1]
        bar_tokens = window_tokens[st:ed]
        model_input = tokens_to_model_input(bar_tokens, tokenizer, max_len=enc_seqlen, device=device)
        lat = get_latent_embedding(model, model_input, use_sampling=use_sampling, sampling_var=sampling_var, device=device)
        latents.append(lat)
    return torch.cat(latents, dim=0)


def extract_bar_latents_cp(model, cp_data, window_bar_positions, enc_seqlen, tokenizer, device, use_sampling, sampling_var):
    type_pad = tokenizer.type_token2idx['<PAD>']
    latents = []
    edges = _bar_edges(window_bar_positions, len(cp_data['cp_type']))

    for bi in range(len(window_bar_positions)):
        st = edges[bi]
        ed = edges[bi + 1]
        bar_cp_type = list(cp_data['cp_type'][st:ed])
        bar_cp_type = bar_cp_type[:enc_seqlen]
        if len(bar_cp_type) < enc_seqlen:
            bar_cp_type.extend([type_pad] * (enc_seqlen - len(bar_cp_type)))

        enc_input = torch.tensor(bar_cp_type, dtype=torch.long, device=device).unsqueeze(1)
        pad_mask = torch.tensor([v == type_pad for v in bar_cp_type], dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            lat = model.get_sampled_latent(
                enc_input,
                padding_mask=pad_mask,
                use_sampling=use_sampling,
                sampling_var=sampling_var,
            )
        latents.append(lat)

    return torch.cat(latents, dim=0)


###########################################
# トークン列から潜在表現を抽出
###########################################
def tokens_to_model_input(tokens, tokenizer, max_len=128, device='cuda'):
    """
    トークン列をモデル入力形式に変換
    
    Args:
        tokens: トークン列
        tokenizer: トークナイザー
        max_len: 最大系列長
        device: デバイス
        
    Returns:
        model_input: dict
    """
    # トークンをインデックスに変換
    token_indices = [tokenizer.token2idx.get(token, tokenizer.token2idx['<PAD>']) 
                     for token in tokens]
    
    # パディングまたは切り詰め
    if len(token_indices) > max_len:
        token_indices = token_indices[:max_len]
    else:
        padding_len = max_len - len(token_indices)
        token_indices = token_indices + [tokenizer.token2idx['<PAD>']] * padding_len
    
    # Tensorに変換
    enc_input = torch.tensor(token_indices, dtype=torch.long).unsqueeze(1).to(device)
    
    # パディングマスク
    padding_mask = torch.tensor(
        [0 if idx != tokenizer.token2idx['<PAD>'] else 1 for idx in token_indices],
        dtype=torch.bool
    ).unsqueeze(0).to(device)
    
    return {
        'enc_input': enc_input,
        'enc_padding_mask': padding_mask
    }


def get_latent_embedding(model, model_input, use_sampling=False, sampling_var=0., device='cuda'):
    """潜在表現を抽出"""
    batch_inp = model_input['enc_input']
    batch_padding_mask = model_input['enc_padding_mask']
    
    with torch.no_grad():
        piece_latents = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask,
            use_sampling=use_sampling, sampling_var=sampling_var
        )
    
    return piece_latents


###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
    """Temperature付きソフトマックス"""
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('[info] overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    """Nucleus (top-p) sampling"""
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]

    if len(candi_index) == 0:
        candi_index = sorted_index[:3]

    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


###########################################
# 生成
###########################################
def generate_with_difficulty_control(
        model, latents,
        s_tech_cls, s_indep_cls, s_hand_cls, s_foot_cls, s_move_cls,
        tokenizer, device='cuda',
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512,
        nucleus_p=0.9, temperature=1.2
):
    """
    5つの難易度パラメータを制御して生成
    
    Args:
        model: MuseMorphoseモデル
        latents: 潜在表現 (n_bars, d_latent)
        s_tech_cls: 特殊奏法クラス (n_bars,) array
        s_indep_cls: 独立性クラス (n_bars,) array
        s_hand_cls: 手スピードクラス (n_bars,) array
        s_foot_cls: 足スピードクラス (n_bars,) array
        s_move_cls: 移動速度クラス (n_bars,) array
        tokenizer: トークナイザー
        device: デバイス
        max_events: 最大イベント数
        primer: プライマー
        max_input_len: 最大入力長
        truncate_len: トランケート長
        nucleus_p: Nucleus sampling の p
        temperature: サンプリング温度
        
    Returns:
        generated: 生成されたトークンインデックス列
        time_elapsed: 生成時間（秒）
        entropies: エントロピー列
    """
    # Tensorに変換
    s_tech_cls = numpy_to_tensor(s_tech_cls, device=device).long()
    s_indep_cls = numpy_to_tensor(s_indep_cls, device=device).long()
    s_hand_cls = numpy_to_tensor(s_hand_cls, device=device).long()
    s_foot_cls = numpy_to_tensor(s_foot_cls, device=device).long()
    s_move_cls = numpy_to_tensor(s_move_cls, device=device).long()
    
    # プレースホルダーを準備
    latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
    s_tech_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_indep_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_hand_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_foot_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_move_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    
    print('[info] Difficulty classes per bar:')
    print(f'  S_tech:  {tensor_to_numpy(s_tech_cls)}')
    print(f'  S_indep: {tensor_to_numpy(s_indep_cls)}')
    print(f'  S_hand:  {tensor_to_numpy(s_hand_cls)}')
    print(f'  S_foot:  {tensor_to_numpy(s_foot_cls)}')
    print(f'  S_move:  {tensor_to_numpy(s_move_cls)}')
    
    # 初期化
    if primer is None:
        generated = [tokenizer.token2idx['<BAR>']]
    else:
        generated = [tokenizer.token2idx[e] for e in primer]
        latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
        s_tech_placeholder[:len(generated), 0] = s_tech_cls[0]
        s_indep_placeholder[:len(generated), 0] = s_indep_cls[0]
        s_hand_placeholder[:len(generated), 0] = s_hand_cls[0]
        s_foot_placeholder[:len(generated), 0] = s_foot_cls[0]
        s_move_placeholder[:len(generated), 0] = s_move_cls[0]
    
    target_bars = latents.size(0)
    generated_bars = 0
    
    time_st = time.time()
    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []
    early_eos_streak = 0
    
    # 生成ループ
    while generated_bars < target_bars:
        # 入力の準備
        if len(generated) == 1:
            dec_input = numpy_to_tensor([generated], device=device).long()
        else:
            dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()
        
        # 現在の小節に対応する潜在表現と難易度クラスを設定
        latent_placeholder[len(generated)-1, 0, :] = latents[generated_bars]
        s_tech_placeholder[len(generated)-1, 0] = s_tech_cls[generated_bars]
        s_indep_placeholder[len(generated)-1, 0] = s_indep_cls[generated_bars]
        s_hand_placeholder[len(generated)-1, 0] = s_hand_cls[generated_bars]
        s_foot_placeholder[len(generated)-1, 0] = s_foot_cls[generated_bars]
        s_move_placeholder[len(generated)-1, 0] = s_move_cls[generated_bars]
        
        dec_seg_emb = latent_placeholder[:len(generated), :]
        dec_s_tech = s_tech_placeholder[:len(generated), :]
        dec_s_indep = s_indep_placeholder[:len(generated), :]
        dec_s_hand = s_hand_placeholder[:len(generated), :]
        dec_s_foot = s_foot_placeholder[:len(generated), :]
        dec_s_move = s_move_placeholder[:len(generated), :]
        
        # サンプリング
        with torch.no_grad():
            logits = model.generate(
                dec_input, dec_seg_emb,
                s_tech_cls=dec_s_tech,
                s_indep_cls=dec_s_indep,
                s_hand_cls=dec_s_hand,
                s_foot_cls=dec_s_foot,
                s_move_cls=dec_s_move
            )
        
        logits = tensor_to_numpy(logits[0])
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)
        word_token = tokenizer.idx2token[word]
        
        # <PAD>はスキップ
        if word_token == '<PAD>':
            continue

        # target_bars 未達の EOS は破棄して継続（早期終了ガード）
        if word_token == '<EOS>':
            if generated_bars < target_bars:
                early_eos_streak += 1
                print(
                    '[warning] Early EOS detected. '
                    f'generated_bars={generated_bars}, target_bars={target_bars}, streak={early_eos_streak}. Continue generation.'
                )
                if early_eos_streak >= 32:
                    forced_bar_idx = tokenizer.token2idx['<BAR>']
                    generated.append(forced_bar_idx)
                    generated_final.append(forced_bar_idx)
                    entropies.append(entropy(probs))
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

        # <BAR>トークンで小節をカウント
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
        entropies.append(entropy(probs))
        
        cur_input_len += 1
        
        # コンテキストウィンドウのトランケート
        assert cur_input_len == len(generated)
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]
            
            # プレースホルダーもシフト
            latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0, :
            ]
            s_tech_placeholder[:len(generated)-1, 0] = s_tech_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_indep_placeholder[:len(generated)-1, 0] = s_indep_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_hand_placeholder[:len(generated)-1, 0] = s_hand_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_foot_placeholder[:len(generated)-1, 0] = s_foot_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_move_placeholder[:len(generated)-1, 0] = s_move_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            
            print(f'[info] Context truncated: accumulated_len={len(generated_final)}')
            cur_input_len = len(generated)
    
    if generated_bars != target_bars:
        raise RuntimeError(
            'DecoderTerminationGuard: generated bar count mismatch at decode end '
            f'(generated_bars={generated_bars}, target_bars={target_bars})'
        )
    print(f'-- Generated {len(generated_final)} events')
    print(f'-- Time elapsed: {time.time() - time_st:.2f} secs')
    
    return generated_final[:-1], time.time() - time_st, np.array(entropies)


###########################################
# main
###########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate drum patterns from input MIDI with difficulty control',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--input_midi', type=str, required=True,
                        help='Path to input drum MIDI file')
    parser.add_argument('--output_midi', type=str, required=True,
                        help='Path to output MIDI file')
    parser.add_argument('--difficulty_mode', type=str,
                        choices=['full_assign', 'global_shift', 'upper_cap'],
                        default='global_shift',
                        help='Difficulty specification mode (default: global_shift)')
    parser.add_argument('--difficulty_shift', type=int, nargs=5, default=[0, 0, 0, 0, 0],
                        help='Global shifts for [s_tech, s_indep, s_hand, s_foot, s_move]')
    parser.add_argument('--difficulty_cap', type=int, nargs=5, default=None,
                        help='Upper caps for [s_tech, s_indep, s_hand, s_foot, s_move] used in upper_cap mode')
    parser.add_argument('--difficulty_labels_file', type=str, default=None,
                        help='Path to full_assign labels JSON file')
    parser.add_argument('--nucleus_p', type=float, default=0.9,
                        help='Nucleus sampling p value (default: 0.9)')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature (default: 1.2)')
    parser.add_argument('--use_latent_sampling', action='store_true',
                        help='Use sampling for latent extraction')
    parser.add_argument('--latent_sampling_var', type=float, default=0.1,
                        help='Latent sampling variance (default: 0.1)')
    parser.add_argument('--bpm', type=int, default=120,
                        help='Tempo in BPM (default: 120)')
    parser.add_argument('--max_bars', type=int, default=None,
                        help='Window size in bars for sliding generation. Defaults to config.generate.max_bars')
    parser.add_argument('--window_stride_bars', type=int, default=None,
                        help='Window stride in bars. Defaults to max_bars')
    parser.add_argument('--window_merge_policy', type=str, default='keep_first',
                        choices=['keep_first', 'keep_last'],
                        help='Merge policy for overlapped windows')
    args = parser.parse_args()

    try:
        validate_difficulty_cli_args(args)
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)
    
    # パスの検証
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f'[ERROR] Model checkpoint not found: {model_path}')
        sys.exit(1)
    
    input_midi_path = Path(args.input_midi)
    if not input_midi_path.exists():
        print(f'[ERROR] Input MIDI file not found: {input_midi_path}')
        sys.exit(1)
    
    # モデルのディレクトリからconfig.yamlを探す
    checkpoint_dir = model_path.parent.parent  # params/ -> checkpoints_xxx/
    config_path = checkpoint_dir / 'config.yaml'
    
    if not config_path.exists():
        print(f'[ERROR] config.yaml not found in {checkpoint_dir}')
        sys.exit(1)
    
    print(f'[info] Loading config from {config_path}')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    
    # トークン設計手法を取得
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method',
    )
    print(f'[info] Tokenization method: {tokenization_method}')
    
    device = config['training']['device']
    
    # 語彙パスを取得
    vocab_path = Path(config['data']['vocab_path'])
    if not vocab_path.exists():
        print(f'[ERROR] Vocabulary file not found: {vocab_path}')
        sys.exit(1)

    data_dir = config['data'].get('data_dir')
    try:
        ensure_runtime_mode_compatibility(
            requested_mode=tokenization_method,
            vocab_path=str(vocab_path),
            data_dir=data_dir,
        )
        artifact_mode, artifact_mode_source = resolve_artifact_tokenization_method(
            str(vocab_path),
            data_dir,
        )
        if artifact_mode is not None:
            print(f'[info] Artifact tokenization mode: {artifact_mode} ({artifact_mode_source})')
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)
    
    print(f'[info] Loading vocabulary from {vocab_path}')
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    try:
        adapter = resolve_mode_adapter(tokenization_method, vocab_data)
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)

    tokenizer = adapter['tokenizer']
    tokenize_fn = adapter['tokenize_fn']
    
    print(f'[info] Vocabulary size: {tokenizer.vocab_size}')
    
    # 難易度境界値を読み込み
    difficulty_bounds_path = Path('data/metadata/difficulty_bounds_standard.pkl')
    if not difficulty_bounds_path.exists():
        print(f'[ERROR] Difficulty bounds file not found: {difficulty_bounds_path}')
        sys.exit(1)
    
    print(f'[info] Loading difficulty bounds from {difficulty_bounds_path}')
    with open(difficulty_bounds_path, 'rb') as f:
        difficulty_bounds = pickle.load(f)
    
    # モデル構築
    print('[info] Building model...')
    mconf = config['model']
    cond_mode = mconf.get('cond_mode', 'in-attn')
    if cond_mode not in ('in-attn', 'pre-attn'):
        print(f"[warning] Unsupported cond_mode={cond_mode}. Falling back to 'in-attn'.")
        cond_mode = 'in-attn'

    bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
    if bar_anchor_mode not in ('first_token', 'bar_token'):
        print(f"[warning] Unsupported bar_anchor_mode={bar_anchor_mode}. Falling back to 'first_token'.")
        bar_anchor_mode = 'first_token'

    runtime_opts_cp = None
    if tokenization_method == 'cp_limb_v1':
        runtime_opts_cp = _resolve_runtime_model_options_cp(mconf, tokenizer)
        model = MuseMorphose(
            mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
            mconf['d_latent'], mconf['d_embed'], tokenizer.vocab_size,
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
        ).to(device)
    else:
        model = MuseMorphose(
            mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
            mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
            mconf['d_latent'], mconf['d_embed'], tokenizer.vocab_size,
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
            cond_mode=cond_mode,
            bar_anchor_mode=bar_anchor_mode,
            bar_token_idx=tokenizer.token2idx.get('<BAR>', None),
            use_hierarchical_pe=bool(mconf.get('use_hierarchical_pe', False)),
            hier_pe_bar_vocab_size=max(1, int(mconf.get('hier_pe_bar_vocab_size', 512))),
            hier_pe_beat_vocab_size=max(1, int(mconf.get('hier_pe_beat_vocab_size', 8))),
            hier_pe_pos_vocab_size=max(1, int(mconf.get('hier_pe_pos_vocab_size', 32))),
            hier_pe_use_struct_bias=bool(mconf.get('hier_pe_use_struct_bias', True)),
        ).to(device)
    
    print(f'[info] Loading model checkpoint from {model_path}')
    #model.load_state_dict(torch.load(model_path, map_location=device))
    #model.eval()
    print(f'[info] Loading model checkpoint from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    
    # checkpointが state_dict（辞書）直接の場合と、
    # Checkpoint型（'state_dict'キーを持つ）の場合に対応
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['state_dict']
    else:
        checkpoint_state_dict = checkpoint

    try:
        validate_vocabulary_compatibility(tokenization_method, tokenizer, checkpoint_state_dict)
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)
    
    # 互換性処理を適用
    adapted_state_dict = adapt_checkpoint_to_model(
        checkpoint_state_dict, 
        tokenizer.vocab_size,
        old_vocab_size=checkpoint_state_dict['token_emb.emb_lookup.weight'].size(0)
    )
    
    # 適応されたstate_dictをロード（strict=Falseで新しい難易度層は初期化）
    model.load_state_dict(adapted_state_dict, strict=False)
    model.eval()
    
    # 入力MIDIを読み込み、トークン化
    print(f'[info] Loading and tokenizing input MIDI: {input_midi_path}')
    mode_payload, mode_bar_positions = tokenize_fn(str(input_midi_path))

    # 難易度計算は standard token 軸で統一
    diff_tokenizer = DrumTokenizer()
    difficulty_tokens, difficulty_bar_positions = diff_tokenizer.midi_to_tokens(str(input_midi_path))

    n_bars = len(mode_bar_positions)
    print(f'[info] Input MIDI contains {n_bars} bars')
    
    if n_bars == 0:
        print('[ERROR] No bars found in input MIDI')
        sys.exit(1)
    
    # 難易度レベル計算
    print('[info] Computing difficulty levels for each bar...')
    difficulty_levels = compute_difficulty_levels(
        difficulty_tokens, difficulty_bar_positions, difficulty_bounds, bpm=args.bpm
    )

    if len(difficulty_levels['s_tech_cls']) != n_bars:
        print(
            '[ERROR] Difficulty/bar mismatch: '
            f'difficulty_bars={len(difficulty_levels["s_tech_cls"])}, mode_bars={n_bars}'
        )
        sys.exit(1)
    
    print('[info] Original difficulty levels:')
    print(f'  S_tech:  {difficulty_levels["s_tech_cls"]}')
    print(f'  S_indep: {difficulty_levels["s_indep_cls"]}')
    print(f'  S_hand:  {difficulty_levels["s_hand_cls"]}')
    print(f'  S_foot:  {difficulty_levels["s_foot_cls"]}')
    print(f'  S_move:  {difficulty_levels["s_move_cls"]}')
    
    difficulty_bounds_by_key = _resolve_difficulty_upper_bounds(mconf)

    try:
        target_difficulty_levels = resolve_target_difficulty_levels(
            difficulty_levels,
            difficulty_mode=args.difficulty_mode,
            bounds=difficulty_bounds_by_key,
            difficulty_shift=args.difficulty_shift,
            difficulty_cap=args.difficulty_cap,
            labels_file=args.difficulty_labels_file,
        )
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)

    print(f'\n[info] Difficulty mode: {args.difficulty_mode}')
    if args.difficulty_mode == 'global_shift':
        print(f'[info] Applied difficulty shifts: {args.difficulty_shift}')
    elif args.difficulty_mode == 'upper_cap':
        print(f'[info] Applied difficulty caps: {args.difficulty_cap}')
    else:
        print(f'[info] Applied per-bar labels file: {args.difficulty_labels_file}')

    print('[info] Target difficulty levels:')
    print(f'  S_tech:  {target_difficulty_levels["s_tech_cls"]}')
    print(f'  S_indep: {target_difficulty_levels["s_indep_cls"]}')
    print(f'  S_hand:  {target_difficulty_levels["s_hand_cls"]}')
    print(f'  S_foot:  {target_difficulty_levels["s_foot_cls"]}')
    print(f'  S_move:  {target_difficulty_levels["s_move_cls"]}')
    
    print('\n[info] Generating drum pattern with difficulty control...')

    # generate設定のデフォルト値
    generate_config = config.get('generate', {})
    max_input_dec_seqlen = generate_config.get('max_input_dec_seqlen', 1280)

    max_bars = args.max_bars if args.max_bars is not None else int(generate_config.get('max_bars', n_bars))
    window_stride_bars = args.window_stride_bars if args.window_stride_bars is not None else int(max_bars)

    try:
        window_plan = build_window_plan(n_bars, int(max_bars), int(window_stride_bars))
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)

    print(f'[info] Window plan: {window_plan}')

    enc_seqlen = int(config['data']['enc_seqlen'])
    all_window_times = []
    all_window_entropies = []
    window_results = []

    for win_idx, (st_bar, ed_bar) in enumerate(window_plan):
        win_n_bars = ed_bar - st_bar
        print(f'[info] Window {win_idx + 1}/{len(window_plan)}: bars [{st_bar}, {ed_bar})')

        win_difficulty = {
            key: target_difficulty_levels[key][st_bar:ed_bar]
            for key in DIFFICULTY_KEYS
        }

        if tokenization_method in ('standard', 'remi'):
            window_tokens = _slice_tokens_by_bar_range(mode_payload, mode_bar_positions, st_bar, ed_bar)
            local_bar_positions = [p - mode_bar_positions[st_bar] for p in mode_bar_positions[st_bar:ed_bar]]

            latents = extract_bar_latents_standard_or_remi(
                model,
                tokenizer,
                window_tokens,
                local_bar_positions,
                enc_seqlen,
                device,
                args.use_latent_sampling,
                args.latent_sampling_var,
            )

            generated_indices, win_time, win_ent = generate_with_difficulty_control(
                model, latents,
                win_difficulty['s_tech_cls'],
                win_difficulty['s_indep_cls'],
                win_difficulty['s_hand_cls'],
                win_difficulty['s_foot_cls'],
                win_difficulty['s_move_cls'],
                tokenizer,
                device=device,
                max_input_len=max_input_dec_seqlen,
                truncate_len=min(512, max(64, max_input_dec_seqlen - 32)),
                nucleus_p=args.nucleus_p,
                temperature=args.temperature
            )
            generated_tokens = [tokenizer.idx2token[idx] for idx in generated_indices]
            chunks = split_tokens_into_bar_chunks(generated_tokens)
        else:
            window_cp_data, local_bar_positions = _slice_cp_data_by_bar_range(
                mode_payload, mode_bar_positions, st_bar, ed_bar
            )

            latents = extract_bar_latents_cp(
                model,
                window_cp_data,
                local_bar_positions,
                enc_seqlen,
                tokenizer,
                device,
                args.use_latent_sampling,
                args.latent_sampling_var,
            )

            cp_generated, win_time, win_ent = generate_cp_sequence(
                model,
                latents,
                tokenizer.struct_token2idx,
                tokenizer.idx2struct_token,
                tokenizer.idx2limb_token,
                tokenizer.pos_pad_value,
                tokenizer.limb_token2idx['<PAD>'],
                tokenizer.event_type2idx['<PAD_EVENT>'],
                device=device,
                max_events=generate_config.get('dec_seqlen', 12800),
                max_input_len=max_input_dec_seqlen,
                truncate_len=min(512, max(64, max_input_dec_seqlen - 32)),
                nucleus_p=args.nucleus_p,
                temperature=args.temperature,
                use_difficulty=True,
                s_tech_cls=win_difficulty['s_tech_cls'],
                s_indep_cls=win_difficulty['s_indep_cls'],
                s_hand_cls=win_difficulty['s_hand_cls'],
                s_foot_cls=win_difficulty['s_foot_cls'],
                s_move_cls=win_difficulty['s_move_cls'],
                use_hierarchical_pe=bool(runtime_opts_cp and runtime_opts_cp['use_hierarchical_pe']),
            )
            chunks = split_cp_data_into_bar_chunks(cp_generated, tokenizer.idx2type_token)

        try:
            verify_bar_count(win_n_bars, len(chunks), context=f'window={win_idx + 1}')
        except ValueError as e:
            print(f'[ERROR] {e}')
            sys.exit(1)

        window_results.append((st_bar, ed_bar, chunks))
        all_window_times.append(float(win_time))
        if win_ent is not None and len(win_ent) > 0:
            all_window_entropies.extend(win_ent.tolist())

    output_midi_path = Path(args.output_midi)
    try:
        ensure_output_parent_dir(output_midi_path)
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)

    print('[info] Output overwrite policy: overwrite existing MIDI/debug files with the same prefix')

    if tokenization_method in ('standard', 'remi'):
        merged_tokens = merge_window_token_chunks(n_bars, window_results, merge_policy=args.window_merge_policy)
        merged_bar_count = count_token_bars(merged_tokens)
        try:
            verify_bar_count(n_bars, merged_bar_count, context='merged_output')
        except ValueError as e:
            print(f'[ERROR] {e}')
            sys.exit(1)
        print(f'\n[info] Saving generated MIDI/debug to prefix {output_midi_path}')
        try:
            debug_txt_path = save_standard_or_remi_outputs(output_midi_path, merged_tokens, bpm=args.bpm)
        except ValueError as e:
            print(f'[ERROR] {e}')
            sys.exit(1)
    else:
        merged_cp_data = merge_window_cp_chunks(
            n_bars,
            window_results,
            tokenizer,
            merge_policy=args.window_merge_policy,
        )
        merged_bar_count = count_cp_bars(merged_cp_data, tokenizer.idx2type_token)
        try:
            verify_bar_count(n_bars, merged_bar_count, context='merged_output')
        except ValueError as e:
            print(f'[ERROR] {e}')
            sys.exit(1)
        print(f'\n[info] Saving generated MIDI/debug to prefix {output_midi_path}')
        try:
            debug_txt_path = save_cp_outputs(
                output_midi_path,
                merged_cp_data,
                tokenizer.idx2type_token,
                tokenizer.idx2limb_token,
                bpm=args.bpm,
            )
        except ValueError as e:
            print(f'[ERROR] {e}')
            sys.exit(1)

    mean_time = float(np.mean(all_window_times)) if all_window_times else 0.0
    mean_entropy = float(np.mean(all_window_entropies)) if all_window_entropies else 0.0
    
    print(f'\n{"="*60}')
    print('Generation completed successfully!')
    print(f'{"="*60}')
    print(f'Model: {model_path}')
    print(f'Input MIDI: {input_midi_path}')
    print(f'Output MIDI: {output_midi_path}')
    print(f'Output debug text: {debug_txt_path}')
    print(f'Number of bars: {n_bars}')
    print(f'Num windows: {len(window_plan)}')
    print(f'Average window generation time: {mean_time:.2f} seconds')
    print(f'Average entropy: {mean_entropy:.3f}')
    print(f'Window settings: max_bars={max_bars}, stride={window_stride_bars}, merge={args.window_merge_policy}')
    if args.difficulty_mode == 'global_shift':
        shift_tech, shift_indep, shift_hand, shift_foot, shift_move = args.difficulty_shift
        print(
            'Difficulty shifts: '
            f'tech={shift_tech:+d}, indep={shift_indep:+d}, hand={shift_hand:+d}, '
            f'foot={shift_foot:+d}, move={shift_move:+d}'
        )
    elif args.difficulty_mode == 'upper_cap':
        cap_tech, cap_indep, cap_hand, cap_foot, cap_move = args.difficulty_cap
        print(
            'Difficulty caps: '
            f'tech<={cap_tech}, indep<={cap_indep}, hand<={cap_hand}, '
            f'foot<={cap_foot}, move<={cap_move}'
        )
    else:
        print(f'Difficulty labels file: {args.difficulty_labels_file}')
