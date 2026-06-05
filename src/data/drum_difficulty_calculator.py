"""
ドラム譜難易度計算モジュール

5つの難易度指標を計算:
- S_tech: 特殊奏法の頻度
- S_indep: 手足の独立性
- S_hand: 手の最大連打スピード
- S_foot: 足の最大連打スピード
- S_move: 打点間の最大移動速度
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


# ============================================================================
# 定数定義
# ============================================================================

# DRUM_NOTE_MAP で定義された2語drum_typeのプレフィックス集合
# トークン形式: {drum_type}_{technique}_{velocity}
# 例: SNARE_RIM_ACCENT_Normal → drum_type='SNARE_RIM'（2語）
_TWO_WORD_DRUM_PREFIXES = {
    'SNARE_RIM',
    'HH_CLOSED', 'HH_PEDAL', 'HH_OPEN', 'HH_HALFOPEN',
    'TOM_LOW', 'TOM_MID', 'TOM_HIGH',
    'RIDE_CYMBAL', 'RIDE_BELL',
}


def _extract_instrument(token: str) -> str:
    """トークン文字列から drum_type（楽器名）を小文字で抽出する。

    2語からなる drum_type（例: TOM_LOW, HH_CLOSED）は parts[:2] を結合して返す。
    1語の drum_type（例: SNARE, KICK）は parts[0] を返す。
    """
    parts = token.split('_')
    if len(parts) >= 2 and f'{parts[0]}_{parts[1]}' in _TWO_WORD_DRUM_PREFIXES:
        return f'{parts[0]}_{parts[1]}'.lower()
    return parts[0].lower()


# 手の楽器（_extract_instrument が返す小文字 drum_type 名）
HAND_INSTRUMENTS = {
    'snare', 'snare_rim',
    'floor',
    'tom_low', 'tom_mid', 'tom_high',
    'hh_closed', 'hh_halfopen', 'hh_open',
    'crash', 'ride_cymbal', 'ride_bell',
    'splash', 'tambourine', 'china', 'cowbell',
}

# 足の楽器
FOOT_INSTRUMENTS = {'kick', 'hh_pedal'}

# 特殊奏法を示すトークンのキーワード
SPECIAL_TECHNIQUE_KEYWORDS = ['ghost', 'rimshot', 'choke', 'flam', 'roll', 'accent', 'xstick']

# 物理距離行列（正規化済み、最大距離=1.0）
# キー配置: high_tom → mid_tom → low_tom → floor_tom（左→右）
DISTANCE_MATRIX = {
    'snare':       {'snare': 0.00, 'hihat': 0.32, 'high_tom': 0.45, 'mid_tom': 0.48,
                    'low_tom': 0.52, 'floor_tom': 0.45, 'crash_cymbal': 0.67,
                    'ride_cymbal': 0.52, 'splash_cymbal': 0.40, 'tambourine': 0.35},
    'hihat':       {'snare': 0.32, 'hihat': 0.00, 'high_tom': 0.32, 'mid_tom': 0.52,
                    'low_tom': 0.63, 'floor_tom': 0.75, 'crash_cymbal': 0.44,
                    'ride_cymbal': 0.71, 'splash_cymbal': 0.35, 'tambourine': 0.10},
    'high_tom':    {'snare': 0.45, 'hihat': 0.32, 'high_tom': 0.00, 'mid_tom': 0.27,
                    'low_tom': 0.45, 'floor_tom': 0.75, 'crash_cymbal': 0.25,
                    'ride_cymbal': 0.56, 'splash_cymbal': 0.15, 'tambourine': 0.30},
    'mid_tom':     {'snare': 0.48, 'hihat': 0.52, 'high_tom': 0.27, 'mid_tom': 0.00,
                    'low_tom': 0.28, 'floor_tom': 0.59, 'crash_cymbal': 0.48,
                    'ride_cymbal': 0.32, 'splash_cymbal': 0.20, 'tambourine': 0.50},
    'low_tom':     {'snare': 0.52, 'hihat': 0.63, 'high_tom': 0.45, 'mid_tom': 0.28,
                    'low_tom': 0.00, 'floor_tom': 0.30, 'crash_cymbal': 0.70,
                    'ride_cymbal': 0.30, 'splash_cymbal': 0.30, 'tambourine': 0.60},
    'floor_tom':   {'snare': 0.45, 'hihat': 0.75, 'high_tom': 0.75, 'mid_tom': 0.59,
                    'low_tom': 0.30, 'floor_tom': 0.00, 'crash_cymbal': 1.00,
                    'ride_cymbal': 0.35, 'splash_cymbal': 0.65, 'tambourine': 0.75},
    'crash_cymbal':{'snare': 0.67, 'hihat': 0.44, 'high_tom': 0.25, 'mid_tom': 0.48,
                    'low_tom': 0.70, 'floor_tom': 1.00, 'crash_cymbal': 0.00,
                    'ride_cymbal': 0.79, 'splash_cymbal': 0.30, 'tambourine': 0.40},
    'ride_cymbal': {'snare': 0.52, 'hihat': 0.71, 'high_tom': 0.56, 'mid_tom': 0.32,
                    'low_tom': 0.30, 'floor_tom': 0.35, 'crash_cymbal': 0.79,
                    'ride_cymbal': 0.00, 'splash_cymbal': 0.50, 'tambourine': 0.70},
    'splash_cymbal':{'snare': 0.40, 'hihat': 0.35, 'high_tom': 0.15, 'mid_tom': 0.20,
                    'low_tom': 0.30, 'floor_tom': 0.65, 'crash_cymbal': 0.30,
                    'ride_cymbal': 0.50, 'splash_cymbal': 0.00, 'tambourine': 0.35},
    'tambourine':  {'snare': 0.35, 'hihat': 0.10, 'high_tom': 0.30, 'mid_tom': 0.50,
                    'low_tom': 0.60, 'floor_tom': 0.75, 'crash_cymbal': 0.40,
                    'ride_cymbal': 0.70, 'splash_cymbal': 0.35, 'tambourine': 0.00},
}

# _extract_instrument が返す drum_type 名 → DISTANCE_MATRIX キー名
TOKEN_TO_DISTANCE_KEY = {
    'snare':       'snare',
    'snare_rim':   'snare',        # スネアと同位置
    'kick':        None,           # 足なのでNone
    'floor':       'floor_tom',
    'tom_low':     'low_tom',
    'tom_mid':     'mid_tom',
    'tom_high':    'high_tom',
    'hh_closed':   'hihat',
    'hh_halfopen': 'hihat',
    'hh_open':     'hihat',
    'hh_pedal':    None,           # 足なのでNone
    'ride_cymbal': 'ride_cymbal',
    'ride_bell':   'ride_cymbal',
    'crash':       'crash_cymbal',
    'china':       'crash_cymbal', # 近似
    'splash':      'splash_cymbal',
    'tambourine':  'tambourine',
    'cowbell':     'crash_cymbal', # 近似
}


# ============================================================================
# 難易度計算関数
# ============================================================================

def compute_s_tech(bar_tokens: List[str]) -> float:
    """
    S_tech (特殊奏法の頻度) を計算

    Args:
        bar_tokens: 1小節分のトークン列

    Returns:
        特殊奏法の割合 (0.0 ~ 1.0)
    """
    # 演奏トークンのみを抽出（<BAR>, <BEAT_*>, <POS_*> を除外）
    note_tokens = [
        t for t in bar_tokens
        if not (t.startswith('<') and t.endswith('>'))
    ]

    if len(note_tokens) == 0:
        return 0.0

    # 特殊奏法を含むトークンをカウント
    special_count = sum(
        1 for token in note_tokens
        if any(keyword.upper() in token.upper() for keyword in SPECIAL_TECHNIQUE_KEYWORDS)
    )

    return special_count / len(note_tokens)


def compute_s_indep(bar_tokens: List[str]) -> float:
    """
    S_indep (手足の独立性) を計算

    Args:
        bar_tokens: 1小節分のトークン列（時系列順）

    Returns:
        独立打撃の割合 (0.0 ~ 1.0)
    """
    # トークンをパースして (time_position, instrument) のリストを作成
    # time_position = beat * 24 + pos_in_beat
    current_beat = 0
    current_pos = 0
    events = []  # [(time_position, part), ...]  part: 'hand' or 'foot'

    for token in bar_tokens:
        if token.startswith('<BEAT_'):
            try:
                current_beat = int(token.replace('<BEAT_', '').replace('>', '')) - 1
            except ValueError:
                continue
        elif token.startswith('<POS_'):
            try:
                current_pos = int(token.replace('<POS_', '').replace('>', ''))
            except ValueError:
                continue
        elif not (token.startswith('<') and token.endswith('>')):
            # 演奏トークン
            time_pos = current_beat * 24 + current_pos

            instrument = _extract_instrument(token)

            # 手か足かを判定
            if instrument in HAND_INSTRUMENTS:
                events.append((time_pos, 'hand'))
            elif instrument in FOOT_INSTRUMENTS:
                events.append((time_pos, 'foot'))

    if len(events) == 0:
        return 0.0

    # 各タイミングごとにグループ化
    timing_groups = defaultdict(list)
    for time_pos, part in events:
        timing_groups[time_pos].append(part)

    unique_timings = len(timing_groups)
    if unique_timings == 0:
        return 0.0

    # 手足が完全同時発音しているタイミングをカウント
    # （手と足の両方が存在する）
    coincidence_count = sum(
        1 for parts in timing_groups.values()
        if 'hand' in parts and 'foot' in parts
    )

    # 独立性 = (全タイミング - 完全同時発音) / 全タイミング
    independence = (unique_timings - coincidence_count) / unique_timings

    return independence


def compute_s_hand(bar_tokens: List[str], bpm: float = 120.0) -> float:
    """
    S_hand (手の最大連打スピード) を計算 (notes per second)

    Args:
        bar_tokens: 1小節分のトークン列
        bpm: テンポ (BPM)

    Returns:
        1秒あたりの最大手打撃数
    """
    # トークンをパースして手の打撃タイミングを抽出
    current_beat = 0
    current_pos = 0
    hand_timings = []  # [time_position, ...]

    for token in bar_tokens:
        if token.startswith('<BEAT_'):
            try:
                current_beat = int(token.replace('<BEAT_', '').replace('>', '')) - 1
            except ValueError:
                continue
        elif token.startswith('<POS_'):
            try:
                current_pos = int(token.replace('<POS_', '').replace('>', ''))
            except ValueError:
                continue
        elif not (token.startswith('<') and token.endswith('>')):
            instrument = _extract_instrument(token)

            if instrument in HAND_INSTRUMENTS:
                time_pos = current_beat * 24 + current_pos
                hand_timings.append(time_pos)

    if len(hand_timings) == 0:
        return 0.0

    # 1拍ごとに区切ってカウント
    # 1小節 = 4拍 = 96 positions (1拍 = 24 positions)
    beat_counts = [0, 0, 0, 0]
    for time_pos in hand_timings:
        beat_idx = min(time_pos // 24, 3)  # 0, 1, 2, 3
        beat_counts[beat_idx] += 1

    max_notes_per_beat = max(beat_counts)

    # BPMからスケール: (bpm / 60) * max_notes_per_beat
    notes_per_second = (bpm / 60.0) * max_notes_per_beat

    return notes_per_second


def compute_s_foot(bar_tokens: List[str], bpm: float = 120.0) -> float:
    """
    S_foot (足の最大連打スピード) を計算 (notes per second)

    S_handと同じロジックを足の楽器に適用

    Args:
        bar_tokens: 1小節分のトークン列
        bpm: テンポ (BPM)

    Returns:
        1秒あたりの最大足打撃数
    """
    current_beat = 0
    current_pos = 0
    foot_timings = []

    for token in bar_tokens:
        if token.startswith('<BEAT_'):
            try:
                current_beat = int(token.replace('<BEAT_', '').replace('>', '')) - 1
            except ValueError:
                continue
        elif token.startswith('<POS_'):
            try:
                current_pos = int(token.replace('<POS_', '').replace('>', ''))
            except ValueError:
                continue
        elif not (token.startswith('<') and token.endswith('>')):
            instrument = _extract_instrument(token)

            if instrument in FOOT_INSTRUMENTS:
                time_pos = current_beat * 24 + current_pos
                foot_timings.append(time_pos)

    if len(foot_timings) == 0:
        return 0.0

    beat_counts = [0, 0, 0, 0]
    for time_pos in foot_timings:
        beat_idx = min(time_pos // 24, 3)
        beat_counts[beat_idx] += 1

    max_notes_per_beat = max(beat_counts)
    notes_per_second = (bpm / 60.0) * max_notes_per_beat

    return notes_per_second


def compute_s_move(bar_tokens: List[str]) -> float:
    """
    S_move (打点間の最大移動速度) を計算

    Args:
        bar_tokens: 1小節分のトークン列

    Returns:
        最大移動速度（正規化距離 / 時間単位）
    """
    # 手の打撃のみを時系列順に抽出
    current_beat = 0
    current_pos = 0
    hand_events = []  # [(time_position, instrument_key), ...]

    for token in bar_tokens:
        if token.startswith('<BEAT_'):
            try:
                current_beat = int(token.replace('<BEAT_', '').replace('>', '')) - 1
            except ValueError:
                continue
        elif token.startswith('<POS_'):
            try:
                current_pos = int(token.replace('<POS_', '').replace('>', ''))
            except ValueError:
                continue
        elif not (token.startswith('<') and token.endswith('>')):
            instrument = _extract_instrument(token)

            if instrument in HAND_INSTRUMENTS:
                time_pos = current_beat * 24 + current_pos

                # 距離行列のキーに変換
                distance_key = TOKEN_TO_DISTANCE_KEY.get(instrument)
                if distance_key and distance_key in DISTANCE_MATRIX:
                    hand_events.append((time_pos, distance_key))

    if len(hand_events) < 2:
        return 0.0

    # 連続する手の打撃間の移動速度を計算
    max_velocity = 0.0

    for i in range(len(hand_events) - 1):
        time_i, inst_i = hand_events[i]
        time_j, inst_j = hand_events[i + 1]

        # 時間差（position単位）
        time_diff = time_j - time_i

        # 完全同時打ち（time_diff == 0）はスキップ
        if time_diff == 0:
            continue

        # 距離を取得
        distance = DISTANCE_MATRIX[inst_i].get(inst_j, 0.0)

        # 移動速度 = 距離 / 時間差
        velocity = distance / time_diff

        max_velocity = max(max_velocity, velocity)

    return max_velocity


def compute_all_difficulty_scores(bar_tokens: List[str], bpm: float = 120.0) -> Dict[str, float]:
    """
    1小節のすべての難易度指標を計算

    Args:
        bar_tokens: 1小節分のトークン列
        bpm: テンポ (BPM)

    Returns:
        {
            's_tech': float,
            's_indep': float,
            's_hand': float,
            's_foot': float,
            's_move': float
        }
    """
    return {
        's_tech': compute_s_tech(bar_tokens),
        's_indep': compute_s_indep(bar_tokens),
        's_hand': compute_s_hand(bar_tokens, bpm),
        's_foot': compute_s_foot(bar_tokens, bpm),
        's_move': compute_s_move(bar_tokens)
    }


def _cp_data_to_tokens(cp_data: Dict, idx2type: Dict, idx2limb: Dict) -> List[str]:
    """
    CP形式データを難易度計算用のトークン列に復元する。

    Args:
        cp_data: CP形式のデータ（cp_type, cp_pos, cp_hand1, etc.）
        idx2type: Typeトークンのインデックス→トークン辞書
        idx2limb: 肢トークンのインデックス→トークン辞書

    Returns:
        復元されたトークンリスト
    """
    tokens = []

    cp_type_seq = cp_data.get('cp_type')
    if cp_type_seq is None and cp_data.get('event_type') is not None and cp_data.get('struct_token') is not None:
        cp_type_seq = []
        for ev_type, struct_id in zip(cp_data['event_type'], cp_data['struct_token']):
            if int(ev_type) == 1:
                cp_type_seq.append(-1)
            elif int(ev_type) == 2:
                cp_type_seq.append(-2)
            else:
                cp_type_seq.append(int(struct_id))

    for type_id, pos_id, h1, h2, rf, lf in zip(
        cp_type_seq,
        cp_data['cp_pos'],
        cp_data['cp_hand1'],
        cp_data['cp_hand2'],
        cp_data['cp_right_foot'],
        cp_data['cp_left_foot'],
    ):
        if int(type_id) == -1:
            cp_type_token = '<NOTE>'
        elif int(type_id) == -2:
            cp_type_token = '<PAD>'
        else:
            cp_type_token = idx2type.get(int(type_id), '<PAD>')

        if cp_type_token != '<NOTE>':
            if cp_type_token != '<PAD>':
                tokens.append(cp_type_token)
            continue

        if int(pos_id) < 24:
            tokens.append(f'<POS_{int(pos_id)}>')

        for limb_id in (h1, h2, rf, lf):
            limb_tok = idx2limb.get(int(limb_id), '<PAD>')
            if limb_tok not in ('<PAD>', '<NONE>', '<NA>'):
                tokens.append(limb_tok)

    return tokens


# ============================================================================
# 境界値計算・離散化
# ============================================================================

def compute_difficulty_bounds(data_dir: str, vocab_path: str, output_path: str, bpm: float = 120.0):
    """
    全データセットから各難易度指標の分布を計算し、8分位点を算出

    Args:
        data_dir: トークン化されたデータのディレクトリ
        vocab_path: 語彙ファイルのパス
        output_path: 境界値を保存するパス
        bpm: デフォルトBPM
    """
    # 語彙を読み込み（CP形式と標準形式の両方に対応）
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    is_cp_vocab = isinstance(vocab_data, dict) and vocab_data.get('tokenization_method') == 'cp_limb_v1'
    
    if is_cp_vocab:
        idx2type = vocab_data.get('idx2type_token', vocab_data['idx2struct_token'])
        idx2limb = vocab_data['idx2limb_token']
        idx2token = None
    else:
        token2idx, idx2token = vocab_data

    all_scores = {
        's_tech': [],
        's_indep': [],
        's_hand': [],
        's_foot': [],
        's_move': []
    }

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl') and f not in ('dataset_stats.pkl', 'train_split.pkl', 'val_split.pkl')])
    print(f'Found {len(files)} pickle files')

    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f'Processing {i}/{len(files)}...')

        try:
            filepath = os.path.join(data_dir, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # データ形式を確認
            if len(data) == 2:
                bar_pos, tokens = data
            elif len(data) == 3:
                bar_pos, tokens, _ = data
            else:
                print(f'Unknown data format in {file}, skipping...')
                continue

            # 各小節ごとに計算
            if len(bar_pos) < 2:
                continue

            for b in range(len(bar_pos) - 1):
                st, ed = bar_pos[b], bar_pos[b + 1]
                token_len = len(tokens['cp_type']) if isinstance(tokens, dict) and 'cp_type' in tokens else (
                    len(tokens['event_type']) if isinstance(tokens, dict) and 'event_type' in tokens else len(tokens)
                )
                if st >= ed or ed > token_len:
                    continue

                # CP形式と標準形式で異なるトークン列復元ロジック
                if is_cp_vocab and isinstance(tokens, dict):
                    # CP形式の復元
                    bar_cp_data = {
                        'cp_type': tokens['cp_type'][st:ed] if 'cp_type' in tokens else None,
                        'event_type': tokens['event_type'][st:ed] if 'event_type' in tokens else None,
                        'struct_token': tokens['struct_token'][st:ed] if 'struct_token' in tokens else None,
                        'cp_pos': tokens['cp_pos'][st:ed],
                        'cp_hand1': tokens['cp_hand1'][st:ed],
                        'cp_hand2': tokens['cp_hand2'][st:ed],
                        'cp_right_foot': tokens['cp_right_foot'][st:ed],
                        'cp_left_foot': tokens['cp_left_foot'][st:ed],
                    }
                    bar_tokens = _cp_data_to_tokens(bar_cp_data, idx2type, idx2limb)
                else:
                    # 標準形式の復元
                    bar_token_indices = tokens[st:ed]
                    bar_tokens = [idx2token.get(int(idx), '<PAD>') for idx in bar_token_indices]

                scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)

                for key, value in scores.items():
                    all_scores[key].append(value)

        except Exception as e:
            print(f'Error processing {file}: {e}')
            continue

    # 各指標の8分位点を計算（7つの境界値）
    bounds = {}
    for key, values in all_scores.items():
        if len(values) == 0:
            print(f'Warning: No valid scores for {key}')
            bounds[key] = [0.0] * 7
            continue

        values = np.array(values)
        # 8分位 (12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%)
        percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
        bounds[key] = np.percentile(values, percentiles).tolist()

        print(f'{key} bounds: {bounds[key]}')
        print(f'  min: {values.min():.4f}, max: {values.max():.4f}, mean: {values.mean():.4f}')

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(bounds, f)

    print(f'Difficulty bounds saved to {output_path}')
    return bounds


def discretize_difficulty_score(score: float, bounds: List[float]) -> int:
    """
    連続値スコアを8クラス (0~7) に離散化

    Args:
        score: 連続値スコア
        bounds: 7つの境界値のリスト

    Returns:
        クラス番号 (0~7)
    """
    return int(np.searchsorted(bounds, score))


# ============================================================================
# テスト用メイン
# ============================================================================

if __name__ == '__main__':
    # 簡単なテスト
    print("Testing difficulty calculation functions...")

    # テストケース1: 特殊奏法あり
    test_bar_1 = [
        '<BAR>', '<BEAT_1>', '<POS_0>', 'SNARE_GHOST_Normal',
        '<POS_6>', 'SNARE_HIT_Normal',
        '<POS_12>', 'KICK_HIT_Normal'
    ]

    scores_1 = compute_all_difficulty_scores(test_bar_1, bpm=120.0)
    print("\nTest Case 1:")
    print(f"  Tokens: {test_bar_1}")
    print(f"  Scores: {scores_1}")

    # テストケース2: 密度高め
    test_bar_2 = [
        '<BAR>', '<BEAT_1>',
        '<POS_0>', 'SNARE_HIT_Normal', 'KICK_HIT_Normal',
        '<POS_6>', 'HH_CLOSED_HIT_Normal',
        '<POS_12>', 'SNARE_HIT_Normal',
        '<POS_18>', 'HH_CLOSED_HIT_Normal',
        '<BEAT_2>',
        '<POS_0>', 'KICK_HIT_Normal',
        '<POS_6>', 'HH_CLOSED_HIT_Normal'
    ]

    scores_2 = compute_all_difficulty_scores(test_bar_2, bpm=120.0)
    print("\nTest Case 2:")
    print(f"  Tokens: {test_bar_2}")
    print(f"  Scores: {scores_2}")

    print("\nAll tests completed!")
