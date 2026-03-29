#!/usr/bin/env python
"""
MIDI → トークン → MIDIの往復変換テスト

このスクリプトは以下を検証します:
1. MIDIファイルをトークン列に変換できるか
2. トークン列をpklファイルとして保存できるか
3. pklファイルからトークン列を読み込めるか
4. トークン列をMIDIファイルに変換できるか
5. 変換前後でMIDIファイルの内容が一致するか

修正内容 (2024年版):
- MIDIファイルのticks_per_beatを考慮した正規化比較
- 異なるticks_per_beatでも正しく比較できるように改善
- より詳細な比較情報の表示
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import miditoolkit
from typing import List, Dict, Tuple, Optional
from src.tokenizers.drum_tokenizer import DrumTokenizer, DRUM_NOTE_MAP
from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
from scripts.preprocess.prepare_drum_dataset import RemiDrumTokenizer
from src.utils.drum_to_midi import DrumToken2MIDI, cp_data_to_midi


DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
REMI_STEPS_PER_BAR = 96  # REMI also uses 96 divisions per bar (same as Standard/CP)


# ピッチ許容リスト（異なる音程でも同じドラム楽器とみなす）
# これは STANDARD_PITCH_MAP により、複数のピッチが同じ楽器に変換される場合に対応
PITCH_TOLERANCE_MAP = {
    35: [35, 36],     # KICK: B1(35) or C2(36)
    36: [35, 36],     # KICK: B1(35) or C2(36)
    37: [37],         # SNARE_ACCENT: C#2(37)
    38: [38, 40],     # SNARE: D2(38) or E2(40)
    40: [38, 40],     # SNARE: D2(38) or E2(40)
    41: [41, 43],     # FLOOR: F2(41) or G2(43)
    43: [41, 43],     # FLOOR: F2(41) or G2(43)
    44: [44],         # HH_PEDAL: G#2(44)
    45: [45, 47, 50], # TOM2: すべての高さ
    47: [45, 47, 50], # TOM2: すべての高さ
    50: [45, 47, 50], # TOM2: すべての高さ
    49: [49, 57],     # CRASH: C#3(49) or A3(57)
    52: [52],         # CHINA: E3(52)
    54: [54],         # TAMBOURINE: F#3(54)
    56: [56],         # COWBELL: G#3(56)
    51: [51, 59],     # RIDE_BOW: D#3(51) or B3(59)
    59: [51, 59],     # RIDE_BOW: D#3(51) or B3(59)
    57: [49, 57],     # CRASH: C#3(49) or A3(57)
}


class MIDIComparator:
    """
    MIDI情報を比較するクラス

    異なるticks_per_beatのMIDIファイルでも正しく比較できるように、
    beat単位に正規化して比較を行います。
    """

    def __init__(self, tolerance_ticks: int = 10):
        """
        Args:
            tolerance_ticks: タイミングの許容誤差（ticks）
                             実際の比較は正規化後の値で行われます
        """
        self.tolerance = tolerance_ticks

    def is_pitch_compatible(self, pitch1: int, pitch2: int) -> bool:
        """
        2つのピッチが互換性があるか（同じドラム楽器として扱えるか）を判定

        Args:
            pitch1: オリジナルMIDIのピッチ
            pitch2: 再構築MIDIのピッチ

        Returns:
            Compatible の場合 True
        """
        # pitch1 が PITCH_TOLERANCE_MAP にある場合、permitted pitches をチェック
        if pitch1 in PITCH_TOLERANCE_MAP:
            return pitch2 in PITCH_TOLERANCE_MAP[pitch1]
        # pitch1 が許容リストにない場合は、完全一致のみ許可
        return pitch1 == pitch2

    def extract_notes_info(self, midi_path: str) -> Tuple[List[Dict], int]:
        """
        MIDIファイルからノート情報を抽出

        Returns:
            (ノート情報のリスト, ticks_per_beat)
            ノート情報: [{pitch, start, end, velocity}, ...]
        """
        midi_obj = miditoolkit.MidiFile(midi_path)

        notes_info = []
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    notes_info.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity
                    })

        # startでソート
        notes_info.sort(key=lambda x: (x['start'], x['pitch']))
        return notes_info, midi_obj.ticks_per_beat

    def compare_notes(
        self,
        notes1: List[Dict],
        notes2: List[Dict],
        ticks_per_beat1: int = 480,
        ticks_per_beat2: int = 480
    ) -> Tuple[bool, Dict]:
        """
        2つのノートリストを比較

        Args:
            notes1: 元のノートリスト
            notes2: 再構築されたノートリスト
            ticks_per_beat1: notes1のticks_per_beat
            ticks_per_beat2: notes2のticks_per_beat

        Returns:
            (比較結果, 詳細情報)
        """
        result = {
            'total_notes_1': len(notes1),
            'total_notes_2': len(notes2),
            'matched_notes': 0,
            'timing_errors': [],
            'missing_notes': [],
            'extra_notes': [],
            'pitch_errors': [],
            'velocity_errors': [],
            'ticks_per_beat_1': ticks_per_beat1,
            'ticks_per_beat_2': ticks_per_beat2
        }

        # 各ノートについてマッチングを試みる
        matched_indices = set()

        for i, note1 in enumerate(notes1):
            best_match = None
            best_distance = float('inf')

            # ===== 追加: 正規化された位置で比較 =====
            # 元のticksをbeat単位に正規化
            note1_normalized_start = note1['start'] / ticks_per_beat1
            # =========================================

            for j, note2 in enumerate(notes2):
                if j in matched_indices:
                    continue

                # ===== 追加: 正規化された位置で比較 =====
                note2_normalized_start = note2['start'] / ticks_per_beat2
                # タイミング差を正規化された単位で計算（beat単位）
                normalized_diff = abs(note2_normalized_start - note1_normalized_start)
                # tick単位の許容範囲を正規化単位に変換
                normalized_tolerance = self.tolerance / ticks_per_beat1
                # ==========================================

                # ピッチが許容範囲内で一致し、タイミング差が許容範囲内
                if self.is_pitch_compatible(note1['pitch'], note2['pitch']) and normalized_diff <= normalized_tolerance:
                    if normalized_diff < best_distance:
                        best_distance = normalized_diff
                        best_match = j

            if best_match is not None:
                result['matched_notes'] += 1
                matched_indices.add(best_match)

                note2 = notes2[best_match]

                # タイミングの誤差を記録（正規化単位）
                if best_distance > 0:
                    orig_bar, orig_beat, orig_pos = tick_to_bar_beat_pos(note1['start'], ticks_per_beat1)
                    conv_bar, conv_beat, conv_pos = tick_to_bar_beat_pos(note2['start'], ticks_per_beat2)
                    result['timing_errors'].append({
                        'note_index': i,
                        'pitch': note1['pitch'],
                        'note_type': pitch_to_note_type(note1['pitch']),
                        'original_bar': orig_bar,
                        'original_beat': orig_beat,
                        'original_position': orig_pos,
                        'converted_bar': conv_bar,
                        'converted_beat': conv_beat,
                        'converted_position': conv_pos,
                        'start_diff_normalized': best_distance,  # beat単位
                        'start_diff_ticks': best_distance * ticks_per_beat1,  # tick単位
                        'original_start': note1['start'],
                        'converted_start': note2['start']
                    })

                # ベロシティの差を記録
                vel_diff = abs(note2['velocity'] - note1['velocity'])
                if vel_diff > 5:  # 5以上の差がある場合
                    bar, beat, pos = tick_to_bar_beat_pos(note1['start'], ticks_per_beat1)
                    result['velocity_errors'].append({
                        'note_index': i,
                        'pitch': note1['pitch'],
                        'note_type': pitch_to_note_type(note1['pitch']),
                        'bar': bar,
                        'beat': beat,
                        'position': pos,
                        'original_velocity': note1['velocity'],
                        'converted_velocity': note2['velocity'],
                        'difference': vel_diff
                    })
            else:
                bar, beat, pos = tick_to_bar_beat_pos(note1['start'], ticks_per_beat1)
                result['missing_notes'].append({
                    'note_index': i,
                    'pitch': note1['pitch'],
                    'note_type': pitch_to_note_type(note1['pitch']),
                    'bar': bar,
                    'beat': beat,
                    'position': pos,
                    'start': note1['start'],
                    'velocity': note1['velocity']
                })

        # マッチしなかった変換後のノートを記録
        for j, note2 in enumerate(notes2):
            if j not in matched_indices:
                bar, beat, pos = tick_to_bar_beat_pos(note2['start'], ticks_per_beat2)
                result['extra_notes'].append({
                    'note_index': j,
                    'pitch': note2['pitch'],
                    'note_type': pitch_to_note_type(note2['pitch']),
                    'bar': bar,
                    'beat': beat,
                    'position': pos,
                    'start': note2['start'],
                    'velocity': note2['velocity']
                })

        # 判定
        is_match = (
            len(result['missing_notes']) == 0 and
            len(result['extra_notes']) == 0 and
            result['matched_notes'] == len(notes1)
        )

        return is_match, result


def get_tokenizer(tokenization_method: str):
    """トークナイズ方式に応じてトークナイザーを返す。"""
    if tokenization_method == 'standard':
        return DrumTokenizer()
    if tokenization_method == 'remi':
        return RemiDrumTokenizer()
    if tokenization_method == 'cp_limb_v1':
        return DrumCPTokenizer()

    raise ValueError(f"Unsupported tokenization_method: {tokenization_method}")


def remi_tokens_to_midi(tokens: List[str], output_path: Optional[str] = None) -> miditoolkit.MidiFile:
    """REMI風トークン列をドラムMIDIへ復元する。"""
    midi_obj = miditoolkit.MidiFile()
    midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL

    drum_track = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')
    step_ticks = DEFAULT_BAR_RESOL // REMI_STEPS_PER_BAR

    current_bar = -1
    current_step = 0
    total_bars = 0
    pending_tempo = 120
    tempo_changes = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == '<BAR>':
            current_bar += 1
            current_step = 0
            total_bars = max(total_bars, current_bar + 1)
            i += 1
            continue

        if token.startswith('<TEMPO_'):
            try:
                pending_tempo = int(token.replace('<TEMPO_', '').replace('>', ''))
                tempo_time = max(0, current_bar) * DEFAULT_BAR_RESOL + current_step * step_ticks
                tempo_changes.append(miditoolkit.TempoChange(pending_tempo, tempo_time))
            except ValueError:
                pass
            i += 1
            continue

        if token.startswith('<BEAT_'):
            try:
                current_step = int(token.replace('<BEAT_', '').replace('>', ''))
            except ValueError:
                current_step = 0
            i += 1
            continue

        if token == '<EOS>':
            break

        if token.startswith('Note_Pitch_') and i + 2 < len(tokens):
            vel_token = tokens[i + 1]
            dur_token = tokens[i + 2]

            if vel_token.startswith('Note_Velocity_') and dur_token.startswith('Note_Duration_'):
                try:
                    pitch = int(token.replace('Note_Pitch_', ''))
                    velocity_bin = int(vel_token.replace('Note_Velocity_', ''))
                    duration_steps = int(dur_token.replace('Note_Duration_', ''))

                    velocity = min(127, max(1, velocity_bin * 4 + 2))
                    start_tick = max(0, current_bar) * DEFAULT_BAR_RESOL + current_step * step_ticks
                    end_tick = start_tick + max(1, duration_steps) * step_ticks

                    drum_track.notes.append(
                        miditoolkit.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_tick,
                            end=end_tick,
                        )
                    )
                    i += 3
                    continue
                except ValueError:
                    pass

        i += 1

    if not tempo_changes:
        tempo_changes.append(miditoolkit.TempoChange(120, 0))

    tempo_changes.sort(key=lambda x: x.time)
    midi_obj.tempo_changes = tempo_changes
    midi_obj.instruments.append(drum_track)

    for bar in range(max(1, total_bars)):
        midi_obj.markers.append(miditoolkit.Marker(f'Bar-{bar + 1}', bar * DEFAULT_BAR_RESOL))

    midi_obj.max_tick = max(max(1, total_bars) * DEFAULT_BAR_RESOL, midi_obj.max_tick)

    if output_path is not None:
        midi_obj.dump(output_path)

    return midi_obj


def pitch_to_note_type(pitch: int) -> str:
    """MIDI pitch をドラム譜上のノート種別へ変換。"""
    if pitch in DRUM_NOTE_MAP:
        drum_type, technique = DRUM_NOTE_MAP[pitch]
        return f"{drum_type}_{technique}"
    return f"UNKNOWN_{pitch}"


def tick_to_bar_beat_pos(tick: int, ticks_per_beat: int) -> Tuple[int, int, int]:
    """tick から小節・拍・位置(24分割)を計算。"""
    bar_ticks = ticks_per_beat * 4
    bar = tick // bar_ticks + 1
    tick_in_bar = tick % bar_ticks
    beat = tick_in_bar // ticks_per_beat + 1
    tick_in_beat = tick_in_bar % ticks_per_beat
    pos = (tick_in_beat * 24) // ticks_per_beat
    return bar, beat, pos


def get_first_drum_note_tick(midi_obj: miditoolkit.MidiFile) -> Optional[int]:
    """ドラムトラック内の最初のノート開始tickを返す。"""
    first_tick = None
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            continue
        for note in instrument.notes:
            if first_tick is None or note.start < first_tick:
                first_tick = note.start
    return first_tick


def align_reconstructed_midi_to_first_note_beat(
    midi_obj: miditoolkit.MidiFile,
    target_first_note_beat: Optional[float],
) -> int:
    """復元MIDIの最初のドラムノートを指定beat位置へ合わせる。"""
    if target_first_note_beat is None or midi_obj.ticks_per_beat <= 0:
        return 0

    first_tick = get_first_drum_note_tick(midi_obj)
    if first_tick is None:
        return 0

    current_first_note_beat = first_tick / midi_obj.ticks_per_beat
    delta_beats = target_first_note_beat - current_first_note_beat
    delta_ticks = int(round(delta_beats * midi_obj.ticks_per_beat))

    if delta_ticks == 0:
        return 0

    # ノート位置をシフトして、先頭空小節を保持する
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            note.start = max(0, note.start + delta_ticks)
            note.end = max(note.start + 1, note.end + delta_ticks)

    # 初期テンポ(0tick)は保持し、それ以降のイベントをシフトする
    for tempo_change in midi_obj.tempo_changes:
        if tempo_change.time > 0:
            tempo_change.time = max(0, tempo_change.time + delta_ticks)

    for marker in midi_obj.markers:
        marker.time = max(0, marker.time + delta_ticks)

    for ts_change in midi_obj.time_signature_changes:
        if ts_change.time > 0:
            ts_change.time = max(0, ts_change.time + delta_ticks)

    for ks_change in midi_obj.key_signature_changes:
        if ks_change.time > 0:
            ks_change.time = max(0, ks_change.time + delta_ticks)

    if hasattr(midi_obj, 'max_tick') and midi_obj.max_tick is not None:
        midi_obj.max_tick = max(0, midi_obj.max_tick + delta_ticks)

    return delta_ticks


def write_comparison_log(
    log_path: str,
    input_file: str,
    reconstructed_file: str,
    tokenization_method: str,
    comparison_result: Dict,
    append: bool = False,
):
    """比較結果を詳細ログとして保存。"""
    mode = 'a' if append else 'w'
    with open(log_path, mode, encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MIDI Round-trip Comparison Log\n")
        f.write("=" * 100 + "\n")
        f.write(f"Input MIDI: {input_file}\n")
        f.write(f"Reconstructed MIDI: {reconstructed_file}\n")
        f.write(f"Tokenization method: {tokenization_method}\n")
        f.write(f"ticks_per_beat (original): {comparison_result['ticks_per_beat_1']}\n")
        f.write(f"ticks_per_beat (reconstructed): {comparison_result['ticks_per_beat_2']}\n")
        f.write("\n")

        f.write("[Summary]\n")
        f.write(f"- total_notes_original: {comparison_result['total_notes_1']}\n")
        f.write(f"- total_notes_reconstructed: {comparison_result['total_notes_2']}\n")
        f.write(f"- matched_notes: {comparison_result['matched_notes']}\n")
        f.write(f"- missing_notes: {len(comparison_result['missing_notes'])}\n")
        f.write(f"- extra_notes: {len(comparison_result['extra_notes'])}\n")
        f.write(f"- timing_errors: {len(comparison_result['timing_errors'])}\n")
        f.write(f"- velocity_errors: {len(comparison_result['velocity_errors'])}\n")
        f.write("\n")

        f.write("[Missing Notes]\n")
        if comparison_result['missing_notes']:
            for n in comparison_result['missing_notes']:
                f.write(
                    "- type=missing "
                    f"bar={n['bar']} beat={n['beat']} pos={n['position']} "
                    f"pitch={n['pitch']} note_type={n['note_type']} "
                    f"start_tick={n['start']} velocity={n['velocity']}\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

        f.write("[Extra Notes]\n")
        if comparison_result['extra_notes']:
            for n in comparison_result['extra_notes']:
                f.write(
                    "- type=extra "
                    f"bar={n['bar']} beat={n['beat']} pos={n['position']} "
                    f"pitch={n['pitch']} note_type={n['note_type']} "
                    f"start_tick={n['start']} velocity={n['velocity']}\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

        f.write("[Timing Errors]\n")
        if comparison_result['timing_errors']:
            for e in comparison_result['timing_errors']:
                f.write(
                    "- "
                    f"pitch={e['pitch']} note_type={e['note_type']} "
                    f"orig(bar={e['original_bar']},beat={e['original_beat']},pos={e['original_position']},tick={e['original_start']}) "
                    f"recon(bar={e['converted_bar']},beat={e['converted_beat']},pos={e['converted_position']},tick={e['converted_start']}) "
                    f"diff_ticks={e['start_diff_ticks']:.2f} diff_beats={e['start_diff_normalized']:.4f}\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

        f.write("[Velocity Errors]\n")
        if comparison_result['velocity_errors']:
            for e in comparison_result['velocity_errors']:
                f.write(
                    "- "
                    f"pitch={e['pitch']} note_type={e['note_type']} "
                    f"bar={e['bar']} beat={e['beat']} pos={e['position']} "
                    f"original_velocity={e['original_velocity']} converted_velocity={e['converted_velocity']} "
                    f"difference={e['difference']}\n"
                )
        else:
            f.write("- none\n")


def test_round_trip(
    midi_path: str,
    output_dir: str,
    vocab_path: str = None,
    tokenization_method: str = 'standard',
    comparison_log_path: Optional[str] = None,
) -> Dict:
    """
    往復変換テストを実行

    Args:
        midi_path: 入力MIDIファイルパス
        output_dir: 出力ディレクトリ
        vocab_path: 語彙ファイルパス（オプション）

    Returns:
        テスト結果の辞書
    """
    print(f"\n{'=' * 80}")
    print(f"テスト対象: {os.path.basename(midi_path)}")
    print(f"{'=' * 80}\n")

    os.makedirs(output_dir, exist_ok=True)

    result = {
        'input_file': midi_path,
        'tokenization_method': tokenization_method,
        'success': False,
        'steps': {}
    }

    try:
        # ステップ1: トークナイザーの初期化
        print("ステップ1: トークナイザーの初期化")
        tokenizer = get_tokenizer(tokenization_method)
        print(f"  トークナイズ方式: {tokenization_method}")

        if tokenization_method == 'standard' and vocab_path and os.path.exists(vocab_path):
            tokenizer.load_vocab(vocab_path)
            print(f"  ✓ 語彙ファイルを読み込み: {vocab_path}")
        else:
            print(f"  ✓ デフォルト語彙を使用")

        print(f"  語彙サイズ: {tokenizer.vocab_size}")
        result['steps']['tokenizer_init'] = True

        # ステップ2: MIDI → トークン変換
        print("\nステップ2: MIDI → トークン列への変換")

        # ===== 追加: 元のMIDI情報を表示 =====
        original_midi = miditoolkit.MidiFile(midi_path)
        print(f"  元のMIDI情報:")
        print(f"    ticks_per_beat: {original_midi.ticks_per_beat}")

        drum_track = None
        for inst in original_midi.instruments:
            if inst.is_drum:
                drum_track = inst
                break

        if drum_track:
            print(f"    ドラム音符数: {len(drum_track.notes)}")
            if drum_track.notes:
                print(f"    音符範囲: {min(n.start for n in drum_track.notes)} ~ {max(n.end for n in drum_track.notes)} ticks")
        # =====================================

        first_note_tick = get_first_drum_note_tick(original_midi)
        first_note_beat = None
        if first_note_tick is not None and original_midi.ticks_per_beat > 0:
            first_note_beat = first_note_tick / original_midi.ticks_per_beat

        tokens = None
        cp_data = None
        indices = None

        if tokenization_method == 'cp_limb_v1':
            cp_data, bar_positions = tokenizer.midi_to_cp_data(midi_path)
            n_events = len(cp_data['cp_type'])
            print(f"  ✓ CPイベント列を生成")
            print(f"  イベント数: {n_events}")
        else:
            tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
            indices = tokenizer.tokens_to_indices(tokens)
            print(f"  ✓ トークン列を生成")
            print(f"  トークン数: {len(tokens)}")

        print(f"  小節数: {len(bar_positions)}")
        result['steps']['midi_to_tokens'] = True
        result['token_count'] = len(cp_data['cp_type']) if cp_data is not None else len(tokens)
        result['bar_count'] = len(bar_positions)

        if tokenization_method != 'cp_limb_v1':
            print(f"  ✓ インデックス列に変換")

        # ステップ3: pklファイルに保存
        print("\nステップ3: トークン列をpklファイルに保存")
        base_name = os.path.splitext(os.path.basename(midi_path))[0]

        tokens_pkl_path = os.path.join(output_dir, f"{base_name}_tokens.pkl")
        with open(tokens_pkl_path, 'wb') as f:
            if tokenization_method == 'cp_limb_v1':
                pickle.dump({
                    'tokenization_method': tokenization_method,
                    'cp_data': cp_data,
                    'bar_positions': bar_positions,
                    'idx2type_token': tokenizer.idx2type_token,
                    'idx2struct_token': tokenizer.idx2struct_token,
                    'idx2limb_token': tokenizer.idx2limb_token,
                    'first_note_beat': first_note_beat,
                }, f)
            else:
                pickle.dump({
                    'tokenization_method': tokenization_method,
                    'tokens': tokens,
                    'indices': indices,
                    'bar_positions': bar_positions,
                    'vocab': tokenizer.idx2token,
                    'first_note_beat': first_note_beat,
                }, f)
        print(f"  ✓ トークン列を保存: {tokens_pkl_path}")
        result['steps']['save_tokens'] = True
        result['tokens_pkl_path'] = tokens_pkl_path

        # ステップ4: pklファイルから読み込み
        print("\nステップ4: pklファイルからトークン列を読み込み")
        with open(tokens_pkl_path, 'rb') as f:
            loaded_data = pickle.load(f)

        loaded_tokens = None
        loaded_cp_data = None

        if tokenization_method == 'cp_limb_v1':
            loaded_cp_data = loaded_data['cp_data']
            print(f"  ✓ CPイベント列を読み込み")
            print(f"  読み込んだイベント数: {len(loaded_cp_data['cp_type'])}")

            if loaded_cp_data == cp_data:
                print(f"  ✓ 保存/読み込みの整合性確認OK")
            else:
                print(f"  ✗ 警告: 保存/読み込みで不一致")
        else:
            loaded_tokens = loaded_data['tokens']
            loaded_indices = loaded_data['indices']
            print(f"  ✓ トークン列を読み込み")
            print(f"  読み込んだトークン数: {len(loaded_tokens)}")

            if loaded_tokens == tokens:
                print(f"  ✓ 保存/読み込みの整合性確認OK")
            else:
                print(f"  ✗ 警告: 保存/読み込みで不一致")

        loaded_first_note_beat = loaded_data.get('first_note_beat')
        if loaded_first_note_beat is not None:
            print(f"  ✓ 先頭ノートbeat情報を読み込み: {loaded_first_note_beat:.4f}")

        result['steps']['load_tokens'] = True

        # ステップ5: トークン → MIDI変換
        print("\nステップ5: トークン列 → MIDIファイルへの変換")
        reconstructed_midi_path = os.path.join(output_dir, f"{base_name}_reconstructed.mid")

        if tokenization_method == 'cp_limb_v1':
            idx2type_token = loaded_data.get('idx2type_token', loaded_data['idx2struct_token'])
            midi_obj = cp_data_to_midi(
                loaded_cp_data,
                idx2type_token,
                loaded_data['idx2limb_token'],
                reconstructed_midi_path,
                bpm=120,
            )
        elif tokenization_method == 'remi':
            midi_obj = remi_tokens_to_midi(loaded_tokens, reconstructed_midi_path)
        else:
            converter = DrumToken2MIDI()
            midi_obj = converter.tokens_to_midi(loaded_tokens, reconstructed_midi_path)

        shift_ticks = align_reconstructed_midi_to_first_note_beat(midi_obj, loaded_first_note_beat)
        if shift_ticks != 0:
            shift_beats = shift_ticks / midi_obj.ticks_per_beat
            print(f"  ✓ 先頭空小節補正を適用: shift={shift_ticks} ticks ({shift_beats:.4f} beats)")

        midi_obj.dump(reconstructed_midi_path)

        print(f"  ✓ MIDIファイルを生成: {reconstructed_midi_path}")

        # ===== 追加: 再構築されたMIDI情報を表示 =====
        print(f"  再構築されたMIDI情報:")
        print(f"    ticks_per_beat: {midi_obj.ticks_per_beat}")
        print(f"    ドラム音符数: {len(midi_obj.instruments[0].notes)}")
        if midi_obj.instruments[0].notes:
            print(f"    音符範囲: {min(n.start for n in midi_obj.instruments[0].notes)} ~ {max(n.end for n in midi_obj.instruments[0].notes)} ticks")
        # ==========================================

        result['steps']['tokens_to_midi'] = True
        result['reconstructed_midi_path'] = reconstructed_midi_path
        result['applied_start_shift_ticks'] = shift_ticks

        # ステップ6: 元のMIDIと比較
        print("\nステップ6: 元のMIDIファイルと復元したMIDIファイルを比較")
        comparator = MIDIComparator(tolerance_ticks=20)

        # ===== 修正: ticks_per_beatも取得 =====
        original_notes, original_tpb = comparator.extract_notes_info(midi_path)
        reconstructed_notes, reconstructed_tpb = comparator.extract_notes_info(reconstructed_midi_path)
        # =====================================

        print(f"  元のノート数: {len(original_notes)} (ticks_per_beat: {original_tpb})")
        print(f"  復元後のノート数: {len(reconstructed_notes)} (ticks_per_beat: {reconstructed_tpb})")

        # ===== 修正: ticks_per_beatを渡す =====
        is_match, comparison_result = comparator.compare_notes(
            original_notes, reconstructed_notes,
            original_tpb, reconstructed_tpb
        )
        # ====================================

        result['steps']['comparison'] = True
        result['comparison'] = comparison_result
        per_file_log_path = os.path.join(output_dir, f"{base_name}_comparison.log")
        target_log_path = comparison_log_path or per_file_log_path
        write_comparison_log(
            target_log_path,
            midi_path,
            reconstructed_midi_path,
            tokenization_method,
            comparison_result,
            append=comparison_log_path is not None,
        )
        result['comparison_log_path'] = target_log_path

        # 比較結果の表示
        print(f"\n  比較結果:")
        print(f"    一致したノート: {comparison_result['matched_notes']}/{comparison_result['total_notes_1']}")
        print(f"    比較ログ: {target_log_path}")

        if comparison_result['timing_errors']:
            print(f"    タイミング誤差: {len(comparison_result['timing_errors'])}件")
            # ===== 修正: 正規化単位での平均誤差を表示 =====
            avg_error_normalized = np.mean([e['start_diff_normalized'] for e in comparison_result['timing_errors']])
            avg_error_ticks = np.mean([e['start_diff_ticks'] for e in comparison_result['timing_errors']])
            print(f"      平均誤差: {avg_error_ticks:.2f} ticks ({avg_error_normalized:.4f} beats)")
            # ============================================

            # ===== 追加: 最初の5個のタイミング誤差を表示 =====
            if len(comparison_result['timing_errors']) <= 5:
                print(f"      詳細:")
                for err in comparison_result['timing_errors']:
                    print(f"        - Pitch {err['pitch']}: {err['start_diff_ticks']:.2f} ticks ({err['start_diff_normalized']:.4f} beats)")
            # ============================================

        if comparison_result['velocity_errors']:
            print(f"    ベロシティ誤差: {len(comparison_result['velocity_errors'])}件")
            avg_vel_error = np.mean([e['difference'] for e in comparison_result['velocity_errors']])
            print(f"      平均誤差: {avg_vel_error:.2f}")

        if comparison_result['missing_notes']:
            print(f"    ✗ 欠損ノート: {len(comparison_result['missing_notes'])}件")
            for note in comparison_result['missing_notes'][:5]:
                print(
                    f"      - Bar {note['bar']}, Beat {note['beat']}, Pos {note['position']:2d}, "
                    f"{note['note_type']} (Pitch {note['pitch']}, Start {note['start']}, Vel {note['velocity']})"
                )
            if len(comparison_result['missing_notes']) > 5:
                print(f"      ... 他 {len(comparison_result['missing_notes']) - 5}件")

        if comparison_result['extra_notes']:
            print(f"    ✗ 余分なノート: {len(comparison_result['extra_notes'])}件")
            for note in comparison_result['extra_notes'][:5]:
                print(
                    f"      - Bar {note['bar']}, Beat {note['beat']}, Pos {note['position']:2d}, "
                    f"{note['note_type']} (Pitch {note['pitch']}, Start {note['start']}, Vel {note['velocity']})"
                )
            if len(comparison_result['extra_notes']) > 5:
                print(f"      ... 他 {len(comparison_result['extra_notes']) - 5}件")

        # ===== 追加: 最初の10個のノート位置比較を表示 =====
        if original_notes and reconstructed_notes:
            print(f"\n  最初の10個のノート位置比較:")
            print(f"    {'元 (ticks)':>15} | {'再構築 (ticks)':>15} | {'差 (beats)':>12}")
            print(f"    {'-'*15}-+-{'-'*15}-+-{'-'*12}")

            for i in range(min(10, len(original_notes), len(reconstructed_notes))):
                orig_start = original_notes[i]['start']
                recon_start = reconstructed_notes[i]['start']

                # 正規化後の位置
                orig_normalized = orig_start / original_tpb
                recon_normalized = recon_start / reconstructed_tpb
                diff = abs(orig_normalized - recon_normalized)

                print(f"    {orig_start:>15} | {recon_start:>15} | {diff:>12.4f}")
        # ==============================================

        if is_match:
            print(f"\n  ✓ 往復変換成功: 元のMIDIファイルとほぼ一致")
            result['success'] = True
        else:
            print(f"\n  △ 往復変換完了: ただし一部のノートが不一致")
            result['success'] = False

    except Exception as e:
        print(f"\n  ✗ エラーが発生: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        result['success'] = False

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MIDI往復変換テスト')
    parser.add_argument('--midi_dir', type=str, default='./test_token/sample_midis',
                        help='テスト用MIDIファイルのディレクトリ')
    parser.add_argument('--output_dir', type=str, default='./test_token/output',
                        help='出力ディレクトリ')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='語彙ファイルパス（オプション）')
    parser.add_argument('--single_file', type=str, default=None,
                        help='単一のMIDIファイルをテスト')
    parser.add_argument('--tokenization_method', type=str, default='standard',
                        choices=['standard', 'remi', 'cp_limb_v1'],
                        help='トークナイズ方式 (default: standard)')
    parser.add_argument('--all_methods', action='store_true',
                        help='3方式 (standard/remi/cp_limb_v1) を連続実行')

    args = parser.parse_args()

    print("=" * 80)
    print("MIDI往復変換テスト")
    print("=" * 80)

    # テストするMIDIファイルのリストを取得
    if args.single_file:
        midi_files = [args.single_file]
    else:
        if not os.path.exists(args.midi_dir):
            print(f"✗ エラー: MIDIディレクトリが見つかりません: {args.midi_dir}")
            print(f"\n最初にサンプルMIDIファイルを生成してください:")
            print(f"  python test_token/sample_midi_generator.py")
            return 1

        midi_files = [
            os.path.join(args.midi_dir, f)
            for f in os.listdir(args.midi_dir)
            if f.endswith(('.mid', '.midi'))
        ]

    if not midi_files:
        print(f"✗ エラー: テスト用MIDIファイルが見つかりません")
        return 1

    methods = [args.tokenization_method]
    if args.all_methods:
        methods = ['standard', 'remi', 'cp_limb_v1']

    print(f"\nテスト対象ファイル数: {len(midi_files)}")
    print(f"実行方式: {', '.join(methods)}")

    os.makedirs(args.output_dir, exist_ok=True)
    global_comparison_log_path = os.path.join(args.output_dir, 'all_comparisons.log')
    if os.path.exists(global_comparison_log_path):
        os.remove(global_comparison_log_path)
    print(f"集約比較ログ(全方式共通): {global_comparison_log_path}")

    results = []
    for method in methods:
        method_output_dir = args.output_dir
        if len(methods) > 1:
            method_output_dir = os.path.join(args.output_dir, method)

        os.makedirs(method_output_dir, exist_ok=True)

        print(f"\n{'-' * 80}")
        print(f"方式: {method}")
        print(f"{'-' * 80}")

        for midi_path in midi_files:
            result = test_round_trip(
                midi_path,
                method_output_dir,
                args.vocab_path,
                tokenization_method=method,
                comparison_log_path=global_comparison_log_path,
            )
            results.append(result)

    # 結果のサマリー
    print(f"\n{'=' * 80}")
    print("テスト結果サマリー")
    print(f"{'=' * 80}\n")

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失敗"
        filename = os.path.basename(result['input_file'])
        method = result.get('tokenization_method', 'standard')
        print(f"{status:8s} - [{method}] {filename}")

        if 'comparison' in result:
            comp = result['comparison']
            print(f"           一致ノート: {comp['matched_notes']}/{comp['total_notes_1']}")
            if comp['missing_notes']:
                print(f"           欠損ノート: {len(comp['missing_notes'])}件")
            if comp['extra_notes']:
                print(f"           余分なノート: {len(comp['extra_notes'])}件")

    print(f"\n総合結果: {success_count}/{total_count} ファイル成功")

    if success_count == total_count:
        print("\n✓ すべてのテストが成功しました!")
        return 0
    else:
        print(f"\n△ {total_count - success_count}件のテストが失敗しました")
        return 1


if __name__ == '__main__':
    sys.exit(main())
