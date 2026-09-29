#!/usr/bin/env python
"""
トークン列の内容検証

pklファイルに保存されたトークン列が、元のMIDI譜面の音符と
正しく対応しているかを詳細に検証するスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import miditoolkit
from typing import List, Dict, Tuple
from collections import Counter
from src.utils.drum_to_midi import cp_events_to_tokens


# 定数
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
POSITIONS_PER_BEAT = 24
REMI_STEPS_PER_BAR = 16

# GMドラムマップ（逆引き用）
NOTE_TO_DRUM = {
    35: 'KICK', 36: 'KICK',
    38: 'SNARE', 40: 'SNARE',
    37: 'SNARE_XSTICK',
    41: 'FLOOR', 43: 'FLOOR',
    45: 'TOM2', 47: 'TOM2', 48: 'TOM1', 50: 'TOM2',
    42: 'HH_CLOSED', 44: 'HH_PEDAL', 46: 'HH_OPEN',
    51: 'RIDE_BOW', 59: 'RIDE_BOW', 53: 'RIDE_BELL',
    49: 'CRASH', 57: 'CRASH',
    52: 'CHINA', 55: 'SPLASH'
}


class TokenVerifier:
    """トークン列と元のMIDIを照合して検証"""

    def __init__(self):
        pass

    def extract_midi_events(self, midi_path: str) -> List[Dict]:
        """
        MIDIファイルからイベント情報を抽出

        Returns:
            イベント情報のリスト [
                {
                    'bar': 小節番号,
                    'beat': 拍番号(0-3),
                    'position': 位置番号(0-23),
                    'tick': 絶対tick位置,
                    'drum': ドラム名,
                    'pitch': MIDIノート番号,
                    'velocity': ベロシティ
                },
                ...
            ]
        """
        midi_obj = miditoolkit.MidiFile(midi_path)

        # ドラムトラックを探す
        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            return []

        events = []

        for note in drum_track.notes:
            # 小節、拍、位置を計算
            bar = note.start // DEFAULT_BAR_RESOL
            tick_in_bar = note.start % DEFAULT_BAR_RESOL
            beat = tick_in_bar // DEFAULT_BEAT_RESOL
            tick_in_beat = tick_in_bar % DEFAULT_BEAT_RESOL
            position = (tick_in_beat * POSITIONS_PER_BEAT) // DEFAULT_BEAT_RESOL

            # ドラム名を取得
            drum_name = NOTE_TO_DRUM.get(note.pitch, f'UNKNOWN_{note.pitch}')

            events.append({
                'bar': bar,
                'beat': beat,
                'position': position,
                'tick': note.start,
                'drum': drum_name,
                'pitch': note.pitch,
                'velocity': note.velocity,
                'duration': note.end - note.start
            })

        # 時系列でソート
        events.sort(key=lambda x: (x['bar'], x['beat'], x['position'], x['drum']))

        return events

    def parse_token_sequence(self, tokens: List[str]) -> List[Dict]:
        """
        トークン列を解析してイベント情報に変換

        Returns:
            イベント情報のリスト [
                {
                    'bar': 小節番号,
                    'beat': 拍番号,
                    'position': 位置番号,
                    'drum_token': ドラムトークン文字列
                },
                ...
            ]
        """
        events = []

        current_bar = -1
        current_beat = -1
        current_position = -1

        for token in tokens:
            if token == '<BAR>':
                current_bar += 1
                current_beat = -1
                current_position = -1

            elif token.startswith('<BEAT_'):
                beat_num = int(token.replace('<BEAT_', '').replace('>', ''))
                current_beat = beat_num - 1
                current_position = -1

            elif token.startswith('<POS_'):
                pos_num = int(token.replace('<POS_', '').replace('>', ''))
                current_position = pos_num

            elif token.startswith('<TEMPO_'):
                # テンポトークンは演奏イベント照合の対象外
                pass

            elif token == '<EOS>':
                break

            elif token not in ['<PAD>']:
                # ドラム演奏トークン
                events.append({
                    'bar': current_bar,
                    'beat': current_beat,
                    'position': current_position,
                    'drum_token': token
                })

        return events

    def classify_velocity(self, velocity: int) -> str:
        """ベロシティをレベル分類"""
        if velocity < 40:
            return 'Ghost'
        elif velocity < 100:
            return 'Normal'
        else:
            return 'Accent'

    def extract_midi_events_remi(self, midi_path: str) -> List[Dict]:
        """REMI検証向けに16分グリッドのMIDIイベントを抽出。"""
        midi_obj = miditoolkit.MidiFile(midi_path)

        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            return []

        events = []
        step_ticks = DEFAULT_BAR_RESOL // REMI_STEPS_PER_BAR

        for note in drum_track.notes:
            bar = note.start // DEFAULT_BAR_RESOL
            in_bar_tick = note.start % DEFAULT_BAR_RESOL
            step = int(round(in_bar_tick / step_ticks))
            step = max(0, min(REMI_STEPS_PER_BAR - 1, step))

            velocity_bin = int(max(0, min(31, note.velocity * 32 // 128)))
            duration_steps = int(round((note.end - note.start) / step_ticks))
            duration_steps = max(1, duration_steps)

            events.append({
                'bar': bar,
                'step': step,
                'pitch': note.pitch,
                'velocity': note.velocity,
                'velocity_bin': velocity_bin,
                'duration_steps': duration_steps,
            })

        events.sort(key=lambda x: (x['bar'], x['step'], x['pitch'], x['velocity']))
        return events

    def parse_remi_token_sequence(self, tokens: List[str]) -> List[Dict]:
        """REMIトークン列をイベントへ変換。"""
        events = []
        current_bar = -1
        current_step = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '<BAR>':
                current_bar += 1
                current_step = 0
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
                        events.append({
                            'bar': current_bar,
                            'step': current_step,
                            'pitch': int(token.replace('Note_Pitch_', '')),
                            'velocity_bin': int(vel_token.replace('Note_Velocity_', '')),
                            'duration_steps': int(dur_token.replace('Note_Duration_', '')),
                        })
                        i += 3
                        continue
                    except ValueError:
                        pass

            i += 1

        return events

    def verify_remi_tokens(self, tokens: List[str], midi_events: List[Dict]) -> Dict:
        """REMIトークン列とMIDIイベントを照合。"""
        token_events = self.parse_remi_token_sequence(tokens)
        result = {
            'total_midi_events': len(midi_events),
            'total_token_events': len(token_events),
            'matched_events': 0,
            'mismatches': [],
            'token_stats': Counter(),
            'drum_stats': Counter(),
        }

        token_map = {}
        for ev in token_events:
            key = (ev['bar'], ev['step'], ev['pitch'])
            token_map.setdefault(key, []).append(ev)
            result['token_stats'][f"Note_Pitch_{ev['pitch']}"] += 1

        for midi_event in midi_events:
            key = (midi_event['bar'], midi_event['step'], midi_event['pitch'])
            result['drum_stats'][f"Pitch_{midi_event['pitch']}"] += 1

            if key not in token_map or not token_map[key]:
                result['mismatches'].append({
                    'type': 'missing_token',
                    'bar': midi_event['bar'],
                    'step': midi_event['step'],
                    'pitch': midi_event['pitch'],
                    'velocity': midi_event['velocity'],
                })
                continue

            token_event = token_map[key].pop(0)
            result['matched_events'] += 1

            if token_event['velocity_bin'] != midi_event['velocity_bin']:
                result['mismatches'].append({
                    'type': 'velocity_bin',
                    'bar': midi_event['bar'],
                    'step': midi_event['step'],
                    'pitch': midi_event['pitch'],
                    'expected_bin': midi_event['velocity_bin'],
                    'actual_bin': token_event['velocity_bin'],
                    'midi_velocity': midi_event['velocity'],
                })

        for key, remaining in token_map.items():
            for ev in remaining:
                result['mismatches'].append({
                    'type': 'extra_token',
                    'bar': ev['bar'],
                    'step': ev['step'],
                    'pitch': ev['pitch'],
                    'velocity_bin': ev['velocity_bin'],
                })

        return result

    def verify_tokens(
        self,
        tokens: List[str],
        midi_events: List[Dict]
    ) -> Dict:
        """
        トークン列とMIDIイベントを照合

        Returns:
            検証結果の辞書
        """
        result = {
            'total_midi_events': len(midi_events),
            'total_token_events': 0,
            'matched_events': 0,
            'mismatches': [],
            'token_stats': Counter(),
            'drum_stats': Counter()
        }

        # トークン列を解析
        token_events = self.parse_token_sequence(tokens)
        result['total_token_events'] = len(token_events)

        # トークンの統計
        for event in token_events:
            result['token_stats'][event['drum_token']] += 1

        # MIDIイベントの統計
        for event in midi_events:
            result['drum_stats'][event['drum']] += 1

        # 各MIDIイベントに対応するトークンを探す
        matched_token_indices = set()

        for midi_event in midi_events:
            found = False
            midi_drum = midi_event['drum']
            expected_vel_level = self.classify_velocity(midi_event['velocity'])

            # 同じ小節、拍、位置の未使用トークンを探す
            for i, token_event in enumerate(token_events):
                if i in matched_token_indices:
                    continue

                if (token_event['bar'] == midi_event['bar'] and
                    token_event['beat'] == midi_event['beat'] and
                    token_event['position'] == midi_event['position']):

                    # ドラムトークンを解析
                    drum_token = token_event['drum_token']
                    parts = drum_token.split('_')

                    # 楽器名を抽出
                    if len(parts) >= 2:
                        if parts[1] in ['CLOSED', 'HALFOPEN', 'OPEN', 'PEDAL', 'BOW', 'BELL']:
                            token_drum = f"{parts[0]}_{parts[1]}"
                        else:
                            token_drum = parts[0]

                        # ベロシティレベルを抽出
                        velocity_level = parts[-1] if len(parts) > 2 else None

                        # ドラム名のマッチング（XSTICKやPEDALなどの特殊ケースも考慮）
                        drum_match = (
                            token_drum == midi_drum or
                            token_drum in midi_drum or
                            midi_drum in token_drum
                        )

                        if drum_match:
                            result['matched_events'] += 1
                            found = True
                            matched_token_indices.add(i)

                            # ベロシティレベルの照合
                            if velocity_level and velocity_level != expected_vel_level:
                                result['mismatches'].append({
                                    'type': 'velocity_level',
                                    'bar': midi_event['bar'],
                                    'beat': midi_event['beat'],
                                    'position': midi_event['position'],
                                    'drum': midi_drum,
                                    'token': drum_token,
                                    'expected_level': expected_vel_level,
                                    'actual_level': velocity_level,
                                    'midi_velocity': midi_event['velocity']
                                })

                            break

            if not found:
                result['mismatches'].append({
                    'type': 'missing_token',
                    'bar': midi_event['bar'],
                    'beat': midi_event['beat'],
                    'position': midi_event['position'],
                    'drum': midi_event['drum'],
                    'pitch': midi_event['pitch'],
                    'velocity': midi_event['velocity']
                })

        for i, token_event in enumerate(token_events):
            if i not in matched_token_indices:
                result['mismatches'].append({
                    'type': 'extra_token',
                    'bar': token_event['bar'],
                    'beat': token_event['beat'],
                    'position': token_event['position'],
                    'token': token_event['drum_token'],
                })

        return result


def verify_pkl_tokens(
    pkl_path: str,
    midi_path: str
) -> Dict:
    """
    pklファイル内のトークン列を検証

    Args:
        pkl_path: pklファイルのパス
        midi_path: 元のMIDIファイルのパス

    Returns:
        検証結果の辞書
    """
    print(f"\n{'=' * 80}")
    print(f"トークン内容検証")
    print(f"{'=' * 80}\n")

    print(f"pklファイル: {os.path.basename(pkl_path)}")
    print(f"MIDIファイル: {os.path.basename(midi_path)}")

    result = {
        'pkl_path': pkl_path,
        'midi_path': midi_path,
        'success': False
    }

    try:
        # pklファイルを読み込み
        print(f"\nステップ1: pklファイルを読み込み")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        tokenization_method = data.get('tokenization_method', 'standard')

        if tokenization_method == 'cp_limb_v1':
            cp_data = data['cp_data']
            idx2type_token = data.get('idx2type_token', data['idx2struct_token'])
            tokens = cp_events_to_tokens(cp_data, idx2type_token, data['idx2limb_token'])
            indices = []
        else:
            tokens = data['tokens']
            indices = data['indices']

        vocab = data.get('vocab', {})
        print(f"  ✓ トークナイズ方式: {tokenization_method}")

        print(f"  ✓ トークン数: {len(tokens)}")
        print(f"  ✓ インデックス数: {len(indices)}")
        print(f"  ✓ 語彙サイズ: {len(vocab)}")

        # トークン列のサンプルを表示
        print(f"\n  トークン列のサンプル（最初の30トークン）:")
        for i, token in enumerate(tokens[:30]):
            print(f"    {i:3d}: {token}")
        if len(tokens) > 30:
            print(f"    ... (残り {len(tokens) - 30} トークン)")

        # MIDIファイルからイベントを抽出
        print(f"\nステップ2: MIDIファイルからイベント情報を抽出")
        verifier = TokenVerifier()
        if tokenization_method == 'remi':
            midi_events = verifier.extract_midi_events_remi(midi_path)
        else:
            midi_events = verifier.extract_midi_events(midi_path)
        print(f"  ✓ MIDIイベント数: {len(midi_events)}")

        # イベントのサンプルを表示
        print(f"\n  MIDIイベントのサンプル（最初の10イベント）:")
        for i, event in enumerate(midi_events[:10]):
            if tokenization_method == 'remi':
                print(f"    {i:3d}: Bar {event['bar']}, Step {event['step']:2d} - "
                      f"Pitch {event['pitch']:2d}, Vel {event['velocity']:3d}")
            else:
                print(f"    {i:3d}: Bar {event['bar']}, Beat {event['beat']}, Pos {event['position']:2d} - "
                      f"{event['drum']:15s} (Pitch {event['pitch']:2d}, Vel {event['velocity']:3d})")
        if len(midi_events) > 10:
            print(f"    ... (残り {len(midi_events) - 10} イベント)")

        # トークン列とMIDIイベントを照合
        print(f"\nステップ3: トークン列とMIDIイベントを照合")
        if tokenization_method == 'remi':
            verification_result = verifier.verify_remi_tokens(tokens, midi_events)
        else:
            verification_result = verifier.verify_tokens(tokens, midi_events)

        print(f"  ✓ 照合完了")
        print(f"\n  照合結果:")
        print(f"    MIDIイベント総数: {verification_result['total_midi_events']}")
        print(f"    トークンイベント総数: {verification_result['total_token_events']}")
        print(f"    一致したイベント: {verification_result['matched_events']}")

        match_rate = (
            verification_result['matched_events'] / verification_result['total_midi_events'] * 100
            if verification_result['total_midi_events'] > 0 else 0
        )
        print(f"    一致率: {match_rate:.1f}%")

        # 不一致の詳細
        if verification_result['mismatches']:
            print(f"\n  不一致の詳細 ({len(verification_result['mismatches'])}件):")

            velocity_mismatches = [m for m in verification_result['mismatches']
                                   if m['type'] in ['velocity_level', 'velocity_bin']]
            missing_tokens = [m for m in verification_result['mismatches']
                             if m['type'] == 'missing_token']
            extra_tokens = [m for m in verification_result['mismatches']
                            if m['type'] == 'extra_token']

            if velocity_mismatches:
                print(f"    ベロシティレベル不一致: {len(velocity_mismatches)}件")
                for m in velocity_mismatches[:5]:
                    if m['type'] == 'velocity_level':
                        print(f"      - Bar {m['bar']}, Beat {m['beat']}, Pos {m['position']:2d} - "
                              f"{m['token']}")
                        print(f"        期待: {m['expected_level']}, 実際: {m['actual_level']} "
                              f"(MIDI Velocity: {m['midi_velocity']})")
                    else:
                        print(f"      - Bar {m['bar']}, Step {m['step']:2d} - Pitch {m['pitch']}")
                        print(f"        期待bin: {m['expected_bin']}, 実際bin: {m['actual_bin']} "
                              f"(MIDI Velocity: {m['midi_velocity']})")
                if len(velocity_mismatches) > 5:
                    print(f"      ... (残り {len(velocity_mismatches) - 5}件)")

            if missing_tokens:
                print(f"    トークンなし: {len(missing_tokens)}件")
                for m in missing_tokens[:5]:
                    if tokenization_method == 'remi':
                        print(f"      - Bar {m['bar']}, Step {m['step']:2d} - "
                              f"Pitch {m['pitch']}, Vel {m['velocity']}")
                    else:
                        print(f"      - Bar {m['bar']}, Beat {m['beat']}, Pos {m['position']:2d} - "
                              f"{m['drum']} (Pitch {m['pitch']}, Vel {m['velocity']})")
                if len(missing_tokens) > 5:
                    print(f"      ... (残り {len(missing_tokens) - 5}件)")

            if extra_tokens:
                print(f"    余分トークン: {len(extra_tokens)}件")
                for m in extra_tokens[:5]:
                    if tokenization_method == 'remi':
                        print(f"      - Bar {m['bar']}, Step {m['step']:2d} - Pitch {m['pitch']}")
                    else:
                        print(f"      - Bar {m['bar']}, Beat {m['beat']}, Pos {m['position']:2d} - {m['token']}")
                if len(extra_tokens) > 5:
                    print(f"      ... (残り {len(extra_tokens) - 5}件)")

        # 統計情報
        print(f"\n  トークン統計:")
        for token, count in verification_result['token_stats'].most_common(10):
            print(f"    {token:30s}: {count:3d}回")

        print(f"\n  ドラム統計:")
        for drum, count in verification_result['drum_stats'].most_common(10):
            print(f"    {drum:20s}: {count:3d}回")

        # 判定
        if match_rate >= 90.0:
            print(f"\n  ✓ 検証成功: トークン列は元のMIDI譜面とよく一致しています")
            result['success'] = True
        elif match_rate >= 70.0:
            print(f"\n  △ 検証部分成功: トークン列は元のMIDI譜面とある程度一致しています")
            result['success'] = True
        else:
            print(f"\n  ✗ 検証失敗: トークン列と元のMIDI譜面の一致率が低すぎます")
            result['success'] = False

        result['verification'] = verification_result
        result['match_rate'] = match_rate

    except Exception as e:
        print(f"\n  ✗ エラーが発生: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        result['success'] = False

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='トークン列の内容検証')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='検証するpklファイルのパス')
    parser.add_argument('--midi_path', type=str, required=True,
                        help='元のMIDIファイルのパス')

    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        print(f"✗ エラー: pklファイルが見つかりません: {args.pkl_path}")
        return 1

    if not os.path.exists(args.midi_path):
        print(f"✗ エラー: MIDIファイルが見つかりません: {args.midi_path}")
        return 1

    result = verify_pkl_tokens(args.pkl_path, args.midi_path)

    print(f"\n{'=' * 80}")
    print("検証結果")
    print(f"{'=' * 80}\n")

    if result['success']:
        print(f"✓ 検証成功")
        if 'match_rate' in result:
            print(f"  一致率: {result['match_rate']:.1f}%")
        return 0
    else:
        print(f"✗ 検証失敗")
        return 1


if __name__ == '__main__':
    sys.exit(main())
