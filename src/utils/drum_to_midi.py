"""
ドラムトークン列からMIDIファイルへの変換
"""
import numpy as np
import miditoolkit
from typing import List, Optional

# 定数
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
POSITIONS_PER_BEAT = 24
DEFAULT_BPM = 120

# ドラム楽器からMIDIノート番号へのマッピング（逆引き）
# drum_tokenizer.pyのSTANDARD_PITCH_MAPと対応させる
DRUM_TO_NOTE = {
    'KICK': 36,            # C2 (MIDI 36)
    'SNARE': 38,           # D2 (MIDI 38)
    'SNARE_ACCENT': 37,    # C#2 (MIDI 37) - Snare accent
    'SNARE_XSTICK': 39,    # G1 (MIDI 39)
    'SNARE_RIMSHOT': 40,   # E2 (MIDI 40)
    'TOM1': 48,            # C3 (MIDI 48)
    'TOM2': 47,            # B2 (MIDI 47)
    'FLOOR': 43,           # G2 (MIDI 43)
    'HH_CLOSED': 42,       # F#2 (MIDI 42)
    'HH_OPEN': 46,         # A#2 (MIDI 46)
    'HH_PEDAL': 44,        # G#2 (MIDI 44)
    'RIDE_BOW': 51,        # D#3 (MIDI 51)
    'RIDE_BELL': 53,       # F#3 (MIDI 53)
    'CRASH': 49,           # C#3 (MIDI 49)
    'SPLASH': 55,          # G#3 (MIDI 55)
    'CHINA': 52,           # E3 (MIDI 52)
    'TAMBOURINE': 54,      # F#3 (MIDI 54)
    'COWBELL': 56,         # G#3 (MIDI 56)
}

# ベロシティレベルのマッピング
VELOCITY_MAP = {
    'Ghost': 30,
    'Normal': 80,
    'Accent': 120,
}


class DrumToken2MIDI:
    """ドラムトークン列をMIDIに変換"""

    def __init__(self, default_bpm: int = 120):
        self.default_bpm = default_bpm

    def _parse_drum_token(self, token: str) -> Optional[tuple]:
        """
        ドラムトークンをパース

        Returns:
            (instrument, technique, velocity_level) or None
        """
        if token.startswith('<') or token in ['<PAD>', '<EOS>']:
            return None
        if token.startswith('LIMB_'):
            return None

        parts = token.split('_')
        if len(parts) < 3:
            return None

        # 楽器名を構築（2単語の可能性がある: HH_CLOSED, RIDE_BOWなど）
        if parts[1] in ['CLOSED', 'OPEN', 'PEDAL', 'BOW', 'BELL', 'ACCENT']:
            instrument = f"{parts[0]}_{parts[1]}"
            technique = parts[2]
            velocity_level = '_'.join(parts[3:]) if len(parts) > 3 else 'Normal'
        else:
            instrument = parts[0]
            technique = parts[1]
            velocity_level = '_'.join(parts[2:]) if len(parts) > 2 else 'Normal'

        return instrument, technique, velocity_level

    def _get_velocity(self, velocity_level: str) -> int:
        """ベロシティレベルをMIDIベロシティ値に変換"""
        return VELOCITY_MAP.get(velocity_level, VELOCITY_MAP['Normal'])

    def _get_note_number(self, instrument: str, technique: str) -> Optional[int]:
        """楽器と技法からMIDIノート番号を取得"""
        if technique == 'XSTICK':
            return DRUM_TO_NOTE.get(f'{instrument}_XSTICK')
        elif technique == 'RIMSHOT':
            return DRUM_TO_NOTE.get(f'{instrument}_RIMSHOT')
        elif technique == 'ACCENT':
            return DRUM_TO_NOTE.get(f'{instrument}_ACCENT')
        else:
            return DRUM_TO_NOTE.get(instrument)

    def _calculate_duration(self, technique: str) -> int:
        """技法に応じた音価を計算（ticks）"""
        if technique == 'ROLL':
            return DEFAULT_BEAT_RESOL * 2  # 2拍分
        elif technique == 'FLAM':
            return DEFAULT_BEAT_RESOL // 4  # 16分音符
        else:
            return DEFAULT_BEAT_RESOL // 8  # 32分音符（デフォルト）

    def _parse_tempo_token(self, token: str) -> Optional[int]:
        """テンポトークンをBPM整数に変換"""
        if not isinstance(token, str) or not token.startswith('<TEMPO_'):
            return None
        try:
            return int(token.replace('<TEMPO_', '').replace('>', ''))
        except ValueError:
            return None

    def tokens_to_midi(
        self,
        tokens: List[str],
        output_path: Optional[str] = None,
        bpm: Optional[int] = None
    ) -> miditoolkit.MidiFile:
        """
        トークン列をMIDIファイルに変換

        Args:
            tokens: トークン列
            output_path: 出力MIDIファイルパス（Noneの場合は保存しない）
            bpm: テンポ（BPM）。Noneの場合はトークン列から自動検出

        Returns:
            MidiFile オブジェクト
        """
        # bpm引数はテンポトークンが存在しない場合のフォールバックとして扱う
        fallback_bpm = self.default_bpm if bpm is None else int(bpm)

        # MIDIオブジェクト作成
        midi_obj = miditoolkit.MidiFile()
        midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL

        # ドラムトラック作成
        drum_track = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')

        # 最初のテンポトークンを検出し、time=0へ初期テンポを必ず設定
        first_token_tempo = None
        for token in tokens:
            parsed_tempo = self._parse_tempo_token(token)
            if parsed_tempo is not None:
                first_token_tempo = parsed_tempo
                break

        initial_tempo = first_token_tempo if first_token_tempo is not None else fallback_bpm
        midi_obj.tempo_changes.append(miditoolkit.TempoChange(initial_tempo, 0))

        # テンポ設定は「楽曲の最初」または「テンポ切替時」のみ追加する
        current_tempo = initial_tempo
        pending_tempo = None

        # トークン列から最大小節番号を取得（<BAR>トークンの総数 = 小節総数）
        max_bar = 0
        for token in tokens:
            if token == '<BAR>':
                max_bar += 1
        
        # 予期される全小節数を記録（空の小節を含む）
        total_bars = max(1, max_bar)

        # トークンを解析してノートイベントを生成
        current_bar = 0
        current_beat = 0
        current_pos = 0
        current_tick = 0

        choke_events = []  # チョークイベントのリスト

        for i, token in enumerate(tokens):
            if token.startswith('<TEMPO_'):
                # テンポトークンは次の時間境界（通常は<BAR>）で適用する
                pending_tempo = self._parse_tempo_token(token)
                continue

            elif token == '<BAR>':
                # 新しい小節を開始（空の小節も含める）
                current_bar += 1
                current_beat = 0
                current_pos = 0
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL

                # 小節先頭でテンポを確定（開始時または変更時のみ追加）
                target_tempo = pending_tempo if pending_tempo is not None else current_tempo
                if target_tempo != current_tempo:
                    midi_obj.tempo_changes.append(miditoolkit.TempoChange(target_tempo, current_tick))
                    current_tempo = target_tempo
                pending_tempo = None

            elif token.startswith('<BEAT_'):
                beat_num = int(token.replace('<BEAT_', '').replace('>', ''))
                current_beat = beat_num - 1
                current_pos = 0
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL + current_beat * DEFAULT_BEAT_RESOL

            elif token.startswith('<POS_'):
                pos_num = int(token.replace('<POS_', '').replace('>', ''))
                current_pos = pos_num
                tick_in_beat = (DEFAULT_BEAT_RESOL * current_pos) // POSITIONS_PER_BEAT
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL + current_beat * DEFAULT_BEAT_RESOL + tick_in_beat

            elif token == '<EOS>':
                break

            elif token not in ['<PAD>']:
                # ドラム演奏トークンの処理
                parsed = self._parse_drum_token(token)
                if parsed is None:
                    continue

                instrument, technique, velocity_level = parsed

                # CHOKEトークンの処理
                if technique == 'CHOKE':
                    # 最後に追加されたこの楽器のノートを探してチョークを適用
                    for note in reversed(drum_track.notes):
                        note_instrument = None
                        for key, val in DRUM_TO_NOTE.items():
                            if val == note.pitch:
                                note_instrument = key.split('_')[0]
                                break

                        if note_instrument == instrument.split('_')[0]:
                            # このノートをチョーク（音を短くする）
                            note.end = min(note.end, current_tick)
                            break
                    continue

                # MIDIノート番号を取得
                note_number = self._get_note_number(instrument, technique)
                if note_number is None:
                    continue

                # ベロシティを取得
                velocity = self._get_velocity(velocity_level)

                # 音価を計算
                duration = self._calculate_duration(technique)

                # FLAMの処理（前打音 + 主音符）
                if technique == 'FLAM':
                    # 前打音（少し早めにゴーストノート）
                    grace_note = miditoolkit.Note(
                        velocity=VELOCITY_MAP['Ghost'],
                        pitch=note_number,
                        start=max(0, current_tick - 30),
                        end=max(0, current_tick - 10)
                    )
                    drum_track.notes.append(grace_note)

                    # 主音符
                    main_note = miditoolkit.Note(
                        velocity=velocity,
                        pitch=note_number,
                        start=current_tick,
                        end=current_tick + duration
                    )
                    drum_track.notes.append(main_note)

                elif technique == 'ROLL':
                    # ロールの表現（複数の短いノートで表現）
                    roll_duration = DEFAULT_BEAT_RESOL * 2
                    roll_subdivision = DEFAULT_BEAT_RESOL // 16  # 64分音符で刻む

                    for t in range(0, roll_duration, roll_subdivision):
                        note = miditoolkit.Note(
                            velocity=max(40, velocity - 20),  # 少し弱めに
                            pitch=note_number,
                            start=current_tick + t,
                            end=current_tick + t + roll_subdivision // 2
                        )
                        drum_track.notes.append(note)

                else:
                    # 通常のノート
                    note = miditoolkit.Note(
                        velocity=velocity,
                        pitch=note_number,
                        start=current_tick,
                        end=current_tick + duration
                    )
                    drum_track.notes.append(note)

        # ドラムトラックを追加
        midi_obj.instruments.append(drum_track)

        # 全小節のマーカーを追加（空の小節を含む）
        # 譜面表示側で先頭空小節を保持しやすくする
        for bar in range(total_bars):
            midi_obj.markers.append(
                miditoolkit.Marker(f'Bar-{bar + 1}', bar * DEFAULT_BAR_RESOL)
            )

        # MIDIの終了時刻を設定（全小節を包含するように）
        midi_obj.end_time = total_bars * DEFAULT_BAR_RESOL

        # ファイル保存
        if output_path is not None:
            midi_obj.dump(output_path)

        return midi_obj


def tokens_to_midi(
    tokens: List[str],
    output_path: str,
    bpm: Optional[int] = None
) -> miditoolkit.MidiFile:
    """
    ユーティリティ関数: トークン列をMIDIファイルに変換

    Args:
        tokens: トークン列
        output_path: 出力MIDIファイルパス
        bpm: テンポ（BPM）。テンポトークンが無い場合のみフォールバックとして使用

    Returns:
        MidiFile オブジェクト
    """
    converter = DrumToken2MIDI(default_bpm=DEFAULT_BPM if bpm is None else bpm)
    return converter.tokens_to_midi(tokens, output_path, bpm)


def cp_events_to_tokens(cp_data: dict, idx2type_token: dict, idx2limb_token: dict) -> List[str]:
    """CPイベント系列を既存トークン列に復元する。"""
    tokens: List[str] = []

    cp_type_seq = cp_data.get('cp_type')
    if cp_type_seq is None and 'event_type' in cp_data and 'struct_token' in cp_data:
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
            cp_type_token = idx2type_token.get(int(type_id), '<PAD>')

        if cp_type_token != '<NOTE>':
            if cp_type_token != '<PAD>':
                tokens.append(cp_type_token)
            continue

        if int(pos_id) < POSITIONS_PER_BEAT:
            tokens.append(f'<POS_{int(pos_id)}>')

        for limb_id in (h1, h2, rf, lf):
            limb_token = idx2limb_token.get(int(limb_id), '<PAD>')
            if limb_token not in ('<PAD>', '<NONE>', '<NA>'):
                tokens.append(limb_token)

    return tokens


def cp_data_to_midi(
    cp_data: dict,
    idx2type_token: dict,
    idx2limb_token: dict,
    output_path: str,
    bpm: Optional[int] = None
) -> miditoolkit.MidiFile:
    """CPデータからMIDIを生成するユーティリティ。"""
    tokens = cp_events_to_tokens(cp_data, idx2type_token, idx2limb_token)
    return tokens_to_midi(tokens, output_path, bpm=bpm)


if __name__ == '__main__':
    # テスト用
    test_tokens = [
        '<BAR>',
        '<TEMPO_66>',
        '<BAR>',
        '<TEMPO_66>',
        '<BAR>',
        '<TEMPO_66>',
        '<BAR>',
        '<TEMPO_66>',
        '<BAR>',
        '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'HH_CLOSED_HIT_Normal',
        '<POS_6>', 'HH_CLOSED_HIT_Ghost',
        '<POS_12>', 'SNARE_HIT_Normal', 'HH_CLOSED_HIT_Normal',
        '<POS_18>', 'HH_CLOSED_HIT_Ghost',
        '<BEAT_2>', '<POS_0>', 'KICK_HIT_Accent', 'HH_CLOSED_HIT_Normal',
        '<POS_6>', 'HH_CLOSED_HIT_Ghost',
        '<POS_12>', 'SNARE_HIT_Accent', 'HH_CLOSED_HIT_Normal',
        '<POS_18>', 'HH_CLOSED_HIT_Ghost',
        '<BAR>',
        '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'CRASH_HIT_Accent',
        '<POS_12>', 'SNARE_FLAM_Normal',
        '<BEAT_2>', '<POS_0>', 'KICK_HIT_Normal',
        '<POS_12>', 'SNARE_HIT_Normal',
        '<EOS>'
    ]

    print("Converting test tokens to MIDI...")
    midi_obj = tokens_to_midi(test_tokens, 'test_drum_output.mid', bpm=120)
    print(f"Created MIDI with {len(midi_obj.instruments[0].notes)} notes")
    print("Saved to: test_drum_output.mid")
