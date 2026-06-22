"""
ドラム譜トークナイザー
MIDIファイルをドラム譜特有のトークン表記法に変換する
"""
import numpy as np
import miditoolkit
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from src.utils.tokenization_policy import (
    POSITIONS_PER_BEAT as SHARED_POSITIONS_PER_BEAT,
    classify_velocity_level,
    quantize_position_in_beat,
)

# 定数定義
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
POSITIONS_PER_BEAT = SHARED_POSITIONS_PER_BEAT  # 1拍を24分割

# ドラム譜用のノートマッピング (カスタムマッピング)
DRUM_NOTE_MAP = {
    35: ('KICK', 'HIT'),
    36: ('KICK', 'HIT'),
    37: ('SNARE_RIM', 'ACCENT'),  # C#2 - Snare accent
    38: ('SNARE', 'HIT'),
    #39: ('SNARE', 'XSTICK'),  # Side stick
    40: ('SNARE', 'HIT'),

    41: ('FLOOR', 'HIT'),  # F2
    42: ('HH_CLOSED', 'HIT'),  # F#2 - Closed hi-hat
    43: ('FLOOR', 'HIT'),  # G2
    44: ('HH_PEDAL', 'HIT'),  # G#2 - Pedal hi-hat
    45: ('TOM_LOW', 'HIT'),  # Low tom
    46: ('HH_OPEN', 'HIT'),  # A#2 - Open hi-hat
    47: ('TOM_MID', 'HIT'),  # B2
    48: ('TOM_MID', 'HIT'),  # C3
    49: ('CRASH', 'HIT'),  # Crash cymbal 1
    50: ('TOM_HIGH', 'HIT'),  # High tom

    51: ('RIDE_CYMBAL', 'HIT'),  # D#3 - Ride cymbal
    52: ('CHINA', 'HIT'),  # E3 - Chinese cymbal
    53: ('RIDE_BELL', 'HIT'),  # Ride bell
    54: ('TAMBOURINE', 'HIT'),  # F#3 - Tambourine
    55: ('SPLASH', 'HIT'),  # Splash cymbal
    56: ('COWBELL', 'HIT'),  # G#3 - Cowbell
    57: ('CRASH', 'HIT'),  # Crash cymbal 2
    #58: ('VIBRASLAP', 'HIT'),   # Vibraslap
    59: ('RIDE_CYMBAL', 'HIT'),  # Ride cymbal 2    
}

# drum_type → LIMB トークン文字列のマッピング
# HH_OPEN/HH_CLOSED は通常時 LIMB_HAND だが、状態切り替え時は midi_to_tokens() 内で上書きする
_DRUM_TYPE_TO_LIMB_TOKEN: Dict[str, str] = {
    'SNARE':       'LIMB_HAND',
    'SNARE_RIM':   'LIMB_HAND',
    'FLOOR':       'LIMB_HAND',
    'TOM_LOW':     'LIMB_HAND',
    'TOM_MID':     'LIMB_HAND',
    'TOM_HIGH':    'LIMB_HAND',
    'HH_CLOSED':   'LIMB_HAND',
    'HH_OPEN':     'LIMB_HAND',
    'RIDE_CYMBAL': 'LIMB_HAND',
    'RIDE_BELL':   'LIMB_HAND',
    'CRASH':       'LIMB_HAND',
    'CHINA':       'LIMB_HAND',
    'SPLASH':      'LIMB_HAND',
    'TAMBOURINE':  'LIMB_HAND',
    'COWBELL':     'LIMB_HAND',
    'KICK':        'LIMB_RFOOT',
    'HH_PEDAL':    'LIMB_LFOOT',
}

# 各楽器の標準ピッチマッピング：複数のピッチをまとめる
# これにより、往復変換でピッチが保持される
STANDARD_PITCH_MAP = {
    'KICK': 36,          # 35 と 36 を 36 に統一
    'SNARE_RIM': 37,     # 標準ピッチ
    'SNARE': 38,         # 38 と 40 を 38 に統一
    'FLOOR': 43,         # 41 と 43 を 43 に統一
    'TOM_LOW': 45,       # 標準ピッチ
    'TOM_MID': 48,       # 47 と 48 を 48 に統一
    'TOM_HIGH': 50,      # 標準ピッチ
    'HH_CLOSED': 42,     # 標準ピッチ
    'HH_PEDAL': 44,      # 標準ピッチ
    'HH_OPEN': 46,       # 標準ピッチ
    'RIDE_CYMBAL': 51,   # 51 と 59 を 51 に統一
    'RIDE_BELL': 53,     # 標準ピッチ
    'CRASH': 49,         # 49 と 57 を 49 に統一
    'CHINA': 52,         # 標準ピッチ
    'SPLASH': 55,        # 標準ピッチ
    'TAMBOURINE': 54,    # 標準ピッチ
    'COWBELL': 56,       # 標準ピッチ
}


class DrumTokenizer:
    """ドラム譜トークナイザー"""

    def __init__(self, use_limb_tokens: bool = False):
        self.use_limb_tokens = use_limb_tokens
        self.vocab = self._build_vocabulary()
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.vocab)

    def _build_vocabulary(self) -> List[str]:
        """トークン語彙を構築"""
        vocab = []

        # 特殊トークン
        vocab.extend(['<PAD>', '<EOS>', '<BAR>'])

        # テンポトークン (60-240 BPM, 5刻み)
        for tempo in range(60, 245, 5):
            vocab.append(f'<TEMPO_{tempo}>')

        # 拍トークン (4拍子のみサポート)
        vocab.extend([f'<BEAT_{i}>' for i in range(1, 5)])

        # 位置トークン (1拍を24分割)
        vocab.extend([f'<POS_{i}>' for i in range(24)])

        # スネアドラム (SNARE: pitch 38, 40)
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'SNARE_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.extend([
                f'SNARE_RIMSHOT_{vel}',
                f'SNARE_XSTICK_{vel}',
                f'SNARE_ACCENT_{vel}',
                f'SNARE_FLAM_{vel}'
            ])
        vocab.append('SNARE_ROLL')

        # スネアリム (SNARE_RIM: pitch 37)
        for vel in ['Normal', 'Accent']:
            vocab.append(f'SNARE_RIM_ACCENT_{vel}')

        # キック (KICK: pitch 35, 36)
        for vel in ['Normal', 'Accent']:
            vocab.append(f'KICK_HIT_{vel}')

        # タム類 (TOM_LOW: pitch 45 / TOM_MID: pitch 47,48 / TOM_HIGH: pitch 50 / FLOOR: pitch 41,43)
        for drum in ['TOM_LOW', 'TOM_MID', 'TOM_HIGH', 'FLOOR']:
            for vel in ['Normal', 'Accent']:
                vocab.extend([
                    f'{drum}_HIT_{vel}',
                    f'{drum}_FLAM_{vel}'
                ])

        # ハイハット (HH_CLOSED: pitch 42 / HH_OPEN: pitch 46 / HH_PEDAL: pitch 44)
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'HH_CLOSED_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.extend([
                f'HH_OPEN_HIT_{vel}',
                f'HH_PEDAL_HIT_{vel}'
            ])

        # ライドシンバル (RIDE_CYMBAL: pitch 51, 59 / RIDE_BELL: pitch 53)
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'RIDE_CYMBAL_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.append(f'RIDE_BELL_HIT_{vel}')

        # クラッシュ・その他シンバル
        for cymbal in ['CRASH', 'SPLASH', 'CHINA', 'TAMBOURINE', 'COWBELL']:
            for vel in ['Normal', 'Accent']:
                vocab.append(f'{cymbal}_HIT_{vel}')
            vocab.append(f'{cymbal}_CHOKE')

        # 四肢トークン（use_limb_tokens=True の場合のみ追加）
        if self.use_limb_tokens:
            vocab.extend(['LIMB_HAND', 'LIMB_RFOOT', 'LIMB_LFOOT', 'LIMB_HAND_LFOOT'])

        return vocab

    def _velocity_to_level(self, velocity: int) -> str:
        """ベロシティをGhost/Normal/Accentに変換"""
        return classify_velocity_level(int(velocity))

    def _detect_flam(self, notes: List[miditoolkit.Note], window_ticks: int = 30) -> Dict[int, List[int]]:
        """
        フラムを検出 (短い間隔で連続する同楽器の打撃)
        
        フラムは通常、非常に短い前打音（10-30 ticks以内）を伴う打撃です。
        このメソッドは、同一楽器で指定された時間以内に発生した
        ノートペアをフラムの（前打音, 主音符）として認識します。

        Returns:
            Dict[主音符のインデックス, 前打音のインデックスのリスト]
        """
        flams = defaultdict(list)
        notes_sorted = sorted(notes, key=lambda x: (x.start, x.pitch))

        for i in range(len(notes_sorted) - 1):
            curr_note = notes_sorted[i]
            next_note = notes_sorted[i + 1]

            # 同じ楽器で極めて短い間隔の場合、フラムとみなす
            if (curr_note.pitch in DRUM_NOTE_MAP and
                next_note.pitch in DRUM_NOTE_MAP):
                curr_drum = DRUM_NOTE_MAP[curr_note.pitch][0]
                next_drum = DRUM_NOTE_MAP[next_note.pitch][0]

                if (curr_drum == next_drum and
                    0 < next_note.start - curr_note.start < window_ticks):
                    flams[i + 1].append(i)

        return flams

    def _detect_choke(self, notes: List[miditoolkit.Note]) -> Dict[int, int]:
        """
        チョークを検出 (Note Offが極端に短い)

        注: CRASH、SPLASH、CHINA、RIDE、TAMBOURINE、COWBELLなどのシンバル類は
        自然に短い音が多いため、自動チョーク検出は無効化
        これにより往復変換時のノート欠損を防ぐ

        Returns:
            Dict[ノートインデックス, チョーク位置のtick]
        """
        # チョーク検出を無効化して空の辞書を返す
        # これにより、全ノートが通常の演奏トークンとしてトークン化される
        return {}

    def _quantize_tempo(self, bpm: float) -> int:
        """
        テンポを5 BPM刻みに量子化

        Args:
            bpm: 元のBPM値

        Returns:
            量子化されたBPM値 (60-240の範囲、5刻み)
        """
        # 5刻みの上側ビンへ量子化
        # 例: 121-125 -> 125, 126-130 -> 130
        quantized = int(np.ceil(float(bpm) / 5.0)) * 5
        return max(60, min(240, quantized))

    def _get_tempo_at_tick(self, midi_obj: miditoolkit.MidiFile, tick: int) -> float:
        """
        指定されたtick位置でのテンポを取得

        Args:
            midi_obj: MIDIファイルオブジェクト
            tick: tick位置

        Returns:
            そのtick位置でのBPM値
        """
        if not midi_obj.tempo_changes:
            return 120.0  # デフォルトBPM

        # 指定されたtick以前の最後のテンポ変更を探す
        current_tempo = 120.0
        for tempo_change in midi_obj.tempo_changes:
            if tempo_change.time <= tick:
                current_tempo = tempo_change.tempo
            else:
                break

        return current_tempo

    def _infer_effective_ticks_per_beat(
        self,
        midi_obj: miditoolkit.MidiFile,
        notes: List[miditoolkit.Note]
    ) -> int:
        """
        ノート配置から実効的なticks_per_beatを推定する。

        一部のMIDIはヘッダ上のticks_per_beatと実データのグリッドが
        食い違うことがあり、そのまま正規化すると時間軸が1/4化する。
        24分割グリッドへの量子化誤差が最小になるTPBを候補から選び、
        ヘッダ値と比べて十分に改善する場合のみ採用する。
        """
        declared_tpb = int(midi_obj.ticks_per_beat) if midi_obj.ticks_per_beat > 0 else DEFAULT_BEAT_RESOL

        if not notes:
            return declared_tpb

        # 実運用で遭遇しやすいTPB候補（ヘッダ値を最優先候補として含める）
        candidates = [declared_tpb]
        for tpb in (120, 240, 480, 960, 1920):
            if tpb not in candidates:
                candidates.append(tpb)

        note_starts = [int(n.start) for n in notes]

        def quantization_error(tpb: int) -> float:
            if tpb <= 0:
                return float('inf')

            errors = []
            for start_tick in note_starts:
                tick_in_beat = start_tick % tpb
                pos_float = (tick_in_beat * POSITIONS_PER_BEAT) / tpb
                pos_nearest = round(pos_float)
                errors.append(abs(pos_float - pos_nearest))

            return float(np.mean(errors)) if errors else float('inf')

        declared_error = quantization_error(declared_tpb)
        best_tpb = declared_tpb
        best_error = declared_error

        for candidate in candidates:
            cand_error = quantization_error(candidate)
            if cand_error < best_error:
                best_error = cand_error
                best_tpb = candidate

        # 微小差分での過剰補正を防ぐため、十分な改善時のみ切り替える
        if best_tpb != declared_tpb and (declared_error - best_error) >= 0.05:
            return best_tpb

        return declared_tpb

    def midi_to_tokens(self, midi_path: str) -> Tuple[List[str], List[int]]:
        """
        MIDIファイルをトークン列に変換

        Returns:
            tokens: トークン列
            bar_positions: 各小節の開始トークン位置
        """
        midi_obj = miditoolkit.MidiFile(midi_path)

        # ドラムトラックを探す
        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            raise ValueError("No drum track found in MIDI file")

        # ===== 重要: ticks_per_beatの正規化 =====
        # ヘッダ値が実データと不整合なMIDIがあるため、実効TPBを推定してから正規化する
        effective_tpb = self._infer_effective_ticks_per_beat(midi_obj, drum_track.notes)
        scale_factor = DEFAULT_BEAT_RESOL / effective_tpb

        if scale_factor != 1.0:
            # すべてのノートの位置を正規化
            for note in drum_track.notes:
                note.start = int(note.start * scale_factor)
                note.end = int(note.end * scale_factor)

            # テンポ変更の位置も正規化
            for tempo_change in midi_obj.tempo_changes:
                tempo_change.time = int(tempo_change.time * scale_factor)
        # ==========================================

        # 小節数を計算
        max_tick = max(note.end for note in drum_track.notes) if drum_track.notes else 0
        n_bars = int(np.ceil(max_tick / DEFAULT_BAR_RESOL))

        # フラムとチョークの検出
        flams = self._detect_flam(drum_track.notes)
        chokes = self._detect_choke(drum_track.notes)
        processed_notes = set()  # フラムの前打音をスキップするため

        # フラムのgrace noteを事前にスキップリストに追加
        for main_idx, grace_indices in flams.items():
            for grace_idx in grace_indices:
                processed_notes.add(grace_idx)

        tokens = []
        bar_positions = []
        hh_state = 'CLOSED'  # ハイハット初期状態（ペダル踏み込み済み）

        for bar in range(n_bars):
            bar_positions.append(len(tokens))

            # 小節開始位置でのテンポを取得して量子化
            bar_start_tick = bar * DEFAULT_BAR_RESOL
            tempo_at_bar = self._get_tempo_at_tick(midi_obj, bar_start_tick)
            quantized_tempo = self._quantize_tempo(tempo_at_bar)

            # テンポトークンを追加
            tempo_token = f'<TEMPO_{quantized_tempo}>'
            tokens.append(tempo_token)

            tokens.append('<BAR>')

            bar_end_tick = (bar + 1) * DEFAULT_BAR_RESOL

            # この小節内のすべてのイベント（ノートとチョーク）を収集
            events = []

            for note_idx, note in enumerate(drum_track.notes):
                if note_idx in processed_notes:
                    continue

                if bar_start_tick <= note.start < bar_end_tick:
                    tick_in_bar = note.start - bar_start_tick
                    beat = tick_in_bar // DEFAULT_BEAT_RESOL
                    pos_in_beat = quantize_position_in_beat(tick_in_bar, DEFAULT_BEAT_RESOL)

                    if note.pitch not in DRUM_NOTE_MAP:
                        continue

                    drum_type, technique = DRUM_NOTE_MAP[note.pitch]
                    velocity_level = self._velocity_to_level(note.velocity)

                    # フラムの処理
                    is_flam = note_idx in flams
                    if is_flam:
                        technique = 'FLAM'
                        # 前打音をスキップリストに追加
                        for grace_idx in flams[note_idx]:
                            processed_notes.add(grace_idx)

                    # ベロシティレベルが対応しているかチェック
                    if velocity_level == 'Ghost' and 'Ghost' not in self.vocab:
                        velocity_level = 'Normal'

                    # トークン生成
                    token = f'{drum_type}_{technique}_{velocity_level}'

                    # トークンが語彙に存在するか確認
                    if token not in self.token2idx:
                        # フォールバック: 基本的なHIT_Normalに戻す
                        token = f'{drum_type}_HIT_Normal'
                        if token not in self.token2idx:
                            continue

                    events.append((beat, pos_in_beat, tick_in_bar, token, 'note', drum_type))

                # チョークイベントの追加
                if note_idx in chokes:
                    choke_tick = chokes[note_idx]
                    if bar_start_tick <= choke_tick < bar_end_tick:
                        tick_in_bar = choke_tick - bar_start_tick
                        beat = tick_in_bar // DEFAULT_BEAT_RESOL
                        pos_in_beat = quantize_position_in_beat(tick_in_bar, DEFAULT_BEAT_RESOL)

                        drum_type = DRUM_NOTE_MAP[note.pitch][0]
                        choke_token = f'{drum_type}_CHOKE'

                        if choke_token in self.token2idx:
                            events.append((beat, pos_in_beat, tick_in_bar, choke_token, 'choke', drum_type))

            # イベントをソート（拍 > 位置 > 楽器の優先順位）
            events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

            # トークン列に追加
            current_beat = -1
            current_pos = -1

            for beat, pos_in_beat, tick_in_bar, token, event_type, drum_type in events:
                # 拍トークン
                if beat != current_beat:
                    tokens.append(f'<BEAT_{beat + 1}>')
                    current_beat = beat
                    current_pos = -1

                # 位置トークン
                if pos_in_beat != current_pos:
                    tokens.append(f'<POS_{pos_in_beat}>')
                    current_pos = pos_in_beat

                # LIMB トークン（use_limb_tokens=True かつ note イベントのみ）
                if self.use_limb_tokens and event_type == 'note':
                    limb_token = _DRUM_TYPE_TO_LIMB_TOKEN.get(drum_type, 'LIMB_HAND')
                    if drum_type == 'HH_OPEN' and hh_state == 'CLOSED':
                        limb_token = 'LIMB_HAND_LFOOT'
                        hh_state = 'OPEN'
                    elif drum_type == 'HH_CLOSED' and hh_state == 'OPEN':
                        limb_token = 'LIMB_HAND_LFOOT'
                        hh_state = 'CLOSED'
                    elif drum_type == 'HH_OPEN':
                        hh_state = 'OPEN'
                    elif drum_type == 'HH_CLOSED':
                        hh_state = 'CLOSED'
                    tokens.append(limb_token)

                # 演奏トークン
                tokens.append(token)

        tokens.append('<EOS>')

        return tokens, bar_positions

    def tokens_to_indices(self, tokens: List[str]) -> np.ndarray:
        """トークン列をインデックス列に変換"""
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                # 未知のトークンはPADとして扱う
                indices.append(self.token2idx['<PAD>'])
        return np.array(indices, dtype=np.int32)

    def indices_to_tokens(self, indices: np.ndarray) -> List[str]:
        """インデックス列をトークン列に変換"""
        return [self.idx2token.get(idx, '<PAD>') for idx in indices]

    def save_vocab(self, vocab_path: str):
        """語彙を保存"""
        import pickle
        with open(vocab_path, 'wb') as f:
            pickle.dump(
                {
                    'tokenization_method': 'standard',
                    'schema_version': 1,
                    'use_limb_tokens': self.use_limb_tokens,
                    'token2idx': self.token2idx,
                    'idx2token': self.idx2token,
                },
                f,
            )

    def load_vocab(self, vocab_path: str):
        """語彙を読み込み"""
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab_payload = pickle.load(f)
            if isinstance(vocab_payload, dict):
                self.token2idx = vocab_payload['token2idx']
                self.idx2token = vocab_payload['idx2token']
            else:
                self.token2idx, self.idx2token = vocab_payload
            self.vocab = list(self.token2idx.keys())
            self.vocab_size = len(self.vocab)


if __name__ == '__main__':
    # テスト用
    tokenizer = DrumTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"\nFirst 50 tokens:")
    for i, token in enumerate(tokenizer.vocab[:50]):
        print(f"  {i:3d}: {token}")

    print(f"\nLast 20 tokens:")
    for i, token in enumerate(tokenizer.vocab[-20:], start=len(tokenizer.vocab)-20):
        print(f"  {i:3d}: {token}")
