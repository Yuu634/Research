"""
四肢ベースCPトークナイザー

CPイベント:
    [Type, Position, Hand1, Hand2, Right_Foot, Left_Foot]

Type:
    <NOTE> または構造トークン（<TEMPO_xxx>, <BAR>, <BEAT_x>, <EOS>）
"""
import pickle
from typing import Dict, List, Tuple

import miditoolkit
import numpy as np

from src.tokenizers.drum_tokenizer import (
    DEFAULT_BAR_RESOL,
    DEFAULT_BEAT_RESOL,
    DRUM_NOTE_MAP,
    POSITIONS_PER_BEAT,
    DrumTokenizer,
)
from src.utils.tokenization_policy import (
    CP_POS_PAD_VALUE,
    quantize_position_in_beat,
)


class DrumCPTokenizer:
    def __init__(self):
        self.base = DrumTokenizer()

        # 学習器側互換のため維持（cp_type から派生する補助系列）
        self.event_type2idx = {
            "STRUCT": 0,
            "CP": 1,
            "<PAD_EVENT>": 2,
        }
        self.idx2event_type = {v: k for k, v in self.event_type2idx.items()}

        self.type_vocab = self._build_type_vocab()
        self.type_token2idx = {tok: i for i, tok in enumerate(self.type_vocab)}
        self.idx2type_token = {i: tok for tok, i in self.type_token2idx.items()}

        # 既存コード互換エイリアス
        self.struct_vocab = self.type_vocab
        self.struct_token2idx = self.type_token2idx
        self.idx2struct_token = self.idx2type_token

        self.limb_vocab = self._build_limb_vocab()
        self.limb_token2idx = {tok: i for i, tok in enumerate(self.limb_vocab)}
        self.idx2limb_token = {i: tok for tok, i in self.limb_token2idx.items()}

        # 既存コード互換: prepare_drum_dataset.py が tokenizer.vocab_size を参照する
        self.struct_vocab_size = len(self.type_vocab)
        self.limb_vocab_size = len(self.limb_vocab)
        self.event_type_vocab_size = len(self.event_type2idx)
        self.vocab_size = self.struct_vocab_size

        # 0-23 を有効POS, 24 を <NA> (非NOTE用) として使用
        self.pos_pad_value = CP_POS_PAD_VALUE

    def _build_type_vocab(self) -> List[str]:
        vocab = ["<PAD>", "<NOTE>", "<EOS>", "<BAR>"]
        for tempo in range(60, 245, 5):
            vocab.append(f"<TEMPO_{tempo}>")
        vocab.extend([f"<BEAT_{i}>" for i in range(1, 5)])
        return vocab

    def _build_limb_vocab(self) -> List[str]:
        perf_tokens = [tok for tok in self.base.vocab if not tok.startswith("<")]
        extra_tokens = [
            "HH_PEDAL_HIT_Down",
            "HH_PEDAL_HIT_Up",
            "TAMBOURINE_HIT_Normal",
            "TAMBOURINE_HIT_Accent",
        ]
        for token in extra_tokens:
            if token not in perf_tokens:
                perf_tokens.append(token)
        return ["<PAD>", "<NA>", "<NONE>"] + sorted(perf_tokens)

    def _validate_cp_schema(
        self,
        cp_type_seq: List[int],
        cp_pos_seq: List[int],
        cp_hand1_seq: List[int],
        cp_hand2_seq: List[int],
        cp_rf_seq: List[int],
        cp_lf_seq: List[int],
        type_note: int,
        type_pad: int,
        limb_na: int,
        limb_pad: int,
    ) -> None:
        seq_lengths = {
            len(cp_type_seq),
            len(cp_pos_seq),
            len(cp_hand1_seq),
            len(cp_hand2_seq),
            len(cp_rf_seq),
            len(cp_lf_seq),
        }
        if len(seq_lengths) != 1:
            raise ValueError(f"CP schema violation: mismatched sequence lengths {sorted(seq_lengths)}")

        for idx, type_id in enumerate(cp_type_seq):
            if type_id == type_note:
                if cp_pos_seq[idx] == self.pos_pad_value:
                    raise ValueError(
                        "CP schema violation: <NOTE> event cannot use <NA> pos value. "
                        f"index={idx}"
                    )
                if cp_pos_seq[idx] < 0 or cp_pos_seq[idx] >= POSITIONS_PER_BEAT:
                    raise ValueError(
                        "CP schema violation: <NOTE> position out of range. "
                        f"index={idx}, pos={cp_pos_seq[idx]}"
                    )
                if (
                    cp_hand1_seq[idx] == limb_na
                    or cp_hand2_seq[idx] == limb_na
                    or cp_rf_seq[idx] == limb_na
                    or cp_lf_seq[idx] == limb_na
                ):
                    raise ValueError(
                        "CP schema violation: <NOTE> event cannot use <NA> in limb slots. "
                        f"index={idx}"
                    )
            elif type_id == type_pad:
                if cp_pos_seq[idx] != self.pos_pad_value:
                    raise ValueError(
                        "CP schema violation: <PAD> event must use <NA> pos value. "
                        f"index={idx}, pos={cp_pos_seq[idx]}"
                    )
                if not (
                    cp_hand1_seq[idx] == limb_pad
                    and cp_hand2_seq[idx] == limb_pad
                    and cp_rf_seq[idx] == limb_pad
                    and cp_lf_seq[idx] == limb_pad
                ):
                    raise ValueError(
                        "CP schema violation: <PAD> event must pad limb slots. "
                        f"index={idx}"
                    )
            else:
                if cp_pos_seq[idx] != self.pos_pad_value:
                    raise ValueError(
                        "CP schema violation: structural type must use <NA> pos value. "
                        f"index={idx}, pos={cp_pos_seq[idx]}"
                    )
                if not (
                    cp_hand1_seq[idx] == limb_na
                    and cp_hand2_seq[idx] == limb_na
                    and cp_rf_seq[idx] == limb_na
                    and cp_lf_seq[idx] == limb_na
                ):
                    raise ValueError(
                        "CP schema violation: structural type must use <NA> in limb slots. "
                        f"index={idx}"
                    )

    def _is_hand_note(self, drum_type: str) -> bool:
        return drum_type not in ("KICK", "HH_PEDAL")

    def _assign_feet(self, drum_type: str, token: str, rf: str, lf: str) -> Tuple[str, str]:
        if drum_type == "HH_PEDAL":
            if lf == "<NONE>":
                return rf, token
            return rf, lf

        if drum_type == "KICK":
            if rf == "<NONE>":
                return token, lf
            if lf == "<NONE>":
                return rf, token
            return rf, lf

        return rf, lf

    def _hand_priority(self, token: str) -> Tuple[int, int, str]:
        # 安定ソートのため、楽器優先とベロシティ優先を組み合わせる
        if token.startswith("SNARE"):
            inst_rank = 0
        elif token.startswith("HH"):
            inst_rank = 1
        elif token.startswith("RIDE"):
            inst_rank = 2
        elif token.startswith("CRASH"):
            inst_rank = 3
        elif token.startswith("TOM") or token.startswith("FLOOR"):
            inst_rank = 4
        else:
            inst_rank = 5

        if token.endswith("_Accent"):
            vel_rank = 0
        elif token.endswith("_Normal"):
            vel_rank = 1
        else:
            vel_rank = 2

        return (inst_rank, vel_rank, token)

    def _parse_note_to_token(self, note: miditoolkit.Note) -> Tuple[str, str]:
        drum_type, technique = DRUM_NOTE_MAP[note.pitch]
        velocity_level = self.base._velocity_to_level(note.velocity)
        token = f"{drum_type}_{technique}_{velocity_level}"

        if token not in self.base.token2idx:
            if drum_type == "TAMBOURINE":
                fallback = "TAMBOURINE_HIT_Normal"
                if fallback in self.base.token2idx:
                    return drum_type, fallback
            fallback = f"{drum_type}_HIT_Normal"
            if fallback in self.base.token2idx:
                token = fallback
            else:
                return drum_type, ""
        return drum_type, token

    def _quantize_tempo(self, bpm: float) -> int:
        return self.base._quantize_tempo(bpm)

    def _get_tempo_at_tick(self, midi_obj: miditoolkit.MidiFile, tick: int) -> float:
        return self.base._get_tempo_at_tick(midi_obj, tick)

    def _infer_effective_ticks_per_beat(
        self, midi_obj: miditoolkit.MidiFile, notes: List[miditoolkit.Note]
    ) -> int:
        return self.base._infer_effective_ticks_per_beat(midi_obj, notes)

    def save_vocab(self, vocab_path: str):
        payload = {
            "format": "cp_limb_v1",
            "tokenization_method": "cp_limb_v1",
            "schema_version": 1,
            "type_token2idx": self.type_token2idx,
            "idx2type_token": self.idx2type_token,
            # backward compatibility aliases
            "event_type2idx": self.event_type2idx,
            "idx2event_type": self.idx2event_type,
            "struct_token2idx": self.struct_token2idx,
            "idx2struct_token": self.idx2struct_token,
            "limb_token2idx": self.limb_token2idx,
            "idx2limb_token": self.idx2limb_token,
            "pos_pad_value": self.pos_pad_value,
            "positions_per_beat": POSITIONS_PER_BEAT,
        }
        with open(vocab_path, "wb") as f:
            pickle.dump(payload, f)

    def midi_to_cp_data(self, midi_path: str) -> Tuple[Dict[str, List[int]], List[int]]:
        midi_obj = miditoolkit.MidiFile(midi_path)

        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            raise ValueError("No drum track found in MIDI file")

        effective_tpb = self._infer_effective_ticks_per_beat(midi_obj, drum_track.notes)
        scale_factor = DEFAULT_BEAT_RESOL / effective_tpb
        if scale_factor != 1.0:
            for note in drum_track.notes:
                note.start = int(note.start * scale_factor)
                note.end = int(note.end * scale_factor)
            for tempo_change in midi_obj.tempo_changes:
                tempo_change.time = int(tempo_change.time * scale_factor)

        max_tick = max(note.end for note in drum_track.notes) if drum_track.notes else 0
        n_bars = int(np.ceil(max_tick / DEFAULT_BAR_RESOL))

        cp_type_seq = []
        cp_pos_seq = []
        cp_hand1_seq = []
        cp_hand2_seq = []
        cp_rf_seq = []
        cp_lf_seq = []
        bar_positions = []

        type_note = self.type_token2idx["<NOTE>"]
        type_pad = self.type_token2idx["<PAD>"]
        limb_pad = self.limb_token2idx["<PAD>"]
        limb_na = self.limb_token2idx["<NA>"]

        def append_struct(type_token: str):
            cp_type_seq.append(self.type_token2idx[type_token])
            cp_pos_seq.append(self.pos_pad_value)
            cp_hand1_seq.append(limb_na)
            cp_hand2_seq.append(limb_na)
            cp_rf_seq.append(limb_na)
            cp_lf_seq.append(limb_na)

        def append_cp(pos: int, hand1: str, hand2: str, rf: str, lf: str):
            cp_type_seq.append(type_note)
            cp_pos_seq.append(pos)
            cp_hand1_seq.append(self.limb_token2idx[hand1])
            cp_hand2_seq.append(self.limb_token2idx[hand2])
            cp_rf_seq.append(self.limb_token2idx[rf])
            cp_lf_seq.append(self.limb_token2idx[lf])

        hh_left_foot_pressed = False

        for bar in range(n_bars):
            bar_positions.append(len(cp_type_seq))

            bar_start_tick = bar * DEFAULT_BAR_RESOL
            bar_end_tick = (bar + 1) * DEFAULT_BAR_RESOL

            quantized_tempo = self._quantize_tempo(self._get_tempo_at_tick(midi_obj, bar_start_tick))
            append_struct(f"<TEMPO_{quantized_tempo}>")
            append_struct("<BAR>")

            tick_events: Dict[int, List[Tuple[str, str]]] = {}
            for note in drum_track.notes:
                if not (bar_start_tick <= note.start < bar_end_tick):
                    continue
                if note.pitch not in DRUM_NOTE_MAP:
                    continue

                drum_type, token = self._parse_note_to_token(note)
                if not token:
                    continue

                tick_in_bar = note.start - bar_start_tick
                tick_events.setdefault(tick_in_bar, []).append((drum_type, token))

            current_beat = -1
            for tick_in_bar in sorted(tick_events.keys()):
                beat = tick_in_bar // DEFAULT_BEAT_RESOL
                pos_in_beat = quantize_position_in_beat(tick_in_bar, DEFAULT_BEAT_RESOL)

                if beat != current_beat:
                    append_struct(f"<BEAT_{beat + 1}>")
                    current_beat = beat

                hand_candidates = []
                rf = "<NONE>"
                lf = "<NONE>"
                saw_hh_open = False
                saw_hh_closed = False

                for drum_type, token in tick_events[tick_in_bar]:
                    if self._is_hand_note(drum_type):
                        hand_candidates.append(token)
                        if token.startswith("HH_OPEN_"):
                            saw_hh_open = True
                        elif token.startswith("HH_CLOSED_"):
                            saw_hh_closed = True
                    else:
                        rf, lf = self._assign_feet(drum_type, token, rf, lf)
                        if drum_type == "HH_PEDAL":
                            hh_left_foot_pressed = True

                # HH_OPEN/HH_CLOSED から左足の開閉状態を推論し、遷移時のみトークンを付与する
                transition_token = None
                if saw_hh_open and hh_left_foot_pressed:
                    transition_token = "HH_PEDAL_HIT_Up"
                    hh_left_foot_pressed = False
                elif saw_hh_closed and not hh_left_foot_pressed:
                    transition_token = "HH_PEDAL_HIT_Down"
                    hh_left_foot_pressed = True

                if lf == "<NONE>" and transition_token:
                    lf = transition_token

                sorted_hands = sorted(hand_candidates, key=self._hand_priority)
                if sorted_hands:
                    for group_index in range(0, len(sorted_hands), 2):
                        group = sorted_hands[group_index:group_index + 2]
                        hand1 = group[0] if len(group) > 0 else "<NONE>"
                        hand2 = group[1] if len(group) > 1 else "<NONE>"
                        if group_index == 0:
                            append_cp(pos_in_beat, hand1, hand2, rf, lf)
                        else:
                            append_cp(pos_in_beat, hand1, hand2, "<NONE>", "<NONE>")
                else:
                    append_cp(pos_in_beat, "<NONE>", "<NONE>", rf, lf)

        append_struct("<EOS>")

        self._validate_cp_schema(
            cp_type_seq,
            cp_pos_seq,
            cp_hand1_seq,
            cp_hand2_seq,
            cp_rf_seq,
            cp_lf_seq,
            type_note,
            type_pad,
            limb_na,
            limb_pad,
        )

        data = {
            "cp_type": cp_type_seq,
            "cp_pos": cp_pos_seq,
            "cp_hand1": cp_hand1_seq,
            "cp_hand2": cp_hand2_seq,
            "cp_right_foot": cp_rf_seq,
            "cp_left_foot": cp_lf_seq,
        }
        return data, bar_positions
