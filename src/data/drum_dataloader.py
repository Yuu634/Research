"""
ドラム譜用データローダー
"""
import os
import pickle
import random
from glob import glob
from typing import List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils.tokenization_policy import POSITIONS_PER_BEAT

DIFFICULTY_KEYS = ('s_tech', 's_indep', 's_hand', 's_foot', 's_move')


def pickle_load(path):
    """Pickleファイルを読み込み"""
    return pickle.load(open(path, 'rb'))


class DrumTransformerDataset(Dataset):
    """ドラム譜用のTransformerデータセット"""

    def __init__(
        self,
        data_dir: str,
        vocab_path: str,
        model_enc_seqlen: int = 128,
        model_dec_seqlen: int = 1280,
        model_max_bars: int = 16,
        pieces: Optional[List[str]] = None,
        pad_to_same: bool = True,
        appoint_st_bar: Optional[int] = None,
        dec_end_pad_value: Optional[str] = None,
        use_difficulty: bool = False,  # ===== 追加: 難易度クラスを使用するか =====
        tokenization_method: str = 'standard'
    ):
        """
        Args:
            data_dir: データディレクトリ
            vocab_path: 語彙ファイルパス
            model_enc_seqlen: エンコーダーの最大系列長
            model_dec_seqlen: デコーダーの最大系列長
            model_max_bars: 最大小節数
            pieces: 使用するデータファイルのリスト（Noneの場合は全て）
            pad_to_same: 系列を同じ長さにパディングするか
            appoint_st_bar: 開始小節を指定（Noneの場合はランダム）
            dec_end_pad_value: デコーダー末尾のパディング値（'EOS' or None）
        """
        self.vocab_path = vocab_path
        self.tokenization_method = tokenization_method
        self.read_vocab()

        self.data_dir = data_dir
        self.pieces = pieces
        self.build_dataset()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars

        self.pad_to_same = pad_to_same
        self.appoint_st_bar = appoint_st_bar
        self.use_difficulty = use_difficulty  # ===== 追加 =====

        if dec_end_pad_value is None:
            self.dec_end_pad_value = self.pad_token
        elif dec_end_pad_value == 'EOS':
            self.dec_end_pad_value = self.eos_token
        else:
            self.dec_end_pad_value = self.pad_token

    def read_vocab(self):
        """語彙を読み込み"""
        vocab_payload = pickle_load(self.vocab_path)

        if isinstance(vocab_payload, dict) and vocab_payload.get('tokenization_method') == 'cp_limb_v1':
            self.tokenization_method = 'cp_limb_v1'

            self.event_type2idx = vocab_payload.get(
                'event_type2idx',
                {'STRUCT': 0, 'CP': 1, '<PAD_EVENT>': 2}
            )
            self.idx2event_type = vocab_payload.get(
                'idx2event_type',
                {v: k for k, v in self.event_type2idx.items()}
            )

            self.type_token2idx = vocab_payload.get('type_token2idx')
            self.idx2type_token = vocab_payload.get('idx2type_token')

            if self.type_token2idx is None:
                self.type_token2idx = vocab_payload['struct_token2idx']
                self.idx2type_token = vocab_payload['idx2struct_token']

            if '<NOTE>' not in self.type_token2idx:
                self.type_token2idx = dict(self.type_token2idx)
                note_id = max(self.type_token2idx.values()) + 1
                self.type_token2idx['<NOTE>'] = note_id
                self.idx2type_token = dict(self.idx2type_token)
                self.idx2type_token[note_id] = '<NOTE>'

            # backward compatibility aliases
            self.struct_token2idx = self.type_token2idx
            self.idx2struct_token = self.idx2type_token

            self.limb_token2idx = vocab_payload['limb_token2idx']
            self.idx2limb_token = vocab_payload['idx2limb_token']

            self.pos_pad_value = vocab_payload.get('pos_pad_value', 24)
            self.positions_per_beat = int(vocab_payload.get('positions_per_beat', POSITIONS_PER_BEAT))

            self.cp_struct_event_idx = self.event_type2idx['STRUCT']
            self.cp_note_event_idx = self.event_type2idx['CP']
            self.cp_event_pad_idx = self.event_type2idx['<PAD_EVENT>']

            self.type_pad_idx = self.type_token2idx['<PAD>']
            self.type_note_idx = self.type_token2idx['<NOTE>']

            self.bar_token = self.type_token2idx['<BAR>']
            self.eos_token = self.type_token2idx['<EOS>']
            self.pad_token = self.type_pad_idx

            # 既存呼び出しとの互換のため、type語彙サイズをvocab_sizeとして公開
            self.vocab_size = len(self.type_token2idx)
        else:
            if isinstance(vocab_payload, dict):
                self.tokenization_method = vocab_payload.get(
                    'tokenization_method',
                    self.tokenization_method,
                )
                token2idx = vocab_payload['token2idx']
                idx2token = vocab_payload['idx2token']
            else:
                token2idx, idx2token = vocab_payload
            self.token2idx = token2idx
            self.idx2token = idx2token

            # 特殊トークン
            self.bar_token = self.token2idx['<BAR>']
            self.eos_token = self.token2idx['<EOS>']
            self.pad_token = len(self.token2idx)  # PADトークンは語彙の末尾に追加
            self.vocab_size = self.pad_token + 1
            self.positions_per_beat = POSITIONS_PER_BEAT

    def _piece_len(self, piece_evs):
        if isinstance(piece_evs, dict):
            if 'cp_type' in piece_evs:
                return len(piece_evs['cp_type'])
            if 'event_type' in piece_evs:
                return len(piece_evs['event_type'])
            first_key = next(iter(piece_evs.keys()))
            return len(piece_evs[first_key])
        return len(piece_evs)

    def _legacy_cp_to_type_seq(self, cp_tokens):
        cp_type_seq = []
        for ev_type, struct_id in zip(cp_tokens['event_type'], cp_tokens['struct_token']):
            ev_type = int(ev_type)
            if ev_type == self.cp_note_event_idx:
                cp_type_seq.append(self.type_note_idx)
            elif ev_type == self.cp_event_pad_idx:
                cp_type_seq.append(self.type_pad_idx)
            else:
                cp_type_seq.append(int(struct_id))
        return cp_type_seq

    def _cp_type_to_aux_sequences(self, cp_type_seq):
        event_seq = []
        struct_seq = []

        for cp_type in cp_type_seq:
            cp_type = int(cp_type)
            if cp_type == self.type_pad_idx:
                event_seq.append(self.cp_event_pad_idx)
                struct_seq.append(self.type_pad_idx)
            elif cp_type == self.type_note_idx:
                event_seq.append(self.cp_note_event_idx)
                struct_seq.append(self.type_pad_idx)
            else:
                event_seq.append(self.cp_struct_event_idx)
                struct_seq.append(cp_type)

        return event_seq, struct_seq

    def _slice_piece_evs(self, piece_evs, st, ed):
        if isinstance(piece_evs, dict):
            return {k: v[st:ed] for k, v in piece_evs.items()}
        return piece_evs[st:ed]

    def build_dataset(self):
        """データセットを構築"""
        if not self.pieces:
            self.pieces = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        else:
            self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])

        # 統計ファイルなどを除外
        self.pieces = [p for p in self.pieces if
                       not os.path.basename(p).startswith('dataset_stats') and
                       not os.path.basename(p).startswith('train_split') and
                       not os.path.basename(p).startswith('val_split') and
                       not os.path.basename(p).startswith('failed_files')]

        self.piece_bar_pos = []

        for i, p in enumerate(self.pieces):
            if not i % 200:
                print(f'[preparing data] now at #{i}')

            try:
                data = pickle_load(p)

                # データ形式を確認（2要素または3要素タプル）
                if len(data) == 2:
                    bar_pos, p_evs = data
                elif len(data) == 3:
                    bar_pos, p_evs, _ = data
                else:
                    print(f'Error loading {p}: Unknown data format')
                    self.piece_bar_pos.append([0])
                    continue

                # 最後の小節位置を調整
                p_len = self._piece_len(p_evs)

                if bar_pos[-1] == p_len:
                    bar_pos = bar_pos[:-1]

                if p_len - bar_pos[-1] <= 2:
                    # 空の末尾小節を除く
                    bar_pos = bar_pos[:-1]

                bar_pos.append(p_len)
                self.piece_bar_pos.append(bar_pos)

            except Exception as e:
                print(f'Error loading {p}: {e}')
                self.piece_bar_pos.append([0])

    def get_sample_from_file(self, piece_idx: int):
        """ファイルからサンプルを取得"""
        # ===== 変更: difficulty_classes も読み込む =====
        data = pickle_load(self.pieces[piece_idx])

        if len(data) == 3:
            # 新形式: (bar_pos, tokens, difficulty_classes)
            _, piece_evs, difficulty_classes = data
        else:
            # 旧形式: (bar_pos, tokens)
            _, piece_evs = data
            difficulty_classes = None
        # =============================================

        # 開始小節を選択
        if len(self.piece_bar_pos[piece_idx]) > self.model_max_bars and self.appoint_st_bar is None:
            picked_st_bar = random.choice(
                range(len(self.piece_bar_pos[piece_idx]) - self.model_max_bars)
            )
        elif self.appoint_st_bar is not None and \
             self.appoint_st_bar < len(self.piece_bar_pos[piece_idx]) - self.model_max_bars:
            picked_st_bar = self.appoint_st_bar
        else:
            picked_st_bar = 0

        piece_bar_pos = self.piece_bar_pos[piece_idx]

        # 小節数に応じてトリミングまたはパディング
        if len(piece_bar_pos) > self.model_max_bars:
            piece_evs = self._slice_piece_evs(
                piece_evs,
                piece_bar_pos[picked_st_bar],
                piece_bar_pos[picked_st_bar + self.model_max_bars]
            )
            picked_bar_pos = np.array(
                piece_bar_pos[picked_st_bar: picked_st_bar + self.model_max_bars]
            ) - piece_bar_pos[picked_st_bar]
            n_bars = self.model_max_bars
            selected_bar_count = self.model_max_bars
        else:
            picked_bar_pos = np.array(
                piece_bar_pos + [piece_bar_pos[-1]] * (self.model_max_bars - len(piece_bar_pos))
            )
            n_bars = len(piece_bar_pos)
            selected_bar_count = max(0, len(piece_bar_pos) - 1)
            assert len(picked_bar_pos) == self.model_max_bars

        # ===== 追加: 難易度クラスの検証・整列 =====
        if self.use_difficulty:
            if difficulty_classes is None:
                raise ValueError(
                    f'Difficulty labels are required when use_difficulty=True: piece={self.pieces[piece_idx]}'
                )

            difficulty_classes = list(difficulty_classes)
            if picked_st_bar > 0 or len(piece_bar_pos) > self.model_max_bars:
                difficulty_classes = difficulty_classes[picked_st_bar: picked_st_bar + selected_bar_count]

            if len(difficulty_classes) != selected_bar_count:
                raise ValueError(
                    'Difficulty/bar count mismatch: '
                    f'piece={self.pieces[piece_idx]}, expected_bars={selected_bar_count}, '
                    f'difficulty_labels={len(difficulty_classes)}'
                )

            normalized_difficulty = []
            for bar_idx, diff in enumerate(difficulty_classes):
                if not isinstance(diff, dict):
                    raise ValueError(
                        'Difficulty label format mismatch: '
                        f'piece={self.pieces[piece_idx]}, bar={bar_idx}, type={type(diff).__name__}'
                    )

                missing_keys = [k for k in DIFFICULTY_KEYS if k not in diff]
                if missing_keys:
                    raise ValueError(
                        'Difficulty label keys mismatch: '
                        f'piece={self.pieces[piece_idx]}, bar={bar_idx}, missing={missing_keys}'
                    )

                normalized_difficulty.append({k: int(diff[k]) for k in DIFFICULTY_KEYS})

            difficulty_classes = normalized_difficulty
        else:
            difficulty_classes = None
        # ==========================================

        return piece_evs, picked_st_bar, picked_bar_pos, n_bars, difficulty_classes

    def pad_sequence(self, seq: List[int], maxlen: int, pad_value: Optional[int] = None) -> List[int]:
        """系列をパディング"""
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend([pad_value for _ in range(maxlen - len(seq))])
        return seq

    def get_encoder_input_data(self, bar_positions: np.ndarray, bar_tokens: List[int]):
        """エンコーダー入力データを生成"""
        assert len(bar_positions) == self.model_max_bars + 1

        enc_padding_mask = np.ones((self.model_max_bars, self.model_enc_seqlen), dtype=bool)
        enc_padding_mask[:, :2] = False
        padded_enc_input = np.full(
            (self.model_max_bars, self.model_enc_seqlen),
            dtype=int,
            fill_value=self.pad_token
        )
        enc_lens = np.zeros((self.model_max_bars,))

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            enc_padding_mask[b, :(ed - st)] = False
            enc_lens[b] = ed - st
            within_bar_tokens = self.pad_sequence(
                bar_tokens[st:ed].copy() if isinstance(bar_tokens, np.ndarray) else bar_tokens[st:ed],
                self.model_enc_seqlen,
                self.pad_token
            )
            within_bar_tokens = np.array(within_bar_tokens)

            padded_enc_input[b, :] = within_bar_tokens[:self.model_enc_seqlen]

        return padded_enc_input, enc_padding_mask, enc_lens

    def get_encoder_hier_input_data(self, bar_positions: np.ndarray, hier_seq: np.ndarray, pad_value: int = 0):
        """階層インデックス用のエンコーダー入力データを生成"""
        assert len(bar_positions) == self.model_max_bars + 1

        padded_enc_hier = np.full(
            (self.model_max_bars, self.model_enc_seqlen),
            dtype=np.int32,
            fill_value=pad_value
        )

        seq_len = len(hier_seq)
        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            st = max(0, min(int(st), seq_len))
            ed = max(st, min(int(ed), seq_len))
            if st >= ed:
                continue

            within_bar_vals = list(hier_seq[st:ed])
            if len(within_bar_vals) < self.model_enc_seqlen:
                within_bar_vals.extend([pad_value] * (self.model_enc_seqlen - len(within_bar_vals)))
            padded_enc_hier[b, :] = np.array(within_bar_vals[:self.model_enc_seqlen], dtype=np.int32)

        return padded_enc_hier

    def _parse_beat_token(self, token: str):
        if not isinstance(token, str):
            return None
        if token.startswith('<BEAT_') and token.endswith('>'):
            try:
                beat_idx = int(token[len('<BEAT_'):-1])
                return max(1, min(4, beat_idx))
            except ValueError:
                return None
        return None

    def _parse_pos_token(self, token: str):
        if not isinstance(token, str):
            return None
        if token.startswith('<POS_') and token.endswith('>'):
            try:
                pos_idx = int(token[len('<POS_'):-1])
                return max(0, min(self.positions_per_beat - 1, pos_idx))
            except ValueError:
                return None
        return None

    def _build_bar_index_seq(self, seq_len: int, bar_positions: List[int]):
        bar_idx_seq = np.zeros((seq_len,), dtype=np.int32)
        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            st = max(0, min(int(st), seq_len))
            ed = max(st, min(int(ed), seq_len))
            if st < ed:
                bar_idx_seq[st:ed] = b
        return bar_idx_seq

    def _build_hier_indices_standard(self, token_seq: List[int], bar_positions: List[int]):
        seq_len = len(token_seq)
        bar_idx_seq = self._build_bar_index_seq(seq_len, bar_positions)
        beat_idx_seq = np.ones((seq_len,), dtype=np.int32)
        pos_idx_seq = np.zeros((seq_len,), dtype=np.int32)
        is_struct_seq = np.zeros((seq_len,), dtype=np.int32)

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            st = max(0, min(int(st), seq_len))
            ed = max(st, min(int(ed), seq_len))
            if st >= ed:
                continue

            current_beat = 1
            current_pos = 0
            for i in range(st, ed):
                token = self.idx2token.get(int(token_seq[i]), '<PAD>')

                if token == '<BAR>':
                    current_beat = 1
                    current_pos = 0

                beat_token = self._parse_beat_token(token)
                if beat_token is not None:
                    current_beat = beat_token
                    current_pos = 0

                pos_token = self._parse_pos_token(token)
                if pos_token is not None:
                    current_pos = pos_token

                beat_idx_seq[i] = current_beat
                pos_idx_seq[i] = current_pos
                is_struct_seq[i] = int(isinstance(token, str) and token.startswith('<') and token != '<PAD>')

        return bar_idx_seq, beat_idx_seq, pos_idx_seq, is_struct_seq

    def _build_hier_indices_cp(self, cp_type_seq: List[int], cp_pos_seq: List[int], bar_positions: List[int]):
        seq_len = len(cp_type_seq)
        bar_idx_seq = self._build_bar_index_seq(seq_len, bar_positions)
        beat_idx_seq = np.ones((seq_len,), dtype=np.int32)
        pos_idx_seq = np.zeros((seq_len,), dtype=np.int32)
        is_struct_seq = np.zeros((seq_len,), dtype=np.int32)

        ticks_per_beat = max(1, self.positions_per_beat // 4)

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            st = max(0, min(int(st), seq_len))
            ed = max(st, min(int(ed), seq_len))
            if st >= ed:
                continue

            current_beat = 1
            current_pos = 0
            for i in range(st, ed):
                cp_type = int(cp_type_seq[i])
                cp_pos = int(cp_pos_seq[i]) if i < len(cp_pos_seq) else self.pos_pad_value
                type_token = self.idx2type_token.get(cp_type, '<PAD>')

                if type_token == '<BAR>':
                    current_beat = 1
                    current_pos = 0

                beat_token = self._parse_beat_token(type_token)
                if beat_token is not None:
                    current_beat = beat_token
                    current_pos = 0

                pos_token = self._parse_pos_token(type_token)
                if pos_token is not None:
                    current_pos = pos_token

                if cp_type == self.type_note_idx and 0 <= cp_pos < self.positions_per_beat:
                    current_pos = cp_pos
                    current_beat = max(1, min(4, (cp_pos // ticks_per_beat) + 1))

                beat_idx_seq[i] = current_beat
                pos_idx_seq[i] = current_pos
                is_struct_seq[i] = int(cp_type != self.type_note_idx and type_token != '<PAD>')

        return bar_idx_seq, beat_idx_seq, pos_idx_seq, is_struct_seq

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        """データセットから1サンプルを取得"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ===== 変更: difficulty_classes も取得 =====
        bar_tokens, st_bar, bar_pos, enc_n_bars, difficulty_classes = self.get_sample_from_file(idx)
        # =========================================

        if self.tokenization_method == 'cp_limb_v1':
            cp_type_seq = bar_tokens.get('cp_type')
            if cp_type_seq is None:
                cp_type_seq = self._legacy_cp_to_type_seq(bar_tokens)

            seq_len = len(cp_type_seq)
            bar_pos = bar_pos.tolist() + [seq_len]

            enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(
                np.array(bar_pos), cp_type_seq
            )

            enc_bar_idx_seq, enc_beat_idx_seq, enc_pos_idx_seq, enc_is_struct_seq = self._build_hier_indices_cp(
                cp_type_seq,
                bar_tokens['cp_pos'],
                bar_pos,
            )
            enc_bar_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_bar_idx_seq, pad_value=0)
            enc_beat_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_beat_idx_seq, pad_value=0)
            enc_pos_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_pos_idx_seq, pad_value=0)
            enc_is_struct = self.get_encoder_hier_input_data(np.array(bar_pos), enc_is_struct_seq, pad_value=0)

            length = seq_len

            def pad_and_shift(seq, pad_value):
                seq = list(seq)
                if self.pad_to_same:
                    inp_full = self.pad_sequence(seq, self.model_dec_seqlen + 1, pad_value=pad_value)
                else:
                    inp_full = self.pad_sequence(seq, len(seq) + 1, pad_value=pad_value)
                tgt = np.array(inp_full[1:], dtype=int)
                inp = np.array(inp_full[:-1], dtype=int)
                return inp[:self.model_dec_seqlen], tgt[:self.model_dec_seqlen]

            type_inp, type_tgt = pad_and_shift(
                cp_type_seq, self.type_pad_idx
            )
            event_seq, struct_seq = self._cp_type_to_aux_sequences(cp_type_seq)
            event_inp, event_tgt = pad_and_shift(
                event_seq, self.cp_event_pad_idx
            )
            struct_inp, struct_tgt = pad_and_shift(
                struct_seq, self.type_pad_idx
            )

            pos_inp, pos_tgt = pad_and_shift(
                bar_tokens['cp_pos'], self.pos_pad_value
            )
            h1_inp, h1_tgt = pad_and_shift(
                bar_tokens['cp_hand1'], self.limb_token2idx['<PAD>']
            )
            h2_inp, h2_tgt = pad_and_shift(
                bar_tokens['cp_hand2'], self.limb_token2idx['<PAD>']
            )
            rf_inp, rf_tgt = pad_and_shift(
                bar_tokens['cp_right_foot'], self.limb_token2idx['<PAD>']
            )
            lf_inp, lf_tgt = pad_and_shift(
                bar_tokens['cp_left_foot'], self.limb_token2idx['<PAD>']
            )

            cp_event_id = self.cp_note_event_idx
            cp_mask = (event_inp == cp_event_id).astype(np.int32)

            dec_bar_idx, _ = pad_and_shift(enc_bar_idx_seq, 0)
            dec_beat_idx, _ = pad_and_shift(enc_beat_idx_seq, 0)
            dec_pos_idx, _ = pad_and_shift(enc_pos_idx_seq, 0)
            dec_is_struct, _ = pad_and_shift(enc_is_struct_seq, 0)

            result = {
                'id': idx,
                'piece_id': int(os.path.basename(self.pieces[idx]).replace('.pkl', '')),
                'st_bar_id': st_bar,
                'bar_pos': np.array(bar_pos, dtype=int),
                'enc_input': enc_inp,
                'enc_bar_idx': enc_bar_idx,
                'enc_beat_idx': enc_beat_idx,
                'enc_pos_idx': enc_pos_idx,
                'enc_is_struct': enc_is_struct,
                # 互換キー: 既存コードが dec_input/target を参照するため構造系列を入れる
                'dec_input': struct_inp,
                'dec_target': struct_tgt,
                'dec_bar_idx': dec_bar_idx,
                'dec_beat_idx': dec_beat_idx,
                'dec_pos_idx': dec_pos_idx,
                'dec_is_struct': dec_is_struct,
                'length': min(length, self.model_dec_seqlen),
                'enc_padding_mask': enc_padding_mask,
                'enc_length': enc_lens,
                'enc_n_bars': enc_n_bars,
                # CP専用系列
                'cp_type_input': type_inp,
                'cp_type_target': type_tgt,
                'cp_event_type_input': event_inp,
                'cp_event_type_target': event_tgt,
                'cp_struct_input': struct_inp,
                'cp_struct_target': struct_tgt,
                'cp_pos_input': pos_inp,
                'cp_pos_target': pos_tgt,
                'cp_hand1_input': h1_inp,
                'cp_hand1_target': h1_tgt,
                'cp_hand2_input': h2_inp,
                'cp_hand2_target': h2_tgt,
                'cp_right_foot_input': rf_inp,
                'cp_right_foot_target': rf_tgt,
                'cp_left_foot_input': lf_inp,
                'cp_left_foot_target': lf_tgt,
                'cp_mask': cp_mask,
            }
        else:
            # トークンのリストに変換
            if isinstance(bar_tokens, np.ndarray):
                bar_tokens = bar_tokens.tolist()

            bar_pos = bar_pos.tolist() + [len(bar_tokens)]

            # エンコーダー入力データを生成
            enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(
                np.array(bar_pos), bar_tokens
            )

            enc_bar_idx_seq, enc_beat_idx_seq, enc_pos_idx_seq, enc_is_struct_seq = self._build_hier_indices_standard(
                bar_tokens,
                bar_pos,
            )
            enc_bar_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_bar_idx_seq, pad_value=0)
            enc_beat_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_beat_idx_seq, pad_value=0)
            enc_pos_idx = self.get_encoder_hier_input_data(np.array(bar_pos), enc_pos_idx_seq, pad_value=0)
            enc_is_struct = self.get_encoder_hier_input_data(np.array(bar_pos), enc_is_struct_seq, pad_value=0)

            # デコーダー入力/ターゲットを生成
            length = len(bar_tokens)
            if self.pad_to_same:
                inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1)
            else:
                inp = self.pad_sequence(
                    bar_tokens,
                    len(bar_tokens) + 1,
                    pad_value=self.dec_end_pad_value
                )

            target = np.array(inp[1:], dtype=int)
            inp = np.array(inp[:-1], dtype=int)
            assert len(inp) == len(target)

            if self.pad_to_same:
                bar_hier = self.pad_sequence(enc_bar_idx_seq.tolist(), self.model_dec_seqlen + 1, pad_value=0)
                beat_hier = self.pad_sequence(enc_beat_idx_seq.tolist(), self.model_dec_seqlen + 1, pad_value=0)
                pos_hier = self.pad_sequence(enc_pos_idx_seq.tolist(), self.model_dec_seqlen + 1, pad_value=0)
                struct_hier = self.pad_sequence(enc_is_struct_seq.tolist(), self.model_dec_seqlen + 1, pad_value=0)
            else:
                bar_hier = self.pad_sequence(enc_bar_idx_seq.tolist(), len(enc_bar_idx_seq) + 1, pad_value=0)
                beat_hier = self.pad_sequence(enc_beat_idx_seq.tolist(), len(enc_beat_idx_seq) + 1, pad_value=0)
                pos_hier = self.pad_sequence(enc_pos_idx_seq.tolist(), len(enc_pos_idx_seq) + 1, pad_value=0)
                struct_hier = self.pad_sequence(enc_is_struct_seq.tolist(), len(enc_is_struct_seq) + 1, pad_value=0)

            dec_bar_idx = np.array(bar_hier[:-1], dtype=np.int32)
            dec_beat_idx = np.array(beat_hier[:-1], dtype=np.int32)
            dec_pos_idx = np.array(pos_hier[:-1], dtype=np.int32)
            dec_is_struct = np.array(struct_hier[:-1], dtype=np.int32)

            result = {
                'id': idx,
                'piece_id': int(os.path.basename(self.pieces[idx]).replace('.pkl', '')),
                'st_bar_id': st_bar,
                'bar_pos': np.array(bar_pos, dtype=int),
                'enc_input': enc_inp,
                'enc_bar_idx': enc_bar_idx,
                'enc_beat_idx': enc_beat_idx,
                'enc_pos_idx': enc_pos_idx,
                'enc_is_struct': enc_is_struct,
                'dec_input': inp[:self.model_dec_seqlen],
                'dec_target': target[:self.model_dec_seqlen],
                'dec_bar_idx': dec_bar_idx[:self.model_dec_seqlen],
                'dec_beat_idx': dec_beat_idx[:self.model_dec_seqlen],
                'dec_pos_idx': dec_pos_idx[:self.model_dec_seqlen],
                'dec_is_struct': dec_is_struct[:self.model_dec_seqlen],
                'length': min(length, self.model_dec_seqlen),
                'enc_padding_mask': enc_padding_mask,
                'enc_length': enc_lens,
                'enc_n_bars': enc_n_bars
            }

        # ===== 追加: 難易度クラスをresultに追加 =====
        if self.use_difficulty and difficulty_classes is not None:
            # 各トークン位置に対応する難易度クラスを展開
            s_tech_cls = np.array([d['s_tech'] for d in difficulty_classes], dtype=np.int32)
            s_indep_cls = np.array([d['s_indep'] for d in difficulty_classes], dtype=np.int32)
            s_hand_cls = np.array([d['s_hand'] for d in difficulty_classes], dtype=np.int32)
            s_foot_cls = np.array([d['s_foot'] for d in difficulty_classes], dtype=np.int32)
            s_move_cls = np.array([d['s_move'] for d in difficulty_classes], dtype=np.int32)

            # (seqlen_per_sample,) の形状に拡張
            s_tech_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=np.int32)
            s_indep_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=np.int32)
            s_hand_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=np.int32)
            s_foot_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=np.int32)
            s_move_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=np.int32)

            for b, (st, ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                if b >= len(difficulty_classes):
                    break
                ed = min(ed, self.model_dec_seqlen)
                st = min(st, self.model_dec_seqlen)
                if st < ed:
                    s_tech_cls_expanded[st:ed] = s_tech_cls[b]
                    s_indep_cls_expanded[st:ed] = s_indep_cls[b]
                    s_hand_cls_expanded[st:ed] = s_hand_cls[b]
                    s_foot_cls_expanded[st:ed] = s_foot_cls[b]
                    s_move_cls_expanded[st:ed] = s_move_cls[b]

            result['s_tech_cls'] = s_tech_cls
            result['s_indep_cls'] = s_indep_cls
            result['s_hand_cls'] = s_hand_cls
            result['s_foot_cls'] = s_foot_cls
            result['s_move_cls'] = s_move_cls
            result['s_tech_cls_seq'] = s_tech_cls_expanded
            result['s_indep_cls_seq'] = s_indep_cls_expanded
            result['s_hand_cls_seq'] = s_hand_cls_expanded
            result['s_foot_cls_seq'] = s_foot_cls_expanded
            result['s_move_cls_seq'] = s_move_cls_expanded
        # ============================================

        return result


def collate_fn(batch):
    """カスタムコレート関数（バッチ処理用）"""
    return {
        key: torch.tensor([item[key] for item in batch]) if key in batch[0] else None
        for key in batch[0].keys()
    }


if __name__ == "__main__":
    # テスト用
    print("Testing DrumTransformerDataset...")

    # ダミーの語彙ファイルを作成
    test_vocab_path = '/tmp/test_drum_vocab.pkl'
    if not os.path.exists(test_vocab_path):
        from drum_tokenizer import DrumTokenizer
        tokenizer = DrumTokenizer()
        tokenizer.save_vocab(test_vocab_path)
        print(f"Created test vocabulary: {test_vocab_path}")

    # データセットディレクトリを指定（実際のデータがある場合）
    data_dir = './drum_dataset'

    if os.path.exists(data_dir):
        dset = DrumTransformerDataset(
            data_dir,
            test_vocab_path,
            model_max_bars=16,
            model_dec_seqlen=1280,
            model_enc_seqlen=128,
            pad_to_same=True
        )

        print(f'Vocabulary size: {dset.vocab_size}')
        print(f'Bar token: {dset.bar_token}')
        print(f'Pad token: {dset.pad_token}')
        print(f'Dataset length: {len(dset)}')

        if len(dset) > 0:
            sample = dset[0]
            print(f'\nSample keys: {sample.keys()}')
            print(f'Encoder input shape: {sample["enc_input"].shape}')
            print(f'Decoder input shape: {sample["dec_input"].shape}')
            print(f'Decoder target shape: {sample["dec_target"].shape}')

            # DataLoaderのテスト
            dloader = DataLoader(dset, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(dloader))
            print(f'\nBatch keys: {batch.keys()}')
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f'  {k}: {v.dtype}, {v.size()}')
    else:
        print(f"Data directory '{data_dir}' not found. Please prepare dataset first.")
