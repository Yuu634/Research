"""
データセット全体から難易度指標の境界値を計算

使用方法:
    python compute_difficulty_bounds.py <data_dir> <vocab_path> <output_path>

例:
    python compute_difficulty_bounds.py ./drum_dataset ./drum_vocab.pkl ./difficulty_bounds.pkl
"""

import sys
import os
import pickle
import argparse
import numpy as np

from src.data.drum_difficulty_calculator import compute_all_difficulty_scores
from src.tokenizers.drum_tokenizer import DRUM_NOTE_MAP


def _is_remi_vocab(idx2token):
    return any(str(v).startswith('Note_Pitch_') for v in idx2token.values())


def _velocity_bin_to_level(bin_value):
    # REMIの32ビンを標準トークンのGhost/Normal/Accentへ近似変換
    if bin_value <= 9:
        return 'Ghost'
    if bin_value <= 24:
        return 'Normal'
    return 'Accent'


def _remi_indices_to_difficulty_tokens(bar_indices, idx2token):
    tokens = [idx2token.get(int(i), '<PAD>') for i in bar_indices]
    out = []
    current_step = 0

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('<BEAT_'):
            try:
                current_step = int(tok.replace('<BEAT_', '').replace('>', ''))
            except ValueError:
                current_step = 0
            i += 1
            continue

        if tok.startswith('Note_Pitch_'):
            try:
                pitch = int(tok.split('_')[-1])
            except ValueError:
                i += 1
                continue

            drum_info = DRUM_NOTE_MAP.get(pitch)
            if drum_info is None:
                i += 1
                continue

            velocity_level = 'Normal'
            if (i + 1) < len(tokens) and tokens[i + 1].startswith('Note_Velocity_'):
                try:
                    vel_bin = int(tokens[i + 1].split('_')[-1])
                    velocity_level = _velocity_bin_to_level(vel_bin)
                except ValueError:
                    velocity_level = 'Normal'

            beat_idx = (current_step // 4) + 1
            pos_idx = (current_step % 4) * 6
            out.append(f'<BEAT_{beat_idx}>')
            out.append(f'<POS_{pos_idx}>')
            out.append(f'{drum_info[0]}_HIT_{velocity_level}')

        i += 1

    return out


def compute_difficulty_bounds(data_dir, vocab_path, output_path, bpm=120.0):
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    is_cp_vocab = isinstance(vocab_data, dict) and vocab_data.get('tokenization_method') == 'cp_limb_v1'
    if is_cp_vocab:
        idx2type = vocab_data.get('idx2type_token', vocab_data['idx2struct_token'])
        idx2limb = vocab_data['idx2limb_token']
        idx2token = None
        is_remi = False
    else:
        # vocab_dataは辞書形式: {'tokenization_method', 'token2idx', 'idx2token'}
        if isinstance(vocab_data, dict):
            idx2token = vocab_data.get('idx2token', {})
            is_remi = vocab_data.get('tokenization_method') == 'remi'
        else:
            # 後方互換性: 古いタプル形式 (token2idx, idx2token)
            _, idx2token = vocab_data
            is_remi = False

    all_scores = {
        's_tech': [],
        's_indep': [],
        's_hand': [],
        's_foot': [],
        's_move': []
    }

    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.pkl') and f not in ('dataset_stats.pkl', 'train_split.pkl', 'val_split.pkl')
    ])
    print(f'Found {len(files)} pickle files')
    print(f'  CP vocab: {is_cp_vocab}')
    print(f'  REMI vocab: {is_remi}')

    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f'Processing {i}/{len(files)}...')

        try:
            with open(os.path.join(data_dir, file), 'rb') as f:
                data = pickle.load(f)

            if len(data) == 2:
                bar_pos, tokens = data
            elif len(data) == 3:
                bar_pos, tokens, _ = data
            else:
                continue

            if len(bar_pos) < 2:
                continue

            for b in range(len(bar_pos) - 1):
                st, ed = bar_pos[b], bar_pos[b + 1]
                token_len = len(tokens['cp_type']) if isinstance(tokens, dict) and 'cp_type' in tokens else (
                    len(tokens['event_type']) if isinstance(tokens, dict) and 'event_type' in tokens else len(tokens)
                )
                if st >= ed or ed > token_len:
                    continue

                if is_cp_vocab and isinstance(tokens, dict):
                    # CP形式は prepare_drum_dataset_with_difficulty.py と同じ復元ロジックを使用
                    bar_tokens = []
                    cp_type_seq = tokens.get('cp_type')
                    if cp_type_seq is None and 'event_type' in tokens and 'struct_token' in tokens:
                        cp_type_seq = []
                        for ev_type, struct_id in zip(tokens['event_type'][st:ed], tokens['struct_token'][st:ed]):
                            if int(ev_type) == 1:
                                cp_type_seq.append(-1)
                            elif int(ev_type) == 2:
                                cp_type_seq.append(-2)
                            else:
                                cp_type_seq.append(int(struct_id))
                    else:
                        cp_type_seq = cp_type_seq[st:ed]

                    for type_id, pos_id, h1, h2, rf, lf in zip(
                        cp_type_seq,
                        tokens['cp_pos'][st:ed],
                        tokens['cp_hand1'][st:ed],
                        tokens['cp_hand2'][st:ed],
                        tokens['cp_right_foot'][st:ed],
                        tokens['cp_left_foot'][st:ed],
                    ):
                        if int(type_id) == -1:
                            cp_type_token = '<NOTE>'
                        elif int(type_id) == -2:
                            cp_type_token = '<PAD>'
                        else:
                            cp_type_token = idx2type.get(int(type_id), '<PAD>')

                        if cp_type_token != '<NOTE>':
                            if cp_type_token != '<PAD>':
                                bar_tokens.append(cp_type_token)
                            continue

                        if int(pos_id) < 24:
                            bar_tokens.append(f'<POS_{int(pos_id)}>')
                        for limb_id in (h1, h2, rf, lf):
                            limb_tok = idx2limb.get(int(limb_id), '<PAD>')
                            if limb_tok not in ('<PAD>', '<NONE>', '<NA>'):
                                bar_tokens.append(limb_tok)
                else:
                    bar_indices = tokens[st:ed]
                    if is_remi:
                        bar_tokens = _remi_indices_to_difficulty_tokens(bar_indices, idx2token)
                    else:
                        bar_tokens = [idx2token.get(int(idx), '<PAD>') for idx in bar_indices]

                scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)
                for key, value in scores.items():
                    all_scores[key].append(value)

        except Exception as e:
            print(f'Error processing {file}: {e}')

    bounds = {}
    for key, values in all_scores.items():
        if len(values) == 0:
            bounds[key] = [0.0] * 7
            continue

        arr = np.array(values)
        bounds[key] = np.percentile(arr, [12.5, 25, 37.5, 50, 62.5, 75, 87.5]).tolist()
        print(f'{key} bounds: {bounds[key]}')

    with open(output_path, 'wb') as f:
        pickle.dump(bounds, f)

    return bounds


def main():
    parser = argparse.ArgumentParser(description='Compute difficulty bounds from dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('vocab_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('bpm', type=float, nargs='?', default=120.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    vocab_path = args.vocab_path
    output_path = args.output_path
    bpm = args.bpm

    # データディレクトリとvocabファイルの存在確認
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' does not exist.")
        sys.exit(1)

    print(f"Computing difficulty bounds...")
    print(f"  Data directory: {data_dir}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Output: {output_path}")
    print(f"  Default BPM: {bpm}")
    print()

    bounds = compute_difficulty_bounds(data_dir, vocab_path, output_path, bpm=bpm)

    print("\n" + "=" * 60)
    print("Difficulty bounds computed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
