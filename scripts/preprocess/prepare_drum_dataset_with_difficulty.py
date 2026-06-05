"""
既存のドラム譜データセットに難易度クラスを付加し、train_split.pkl/val_split.pklを作成

使用方法:
    python prepare_drum_dataset_with_difficulty.py <input_dir> <vocab_path> <bounds_path> <output_dir> [--train_ratio 0.9] [--seed 42]

例:
    python prepare_drum_dataset_with_difficulty.py ./drum_prepare ./drum_vocab.pkl ./difficulty_bounds.pkl ./drum_dataset_with_difficulty
"""

import sys
import os
import pickle
from tqdm import tqdm
import numpy as np
import argparse

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.drum_difficulty_calculator import (
    compute_all_difficulty_scores,
    discretize_difficulty_score
)
from src.tokenizers.drum_tokenizer import DRUM_NOTE_MAP

try:
    from src.utils.tokenization_mode import write_dataset_tokenization_metadata
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.utils.tokenization_mode import write_dataset_tokenization_metadata


def cp_data_to_tokens(cp_data, idx2type, idx2limb):
    """CP形式データを難易度計算用のトークン列に近似復元する。"""
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


def is_remi_vocab(idx2token):
    return any(str(v).startswith('Note_Pitch_') for v in idx2token.values())


def velocity_bin_to_level(bin_value):
    if bin_value <= 9:
        return 'Ghost'
    if bin_value <= 24:
        return 'Normal'
    return 'Accent'


def remi_indices_to_difficulty_tokens(bar_indices, idx2token):
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
                    velocity_level = velocity_bin_to_level(vel_bin)
                except ValueError:
                    velocity_level = 'Normal'

            beat_idx = (current_step // 4) + 1
            pos_idx = (current_step % 4) * 6
            out.append(f'<BEAT_{beat_idx}>')
            out.append(f'<POS_{pos_idx}>')
            out.append(f'{drum_info[0]}_{drum_info[1]}_{velocity_level}')

        i += 1

    return out


def prepare_dataset_with_difficulty(
    input_dir: str,
    vocab_path: str,
    difficulty_bounds_path: str,
    output_dir: str,
    bpm: float = 120.0
):
    """
    各pkl ファイルに難易度クラスを付加して保存

    出力形式:
        output_dir/
            <piece_id>.pkl  -> (bar_pos, tokens, difficulty_classes)
            difficulty_classes: List[Dict[str, int]]  # 各小節の5つの難易度クラス
    """
    # 語彙を読み込み
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    is_cp_vocab = isinstance(vocab_data, dict) and vocab_data.get('tokenization_method') == 'cp_limb_v1'
    if is_cp_vocab:
        idx2type = vocab_data.get('idx2type_token', vocab_data['idx2struct_token'])
        idx2limb = vocab_data['idx2limb_token']
        idx2token = None
        remi_vocab = False
    else:
        # vocab_dataは辞書形式: {'tokenization_method', 'token2idx', 'idx2token'}
        if isinstance(vocab_data, dict):
            idx2token = vocab_data.get('idx2token', {})
            remi_vocab = vocab_data.get('tokenization_method') == 'remi'
        else:
            # 後方互換性: 古いタプル形式 (token2idx, idx2token)
            _, idx2token = vocab_data
            remi_vocab = False

    # 境界値を読み込み
    print(f"Loading difficulty bounds from {difficulty_bounds_path}...")
    with open(difficulty_bounds_path, 'rb') as f:
        bounds = pickle.load(f)

    print(f"Difficulty bounds:")
    for key, values in bounds.items():
        print(f"  {key}: {values}")

    os.makedirs(output_dir, exist_ok=True)

    tokenization_method = 'cp_limb_v1' if is_cp_vocab else ('remi' if remi_vocab else 'standard')
    metadata_path = write_dataset_tokenization_metadata(output_dir, tokenization_method)
    print(f"Tokenization metadata saved to {metadata_path}")

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pkl')])
    print(f"\nFound {len(files)} files to process")
    print(f"  CP vocab: {is_cp_vocab}")
    print(f"  REMI vocab: {remi_vocab}")

    success_count = 0
    error_count = 0

    for i, file in enumerate(tqdm(files, desc="Processing files")):
        try:
            input_path = os.path.join(input_dir, file)

            # データを読み込み
            with open(input_path, 'rb') as f:
                data = pickle.load(f)

            # データ形式を確認
            if len(data) == 2:
                bar_pos, tokens = data
            elif len(data) == 3:
                bar_pos, tokens, _ = data
            else:
                print(f"\nWarning: Unknown data format in {file}, skipping...")
                error_count += 1
                continue

            difficulty_classes = []

            # 各小節ごとに難易度を計算
            if len(bar_pos) < 2:
                print(f"\nWarning: Not enough bars in {file}, skipping...")
                error_count += 1
                continue

            for b in range(len(bar_pos) - 1):
                st, ed = bar_pos[b], bar_pos[b + 1]
                token_len = len(tokens['cp_type']) if isinstance(tokens, dict) and 'cp_type' in tokens else (
                    len(tokens['event_type']) if isinstance(tokens, dict) and 'event_type' in tokens else len(tokens)
                )

                if st >= ed or ed > token_len:
                    # 不正な範囲の場合はデフォルト値を使用
                    bar_difficulty = {
                        's_tech': 0,
                        's_indep': 0,
                        's_hand': 0,
                        's_foot': 0,
                        's_move': 0
                    }
                else:
                    if is_cp_vocab and isinstance(tokens, dict):
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
                        bar_tokens = cp_data_to_tokens(bar_cp_data, idx2type, idx2limb)
                    else:
                        bar_token_indices = tokens[st:ed]
                        if remi_vocab:
                            bar_tokens = remi_indices_to_difficulty_tokens(bar_token_indices, idx2token)
                        else:
                            bar_tokens = [idx2token.get(int(idx), '<PAD>') for idx in bar_token_indices]

                    # 連続値スコアを計算
                    scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)

                    # 離散化
                    bar_difficulty = {
                        key: discretize_difficulty_score(value, bounds[key])
                        for key, value in scores.items()
                    }

                difficulty_classes.append(bar_difficulty)

            # 保存
            output_path = os.path.join(output_dir, file)
            with open(output_path, 'wb') as f:
                pickle.dump((bar_pos, tokens, difficulty_classes), f)

            success_count += 1

        except Exception as e:
            print(f'\nError processing {file}: {e}')
            error_count += 1
            continue

    print(f'\n{"=" * 60}')
    print(f'Dataset preparation completed!')
    print(f'  Successfully processed: {success_count} files')
    print(f'  Errors: {error_count} files')
    print(f'  Output directory: {output_dir}')
    print(f'{"=" * 60}')

    return success_count, error_count


def create_train_val_split(
    data_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42
):
    """
    難易度付きデータセットを訓練/検証セットに分割し、ファイルリストを保存

    Args:
        data_dir: 難易度付きデータが保存されているディレクトリ
        train_ratio: 訓練データの割合
        seed: ランダムシード
    """
    np.random.seed(seed)

    # すべてのpickleファイルを取得（難易度付きデータファイルのみ）
    pkl_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.endswith('.pkl') 
        and f not in ('dataset_stats.pkl', 'train_split.pkl', 'val_split.pkl')
    ])

    if len(pkl_files) == 0:
        print(f"No pickle files found in {data_dir}")
        return

    print(f"\nFound {len(pkl_files)} data files in {data_dir}")

    # シャッフル
    indices = np.arange(len(pkl_files))
    np.random.shuffle(indices)

    # 分割
    n_train = int(len(pkl_files) * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_files = [pkl_files[i] for i in train_indices]
    val_files = [pkl_files[i] for i in val_indices]

    # 分割情報を保存
    train_split_path = os.path.join(data_dir, 'train_split.pkl')
    val_split_path = os.path.join(data_dir, 'val_split.pkl')

    with open(train_split_path, 'wb') as f:
        pickle.dump(train_files, f)

    with open(val_split_path, 'wb') as f:
        pickle.dump(val_files, f)

    print(f"\nTrain/Val split created:")
    print(f"  Train: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_files)} files ({(1-train_ratio)*100:.1f}%)")
    print(f"  Train split saved to: {train_split_path}")
    print(f"  Val split saved to:   {val_split_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare drum dataset with difficulty labels')
    parser.add_argument('input_dir', type=str,
                        help='Input directory with tokenized data')
    parser.add_argument('vocab_path', type=str,
                        help='Path to vocabulary file')
    parser.add_argument('bounds_path', type=str,
                        help='Path to difficulty bounds pickle file')
    parser.add_argument('output_dir', type=str,
                        help='Output directory for difficulty-labeled data')
    parser.add_argument('bpm', type=float, nargs='?', default=120.0,
                        help='Default BPM (default: 120.0)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split (default: 42)')

    args = parser.parse_args()

    input_dir = args.input_dir
    vocab_path = args.vocab_path
    bounds_path = args.bounds_path
    output_dir = args.output_dir
    bpm = args.bpm
    train_ratio = args.train_ratio
    seed = args.seed

    # 入力の存在確認
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(bounds_path):
        print(f"Error: Bounds file '{bounds_path}' does not exist.")
        sys.exit(1)

    print(f"Preparing dataset with difficulty classes...")
    print(f"  Input directory: {input_dir}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Bounds: {bounds_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Default BPM: {bpm}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Seed: {seed}")
    print()

    # 難易度クラスを付加
    success_count, error_count = prepare_dataset_with_difficulty(
        input_dir,
        vocab_path,
        bounds_path,
        output_dir,
        bpm=bpm
    )

    # Train/Val splitを作成
    if success_count > 0:
        create_train_val_split(
            data_dir=output_dir,
            train_ratio=train_ratio,
            seed=seed
        )

    print("\n" + "=" * 60)
    print("Dataset preparation with difficulty completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
