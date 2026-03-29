"""
ドラム譜MIDIデータセットの準備スクリプト
MIDIファイルをトークン化してpickleファイルとして保存
"""
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import miditoolkit
import sys

try:
    from src.tokenizers.drum_tokenizer import DrumTokenizer, DRUM_NOTE_MAP
    from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
    from src.utils.tokenization_mode import (
        validate_tokenization_method,
        write_dataset_tokenization_metadata,
    )
    from src.utils.tokenization_policy import (
        POSITIONS_PER_BEAT,
        classify_velocity_level,
        quantize_step_in_bar,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.tokenizers.drum_tokenizer import DrumTokenizer, DRUM_NOTE_MAP
    from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
    from src.utils.tokenization_mode import (
        validate_tokenization_method,
        write_dataset_tokenization_metadata,
    )
    from src.utils.tokenization_policy import (
        POSITIONS_PER_BEAT,
        classify_velocity_level,
        quantize_step_in_bar,
    )


class RemiDrumTokenizer:
    """ドラムMIDIをREMI風イベント列に変換する簡易トークナイザー"""

    def __init__(self):
        self.beats_per_bar = 4 * POSITIONS_PER_BEAT  # 4/4を96分割（1拍24分割、Standard/CPと統一）
        self.max_duration_steps = 256
        self.drum_pitches = sorted(set(DRUM_NOTE_MAP.keys()))
        self.last_conversion_info = {
            'drum_track_count': 0,
            'non_drum_tracks_excluded': 0,
            'total_drum_notes': 0,
        }
        self.velocity_level_to_bin = {
            'Ghost': 0,
            'Normal': 16,
            'Accent': 31,
        }

        self.vocab = self._build_vocabulary()
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.vocab)

    def _build_vocabulary(self):
        vocab = ['<PAD>', '<EOS>', '<BAR>']
        vocab.extend([f'<TEMPO_{t}>' for t in range(30, 241)])
        vocab.extend([f'<BEAT_{i}>' for i in range(self.beats_per_bar)])
        vocab.extend([f'Note_Pitch_{p}' for p in self.drum_pitches])
        vocab.extend([f'Note_Velocity_{i}' for i in range(32)])
        vocab.extend([f'Note_Duration_{i}' for i in range(1, self.max_duration_steps + 1)])
        return vocab

    def save_vocab(self, path):
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'tokenization_method': 'remi',
                    'schema_version': 1,
                    'token2idx': self.token2idx,
                    'idx2token': self.idx2token,
                },
                f,
            )

    def _get_tempo_at_tick(self, midi_obj, tick):
        if not midi_obj.tempo_changes:
            return 120
        curr = int(round(midi_obj.tempo_changes[0].tempo))
        for tc in midi_obj.tempo_changes:
            if tc.time <= tick:
                curr = int(round(tc.tempo))
            else:
                break
        return max(30, min(240, curr))

    def _collect_drum_notes(self, midi_obj):
        drum_tracks = [inst for inst in midi_obj.instruments if inst.is_drum]
        non_drum_tracks_with_notes = [inst for inst in midi_obj.instruments if not inst.is_drum and len(inst.notes) > 0]

        if not drum_tracks:
            raise ValueError('No drum track found in MIDI file')

        drum_notes = []
        for inst in drum_tracks:
            drum_notes.extend(inst.notes)

        drum_notes.sort(key=lambda n: (int(n.start), int(n.pitch), int(n.end)))

        self.last_conversion_info = {
            'drum_track_count': len(drum_tracks),
            'non_drum_tracks_excluded': len(non_drum_tracks_with_notes),
            'total_drum_notes': len(drum_notes),
        }
        return drum_notes

    def midi_to_tokens(self, midi_path):
        midi_obj = miditoolkit.MidiFile(midi_path)
        drum_notes = self._collect_drum_notes(midi_obj)

        ticks_per_beat = midi_obj.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4
        step_ticks = max(1, ticks_per_beat // POSITIONS_PER_BEAT)  # 1拍を24分割

        max_tick = max((n.end for n in drum_notes), default=0)
        n_bars = int(np.ceil(max_tick / ticks_per_bar))
        if n_bars == 0:
            n_bars = 1

        notes_by_bar_step = {}
        for note in drum_notes:
            if note.pitch not in DRUM_NOTE_MAP:
                continue
            bar_idx = max(0, int(note.start // ticks_per_bar))
            in_bar_tick = note.start - bar_idx * ticks_per_bar
            step = quantize_step_in_bar(in_bar_tick, ticks_per_beat)
            step = max(0, min(self.beats_per_bar - 1, step))
            notes_by_bar_step.setdefault((bar_idx, step), []).append(note)

        tokens = []
        bar_positions = []

        for bar in range(n_bars):
            bar_positions.append(len(tokens))
            tokens.append('<BAR>')

            tempo = self._get_tempo_at_tick(midi_obj, bar * ticks_per_bar)
            tokens.append(f'<TEMPO_{tempo}>')

            steps = sorted([s for (b, s) in notes_by_bar_step.keys() if b == bar])
            for step in steps:
                tokens.append(f'<BEAT_{step}>')
                for note in sorted(notes_by_bar_step[(bar, step)], key=lambda x: (x.start, x.pitch)):
                    velocity_level = classify_velocity_level(int(note.velocity))
                    velocity_bin = self.velocity_level_to_bin[velocity_level]
                    duration_steps = int(round((note.end - note.start) / step_ticks))
                    duration_steps = int(np.clip(duration_steps, 1, self.max_duration_steps))

                    tokens.append(f'Note_Pitch_{note.pitch}')
                    tokens.append(f'Note_Velocity_{velocity_bin}')
                    tokens.append(f'Note_Duration_{duration_steps}')

        tokens.append('<EOS>')
        return tokens, bar_positions

    def tokens_to_indices(self, tokens):
        unk = self.token2idx['<PAD>']
        return np.array([self.token2idx.get(tok, unk) for tok in tokens], dtype=np.int32)


def prepare_drum_dataset(
    midi_dir: str,
    output_dir: str,
    vocab_path: str,
    file_extension: str = '.midi',
    tokenization_method: str = 'standard'
):
    """
    MIDIファイルをトークン化してデータセットを準備

    Args:
        midi_dir: MIDIファイルが格納されているディレクトリ
        output_dir: トークン化されたデータを保存するディレクトリ
        vocab_path: 語彙ファイルの保存先パス
        file_extension: MIDIファイルの拡張子
    """
    tokenization_method = validate_tokenization_method(
        tokenization_method,
        context='--tokenization_method'
    )

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # トークナイザーを初期化
    if tokenization_method == 'cp_limb_v1':
        tokenizer = DrumCPTokenizer()
    elif tokenization_method == 'remi':
        tokenizer = RemiDrumTokenizer()
    elif tokenization_method == 'standard':
        tokenizer = DrumTokenizer()
    else:
        raise ValueError(
            f"Unsupported --tokenization_method: {tokenization_method}. "
            "Supported: cp_limb_v1, remi, standard"
        )

    # 語彙を保存
    tokenizer.save_vocab(vocab_path)
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    metadata_extra = None
    if tokenization_method == 'remi':
        metadata_extra = {
            'input_scope': 'drum_only',
            'non_drum_tracks_policy': 'excluded',
            'velocity_policy': 'shared_3level',
            'position_policy': 'standard_24div',
            'remi_velocity_representation': 'discrete_bins',
        }

    metadata_path = write_dataset_tokenization_metadata(
        output_dir,
        tokenization_method,
        extra_metadata=metadata_extra,
    )
    print(f"Tokenization metadata saved to: {metadata_path}")

    # MIDIファイルのリストを取得
    midi_files = list(Path(midi_dir).rglob(f'*{file_extension}'))
    print(f"\nFound {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        print(f"Warning: No MIDI files found in {midi_dir}")
        return

    # 統計情報
    stats = {
        'total_files': len(midi_files),
        'successful': 0,
        'failed': 0,
        'total_tokens': 0,
        'total_bars': 0,
        'avg_tokens_per_file': 0,
        'avg_bars_per_file': 0,
        'token_lengths': [],
        'bar_counts': [],
        'remi_drum_tracks_used_total': 0,
        'remi_non_drum_tracks_excluded_total': 0,
        'remi_files_with_non_drum_excluded': 0,
    }

    # 各MIDIファイルを処理
    print("\nProcessing MIDI files...")
    failed_files = []

    for i, midi_file in enumerate(tqdm(midi_files)):
        try:
            if tokenization_method == 'cp_limb_v1':
                cp_data, bar_positions = tokenizer.midi_to_cp_data(str(midi_file))
                n_tokens = len(cp_data['cp_type'])
            else:
                # MIDIをトークン列に変換
                tokens, bar_positions = tokenizer.midi_to_tokens(str(midi_file))

                if tokenization_method == 'remi':
                    info = getattr(tokenizer, 'last_conversion_info', {})
                    excluded_non_drum = int(info.get('non_drum_tracks_excluded', 0))
                    stats['remi_drum_tracks_used_total'] += int(info.get('drum_track_count', 0))
                    stats['remi_non_drum_tracks_excluded_total'] += excluded_non_drum
                    if excluded_non_drum > 0:
                        stats['remi_files_with_non_drum_excluded'] += 1

                # インデックス列に変換
                token_indices = tokenizer.tokens_to_indices(tokens)
                n_tokens = len(token_indices)

            # 統計を更新
            n_bars = len(bar_positions)

            stats['successful'] += 1
            stats['total_tokens'] += n_tokens
            stats['total_bars'] += n_bars
            stats['token_lengths'].append(n_tokens)
            stats['bar_counts'].append(n_bars)

            # ファイル名を生成（連番）
            output_filename = f"{i:06d}.pkl"
            output_path = os.path.join(output_dir, output_filename)

            # データを保存
            with open(output_path, 'wb') as f:
                if tokenization_method == 'cp_limb_v1':
                    pickle.dump((bar_positions, cp_data), f)
                else:
                    pickle.dump((bar_positions, token_indices.tolist()), f)

        except Exception as e:
            stats['failed'] += 1
            failed_files.append((str(midi_file), str(e)))
            print(f"\nError processing {midi_file}: {e}")
            continue

    # 統計情報を計算
    if stats['successful'] > 0:
        stats['avg_tokens_per_file'] = stats['total_tokens'] / stats['successful']
        stats['avg_bars_per_file'] = stats['total_bars'] / stats['successful']

    # 統計情報を表示
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"Total files processed:    {stats['total_files']}")
    print(f"Successful:               {stats['successful']}")
    print(f"Failed:                   {stats['failed']}")
    print(f"\nTotal tokens:             {stats['total_tokens']}")
    print(f"Total bars:               {stats['total_bars']}")
    print(f"Avg tokens per file:      {stats['avg_tokens_per_file']:.2f}")
    print(f"Avg bars per file:        {stats['avg_bars_per_file']:.2f}")

    if stats['token_lengths']:
        print(f"\nToken length statistics:")
        print(f"  Min:  {min(stats['token_lengths'])}")
        print(f"  Max:  {max(stats['token_lengths'])}")
        print(f"  Mean: {np.mean(stats['token_lengths']):.2f}")
        print(f"  Std:  {np.std(stats['token_lengths']):.2f}")

    if stats['bar_counts']:
        print(f"\nBar count statistics:")
        print(f"  Min:  {min(stats['bar_counts'])}")
        print(f"  Max:  {max(stats['bar_counts'])}")
        print(f"  Mean: {np.mean(stats['bar_counts']):.2f}")
        print(f"  Std:  {np.std(stats['bar_counts']):.2f}")

    if tokenization_method == 'remi':
        print(f"\nREMI drum-only policy summary:")
        print(f"  Drum tracks used (total):            {stats['remi_drum_tracks_used_total']}")
        print(f"  Non-drum tracks excluded (total):    {stats['remi_non_drum_tracks_excluded_total']}")
        print(f"  Files with non-drum exclusions:      {stats['remi_files_with_non_drum_excluded']}")

    # 失敗したファイルのリストを保存
    if failed_files:
        error_log_path = os.path.join(output_dir, 'failed_files.txt')
        with open(error_log_path, 'w') as f:
            f.write("Failed files:\n")
            for filepath, error in failed_files:
                f.write(f"{filepath}: {error}\n")
        print(f"\nFailed files logged to: {error_log_path}")

    # 統計情報を保存
    stats_path = os.path.join(output_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Statistics saved to: {stats_path}")

    print("=" * 60)


def create_train_val_split(
    data_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42
):
    """
    データセットを訓練/検証セットに分割

    Args:
        data_dir: トークン化されたデータが保存されているディレクトリ
        train_ratio: 訓練データの割合
        seed: ランダムシード
    """
    np.random.seed(seed)

    # すべてのpickleファイルを取得
    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl') and f != 'dataset_stats.pkl'])

    if len(pkl_files) == 0:
        print(f"No pickle files found in {data_dir}")
        return

    print(f"\nFound {len(pkl_files)} data files")

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
    parser = argparse.ArgumentParser(description='Prepare drum MIDI dataset for training')
    parser.add_argument('--midi_dir', type=str, required=True,
                        help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for tokenized data')
    parser.add_argument('--vocab_path', type=str, default='drum_vocab.pkl',
                        help='Path to save vocabulary file')
    parser.add_argument('--file_extension', type=str, default='.midi',
                        help='MIDI file extension (default: .midi)')
    parser.add_argument('--tokenization_method', type=str, default='standard',
                        choices=['standard', 'cp_limb_v1', 'remi'],
                        help='Tokenization method (default: standard)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split (default: 42)')
    parser.add_argument('--skip_preparation', action='store_true',
                        help='Skip dataset preparation and only create train/val split')

    args = parser.parse_args()

    if not args.skip_preparation:
        # データセットを準備
        prepare_drum_dataset(
            midi_dir=args.midi_dir,
            output_dir=args.output_dir,
            vocab_path=args.vocab_path,
            file_extension=args.file_extension,
            tokenization_method=args.tokenization_method
        )

    # 訓練/検証セットに分割
    create_train_val_split(
        data_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    print("\nDataset preparation completed!")


if __name__ == '__main__':
    main()
