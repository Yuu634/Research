"""
LakhMIDI からドラム譜を抽出・クリーニングするスクリプト

クリーニング内容:
  1. 4/4 拍子のファイルのみ対象
  2. グリッド外音符を削除（小節内のオングリッド音符は保持）
  3. 対応外ピッチを含む小節を削除

使用方法:
    python data/cleaning_lakhmidi.py \
        --lakhmidi_dir data/raw/lmd_full \
        --output_dir data/filtered/lakhmidi
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import miditoolkit

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tokenizers.drum_tokenizer import DRUM_NOTE_MAP
from src.utils.tokenization_policy import POSITIONS_PER_BEAT

SUPPORTED_PITCHES = set(DRUM_NOTE_MAP.keys())


def is_44_time(midi_obj: miditoolkit.MidiFile) -> bool:
    if not midi_obj.time_signature_changes:
        return True  # デフォルトは4/4とみなす
    return all(
        ts.numerator == 4 and ts.denominator == 4
        for ts in midi_obj.time_signature_changes
    )


def is_on_grid(tick: int, ticks_per_beat: int) -> bool:
    """tick が1拍24分割グリッド上にあるか判定（浮動小数点を使わない）"""
    return (tick * POSITIONS_PER_BEAT) % ticks_per_beat == 0


def get_tempo_at_tick(tempo_changes: list, tick: int) -> float:
    """指定 tick 時点のテンポ（BPM）を返す"""
    if not tempo_changes:
        return 120.0
    curr = tempo_changes[0].tempo
    for tc in tempo_changes:
        if tc.time <= tick:
            curr = tc.tempo
        else:
            break
    return curr


def clean_midi(midi_path: Path):
    """
    1ファイルのクリーニングを実行。

    Returns:
        (bars, tempos, ticks_per_beat) or (None, reason)
        bars: list of (bar_idx, notes)
    """
    try:
        midi_obj = miditoolkit.MidiFile(str(midi_path))
    except Exception as e:
        return None, f"parse error: {e}"

    # 1. 4/4拍子チェック
    if not is_44_time(midi_obj):
        return None, "not 4/4"

    # テンポ変化チェック（set_tempo が複数あるものは除外）
    if len(midi_obj.tempo_changes) > 1:
        return None, "tempo changes"

    # ドラムトラックの音符を収集
    drum_notes = []
    for inst in midi_obj.instruments:
        if inst.is_drum:
            drum_notes.extend(inst.notes)

    if not drum_notes:
        return None, "no drum track"

    ticks_per_beat = midi_obj.ticks_per_beat
    ticks_per_bar = ticks_per_beat * 4

    drum_notes.sort(key=lambda n: n.start)

    # 小節ごとに振り分け
    bars_dict: dict[int, list] = {}
    for note in drum_notes:
        bar_idx = note.start // ticks_per_bar
        bars_dict.setdefault(bar_idx, []).append(note)

    result_bars = []  # list of (bar_idx, notes)
    result_tempos = []

    for bar_idx in sorted(bars_dict.keys()):
        bar_notes = bars_dict[bar_idx]
        bar_start = bar_idx * ticks_per_bar
        tempo = get_tempo_at_tick(midi_obj.tempo_changes, bar_start)

        # 3. 対応外ピッチを含む小節は削除
        if any(n.pitch not in SUPPORTED_PITCHES for n in bar_notes):
            continue

        # 2. グリッド外音符のみ削除（小節は保持）
        on_grid_notes = [n for n in bar_notes if is_on_grid(n.start, ticks_per_beat)]
        if not on_grid_notes:
            continue

        result_bars.append((bar_idx, on_grid_notes))
        result_tempos.append(tempo)

    if not result_bars:
        return None, "no valid bars"

    if len(result_bars) <= 16:
        return None, "too few bars"

    return (result_bars, result_tempos, ticks_per_beat), "ok"


def save_drum_midi(
    bars: list,
    tempos: list,
    ticks_per_beat: int,
    output_path: str,
) -> None:
    """クリーニング済み小節を連番で詰めた MIDI ファイルとして保存"""
    ticks_per_bar = ticks_per_beat * 4
    midi_out = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')

    notes_out = []
    tempo_changes_out = []
    prev_tempo = None

    for new_bar_idx, ((orig_bar_idx, bar_notes), tempo) in enumerate(zip(bars, tempos)):
        new_bar_start = new_bar_idx * ticks_per_bar
        orig_bar_start = orig_bar_idx * ticks_per_bar
        offset = new_bar_start - orig_bar_start

        if tempo != prev_tempo:
            tempo_changes_out.append(miditoolkit.TempoChange(tempo=tempo, time=new_bar_start))
            prev_tempo = tempo

        for note in bar_notes:
            notes_out.append(miditoolkit.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start + offset,
                end=note.end + offset,
            ))

    if not tempo_changes_out:
        tempo_changes_out.append(miditoolkit.TempoChange(tempo=120.0, time=0))

    drum_inst.notes = sorted(notes_out, key=lambda n: n.start)
    midi_out.instruments.append(drum_inst)
    midi_out.tempo_changes = tempo_changes_out
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    midi_out.dump(str(output_path))


def resolve_output_path(output_dir: str, midi_path: Path) -> str:
    """出力先パスを決定（ファイル名衝突時は連番を付与）"""
    candidate = os.path.join(output_dir, midi_path.stem + '.mid')
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        candidate = os.path.join(output_dir, f"{midi_path.stem}_{counter}.mid")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def main():
    parser = argparse.ArgumentParser(description='LakhMIDI ドラム譜クリーニング')
    parser.add_argument('--lakhmidi_dir', type=str, required=True,
                        help='LakhMIDI のルートディレクトリ')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='クリーニング済みMIDIの保存先ディレクトリ')
    args = parser.parse_args()

    midi_files = sorted(
        list(Path(args.lakhmidi_dir).rglob('*.mid')) +
        list(Path(args.lakhmidi_dir).rglob('*.midi'))
    )
    print(f"Found {len(midi_files)} MIDI files in {args.lakhmidi_dir}")

    stats = {
        'total': len(midi_files),
        'saved': 0,
        'skipped_not_44': 0,
        'skipped_tempo_changes': 0,
        'skipped_no_drum': 0,
        'skipped_no_valid_bars': 0,
        'skipped_too_short': 0,
        'skipped_parse_error': 0,
        'skipped_other': 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for midi_path in tqdm(midi_files):
        result, reason = clean_midi(midi_path)

        if result is None:
            if 'not 4/4' in reason:
                stats['skipped_not_44'] += 1
            elif 'tempo changes' in reason:
                stats['skipped_tempo_changes'] += 1
            elif 'no drum' in reason:
                stats['skipped_no_drum'] += 1
            elif 'parse error' in reason:
                stats['skipped_parse_error'] += 1
            elif 'no valid bars' in reason:
                stats['skipped_no_valid_bars'] += 1
            elif 'too few bars' in reason:
                stats['skipped_too_short'] += 1
            else:
                stats['skipped_other'] += 1
            continue

        bars, tempos, ticks_per_beat = result
        output_path = resolve_output_path(args.output_dir, midi_path)

        try:
            save_drum_midi(bars, tempos, ticks_per_beat, output_path)
            stats['saved'] += 1
        except Exception as e:
            print(f"\nSave error {midi_path}: {e}")
            stats['skipped_other'] += 1

    print("\n" + "=" * 60)
    print("Cleaning Summary")
    print("=" * 60)
    print(f"Total:                   {stats['total']}")
    print(f"Saved:                   {stats['saved']}")
    print(f"Skipped (not 4/4):       {stats['skipped_not_44']}")
    print(f"Skipped (tempo changes): {stats['skipped_tempo_changes']}")
    print(f"Skipped (no drum):       {stats['skipped_no_drum']}")
    print(f"Skipped (no valid bars): {stats['skipped_no_valid_bars']}")
    print(f"Skipped (<=16 bars):     {stats['skipped_too_short']}")
    print(f"Skipped (parse error):   {stats['skipped_parse_error']}")
    print(f"Skipped (other):         {stats['skipped_other']}")


if __name__ == '__main__':
    main()
