"""
ドラム MIDI データ増強スクリプト

増強手法:
  1. ハイハット頻度変更 × BPM 変更:
     一定のリズムを刻む小節のハイハットパターンを 4分音符/8分音符/16分音符に変更し、
     それぞれ N 種類のランダム BPM と組み合わせる
  2. 打撃場所変更 × BPM 変更:
     一定のリズムを刻む小節のハイハット・シンバル系・フロアタムを
     5種類の楽器に置換し、それぞれ N 種類のランダム BPM と組み合わせる

使用方法:
    python data/augment_drum_midi.py \
        --input_dir  data/raw/test \
        --output_dir data/filtered/test \
        --augment_hihat --augment_location \
        --n_bpm_versions 3 --seed 42
"""

import copy
import math
import random
import statistics
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import mido

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.tokenization_policy import POSITIONS_PER_BEAT

# トークナイザと共通の定数
DEFAULT_BEAT_RESOL = 480                         # 正規化後の ticks_per_beat
GRID_STEP = DEFAULT_BEAT_RESOL // POSITIONS_PER_BEAT  # = 20 ticks（24分割グリッド1ステップ）

# BPM サンプリング候補（トークナイザの語彙と一致: 60〜240、5刻み）
BPM_CHOICES = list(range(60, 245, 5))

# ハイハット頻度変更の対象ピッチ
HH_PITCHES      = frozenset({42, 44, 46})   # HH_CLOSED, HH_PEDAL, HH_OPEN
HH_CLOSED_PITCH = 42                          # 新規ノートに使う標準ピッチ

# 打撃場所変更の対象ピッチ (FLOOR, HH_CLOSED, HH_OPEN, CRASH, RIDE_CYMBAL)
LOC_TARGET_PITCHES = frozenset({41, 42, 43, 46, 49, 51, 57, 59})

# 打撃場所バリアント: (サフィックス, 置換後の標準ピッチ)
LOC_CANONICAL = [
    ("hh_closed", 42),
    ("hh_open",   46),
    ("crash",     49),
    ("ride",      51),
    ("floor",     43),
]


# ─────────────────────────────────────────────
# mido ヘルパー
# ─────────────────────────────────────────────

def _msgs_to_abs(track):
    """トラックのメッセージを (絶対tick, msg) リストに変換"""
    result = []
    t = 0
    for msg in track:
        t += msg.time
        result.append((t, msg))
    return result


def _abs_to_delta(abs_msgs):
    """(絶対tick, msg) リストをデルタ時刻メッセージリストに変換（時刻昇順ソート）。
    同一 tick では note_off を note_on より先に配置する。"""
    def sort_key(item):
        t, msg = item
        # note_off (velocity==0 の note_on 含む) を優先
        if not msg.is_meta and msg.type == 'note_off':
            priority = 0
        elif not msg.is_meta and msg.type == 'note_on' and msg.velocity == 0:
            priority = 0
        else:
            priority = 1
        return (t, priority)

    abs_msgs = sorted(abs_msgs, key=sort_key)
    result = []
    prev_t = 0
    for t, msg in abs_msgs:
        result.append(msg.copy(time=t - prev_t))
        prev_t = t
    return result


def is_drum_track(track):
    """チャンネル9の note_on/note_off を含むトラックをドラムトラックとみなす"""
    return any(
        not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9
        for msg in track
    )


# ─────────────────────────────────────────────
# TPB 正規化・グリッドスナップ（トークナイザと同一ロジック）
# ─────────────────────────────────────────────

def _quantization_error(note_starts: list, tpb: int) -> float:
    """24 分割グリッドへの平均量子化誤差を計算"""
    if tpb <= 0 or not note_starts:
        return float('inf')
    errors = []
    for start in note_starts:
        tick_in_beat = start % tpb
        pos_float = (tick_in_beat * POSITIONS_PER_BEAT) / tpb
        errors.append(abs(pos_float - round(pos_float)))
    return sum(errors) / len(errors)


def _infer_effective_tpb(declared_tpb: int, note_starts: list) -> int:
    """実効 ticks_per_beat を推定する。DrumTokenizer と同一ロジック。"""
    declared_tpb = declared_tpb if declared_tpb > 0 else DEFAULT_BEAT_RESOL
    if not note_starts:
        return declared_tpb

    candidates = [declared_tpb]
    for tpb in (120, 240, 480, 960, 1920):
        if tpb not in candidates:
            candidates.append(tpb)

    declared_error = _quantization_error(note_starts, declared_tpb)
    best_tpb, best_error = declared_tpb, declared_error

    for candidate in candidates:
        err = _quantization_error(note_starts, candidate)
        if err < best_error:
            best_error = err
            best_tpb = candidate

    if best_tpb != declared_tpb and (declared_error - best_error) >= 0.05:
        return best_tpb
    return declared_tpb


def normalize_and_snap(midi_obj: mido.MidiFile) -> None:
    """
    全ドラムノートを DEFAULT_BEAT_RESOL=480 に正規化し、
    24 分割グリッドの最近傍 tick に丸める（in-place）。
    """
    # ドラムトラックの全 note_on の絶対 tick を収集
    all_note_starts = []
    for track in midi_obj.tracks:
        if not is_drum_track(track):
            continue
        t = 0
        for msg in track:
            t += msg.time
            if not msg.is_meta and msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                all_note_starts.append(t)

    if not all_note_starts:
        return

    effective_tpb = _infer_effective_tpb(midi_obj.ticks_per_beat, all_note_starts)
    scale_factor = DEFAULT_BEAT_RESOL / effective_tpb

    for track in midi_obj.tracks:
        abs_msgs = _msgs_to_abs(track)
        new_abs = []
        if is_drum_track(track):
            for t, msg in abs_msgs:
                if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                    snapped = round(t * scale_factor / GRID_STEP) * GRID_STEP
                    new_abs.append((snapped, msg))
                else:
                    new_abs.append((int(t * scale_factor), msg))
        else:
            for t, msg in abs_msgs:
                new_abs.append((int(t * scale_factor), msg))
        track[:] = _abs_to_delta(new_abs)

    midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def get_drum_note_events(midi_obj: mido.MidiFile) -> list:
    """全ドラムトラックの note_on を (abs_tick, pitch, velocity) のリストで返す（時刻昇順）"""
    events = []
    for track in midi_obj.tracks:
        if not is_drum_track(track):
            continue
        t = 0
        for msg in track:
            t += msg.time
            if not msg.is_meta and msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                events.append((t, msg.note, msg.velocity))
    events.sort(key=lambda x: x[0])
    return events


def get_bar_count(drum_events: list, ticks_per_bar: int) -> int:
    if not drum_events:
        return 0
    return drum_events[-1][0] // ticks_per_bar + 1


def detect_steady_rhythm(events: list, bar_start: int, tpb: int):
    """
    events が sixteenth/eighth/quarter グリッドの全ポジションを過不足なく埋めているか判定。
    events は (abs_tick, pitch, velocity) のリスト。
    細かい方から検査し、最初に一致した名称を返す。一致なしは None。

    粗い方から検査すると、8分音符パターンが quarter スナップで 4点に集約されて
    'quarter' と誤検出される。細かい方から検査することで正しく 'eighth' を返す。
    """
    if not events:
        return None
    tpbar = tpb * 4
    raw_ticks = [e[0] - bar_start for e in events]

    for interval, expected_count, name in [
        (tpb // 4, 16, 'sixteenth'),  # 細かい方から検査
        (tpb // 2, 8,  'eighth'),
        (tpb,      4,  'quarter'),
    ]:
        snapped = set()
        for t in raw_ticks:
            grid_pos = round(t / interval) * interval
            grid_pos = max(0, min(tpbar - interval, grid_pos))
            snapped.add(grid_pos)
        if len(snapped) == expected_count and all(s % interval == 0 for s in snapped):
            return name

    return None


def make_grid_note_msgs(bar_start: int, grid: str, tpb: int, pitch: int, velocity: int) -> list:
    """指定グリッドで 1小節分の (abs_tick, mido.Message) ペアリストを生成（note_on + note_off）"""
    ticks_per_bar = tpb * 4
    step = {'quarter': tpb, 'eighth': tpb // 2, 'sixteenth': tpb // 4}[grid]
    dur  = max(1, tpb // 24)
    msgs = []
    tick = bar_start
    while tick < bar_start + ticks_per_bar:
        msgs.append((tick, mido.Message('note_on',  channel=9, note=pitch, velocity=velocity, time=0)))
        msgs.append((tick + dur, mido.Message('note_off', channel=9, note=pitch, velocity=0, time=0)))
        tick += step
    return msgs


def apply_random_bpm(midi_obj: mido.MidiFile, rng: random.Random) -> None:
    """
    全 set_tempo メタメッセージに同一のランダムオフセットを加算する。
    tempo_changes が空の場合は BPM_CHOICES からランダムに 1 件挿入する。
    """
    # 全トラックの set_tempo を収集
    tempo_vals = []
    for track in midi_obj.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                tempo_vals.append(mido.tempo2bpm(msg.tempo))

    if not tempo_vals:
        bpm = rng.choice(BPM_CHOICES)
        midi_obj.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        return

    bpm_min = min(tempo_vals)
    bpm_max = max(tempo_vals)

    lo = math.ceil(60.0 - bpm_min)
    hi = math.floor(240.0 - bpm_max)

    if lo > hi:
        return

    step5 = [o for o in range(lo, hi + 1) if o % 5 == 0]
    choices = step5 if step5 else list(range(lo, hi + 1))
    offset = rng.choice(choices)

    for track in midi_obj.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                new_bpm = mido.tempo2bpm(msg.tempo) + offset
                new_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(new_bpm), time=msg.time))
            else:
                new_track.append(msg)
        track[:] = new_track


def make_output_path(input_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    rel = input_path.relative_to(input_root)
    return output_root / rel.parent / f"{input_path.stem}_{suffix}{input_path.suffix}"


# ─────────────────────────────────────────────
# 増強関数
# ─────────────────────────────────────────────

def augment_hihat_with_bpm(
    midi_obj: mido.MidiFile,
    n_bpm_versions: int,
    rng: random.Random,
) -> list:
    """
    ハイハット頻度変更 × BPM 変更。
    qualifying bars の HH note_on/note_off を削除し、新規グリッドノートを追加。
    元譜面の他の音符はそのまま保持される。

    各小節の検出済みリズムを記録し、ターゲットグリッドがそのリズムと異なる小節のみを
    変換する。同一の場合は既存 HH を取り除かず、BPM 変更のみを適用する。
    （例: 8分音符小節 → sixteenth バリアント で 16分音符化 = 8ビート→16ビート変換）

    戻り値: list of (midi_obj_copy, suffix_str)
    """
    tpb   = midi_obj.ticks_per_beat
    tpbar = tpb * 4
    drum_events = get_drum_note_events(midi_obj)
    n_bars = get_bar_count(drum_events, tpbar)
    if n_bars == 0:
        return []

    # qualifying bars を特定（HH が一定グリッドに乗っている小節）
    # qualifying: bar_start -> (detected_rhythm, median_velocity)
    qualifying = {}
    for bar_idx in range(n_bars):
        bar_start = bar_idx * tpbar
        bar_end   = bar_start + tpbar
        hh_in_bar = [
            e for e in drum_events
            if e[1] in HH_PITCHES and bar_start <= e[0] < bar_end
        ]
        if hh_in_bar:
            rhythm = detect_steady_rhythm(hh_in_bar, bar_start, tpb)
            if rhythm is not None:
                vel = int(statistics.median(e[2] for e in hh_in_bar))
                qualifying[bar_start] = (rhythm, max(1, min(127, vel)))

    if not qualifying:
        return []

    results = []
    for target_grid in ('quarter', 'eighth', 'sixteenth'):
        # target_grid に変換する小節: 元のリズムが target_grid と異なる小節のみ
        # 元と同一の小節は HH を削除せずそのまま保持し、BPM 変更のみ適用する
        convert_bars = {
            bs: vel
            for bs, (rhythm, vel) in qualifying.items()
            if rhythm != target_grid
        }
        keep_bars = {
            bs for bs, (rhythm, _) in qualifying.items()
            if rhythm == target_grid
        }

        for bpm_idx in range(1, n_bpm_versions + 1):
            midi_c = copy.deepcopy(midi_obj)

            for track in midi_c.tracks:
                if not is_drum_track(track):
                    continue

                abs_msgs = _msgs_to_abs(track)
                new_abs = []

                for t, msg in abs_msgs:
                    if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                        bar_start = (t // tpbar) * tpbar
                        # 変換対象の小節の HH のみ削除（keep_bars の小節はそのまま）
                        if msg.note in HH_PITCHES and bar_start in convert_bars:
                            continue
                    new_abs.append((t, msg))

                # 変換対象の小節に新規グリッドノートを追加
                for bar_start, vel in convert_bars.items():
                    new_abs.extend(make_grid_note_msgs(bar_start, target_grid, tpb, HH_CLOSED_PITCH, vel))

                track[:] = _abs_to_delta(new_abs)

            apply_random_bpm(midi_c, rng)
            results.append((midi_c, f"hh_{target_grid}_bpm{bpm_idx}"))

    return results


def augment_location_with_bpm(
    midi_obj: mido.MidiFile,
    n_bpm_versions: int,
    rng: random.Random,
) -> list:
    """
    打撃場所変更 × BPM 変更。
    qualifying bars の HH 音符を削除し、置換ピッチの音符を追加。
    元のタイミング・ベロシティを保持し、ピッチのみを変更する。

    検出は HH_PITCHES のみで行う。LOC_TARGET_PITCHES（HH/クラッシュ/ライド/フロアタム混在）
    を合算してリズム検出すると異なる楽器のタイミングが混在して均一リズムとみなされず
    ほぼすべての小節がスキップされるため。

    戻り値: list of (midi_obj_copy, suffix_str)
    """
    tpb   = midi_obj.ticks_per_beat
    tpbar = tpb * 4
    drum_events = get_drum_note_events(midi_obj)
    n_bars = get_bar_count(drum_events, tpbar)
    if n_bars == 0:
        return []

    # qualifying bars: HH が一定グリッドに乗っている小節
    qualifying_starts = set()
    for bar_idx in range(n_bars):
        bar_start = bar_idx * tpbar
        bar_end   = bar_start + tpbar
        hh_in_bar = [
            e for e in drum_events
            if e[1] in HH_PITCHES and bar_start <= e[0] < bar_end
        ]
        if hh_in_bar and detect_steady_rhythm(hh_in_bar, bar_start, tpb) is not None:
            qualifying_starts.add(bar_start)

    if not qualifying_starts:
        return []

    dur = max(1, tpb // 24)

    results = []
    for loc_name, canonical_pitch in LOC_CANONICAL:
        for bpm_idx in range(1, n_bpm_versions + 1):
            midi_c = copy.deepcopy(midi_obj)

            for track in midi_c.tracks:
                if not is_drum_track(track):
                    continue

                abs_msgs = _msgs_to_abs(track)
                new_abs = []
                # 削除した HH note_on の (abs_tick, velocity) を記録して後で追加
                replaced_note_ons = []

                for t, msg in abs_msgs:
                    if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                        bar_start = (t // tpbar) * tpbar
                        if msg.note in HH_PITCHES and bar_start in qualifying_starts:
                            # HH 音符を削除し、note_on は置換用に記録
                            if msg.type == 'note_on' and msg.velocity > 0:
                                replaced_note_ons.append((t, msg.velocity))
                            continue
                    new_abs.append((t, msg))

                # 置換ピッチの音符を追加（元のタイミング・ベロシティを保持）
                for t, vel in replaced_note_ons:
                    new_abs.append((t, mido.Message('note_on',  channel=9, note=canonical_pitch, velocity=vel, time=0)))
                    new_abs.append((t + dur, mido.Message('note_off', channel=9, note=canonical_pitch, velocity=0, time=0)))

                track[:] = _abs_to_delta(new_abs)

            apply_random_bpm(midi_c, rng)
            results.append((midi_c, f"loc_{loc_name}_bpm{bpm_idx}"))

    return results


# ─────────────────────────────────────────────
# ファイル単位処理
# ─────────────────────────────────────────────

def process_file(
    midi_path: Path,
    input_root: Path,
    output_root: Path,
    args,
    rng: random.Random,
) -> dict:
    stats = {'hh_saved': 0, 'loc_saved': 0, 'hh_skipped': 0, 'loc_skipped': 0, 'error': False}

    try:
        midi_obj = mido.MidiFile(str(midi_path))
        normalize_and_snap(midi_obj)
    except Exception as e:
        print(f"\n[WARN] Parse error {midi_path}: {e}")
        stats['error'] = True
        return stats

    if args.augment_hihat:
        variants = augment_hihat_with_bpm(midi_obj, args.n_bpm_versions, rng)
        if not variants:
            stats['hh_skipped'] += 1
        for midi_v, suffix in variants:
            out_path = make_output_path(midi_path, input_root, output_root, suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            midi_v.save(str(out_path))
            stats['hh_saved'] += 1

    if args.augment_location:
        variants = augment_location_with_bpm(midi_obj, args.n_bpm_versions, rng)
        if not variants:
            stats['loc_skipped'] += 1
        for midi_v, suffix in variants:
            out_path = make_output_path(midi_path, input_root, output_root, suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            midi_v.save(str(out_path))
            stats['loc_saved'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='ドラム MIDI データ増強スクリプト')
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='クリーニング済み MIDI ファイルのルートディレクトリ')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='増強済み MIDI の出力先ディレクトリ')
    parser.add_argument('--augment_hihat',    action='store_true',
                        help='ハイハット頻度変更 × BPM 増強を有効化')
    parser.add_argument('--augment_location', action='store_true',
                        help='打撃場所変更 × BPM 増強を有効化')
    parser.add_argument('--n_bpm_versions', type=int, default=3,
                        help='各構造バリアントに対するBPMバリアント数 (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (default: 42)')
    args = parser.parse_args()

    if not args.augment_hihat and not args.augment_location:
        parser.error('--augment_hihat または --augment_location の少なくとも一方を指定してください')

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(
        list(input_root.rglob('*.mid')) +
        list(input_root.rglob('*.midi'))
    )
    print(f"Found {len(midi_files)} MIDI files in {input_root}")

    total = {
        'total':       len(midi_files),
        'hh_saved':    0,
        'loc_saved':   0,
        'hh_skipped':  0,
        'loc_skipped': 0,
        'errors':      0,
    }

    for file_idx, midi_path in enumerate(tqdm(midi_files)):
        file_rng = random.Random(args.seed ^ file_idx)
        stats = process_file(midi_path, input_root, output_root, args, file_rng)
        total['hh_saved']    += stats['hh_saved']
        total['loc_saved']   += stats['loc_saved']
        total['hh_skipped']  += stats['hh_skipped']
        total['loc_skipped'] += stats['loc_skipped']
        if stats['error']:
            total['errors'] += 1

    print("\n" + "=" * 60)
    print("Augmentation Summary")
    print("=" * 60)
    print(f"Total files:               {total['total']}")
    print(f"Hihat variants saved:      {total['hh_saved']}  "
          f"(skipped: {total['hh_skipped']} files)")
    print(f"Location variants saved:   {total['loc_saved']}  "
          f"(skipped: {total['loc_skipped']} files)")
    print(f"Parse errors:              {total['errors']}")


if __name__ == '__main__':
    main()
