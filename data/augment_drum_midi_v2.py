"""
ドラム MIDI データ増強スクリプト v2

増強手法:
  1. ハイハット頻度変更 × BPM 変更
       HH_CLOSED(42) または HH_OPEN(46) の合計が 4分/8分/16分の一定リズムを
       刻んでいる小節において、そのハイハット音符のみを別の密度に置き換える。
       HH_PEDAL(44)・キック・スネア等のハイハット以外は一切変更しない。

  2. 打撃場所変更 × BPM 変更
       HH_CLOSED / HH_OPEN / シンバル / ライド / フロアタムのいずれかが
       一定リズムを刻んでいる小節において、そのグループの音符のみピッチを置換する。
       一定リズムを刻んでいないノート（アクセント等）は変更しない。

  3. BPM 変更: 上記 1・2 の各バリアントにランダムな BPM オフセットを付与する。

使用方法:
    python data/augment_drum_midi_v2.py \
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
DEFAULT_BEAT_RESOL = 480
GRID_STEP = DEFAULT_BEAT_RESOL // POSITIONS_PER_BEAT   # 20 ticks (24 分割グリッド)

# BPM サンプリング候補（トークナイザの語彙と一致: 60〜240、5 刻み）
BPM_CHOICES = list(range(60, 245, 5))

# ── ハイハット頻度変更 ─────────────────────────────
# 検出・削除対象: HH_CLOSED + HH_OPEN のみ (HH_PEDAL=44 は除外)
HH_FREQ_PITCHES  = frozenset({42, 46})
HH_FREQ_NEW_PITCH = 42                 # 新規グリッドノートは HH_CLOSED に統一

# ── 打撃場所変更 ──────────────────────────────────
# 検出対象の楽器グループ (グループ単位で一定リズム判定)
LOC_GROUPS: list[tuple[str, frozenset]] = [
    ("hh_closed", frozenset({42})),
    ("hh_open",   frozenset({46})),
    ("crash",     frozenset({49, 57})),
    ("ride",      frozenset({51, 59})),
    ("floor",     frozenset({41, 43})),
]

# 打撃場所変更の置換先ピッチ
LOC_CANONICAL: list[tuple[str, int]] = [
    ("hh_closed", 42),
    ("hh_open",   46),
    ("crash",     49),
    ("ride",      51),
    ("floor",     43),
]


# ─────────────────────────────────────────────
# mido ヘルパー
# ─────────────────────────────────────────────

def _msgs_to_abs(track: mido.MidiTrack) -> list[tuple[int, mido.Message]]:
    """トラックの全メッセージを (絶対 tick, msg) リストに変換する"""
    result, t = [], 0
    for msg in track:
        t += msg.time
        result.append((t, msg))
    return result


def _abs_to_delta(abs_msgs: list[tuple[int, mido.Message]]) -> list[mido.Message]:
    """(絶対 tick, msg) リストをデルタ時刻メッセージリストに変換する。
    同一 tick では note_off を note_on より先に配置する。"""
    def _sort_key(item: tuple[int, mido.Message]) -> tuple[int, int]:
        t, msg = item
        is_note_off = (
            not msg.is_meta
            and (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))
        )
        return (t, 0 if is_note_off else 1)

    result, prev_t = [], 0
    for t, msg in sorted(abs_msgs, key=_sort_key):
        result.append(msg.copy(time=t - prev_t))
        prev_t = t
    return result


def is_drum_track(track: mido.MidiTrack) -> bool:
    """channel 9 の note_on/note_off を含むトラックをドラムトラックとみなす"""
    return any(
        not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9
        for msg in track
    )


# ─────────────────────────────────────────────
# TPB 正規化・グリッドスナップ
# ─────────────────────────────────────────────

def _quantization_error(note_starts: list[int], tpb: int) -> float:
    """24 分割グリッドへの平均量子化誤差を計算する"""
    if tpb <= 0 or not note_starts:
        return float('inf')
    errors = []
    for s in note_starts:
        pf = (s % tpb) * POSITIONS_PER_BEAT / tpb
        errors.append(abs(pf - round(pf)))
    return sum(errors) / len(errors)


def _infer_effective_tpb(declared_tpb: int, note_starts: list[int]) -> int:
    """DrumTokenizer と同一ロジックで実効 ticks_per_beat を推定する"""
    declared_tpb = declared_tpb if declared_tpb > 0 else DEFAULT_BEAT_RESOL
    if not note_starts:
        return declared_tpb
    candidates = [declared_tpb] + [t for t in (120, 240, 480, 960, 1920) if t != declared_tpb]
    declared_err = _quantization_error(note_starts, declared_tpb)
    best_tpb, best_err = declared_tpb, declared_err
    for tpb in candidates:
        err = _quantization_error(note_starts, tpb)
        if err < best_err:
            best_err, best_tpb = err, tpb
    if best_tpb != declared_tpb and (declared_err - best_err) >= 0.05:
        return best_tpb
    return declared_tpb


def normalize_and_snap(midi_obj: mido.MidiFile) -> None:
    """全ドラムノートを DEFAULT_BEAT_RESOL=480 に正規化し 24 分割グリッドにスナップする (in-place)"""
    all_starts: list[int] = []
    for track in midi_obj.tracks:
        if not is_drum_track(track):
            continue
        t = 0
        for msg in track:
            t += msg.time
            if not msg.is_meta and msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                all_starts.append(t)

    if not all_starts:
        return

    scale = DEFAULT_BEAT_RESOL / _infer_effective_tpb(midi_obj.ticks_per_beat, all_starts)

    for track in midi_obj.tracks:
        abs_msgs = _msgs_to_abs(track)
        if is_drum_track(track):
            new_abs = []
            for t, msg in abs_msgs:
                if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                    snapped = round(t * scale / GRID_STEP) * GRID_STEP
                    new_abs.append((snapped, msg))
                else:
                    new_abs.append((int(t * scale), msg))
        else:
            new_abs = [(int(t * scale), msg) for t, msg in abs_msgs]
        track[:] = _abs_to_delta(new_abs)

    midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def get_drum_note_events(midi_obj: mido.MidiFile) -> list[tuple[int, int, int]]:
    """全ドラムトラックの note_on を (abs_tick, pitch, velocity) の昇順リストで返す"""
    events: list[tuple[int, int, int]] = []
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


def get_bar_count(drum_events: list[tuple[int, int, int]], ticks_per_bar: int) -> int:
    if not drum_events:
        return 0
    return drum_events[-1][0] // ticks_per_bar + 1


def detect_steady_rhythm(
    events: list[tuple[int, int, int]],
    bar_start: int,
    tpb: int,
) -> str | None:
    """
    events が 16分/8分/4分グリッドの全ポジションを過不足なく埋めているか判定する。
    細かい方から検査し、最初に一致したグリッド名を返す。一致なしは None。

    粗い方から検査すると 8 分音符パターンが quarter スナップで 4 点に集約されて
    'quarter' と誤検出されるため、細かい方（sixteenth）から順に検査する。
    """
    if not events:
        return None
    tpbar = tpb * 4
    raw_ticks = [e[0] - bar_start for e in events]

    for interval, expected, name in [
        (tpb // 4, 16, 'sixteenth'),
        (tpb // 2,  8, 'eighth'),
        (tpb,       4, 'quarter'),
    ]:
        snapped: set[int] = set()
        for t in raw_ticks:
            g = round(t / interval) * interval
            snapped.add(max(0, min(tpbar - interval, g)))
        if len(snapped) == expected and all(g % interval == 0 for g in snapped):
            return name

    return None


def make_grid_msgs(
    bar_start: int, grid: str, tpb: int, pitch: int, velocity: int
) -> list[tuple[int, mido.Message]]:
    """指定グリッドで 1 小節分の (abs_tick, Message) ペアを生成する (note_on + note_off)"""
    tpbar = tpb * 4
    step  = {'quarter': tpb, 'eighth': tpb // 2, 'sixteenth': tpb // 4}[grid]
    dur   = max(1, tpb // 24)
    msgs: list[tuple[int, mido.Message]] = []
    tick = bar_start
    while tick < bar_start + tpbar:
        msgs.append((tick,       mido.Message('note_on',  channel=9, note=pitch, velocity=velocity, time=0)))
        msgs.append((tick + dur, mido.Message('note_off', channel=9, note=pitch, velocity=0,        time=0)))
        tick += step
    return msgs


def apply_random_bpm(midi_obj: mido.MidiFile, rng: random.Random) -> None:
    """全 set_tempo に同一のランダムオフセットを加算する。情報がなければランダム挿入。"""
    bpms: list[float] = []
    for track in midi_obj.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                bpms.append(mido.tempo2bpm(msg.tempo))

    if not bpms:
        midi_obj.tracks[0].insert(
            0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(rng.choice(BPM_CHOICES)), time=0)
        )
        return

    lo = math.ceil(60.0  - min(bpms))
    hi = math.floor(240.0 - max(bpms))
    if lo > hi:
        return

    multiples_of_5 = [o for o in range(lo, hi + 1) if o % 5 == 0]
    offset = rng.choice(multiples_of_5 if multiples_of_5 else list(range(lo, hi + 1)))

    for track in midi_obj.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                new_bpm = mido.tempo2bpm(msg.tempo) + offset
                new_track.append(
                    mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(new_bpm), time=msg.time)
                )
            else:
                new_track.append(msg)
        track[:] = new_track


def make_output_path(
    input_path: Path, input_root: Path, output_root: Path, suffix: str
) -> Path:
    rel = input_path.relative_to(input_root)
    return output_root / rel.parent / f"{input_path.stem}_{suffix}{input_path.suffix}"


# ─────────────────────────────────────────────
# 増強関数
# ─────────────────────────────────────────────

def augment_hihat_freq_with_bpm(
    midi_obj: mido.MidiFile,
    n_bpm_versions: int,
    rng: random.Random,
) -> list[tuple[mido.MidiFile, str]]:
    """
    ハイハット頻度変更 × BPM 変更。

    検出: 各小節の HH_CLOSED(42) と HH_OPEN(46) の音符を合算し、
          4分/8分/16分のいずれか一定リズムを形成しているか確認する。

    変換: qualifying 小節のハイハット音符（42/46）を削除し、
          ターゲット密度の HH_CLOSED ノートを追加する。
          HH_PEDAL(44)・キック・スネア等は一切変更しない。

    元リズムと同一ターゲットグリッドの小節はハイハットを置換せず、
    BPM 変更のみ適用する（ターゲットが異なる小節のみ置換）。

    戻り値: list of (mido.MidiFile copy, suffix_str)
    """
    tpb   = midi_obj.ticks_per_beat
    tpbar = tpb * 4
    drum_events = get_drum_note_events(midi_obj)
    n_bars = get_bar_count(drum_events, tpbar)
    if n_bars == 0:
        return []

    # qualifying: bar_start -> (detected_rhythm, median_velocity)
    # 合算した HH_CLOSED + HH_OPEN が一定リズムを形成する小節のみ収集
    qualifying: dict[int, tuple[str, int]] = {}
    for bar_idx in range(n_bars):
        bar_start = bar_idx * tpbar
        bar_end   = bar_start + tpbar
        hh_events = [
            e for e in drum_events
            if e[1] in HH_FREQ_PITCHES and bar_start <= e[0] < bar_end
        ]
        if hh_events:
            rhythm = detect_steady_rhythm(hh_events, bar_start, tpb)
            if rhythm is not None:
                vel = int(statistics.median(e[2] for e in hh_events))
                qualifying[bar_start] = (rhythm, max(1, min(127, vel)))

    if not qualifying:
        return []

    results: list[tuple[mido.MidiFile, str]] = []
    for target_grid in ('quarter', 'eighth', 'sixteenth'):
        # 元リズムが target_grid と異なる小節のみ置換（同一の場合は BPM のみ変更）
        convert_bars = {
            bs: vel
            for bs, (rhythm, vel) in qualifying.items()
            if rhythm != target_grid
        }

        for bpm_idx in range(1, n_bpm_versions + 1):
            midi_c = copy.deepcopy(midi_obj)

            for track in midi_c.tracks:
                if not is_drum_track(track):
                    continue

                abs_msgs = _msgs_to_abs(track)
                new_abs: list[tuple[int, mido.Message]] = []

                for t, msg in abs_msgs:
                    if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                        bar_start = (t // tpbar) * tpbar
                        # convert_bars の小節の HH のみ削除
                        if msg.note in HH_FREQ_PITCHES and bar_start in convert_bars:
                            continue
                    new_abs.append((t, msg))

                # 変換対象小節に新規グリッドノートを追加
                for bar_start, vel in convert_bars.items():
                    new_abs.extend(make_grid_msgs(bar_start, target_grid, tpb, HH_FREQ_NEW_PITCH, vel))

                track[:] = _abs_to_delta(new_abs)

            apply_random_bpm(midi_c, rng)
            results.append((midi_c, f"hh_{target_grid}_bpm{bpm_idx}"))

    return results


def augment_location_with_bpm(
    midi_obj: mido.MidiFile,
    n_bpm_versions: int,
    rng: random.Random,
) -> list[tuple[mido.MidiFile, str]]:
    """
    打撃場所変更 × BPM 変更。

    検出: LOC_GROUPS（HH_CLOSED / HH_OPEN / シンバル / ライド / フロアタム）を
          グループ単位で独立に検査し、一定リズムを形成するグループを小節ごとに特定する。

    変換: qualifying グループの音符のみを削除し、同一タイミング・ベロシティで
          canonical ピッチのノートを追加する。
          一定リズムを形成していない音符（アクセント等）は変更しない。

    戻り値: list of (mido.MidiFile copy, suffix_str)
    """
    tpb   = midi_obj.ticks_per_beat
    tpbar = tpb * 4
    drum_events = get_drum_note_events(midi_obj)
    n_bars = get_bar_count(drum_events, tpbar)
    if n_bars == 0:
        return []

    # bar_replace_pitches: bar_start -> set of pitches を置換対象として収集
    # 各グループが独立に一定リズムを形成していれば、そのピッチ全体が対象になる
    bar_replace_pitches: dict[int, set[int]] = {}
    for bar_idx in range(n_bars):
        bar_start = bar_idx * tpbar
        bar_end   = bar_start + tpbar
        for _grp_name, grp_pitches in LOC_GROUPS:
            grp_events = [
                e for e in drum_events
                if e[1] in grp_pitches and bar_start <= e[0] < bar_end
            ]
            if grp_events and detect_steady_rhythm(grp_events, bar_start, tpb) is not None:
                if bar_start not in bar_replace_pitches:
                    bar_replace_pitches[bar_start] = set()
                bar_replace_pitches[bar_start].update(grp_pitches)

    if not bar_replace_pitches:
        return []

    dur = max(1, tpb // 24)
    results: list[tuple[mido.MidiFile, str]] = []

    for canonical_name, canonical_pitch in LOC_CANONICAL:
        for bpm_idx in range(1, n_bpm_versions + 1):
            midi_c = copy.deepcopy(midi_obj)

            for track in midi_c.tracks:
                if not is_drum_track(track):
                    continue

                abs_msgs = _msgs_to_abs(track)
                new_abs: list[tuple[int, mido.Message]] = []
                replaced: list[tuple[int, int]] = []   # (abs_tick, velocity)

                for t, msg in abs_msgs:
                    if not msg.is_meta and msg.type in ('note_on', 'note_off') and msg.channel == 9:
                        bar_start = (t // tpbar) * tpbar
                        replace_pitches = bar_replace_pitches.get(bar_start, set())
                        if msg.note in replace_pitches:
                            # qualifying グループの音符を削除し、note_on は後で置換追加
                            if msg.type == 'note_on' and msg.velocity > 0:
                                replaced.append((t, msg.velocity))
                            continue
                    new_abs.append((t, msg))

                # 同一タイミング・ベロシティで canonical ピッチのノートを追加
                for t, vel in replaced:
                    new_abs.append((t,       mido.Message('note_on',  channel=9, note=canonical_pitch, velocity=vel, time=0)))
                    new_abs.append((t + dur, mido.Message('note_off', channel=9, note=canonical_pitch, velocity=0,   time=0)))

                track[:] = _abs_to_delta(new_abs)

            apply_random_bpm(midi_c, rng)
            results.append((midi_c, f"loc_{canonical_name}_bpm{bpm_idx}"))

    return results


# ─────────────────────────────────────────────
# ファイル単位処理
# ─────────────────────────────────────────────

def process_file(
    midi_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
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
        variants = augment_hihat_freq_with_bpm(midi_obj, args.n_bpm_versions, rng)
        if not variants:
            stats['hh_skipped'] += 1
        for midi_v, suffix in variants:
            out = make_output_path(midi_path, input_root, output_root, suffix)
            out.parent.mkdir(parents=True, exist_ok=True)
            midi_v.save(str(out))
            stats['hh_saved'] += 1

    if args.augment_location:
        variants = augment_location_with_bpm(midi_obj, args.n_bpm_versions, rng)
        if not variants:
            stats['loc_skipped'] += 1
        for midi_v, suffix in variants:
            out = make_output_path(midi_path, input_root, output_root, suffix)
            out.parent.mkdir(parents=True, exist_ok=True)
            midi_v.save(str(out))
            stats['loc_saved'] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='ドラム MIDI データ増強スクリプト v2')
    parser.add_argument('--input_dir',       required=True, help='入力 MIDI ディレクトリ')
    parser.add_argument('--output_dir',      required=True, help='出力 MIDI ディレクトリ')
    parser.add_argument('--augment_hihat',   action='store_true', help='ハイハット頻度変更を有効化')
    parser.add_argument('--augment_location',action='store_true', help='打撃場所変更を有効化')
    parser.add_argument('--n_bpm_versions',  type=int, default=3, help='BPM バリアント数 (default: 3)')
    parser.add_argument('--seed',            type=int, default=42, help='乱数シード (default: 42)')
    args = parser.parse_args()

    if not args.augment_hihat and not args.augment_location:
        parser.error('--augment_hihat または --augment_location の少なくとも一方を指定してください')

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(
        list(input_root.rglob('*.mid')) + list(input_root.rglob('*.midi'))
    )
    print(f"Found {len(midi_files)} MIDI files in {input_root}")

    total = {
        'total': len(midi_files), 'hh_saved': 0, 'loc_saved': 0,
        'hh_skipped': 0, 'loc_skipped': 0, 'errors': 0,
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
    print(f"Total files:                {total['total']}")
    print(f"Hihat freq variants saved:  {total['hh_saved']}  (skipped: {total['hh_skipped']} files)")
    print(f"Location variants saved:    {total['loc_saved']}  (skipped: {total['loc_skipped']} files)")
    print(f"Parse errors:               {total['errors']}")


if __name__ == '__main__':
    main()
