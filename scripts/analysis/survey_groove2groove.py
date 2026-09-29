"""Groove2Groove アーカイブを展開せず、収録構成と MIDI サンプルを調査する。"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import tarfile

import mido

from src.tokenizers.drum_tokenizer import DRUM_NOTE_MAP


def summarize_midi(payload, name):
    midi = mido.MidiFile(file=io.BytesIO(payload))
    notes = []
    signatures = set()
    tempos = set()
    programs = set()
    track_names = []
    last_tick = 0
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'time_signature':
                signatures.add(f'{msg.numerator}/{msg.denominator}')
            elif msg.type == 'set_tempo':
                tempos.add(round(mido.tempo2bpm(msg.tempo), 3))
            elif msg.type == 'program_change' and msg.channel != 9:
                programs.add(msg.program)
            elif msg.type == 'track_name':
                track_names.append(msg.name)
            elif msg.type == 'note_on' and msg.velocity:
                notes.append((tick, msg.channel, msg.note, msg.velocity))
        last_tick = max(last_tick, tick)
    # 公式仕様の先頭2小節を除く。4/4以外の場合は拍子に合わせた精査が必要。
    count_in_ticks = 8 * midi.ticks_per_beat
    drum = [(t - count_in_ticks, p, v) for t, ch, p, v in notes if ch == 9 and t >= count_in_ticks]
    non_drum = sum(ch != 9 and t >= count_in_ticks for t, ch, p, v in notes)
    pitches = Counter(p for t, p, v in drum)
    unsupported = {str(p): n for p, n in pitches.items() if p not in DRUM_NOTE_MAP}
    bars = defaultdict(list)
    for tick, pitch, velocity in drum:
        bar, position = divmod(tick, 4 * midi.ticks_per_beat)
        bars[bar].append((position, pitch))
    patterns = Counter(tuple(sorted(v)) for v in bars.values())
    # 完全一致の判定。音長は含めず、開始位置・ピッチ・強弱を対象とする。
    canonical = [(t / midi.ticks_per_beat, p, v) for t, p, v in sorted(drum)]
    fingerprint = hashlib.sha256(json.dumps(canonical).encode()).hexdigest() if drum else None
    return {
        'file': name, 'ticks_per_beat': midi.ticks_per_beat,
        'time_signatures': sorted(signatures), 'tempos_bpm': sorted(tempos),
        'tracks': len(midi.tracks), 'track_names': track_names,
        'non_drum_programs_0based': sorted(programs),
        'quarter_notes_after_count_in': max(0, last_tick - count_in_ticks) / midi.ticks_per_beat,
        'drum_notes_in_count_in': sum(ch == 9 and t < count_in_ticks for t, ch, p, v in notes),
        'drum_notes_after_count_in': len(drum), 'non_drum_notes_after_count_in': non_drum,
        'pitches': dict(sorted(pitches.items())), 'unsupported_pitches': unsupported,
        'velocity_min': min((v for t, p, v in drum), default=None),
        'velocity_max': max((v for t, p, v in drum), default=None),
        'velocity_values': len(set(v for t, p, v in drum)),
        'off_24_grid_notes': sum((t * 24) % midi.ticks_per_beat != 0 for t, p, v in drum),
        'active_4quarter_bars': len(bars), 'unique_exact_rhythm_bars': len(patterns),
        'most_common_rhythm_bar_count': patterns.most_common(1)[0][1] if patterns else 0,
        'drum_fingerprint': fingerprint,
        'first_drum_bar': [[pos / midi.ticks_per_beat, pitch] for pos, pitch in sorted(bars[min(bars)])] if bars else [],
    }


def audit(archive, samples_per_split):
    counts = Counter()
    selected = defaultdict(list)
    metadata = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if member.name.endswith('.tsv') or Path(member.name).name.lower().startswith(('readme', 'license')):
                metadata[member.name] = tar.extractfile(member).read().decode('utf-8', errors='replace')
            if len(parts) < 5 or parts[1] != 'midi' or not member.name.lower().endswith('.mid'):
                continue
            split, version = parts[2:4]
            counts[f'{split}/{version}'] += 1
            if version != 'fixed':
                continue
            # ファイル名の SHA-256 が小さい順。処理順に依存しないサンプル選択。
            rank = hashlib.sha256(member.name.encode()).hexdigest()
            bucket = selected[split]
            if len(bucket) < samples_per_split or rank < bucket[-1][0]:
                payload = tar.extractfile(member).read()
                bucket.append((rank, member.name, payload))
                bucket.sort(key=lambda x: x[0])
                del bucket[samples_per_split:]
    results = []
    for split, entries in sorted(selected.items()):
        for rank, name, payload in entries:
            try:
                results.append(dict(split=split, **summarize_midi(payload, name)))
            except Exception as exc:
                results.append({'file': name, 'split': split, 'error': str(exc)})
        print(f'Parsed {split}: {len(entries)} samples', flush=True)
    styles_name = next((name for name in metadata if name.endswith('/styles.tsv')), None)
    styles = list(csv.DictReader(io.StringIO(metadata[styles_name]), delimiter='\t')) if styles_name else []
    valid = [r for r in results if 'error' not in r]
    summary = {}
    for split in sorted(selected):
        rows = [r for r in valid if r['split'] == split]
        summary[split] = {
            'parsed': len(rows), 'with_musical_drums': sum(r['drum_notes_after_count_in'] > 0 for r in rows),
            'without_musical_drums': sum(r['drum_notes_after_count_in'] == 0 for r in rows),
            'with_unsupported_pitches': sum(bool(r['unsupported_pitches']) for r in rows),
            'drum_notes': sum(r['drum_notes_after_count_in'] for r in rows),
            'off_24_grid_notes': sum(r['off_24_grid_notes'] for r in rows),
        }
    return {'archive': str(archive), 'sampling': 'lowest SHA-256 of member names per fixed split',
            'samples_per_split': samples_per_split, 'counts': dict(sorted(counts.items())),
            'style_rows': len(styles),
            'style_distributions': {key: dict(Counter(row[key] for row in styles)) for key in ['TSig', 'EvSw', 'Genre', 'Group']} if styles else {},
            'metadata_files': sorted(metadata), 'sample_summary': summary, 'samples': results}, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--samples_per_split', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error('--samples_per_split は1以上にしてください')
    report, metadata = audit(args.archive, args.samples_per_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False, indent=2))
    for name, contents in metadata.items():
        if Path(name).name.lower().startswith(('readme', 'license')):
            dest = args.output.parent / ('archive_' + Path(name).name)
            dest.write_text(contents)


if __name__ == '__main__':
    main()
