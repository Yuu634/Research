"""3データセットから原表現を残したドラムMIDIとstandardトークンを作る。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from functools import lru_cache
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import signal
import time
import tarfile

import miditoolkit
import mido

from scripts.preprocess.download_drum_sources import SOURCES
from src.tokenizers.drum_tokenizer import DrumTokenizer, DRUM_NOTE_MAP
from src.utils.tokenization_mode import write_dataset_tokenization_metadata
from src.utils.tokenization_policy import quantize_position_in_beat

# Roland固有音高のみならず、タム/シンバルの奏法も公式対応表に従う。
# HHペダルとRideベルは既存語彙が区別できるため維持する。
GROOVE_PITCH_MAP = {22: 42, 26: 46, 40: 38, 48: 50, 50: 50, 45: 47,
                    47: 47, 58: 43, 55: 49, 52: 57, 59: 51}
PROFILE = 'standard_declared_tpb_no_flam_v1'


class CollectionTokenizer(DrumTokenizer):
    """語彙を維持し、取得データの宣言TPBを尊重。推測による音符統合は行わない。"""
    def midi_to_tokens(self, midi_path):
        return self.midi_object_to_tokens(miditoolkit.MidiFile(midi_path))

    def midi_object_to_tokens(self, obj):
        if self.use_limb_tokens:
            raise ValueError('この収集プロファイルはLIMBトークンを使用しません')
        notes = [n for inst in obj.instruments if inst.is_drum for n in inst.notes]
        if not notes:
            raise ValueError('ドラムノートがありません')
        scale = 480 / obj.ticks_per_beat
        events = defaultdict(list)
        for note in notes:
            if note.pitch not in DRUM_NOTE_MAP:
                continue
            start = int(note.start * scale)
            bar, local = divmod(start, 1920)
            beat = local // 480
            position = quantize_position_in_beat(local, 480)
            drum, technique = DRUM_NOTE_MAP[note.pitch]
            velocity = self._velocity_to_level(note.velocity)
            token = f'{drum}_{technique}_{velocity}'
            if token not in self.token2idx:
                token = f'{drum}_{technique}_Normal'
            if token not in self.token2idx:
                raise ValueError(f'既存語彙に対応できない音符: {note.pitch}')
            events[bar].append((beat, position, local, token))
        n_bars = max(1, math.ceil(max(n.end for n in notes) * scale / 1920), max(events, default=0) + 1)
        tempos = sorted(((int(t.time * scale), t.tempo) for t in obj.tempo_changes), key=lambda x: x[0])
        tempo_index, tempo = 0, 120.0
        tokens, bars = [], []
        for bar in range(n_bars):
            while tempo_index < len(tempos) and tempos[tempo_index][0] <= bar * 1920:
                tempo = tempos[tempo_index][1]
                tempo_index += 1
            bars.append(len(tokens))
            tokens.extend([f'<TEMPO_{self._quantize_tempo(tempo)}>', '<BAR>'])
            last_beat = last_position = -1
            for beat, position, local, token in sorted(events[bar]):
                if beat != last_beat:
                    tokens.append(f'<BEAT_{beat + 1}>')
                    last_beat, last_position = beat, -1
                if position != last_position:
                    tokens.append(f'<POS_{position}>')
                    last_position = position
                tokens.append(token)
        tokens.append('<EOS>')
        return tokens, bars


@lru_cache(maxsize=128)
def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    ensure_directory(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_pickle(path, obj):
    ensure_directory(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def extract_drums(midi, count_in_quarters=0):
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('非同期トラック/非PPQ MIDIは対象外')
    cutoff = count_in_quarters * midi.ticks_per_beat
    out = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    pitches = Counter()
    track_info = []
    # track単位で保持し、非ドラムのチャンネルイベントだけを除く。
    for index, track in enumerate(midi.tracks):
        events, state = [], {}
        tick = 0
        source_notes = kept_notes = 0
        track_name = ''
        active = Counter()
        for msg in track:
            tick += msg.time
            if msg.type == 'track_name':
                track_name = msg.name
            if msg.type == 'end_of_track':
                continue
            drum = getattr(msg, 'channel', None) == 9
            if not msg.is_meta and not drum:
                continue
            is_on = msg.type == 'note_on' and msg.velocity > 0
            is_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
            if is_on:
                source_notes += 1
            if tick < cutoff:
                if not is_on and not is_off:
                    key = (msg.type, getattr(msg, 'control', None), getattr(msg, 'channel', None))
                    state[key] = msg.copy(time=0)
                continue
            if is_on:
                active[msg.note] += 1
                pitches[msg.note] += 1
                kept_notes += 1
            if is_off:
                if cutoff and not active[msg.note]:
                    continue
                active[msg.note] = max(0, active[msg.note] - 1)
            events.append((tick - cutoff, msg.copy(time=0)))
        if not events and not state:
            continue
        dest = mido.MidiTrack()
        dest.extend(state.values())
        end_tick = max(0, tick - cutoff)
        last = 0
        for tick, msg in events:
            dest.append(msg.copy(time=tick - last))
            last = tick
        dest.append(mido.MetaMessage('end_of_track', time=max(0, end_tick - last)))
        out.tracks.append(dest)
        track_info.append({'source_track': index, 'name': track_name,
                           'source_drum_onsets': source_notes, 'retained_drum_onsets': kept_notes})
    return out, dict(pitches), track_info


def discover(data_root):
    records = []
    # metadataを基にして欠損を即座に検出する。
    groove_root = data_root / 'raw/groove/groove'
    with (groove_root / 'info.csv').open() as f:
        for row in csv.DictReader(f):
            records.append({'dataset': 'groove', 'source': str(groove_root / row['midi_filename']),
                            'official_split': row['split'].replace('validation', 'val'),
                            'source_metadata': row})
    g2g = data_root / 'raw/groove2groove/groove2groove-data-v1.0.0'
    with (g2g / 'styles.tsv').open() as f:
        styles = {row['Name']: row for row in csv.DictReader(f, delimiter='\t')}
    for split in ('train', 'val', 'test', 'itest'):
        for path in sorted((g2g / 'midi' / split / 'fixed').glob('*.mid')):
            chart, style = path.stem.split('.')
            base, sub = style.rsplit('_', 1)
            records.append({'dataset': 'groove2groove', 'source': str(path), 'official_split': split,
                            'source_metadata': {'chart': chart, 'style': base, 'substyle': sub,
                                                **styles.get(base, {})}})
    original_names = json.loads((data_root / 'raw/lakh/md5_to_paths.json').read_text())
    lakh_root = data_root / 'raw/lakh/lmd_full'
    for md5 in sorted(original_names):
        path = lakh_root / md5[0] / (md5 + '.mid')
        records.append({'dataset': 'lakh', 'source': str(path), 'official_split': None,
                        'source_archive': str(data_root / 'raw/lakh/lmd_full.tar.gz'),
                        'archive_member': f'lmd_full/{md5[0]}/{md5}.mid',
                        'source_metadata': {'source_md5': md5, 'original_paths': original_names[md5]}})
    counts = Counter(r['dataset'] for r in records)
    if counts != {'groove': 1150, 'groove2groove': 9333, 'lakh': 178561}:
        # 配布の実ファイル数は概要記載176,581と異なる。検証済みアーカイブの件数。
        raise ValueError(f'原データ件数を確認してください: {dict(counts)}')
    for index, row in enumerate(records):
        row['id'] = f'{index:06d}'
        row['license'] = SOURCES[row['dataset']]['license']
    return records


def timeout_handler(signum, frame):
    raise TimeoutError('1ファイルの処理時間が120秒を超過')


def process_one(task):
    row, data_root, collection = task[:3]
    row = dict(row)
    root = Path(data_root)
    extracted = root / 'extracted' / collection / row['dataset'] / (row['id'] + '.mid')
    token_path = root / 'tokens' / collection / (row['id'] + '.pkl')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)
    try:
        payload = task[3] if len(task) == 4 else Path(row['source']).read_bytes()
        row['source_sha256'] = hashlib.sha256(payload).hexdigest()
        if row['dataset'] == 'lakh' and hashlib.md5(payload).hexdigest() != Path(row['source']).stem:
            raise ValueError('Lakhのファイル名MD5と内容が不一致')
        midi = mido.MidiFile(file=io.BytesIO(payload))
        drum, pitches, tracks = extract_drums(midi, 8 if row['dataset'] == 'groove2groove' else 0)
        row.update(source_tracks=tracks, source_ticks_per_beat=midi.ticks_per_beat,
                   pitches=pitches, extracted_onsets=sum(pitches.values()),
                   count_in_quarters_removed=8 if row['dataset'] == 'groove2groove' else 0)
        if not pitches:
            row['status'] = 'no_drums'
            return row
        ensure_directory(extracted.parent)
        buffer = io.BytesIO()
        drum.save(file=buffer)
        extracted_bytes = buffer.getvalue()
        extracted.write_bytes(extracted_bytes)
        row['extracted'] = str(extracted)
        row['extracted_sha256'] = hashlib.sha256(extracted_bytes).hexdigest()
        row['status'] = 'extracted'
        signatures = sorted({f'{m.numerator}/{m.denominator}' for tr in drum.tracks for m in tr if m.type == 'time_signature'})
        row['time_signatures'] = signatures
        if any(s != '4/4' for s in signatures):
            row['token_status'] = 'unsupported_time_signature'
            return row
        obj = miditoolkit.MidiFile(file=io.BytesIO(extracted_bytes))
        notes = [n for inst in obj.instruments if inst.is_drum for n in inst.notes]
        row['parsed_note_count'] = len(notes)
        if not notes:
            row['token_status'] = 'no_complete_notes'
            return row
        if len(notes) > 500000 or max(n.end for n in notes) / obj.ticks_per_beat > 16384:
            row['token_status'] = 'resource_limit'
            return row
        mapped = Counter()
        for n in notes:
            before = n.pitch
            if row['dataset'] == 'groove':
                n.pitch = GROOVE_PITCH_MAP.get(n.pitch, n.pitch)
            if n.pitch != before:
                mapped[f'{before}->{n.pitch}'] += 1
        unsupported = Counter(n.pitch for n in notes if n.pitch not in DRUM_NOTE_MAP)
        row.update(pitch_mapping_counts=dict(mapped), unsupported_pitches=dict(unsupported),
                   off_grid_onsets=sum(n.start * 24 % obj.ticks_per_beat != 0 for n in notes),
                   tempo_bpm=[float(t.tempo) for t in obj.tempo_changes])
        supported = [n for n in notes if n.pitch in DRUM_NOTE_MAP]
        row['supported_note_count'] = len(supported)
        if not supported:
            row['token_status'] = 'no_supported_notes'
            return row
        inst = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')
        inst.notes = sorted(supported, key=lambda n: (n.start, n.pitch, n.end, n.velocity))
        obj.instruments = [inst]
        tokenizer = CollectionTokenizer()
        tokens, bars = tokenizer.midi_object_to_tokens(obj)
        assert all(t in tokenizer.token2idx for t in tokens)
        assert tokens[-1] == '<EOS>' and bars
        hits = sum(not t.startswith('<') for t in tokens)
        row['tokenized_hits'] = hits
        row['additional_tokenizer_dropped_notes'] = len(supported) - hits
        if hits == 0:
            row['token_status'] = 'no_token_hits'
            return row
        indices = tokenizer.tokens_to_indices(tokens)
        write_pickle(token_path, (bars, indices.tolist()))
        row.update(token_status='ok', tokens=str(token_path), bars=len(bars), token_count=len(tokens),
                   token_sha256=hashlib.sha256(indices.astype('<i4').tobytes()).hexdigest())
        return row
    except Exception as exc:
        row.setdefault('status', 'error')
        row['token_status'] = 'error'
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row
    finally:
        signal.alarm(0)


def assign_splits(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in ('split', 'split_exclusion', 'duplicate_of'):
            row.pop(key, None)
        if row.get('token_status') == 'ok':
            groups[row['token_sha256']].append(row)
    for fingerprint, group in groups.items():
        official = {r['official_split'] for r in group if r['official_split']}
        # 公式集合間で同じ入力が現れる場合は双方を学習評価リストから除外。
        if len(official) > 1:
            for r in group:
                r.update(split=None, split_exclusion='duplicate_across_official_splits')
            continue
        if official:
            split = next(iter(official))
        else:
            fraction = int(fingerprint[:8], 16) / 2**32
            split = 'test' if fraction < .1 else 'val' if fraction < .2 else 'train'
        # 原本と成果物は全件保持。学習評価リストは内容ごとに代表1件のみ。
        chosen = min(group, key=lambda r: (r['official_split'] is None, r['id']))
        for r in group:
            r['split'] = split
            if r is not chosen:
                r['split_exclusion'] = 'duplicate_token_sequence'
                r['duplicate_of'] = chosen['id']


def finalize(rows, meta, token_root):
    assign_splits(rows)
    rows.sort(key=lambda r: r['id'])
    with (meta / 'manifest.jsonl').open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {}
    for dataset in SOURCES:
        selected = [r for r in rows if r['dataset'] == dataset]
        summary[dataset] = {
            'source_files': len(selected),
            'extracted_files': sum('extracted' in r for r in selected),
            'token_files': sum(r.get('token_status') == 'ok' for r in selected),
            'statuses': dict(Counter(r['status'] for r in selected)),
            'token_statuses': dict(Counter(r.get('token_status', 'not_applicable') for r in selected)),
            'split_exclusions': dict(Counter(r['split_exclusion'] for r in selected if r.get('split_exclusion'))),
            'split_counts': dict(Counter(r['split'] for r in selected if r.get('split') and not r.get('split_exclusion'))),
            'unsupported_notes': sum(sum(r.get('unsupported_pitches', {}).values()) for r in selected),
            'additional_tokenizer_dropped_notes': sum(r.get('additional_tokenizer_dropped_notes', 0) for r in selected),
        }
    for split in ('train', 'val', 'test', 'itest'):
        eligible = [r for r in rows if r.get('split') == split and not r.get('split_exclusion')]
        write_pickle(meta / f'{split}_split.pkl', [Path(r['tokens']).name for r in eligible])
        for dataset in SOURCES:
            write_pickle(meta / f'{dataset}_{split}_split.pkl', [Path(r['tokens']).name for r in eligible if r['dataset'] == dataset])
    write_json(meta / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--collection', default='drum_collection_v1')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers は1以上')
    meta = args.data_root / 'metadata' / args.collection
    token_root = args.data_root / 'tokens' / args.collection
    meta.mkdir(parents=True, exist_ok=True)
    token_root.mkdir(parents=True, exist_ok=True)
    records = discover(args.data_root)
    config = {'profile': PROFILE, 'sources': SOURCES, 'groove_pitch_map': GROOVE_PITCH_MAP,
              'expected_inputs': len(records), 'difficulty_labels': False,
              'non_4_4_policy': 'extract_but_do_not_tokenize',
              'unsupported_pitch_policy': 'preserve_in_midi_exclude_from_standard_tokens'}
    if (meta / 'policy.json').exists() and json.loads((meta / 'policy.json').read_text()) != json.loads(json.dumps(config)):
        raise ValueError('処理方針が変わっています。新しい --collection を指定してください')
    write_json(meta / 'policy.json', config)
    CollectionTokenizer().save_vocab(str(meta / 'drum_vocab.pkl'))
    write_dataset_tokenization_metadata(str(token_root), 'standard', {'profile': PROFILE, 'use_limb_tokens': False})
    checkpoint = meta / 'progress.jsonl'
    done = {}
    if checkpoint.exists():
        lines = checkpoint.read_text().split('\n')
        if lines and not lines[-1]:
            lines.pop()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                continue
            done[row['id']] = row
        # 末尾の不完全な行を除いて再開する。
        with checkpoint.open('w') as f:
            for row in done.values():
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    expected_records = {r['id']: r for r in records}
    for key, row in list(done.items()):
        if key not in expected_records or row['source'] != expected_records[key]['source']:
            raise ValueError('再開元と入力ファイル一覧が一致しません')
        if any(not Path(row[field]).is_file() for field in ('extracted', 'tokens') if field in row):
            del done[key]
    pending = [r for r in records if r['id'] not in done]
    print(f'全{len(records)}件、完了{len(done)}件、残り{len(pending)}件', flush=True)
    start = time.monotonic()
    with checkpoint.open('a', buffering=1) as out, mp.Pool(args.workers) as pool:
        def input_tasks():
            lakh = {r['archive_member']: r for r in pending if r['dataset'] == 'lakh'}
            for r in pending:
                if r['dataset'] != 'lakh':
                    yield r, str(args.data_root), args.collection
            if lakh:
                archive = args.data_root / 'raw/lakh/lmd_full.tar.gz'
                with tarfile.open(archive, 'r|gz') as tar:
                    for member in tar:
                        r = lakh.pop(member.name, None)
                        if r is not None:
                            yield r, str(args.data_root), args.collection, tar.extractfile(member).read()
                if lakh:
                    raise ValueError(f'アーカイブに未収録の入力: {len(lakh)}件')
        tasks = input_tasks()
        for index, row in enumerate(pool.imap_unordered(process_one, tasks, chunksize=1), 1):
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            done[row['id']] = row
            if index % 500 == 0:
                print(f'{len(done)}/{len(records)} ({index / (time.monotonic()-start):.1f} files/s)', flush=True)
    finalize(list(done.values()), meta, token_root)


if __name__ == '__main__':
    main()
