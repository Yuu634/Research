from pathlib import Path
import pickle

import mido

from scripts.preprocess.build_drum_collection import extract_drums, process_one, assign_splits
from scripts.preprocess.download_drum_sources import safe_target


def save_fixture(path, signature=(4, 4)):
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=600000),
        mido.MetaMessage('time_signature', numerator=signature[0], denominator=signature[1]),
    ])
    notes = mido.MidiTrack([
        mido.Message('note_on', channel=0, note=60, velocity=90),
        mido.Message('note_off', channel=0, note=60, time=120),
        mido.Message('note_on', channel=9, note=36, velocity=90, time=3720),
        mido.Message('note_off', channel=9, note=36, time=120),
        mido.Message('note_on', channel=9, note=70, velocity=70, time=360),
        mido.Message('note_off', channel=9, note=70, time=120),
    ])
    other = mido.MidiTrack([
        mido.Message('note_on', channel=9, note=38, velocity=90, time=4320),
        mido.Message('note_off', channel=9, note=38, time=120),
    ])
    midi.tracks.extend([meta, notes, other])
    midi.save(path)
    return midi


def test_extraction_preserves_ticks_tempo_and_all_drum_tracks(tmp_path):
    midi = save_fixture(tmp_path / 'source.mid')
    drum, pitches, tracks = extract_drums(midi, 8)
    assert pitches == {36: 1, 70: 1, 38: 1}
    assert all(getattr(m, 'channel', 9) == 9 for tr in drum.tracks for m in tr)
    onsets = []
    for tr in drum.tracks:
        tick = 0
        for m in tr:
            tick += m.time
            if m.type == 'note_on' and m.velocity:
                onsets.append((tick, m.note, m.velocity))
    assert sorted(onsets) == [(0, 36, 90), (480, 38, 90), (480, 70, 70)]
    assert drum.tracks[0][0].tempo == 600000


def test_pipeline_retains_unsupported_notes_in_midi_and_audits_token_projection(tmp_path):
    source = tmp_path / 'source.mid'
    save_fixture(source)
    result = process_one(({'id': '000000', 'source': str(source), 'dataset': 'groove2groove'}, str(tmp_path), 'test'))
    assert result['token_status'] == 'ok'
    assert result['unsupported_pitches'] == {70: 1}
    assert result['tokenized_hits'] == 2
    assert result['source_ticks_per_beat'] == 480
    with Path(result['tokens']).open('rb') as f:
        bars, tokens = pickle.load(f)
    assert bars == [0]
    assert tokens[-1] == 1
    assert result['extracted_onsets'] == 3


def test_non_four_four_is_extracted_but_not_tokenized(tmp_path):
    source = tmp_path / 'source.mid'
    save_fixture(source, (3, 4))
    result = process_one(({'id': '000000', 'source': str(source), 'dataset': 'groove'}, str(tmp_path), 'test'))
    assert result['token_status'] == 'unsupported_time_signature'
    assert Path(result['extracted']).exists()
    assert 'tokens' not in result


def test_official_split_conflicts_and_lakh_duplicates():
    rows = [
        {'id': '1', 'token_status': 'ok', 'token_sha256': 'a'*64, 'official_split': 'train'},
        {'id': '2', 'token_status': 'ok', 'token_sha256': 'a'*64, 'official_split': 'test'},
        {'id': '3', 'token_status': 'ok', 'token_sha256': 'b'*64, 'official_split': 'val'},
        {'id': '4', 'token_status': 'ok', 'token_sha256': 'b'*64, 'official_split': None},
    ]
    assign_splits(rows)
    assert rows[0]['split'] is None and rows[1]['split'] is None
    assert rows[2]['split'] == 'val' and 'split_exclusion' not in rows[2]
    assert rows[3]['duplicate_of'] == '3'


def test_reject_archive_path_escape(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        safe_target(tmp_path, '../escape.mid')


def test_full_validation_detects_corrupted_pickle(tmp_path):
    import pytest
    from scripts.preprocess.build_drum_collection import finalize, write_json
    from scripts.preprocess.validate_drum_collection import validate
    source = tmp_path / 'source.mid'
    save_fixture(source)
    row = process_one(({'id': '000000', 'source': str(source), 'dataset': 'groove2groove',
                        'official_split': 'train'}, str(tmp_path), 'test'))
    meta = tmp_path / 'metadata/test'
    write_json(meta / 'policy.json', {'expected_inputs': 1})
    finalize([row], meta, tmp_path / 'tokens/test')
    assert validate(tmp_path, 'test', 1)['token_files_verified'] == 1
    stray = tmp_path / 'tokens/test/999999.pkl'
    stray.write_bytes(b'untracked artifact')
    with pytest.raises(AssertionError, match='manifest'):
        validate(tmp_path, 'test', 1)
    stray.unlink()
    token_path = Path(row['tokens'])
    with token_path.open('rb') as f:
        bars, tokens = pickle.load(f)
    tokens[0] = 0
    with token_path.open('wb') as f:
        pickle.dump((bars, tokens), f)
    with pytest.raises(AssertionError):
        validate(tmp_path, 'test', 1)


def test_collection_tokens_preserve_ghost_and_overlapping_notes():
    import miditoolkit
    from scripts.preprocess.build_drum_collection import CollectionTokenizer
    obj = miditoolkit.MidiFile(ticks_per_beat=480)
    inst = miditoolkit.Instrument(program=0, is_drum=True)
    inst.notes = [miditoolkit.Note(velocity=10, pitch=38, start=0, end=240),
                  miditoolkit.Note(velocity=80, pitch=38, start=120, end=300),
                  miditoolkit.Note(velocity=10, pitch=36, start=480, end=600)]
    obj.instruments = [inst]
    tokenizer = CollectionTokenizer()
    tokens, bars = tokenizer.midi_object_to_tokens(obj)
    assert 'SNARE_HIT_Ghost' in tokens
    assert 'SNARE_HIT_Normal' in tokens
    assert 'KICK_HIT_Normal' in tokens
    assert sum(not t.startswith('<') for t in tokens) == 3
    assert tokenizer.vocab == __import__('src.tokenizers.drum_tokenizer', fromlist=['DrumTokenizer']).DrumTokenizer().vocab


def test_archive_payload_does_not_require_expanded_source(tmp_path):
    import hashlib
    source = tmp_path / 'fixture.mid'
    save_fixture(source)
    payload = source.read_bytes()
    nonexistent = tmp_path / 'not-expanded' / (hashlib.md5(payload).hexdigest() + '.mid')
    result = process_one(({'id': '000000', 'source': str(nonexistent), 'dataset': 'lakh'},
                          str(tmp_path), 'test', payload))
    assert result['token_status'] == 'ok'
    assert result['extracted_onsets'] == 3
    assert not nonexistent.exists()


def test_resume_preserves_unicode_line_separator_in_metadata(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.preprocess import build_drum_collection as builder
    row = {'id': '000000', 'dataset': 'lakh', 'source': 'fixture.mid',
           'official_split': None, 'status': 'no_drums',
           'source_metadata': {'name': 'first\u2028second\u0085third'}}
    meta = tmp_path / 'metadata/drum_collection_v1'
    meta.mkdir(parents=True)
    (meta / 'progress.jsonl').write_text(json.dumps(row, ensure_ascii=False) + '\n')
    monkeypatch.setattr(builder, 'discover', lambda root: [dict(row)])
    monkeypatch.setattr(sys, 'argv', ['build', '--data-root', str(tmp_path), '--workers', '1'])
    builder.main()
    result = json.loads((meta / 'manifest.jsonl').read_text())
    assert result['source_metadata'] == row['source_metadata']


def test_zero_duration_final_boundary_note_is_retained():
    import miditoolkit
    from scripts.preprocess.build_drum_collection import CollectionTokenizer
    obj = miditoolkit.MidiFile(ticks_per_beat=192)
    inst = miditoolkit.Instrument(program=0, is_drum=True)
    inst.notes = [miditoolkit.Note(velocity=90, pitch=38, start=0, end=50),
                  miditoolkit.Note(velocity=90, pitch=36, start=768, end=768)]
    obj.instruments = [inst]
    tokens, bars = CollectionTokenizer().midi_object_to_tokens(obj)
    assert len(bars) == 2
    assert tokens.count('KICK_HIT_Normal') == 1
    assert sum(not t.startswith('<') for t in tokens) == 2


def test_split_rebuild_clears_previous_duplicate_exclusion():
    rows = [dict(id=str(i), token_status='ok', token_sha256='a'*64, official_split=None) for i in (1, 2)]
    assign_splits(rows)
    assert rows[1]['duplicate_of'] == '1'
    rows[1]['token_sha256'] = 'b'*64
    assign_splits(rows)
    assert all('split_exclusion' not in r and 'duplicate_of' not in r for r in rows)


def test_stratified_validation_sample_includes_required_ids():
    from scripts.preprocess.validate_drum_collection import choose_rows
    rows = [dict(id=str(i), dataset='groove', split='train' if i < 5 else 'test',
                 status='extracted', token_status='ok') for i in range(10)]
    selected = choose_rows(rows, 1, ['4', '9'])
    assert {'4', '9'} <= {r['id'] for r in selected}
    assert {r['split'] for r in selected} == {'train', 'test'}
    assert selected == choose_rows(list(reversed(rows)), 1, ['4', '9'])
    assert len(choose_rows(rows)) == len(rows)
