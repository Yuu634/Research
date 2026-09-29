import io
import tarfile

import mido

from scripts.analysis.survey_groove2groove import audit, summarize_midi


def midi_bytes():
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.Message('note_on', channel=9, note=37, velocity=90, time=0),
        mido.Message('note_on', channel=9, note=36, velocity=80, time=3840),
        mido.Message('note_on', channel=0, note=60, velocity=80, time=0),
        mido.Message('note_on', channel=9, note=82, velocity=50, time=1),
        mido.Message('note_on', channel=9, note=38, velocity=0, time=0),
    ])
    buf = io.BytesIO()
    midi.save(file=buf)
    return buf.getvalue()


def test_count_in_channels_grid_and_unsupported_pitch():
    result = summarize_midi(midi_bytes(), 'example.mid')
    assert result['drum_notes_in_count_in'] == 1
    assert result['drum_notes_after_count_in'] == 2
    assert result['non_drum_notes_after_count_in'] == 1
    assert result['unsupported_pitches'] == {'82': 1}
    assert result['off_24_grid_notes'] == 1
    assert result['velocity_min'] == 50
    assert result['velocity_max'] == 80


def test_archive_counts_and_order_independent_sample(tmp_path):
    members = [
        (f'dataset/midi/train/{version}/{n}.mid', midi_bytes())
        for version in ('raw', 'fixed') for n in range(3)
    ]
    members.append(('dataset/styles.tsv', b'Name\tTSig\tEvSw\tGenre\tGroup\nDemo\t4/4\tev8\tJazz\tJazz\n'))
    results = []
    for index, entries in enumerate((members, list(reversed(members)))):
        archive = tmp_path / f'{index}.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, payload in entries:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        result, _ = audit(archive, 2)
        assert result['counts'] == {'train/fixed': 3, 'train/raw': 3}
        assert result['style_distributions']['TSig'] == {'4/4': 1}
        assert result['sample_summary']['train']['parsed'] == 2
        results.append([s['file'] for s in result['samples']])
    assert results[0] == results[1]
