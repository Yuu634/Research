"""選定済み設定の復元と、語彙・境界を取り違えない推論CLIの検証。"""
import hashlib
import json
from pathlib import Path
import pickle

import mido
import pytest

from scripts.generate.arrange_beam_search import main
from src.arrangement.beam_search import Action
from src.arrangement.examples import make_bar
from src.arrangement.inputs import bar_to_payload
from src.arrangement.representation import to_tokens
from src.arrangement.scoring import ScoringModel
from src.arrangement.selected_model import load_selected_model
from src.data.drum_difficulty_v2 import ATTRS


@pytest.fixture
def model_files(tmp_path):
    original = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] + [(0, 'kick'), (24, 'snare')])
    tokens = sorted(set(to_tokens(original)))
    vocab = dict(tokenization_method='standard', token2idx=dict(zip(tokens, range(len(tokens)))))
    meta = tmp_path/'meta'
    meta.mkdir()
    (meta/'bounds.json').write_text(json.dumps(dict(boundaries={a: [.5, 1, 2, 4, 8, 16] for a in ATTRS})))
    (meta/'drum_vocab.pkl').write_bytes(pickle.dumps(vocab))
    selected = dict(schema_version=1, selected_on='val', candidate=dict(name='fixture', score={}, search=dict(beam_width=2, max_depth=2)),
                    evaluation=dict(target_tolerance=.5/7, skeleton_threshold=.9),
                    actions=[Action('hh_16_to_8').as_dict()],
                    bounds_sha256=hashlib.sha256((meta/'bounds.json').read_bytes()).hexdigest(),
                    vocabulary_sha256=hashlib.sha256((meta/'drum_vocab.pkl').read_bytes()).hexdigest())
    path = tmp_path/'model.json'
    path.write_text(json.dumps(selected))
    return meta, path, original


def test_load_model_preserves_selected_settings(model_files):
    meta, path, _ = model_files
    model, search, actions, evaluation, selected = load_selected_model(path, meta)
    assert isinstance(model, ScoringModel) and search.beam_width == 2 and search.max_depth == 2
    assert actions == (Action('hh_16_to_8'),)
    assert evaluation.skeleton_threshold == .9 and selected['selected_on'] == 'val'


@pytest.mark.parametrize('filename', ['bounds.json', 'drum_vocab.pkl'])
def test_mismatched_metadata_rejected(model_files, filename):
    meta, path, _ = model_files
    (meta/filename).write_bytes((meta/filename).read_bytes()+b' ')
    with pytest.raises(ValueError, match='メタデータ'):
        load_selected_model(path, meta)


@pytest.mark.parametrize('change', [dict(selected_on='test'), dict(schema_version=True), dict(actions=[]),
                                   dict(actions=[dict(name='unknown', parameters={})])])
def test_invalid_selected_settings(model_files, change):
    meta, path, _ = model_files
    content = json.loads(path.read_text())
    content.update(change)
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_selected_model(path, meta)


def test_inference_cli_writes_valid_tokens_preserved_ids_and_midi(model_files, tmp_path, monkeypatch):
    meta, path, original = model_files
    model, *_ = load_selected_model(path, meta)
    request = dict(schema_version=1, original=bar_to_payload(original), previous=None,
                   target=dict(mode='absolute', values={k: 0 for k in model.difficulty(original).normalized}))
    input_path = tmp_path/'request.json'
    input_path.write_text(json.dumps(request))
    output = tmp_path/'result'
    monkeypatch.setattr('sys.argv', ['arrange_beam_search', '--input', str(input_path), '--model', str(path),
                                    '--meta', str(meta), '--output', str(output)])
    main()
    report = json.loads((output/'result.json').read_text())
    assert report['evaluation']['status'] == 'valid'
    assert set(report['tokens']) <= model.vocabulary
    assert len(report['history']) <= 2
    for name in ('before.mid', 'after.mid'):
        midi = mido.MidiFile(output/name)
        assert sum(message.time for message in midi.tracks[0]) == 1920
        assert all(message.channel == 9 for message in midi.tracks[0] if message.type in ('note_on', 'note_off'))
    assert (output/'index.html').exists()
