"""移動後の依存方向、旧pickleの型復元、実装ハッシュの照合を検証する。"""
import ast
import gzip
import importlib
import json
from pathlib import Path
import pickle

import pytest

from data.pairdata.io import read_pickle
from src.common.examples import make_bar
from src.common.provenance import implementation_hashes, validate_sources
from src.common.serialization import loads_local_pickle


def test_shared_and_standalone_modules_do_not_import_hybrid_or_legacy_packages():
    for package, prohibited in (
        ('common', ('src.beamsearch', 'src.musemorphose', 'src.beam_muse', 'src.arrangement', 'src.models')),
        ('musemorphose', ('src.beamsearch', 'src.beam_muse', 'src.arrangement', 'src.models')),
    ):
        for path in Path('src', package).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                imports = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                assert not any(name == prefix or name.startswith(prefix+'.') for name in imports for prefix in prohibited), path
    assert not list(Path('src/arrangement').glob('*.py'))
    assert not list(Path('src/models').glob('*.py'))


@pytest.mark.parametrize('before,after,name', [
    ('src.arrangement.representation', 'src.common.representation', 'Bar'),
    ('src.arrangement.safety', 'src.common.safety', 'Guards'),
    ('src.arrangement.inputs', 'src.common.inputs', 'PreparedInput'),
    ('src.arrangement.scoring', 'src.common.difficulty', 'Difficulty'),
    ('src.arrangement.scoring', 'src.beamsearch.scoring', 'ScoreBreakdown'),
    ('src.arrangement.beam_search', 'src.beamsearch.beam_search', 'Action'),
    ('src.models.musemorphose', 'src.musemorphose.model', 'MuseMorphose'),
    ('src.beam_muse.features', 'src.common.skeleton', 'SkeletonFeatures'),
])
def test_old_pickle_global_reference_resolves_to_canonical_class(before, after, name):
    payload = f'c{before}\n{name}\n.'.encode()
    assert loads_local_pickle(payload) is getattr(importlib.import_module(after), name)


def test_old_gzip_pickle_restores_actual_bar_and_notes(tmp_path):
    bar = make_bar([(0, 'kick'), (24, 'snare')])
    payload = pickle.dumps(bar, protocol=0).replace(b'src.common.representation', b'src.arrangement.representation')
    path = tmp_path/'old.pkl.gz'
    path.write_bytes(gzip.compress(payload))
    assert read_pickle(path) == bar
    assert type(read_pickle(path)) is type(bar)


def test_migration_hashes_require_full_match_on_both_sides(tmp_path):
    before, after = {'old.py': 'a'}, {'new.py': 'b', 'common.py': 'c'}
    path = tmp_path/'migration.json'
    path.write_text(json.dumps(dict(schema_version=1, snapshots=[dict(before=before, after=after)])))
    validate_sources(before, after, migration_path=path)
    validate_sources(after, after, migration_path=path)
    for recorded, current in (({'old.py': 'changed'}, after), (before, {'new.py': 'b'}),
                              (before, dict(after, **{'common.py': 'changed'}))):
        with pytest.raises(ValueError, match='実装'):
            validate_sources(recorded, current, migration_path=path)


def test_reviewed_migration_describes_current_sources():
    migration = json.loads(Path('config/source_layout_migration.json').read_text())
    for snapshot in migration['snapshots']:
        validate_sources(snapshot['before'], implementation_hashes(snapshot['kind']))
