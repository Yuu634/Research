"""実装ハッシュを検証し、記録済みのディレクトリ移行だけを明示的に許容する。"""
import json
from pathlib import Path

from data.pairdata.io import file_hash


def implementation_files(kind):
    if kind not in ('beamsearch', 'beam_muse'):
        raise ValueError('未知の実装記録種別')
    packages = ['common', 'beamsearch']
    if kind == 'beam_muse':
        packages += ['beam_muse', 'musemorphose']
    paths = {p for name in packages for p in Path('src', name).glob('*.py')}
    paths.update(Path(p) for p in ('src/data/drum_difficulty_v2.py', 'src/data/drum_difficulty_calculator.py',
                                  'data/pairdata/io.py'))
    scripts = ('scripts/train/tune_beam_muse.py', 'scripts/generate/arrange_beam_muse.py') if kind == 'beam_muse' else (
        'scripts/evaluate/review_arrangement_metrics.py', 'scripts/evaluate/compare_arrangement_models.py')
    paths.update(Path(p) for p in scripts)
    return [str(p) for p in sorted(paths)]


def implementation_hashes(kind):
    return {p: file_hash(p) for p in implementation_files(kind)}


def validate_sources(recorded, current, *, migration_path=None):
    if recorded == current:
        return
    path = Path(migration_path) if migration_path is not None else Path(__file__).resolve().parents[2]/'config/source_layout_migration.json'
    if path.exists():
        migration = json.loads(path.read_text(encoding='utf-8'))
        if migration.get('schema_version') == 1:
            for snapshot in migration.get('snapshots', []):
                # 部分一致やパス置換だけで検査を省略しない。両側の全ファイルを照合する。
                if recorded == snapshot['before'] and current == snapshot['after']:
                    return
    raise ValueError('実装が記録時と異なる。検証済みのディレクトリ移行にも一致しない')
