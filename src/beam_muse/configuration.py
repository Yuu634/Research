"""実験configとCLIから入力先を確定し、pairdata内のメタデータを照合する。"""
from copy import deepcopy
import json
from pathlib import Path
import re

from data.pairdata.io import file_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA = dict(pairdata='data/processed/drum_pairdata_v1',
                   beam_model='config/beamsearch_selected.json')


def load_experiment_config(path):
    """実験設定だけJSONCとして読む。文字列を保持し、コメントと末尾カンマを許容する。"""
    text = Path(path).read_text(encoding='utf-8')
    # 文字列を先に認識し、パス中の // やエスケープされた引用符を壊さない。
    string = r'"(?:\\.|[^"\\])*"'
    def remove_comment(match):
        value = match.group()
        return value if value.startswith('"') else re.sub(r'[^\r\n]', ' ', value)
    text = re.sub(string + r'|//[^\r\n]*|/\*[\s\S]*?\*/', remove_comment, text)
    # 最後の候補をコメントアウトした際に残るカンマも許容する。
    text = re.sub(string + r'|,(?=\s*[}\]])',
                  lambda match: ' ' if match.group() == ',' else match.group(), text)
    return json.loads(text)


def validate_data_config(config):
    """dataのない既存configも許容する。指定時は2項目とも必須。"""
    if 'data' not in config:
        return
    data = config['data']
    if not isinstance(data, dict) or set(data) != {'pairdata', 'beam_model'}:
        raise ValueError('dataにはpairdataとbeam_modelの2項目を指定する')
    if any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError('dataのパスは空でない文字列を指定する')


def repository_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def resolve_inputs(config, *, pairdata=None, beam_model=None, meta=None):
    """優先順位はCLI、config、旧既定値。相対パスはリポジトリ直下基準。"""
    validate_data_config(config)
    data = config.get('data', LEGACY_DATA)
    pairdata = repository_path(pairdata if pairdata is not None else data['pairdata'])
    beam_model = repository_path(beam_model if beam_model is not None else data['beam_model'])
    meta = pairdata if meta is None else repository_path(meta)
    for path in (pairdata / 'index.json', pairdata / 'drum_vocab.pkl', pairdata / 'bounds.json', beam_model):
        if not path.is_file():
            raise ValueError('入力ファイルが存在しない: ' + str(path))
    # --metaは既存コマンド用に残すが、別の語彙・境界への差替えは認めない。
    for name in ('drum_vocab.pkl', 'bounds.json'):
        if file_hash(meta / name) != file_hash(pairdata / name):
            raise ValueError('指定メタデータがpairdata内のファイルと異なる: ' + name)
    selected = json.loads(beam_model.read_text())
    for key, name in (('vocabulary_sha256', 'drum_vocab.pkl'), ('bounds_sha256', 'bounds.json')):
        if selected.get(key) != file_hash(pairdata / name):
            raise ValueError('BeamSearch選定時とpairdataのメタデータが異なる: ' + name)
    resolved = deepcopy(config)
    resolved['data'] = dict(pairdata=str(pairdata), beam_model=str(beam_model))
    return resolved, pairdata, meta, beam_model


def validate_cache_input(cache, pairdata):
    """同じ語彙でも異なるデータセットのキャッシュを取り違えない。"""
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    if manifest['identity']['pair_index'] != file_hash(Path(pairdata) / 'index.json'):
        raise ValueError('キャッシュと指定pairdataが異なる。新しい出力先でprepareを実行する')
