"""valで選定した設定と対応する語彙・難易度境界を読み込む。"""
import hashlib
import json
from pathlib import Path
import pickle

from src.arrangement.beam_search import Action, SearchConfig
from src.arrangement.evaluation import EvaluationConfig
from src.arrangement.scoring import ScoreConfig, ScoringModel


def load_selected_model(path, meta):
    selected = json.loads(Path(path).read_text(encoding='utf-8'))
    if type(selected.get('schema_version')) is not int or selected['schema_version'] != 1 or selected.get('selected_on') != 'val':
        raise ValueError('valで選定済みの設定ファイルが必要')
    meta = Path(meta)
    for name, filename in (('bounds_sha256', 'bounds.json'), ('vocabulary_sha256', 'drum_vocab.pkl')):
        if hashlib.sha256((meta/filename).read_bytes()).hexdigest() != selected[name]:
            raise ValueError('選定時とメタデータが異なる: ' + filename)
    vocabulary = pickle.loads((meta/'drum_vocab.pkl').read_bytes())
    if vocabulary['tokenization_method'] != 'standard':
        raise ValueError('standard語彙が必要')
    candidate = selected['candidate']
    model = ScoringModel(json.loads((meta/'bounds.json').read_text())['boundaries'],
                         vocabulary['token2idx'], ScoreConfig(**candidate['score']))
    search = SearchConfig(**candidate['search']).validate()
    evaluation = EvaluationConfig(**selected['evaluation']).validate()
    actions = tuple(Action(row['name'], tuple(sorted(row['parameters'].items()))).validate()
                    for row in selected['actions'])
    if not actions or len({action.key for action in actions}) != len(actions):
        raise ValueError('アクション集合が空または重複')
    return model, search, actions, evaluation, selected
