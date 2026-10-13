"""追加モデル評価の生成共有・入力照合・CSV分母を検証する。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from data.pairdata.io import file_hash
from scripts.evaluate import collect_arrangement_evaluations as collect
from src.beamsearch.scoring import ScoringModel
from src.common.comparison import summarize_outputs
from src.common.evaluation import EvaluationConfig
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.representation import to_tokens
from src.data.drum_difficulty_v2 import ATTRS


@pytest.fixture
def prepared():
    original = make_bar([(0, 'kick'), (24, 'snare')])
    scorer = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, to_tokens(original))
    example = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
        target=dict(mode='absolute', values=scorer.difficulty(original).normalized)), scorer)
    return example, scorer


def test_input_comparison_checks_guards_and_target_not_just_ids(prepared):
    p, _ = prepared
    def example(value):
        return SimpleNamespace(case_id='id', source_id='song', prepared=value)
    collect.validate_inputs([example(p)], [example(p)])
    for changed in (replace(p, target=dict(p.target, s_tech=.5)),
                    replace(p, guards=replace(p.guards, max_added=0))):
        with pytest.raises(ValueError, match='原譜・文脈・目標・制約'):
            collect.validate_inputs([example(changed)], [example(p)])
    with pytest.raises(ValueError, match='重複'):
        collect.case_contract([dict(case_id='x', source_id='song', target={})]*2)


def test_one_generation_supplies_valid_and_invalid_metrics_and_tables(prepared, monkeypatch, tmp_path):
    p, scorer = prepared
    invalid = replace(p.original, notes=tuple(n for n in p.original.notes if n.instrument != 'kick'))
    candidates = iter([p.original, invalid, None])
    calls = []
    def generate(*args, **kwargs):
        candidate = next(candidates)
        calls.append(candidate)
        return SimpleNamespace(candidate=candidate, tokens=[] if candidate is None else to_tokens(candidate),
                               failure_reason='no_output' if candidate is None else None)
    monkeypatch.setattr(collect, 'generate', generate)
    monkeypatch.setattr(collect, 'Grammar', lambda v: None)
    monkeypatch.setattr(collect, 'collate', lambda *args, **kwargs: {})
    examples = [SimpleNamespace(case_id=str(i), source_id='song', prepared=p, history=[], beam_seconds=0.)
                for i in range(3)]
    monkeypatch.setattr(collect, 'WORKER', dict(model=object(), vocabulary={}, scorer=scorer,
        examples=examples, variant='beam_muse', evaluation=EvaluationConfig()))
    path = tmp_path/'batch.json'
    assert collect.evaluate_batch(0, 3, path) == 3
    assert len(calls) == 3
    saved = collect.read_json(path)
    assert [r['evaluation']['status'] for r in saved] == ['valid', 'invalid', 'failed']
    assert saved[1]['evaluation']['metrics'] is None
    assert saved[1]['including_invalid']['metrics'] is not None
    assert saved[2]['including_invalid']['metrics'] is None
    assert collect.read_json(path.with_suffix('.receipt.json')) == dict(start=0, end=3, sha256=file_hash(path))
    regular = summarize_outputs([r['evaluation'] for r in saved])
    inclusive = collect.summarize([r['including_invalid'] for r in saved], .5/7)
    collect.verify_valid_summary(inclusive['valid_only'], regular)
    v = collect.table_row('experiment/model', True, 'valid_only', inclusive, regular)
    a = collect.table_row('experiment/model', True, 'all_outputs', inclusive, regular)
    assert v['n'] == a['n'] == 3
    assert v['label_distance_n'] == 1 and a['label_distance_n'] == 2
    assert v['valid_rate'] == a['valid_rate'] == 1/3
    assert v['target_success_rate'] == a['target_success_rate'] == 1/3
    assert a['target_success_ignoring_validity_denominator'] == 3
