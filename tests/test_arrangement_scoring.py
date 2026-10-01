"""探索スコアの手計算、方向性、欠損、前小節依存、ハード制約を検証する。"""
from dataclasses import replace
import math

import numpy as np
import pytest

from src.common.examples import make_bar
from src.common.representation import Bar, Note, to_tokens
from src.beamsearch.rules import apply_rule
from src.common.safety import Guards
from src.beamsearch.scoring import (
    DIFFICULTY_KEYS, ScoreConfig, ScoringModel, musical_edit_distance, skeleton_similarity,
)
from src.data.drum_difficulty_v2 import ATTRS

RELAXED = Guards(max_added=None, max_deleted=None, max_edit_ratio=None)


def fixture(config=ScoreConfig()):
    original = make_bar([(p,'hh_closed') for p in range(0,96,6)] +
                        [(0,'kick'),(48,'kick'),(24,'snare'),(72,'snare')])
    after = apply_rule(original,'hh_16_to_8').after
    vocabulary = set(to_tokens(original)) | set(to_tokens(after))
    for bpm in (120,240):vocabulary.add(f'<TEMPO_{bpm}>')
    for beat in range(1,5):vocabulary.add(f'<BEAT_{beat}>')
    for p in range(24):vocabulary.add(f'<POS_{p}>')
    vocabulary.update(['SNARE_HIT_Ghost','SNARE_HIT_Accent'])
    model = ScoringModel({a:[.5,1,2,4,8,16] for a in ATTRS},vocabulary,config)
    return original,after,model


def test_four_terms_match_hand_computed_formula():
    original,after,model = fixture()
    target = model.difficulty(after).normalized
    result = model.score(original,after,target,rule_count=1)
    assert result.valid and result.difficulty_loss == 0
    assert result.skeleton_similarity == 1
    assert result.edit_distance == pytest.approx(8/20)
    assert result.total == pytest.approx(.5-.02-.1*(8/20))
    assert sum(result.contributions.values()) == result.total


def test_closer_target_is_better_when_other_terms_equal():
    original,after,model = fixture()
    exact = model.difficulty(after).normalized
    near = model.score(original,after,exact,rule_count=1)
    far = model.score(original,after,{k:1-v for k,v in exact.items()},rule_count=1)
    assert near.difficulty_loss < far.difficulty_loss and near.total > far.total


def test_rule_count_penalty_and_weight_scaling():
    original,after,model = fixture()
    target = model.difficulty(after).normalized
    a = model.score(original,after,target,rule_count=1)
    b = model.score(original,after,target,rule_count=2)
    assert a.total-b.total == pytest.approx(.02)
    config=replace(model.config,lambda_rules=.2)
    other=ScoringModel(model.boundaries,model.vocabulary,config)
    assert other.score(original,after,target,rule_count=1).total == pytest.approx(a.total-.18)


def test_single_attribute_weight_and_common_scaling():
    original,after,model=fixture(ScoreConfig(difficulty_weights=(0,0,1,0,0)))
    target=dict(model.difficulty(after).normalized); target['s_hand']=0
    result=model.score(original,after,target,rule_count=1)
    assert result.difficulty_loss==model.difficulty(after).normalized['s_hand']
    other=ScoringModel(model.boundaries,model.vocabulary,replace(model.config,difficulty_weights=(0,0,10,0,0)))
    assert other.score(original,after,target,rule_count=1).total==pytest.approx(result.total)


def test_same_music_with_reordered_renamed_notes_has_zero_edit_distance():
    original,_,_=fixture()
    changed=original.with_notes(replace(n,id='new_'+str(k)) for k,n in enumerate(reversed(original.notes)))
    assert musical_edit_distance(original,changed)==0
    assert musical_edit_distance(original,replace(original,fill_regions=((48,96),)))==0


def test_edit_distance_counts_replacement_once_and_retains_multiplicity():
    a=make_bar([(0,'kick'),(24,'snare')])
    b=make_bar([(12,'kick'),(24,'snare')])
    assert musical_edit_distance(a,b)==.5
    duplicate=Bar(a.notes+(replace(a.notes[0],id='extra'),))
    assert musical_edit_distance(a,duplicate)==.5
    assert musical_edit_distance(Bar(()),a)==2


def test_missing_skeleton_does_not_gain_perfect_similarity():
    a=make_bar([(0,'hh_closed'),(12,'hh_closed')])
    score,components,weights=skeleton_similarity(a,a)
    assert score==0 and all(v is None for v in components.values()) and sum(weights.values())==0


def test_offbeat_kick_backbeat_and_accent_changes_reduce_similarity():
    a=make_bar([(12,'kick'),(24,'snare','Accent')])
    for candidate in (make_bar([(18,'kick'),(24,'snare','Accent')]),
                      make_bar([(12,'kick'),(30,'snare','Accent')]),
                      make_bar([(12,'kick'),(24,'snare','Normal')])):
        assert skeleton_similarity(a,candidate)[0]<1
    assert skeleton_similarity(a,a)[0]==pytest.approx(1)


def test_explicit_fill_uses_actual_endpoints_and_original_regions():
    a=make_bar([(48,'floor'),(60,'tom_mid'),(84,'floor')],fill_regions=((48,96),))
    b=a.with_notes(a.notes[:-1])
    score,components,_=skeleton_similarity(a,b)
    assert components['fill']==.5 and score==.5
    untrusted=replace(b,fill_regions=())
    assert skeleton_similarity(a,untrusted)[0]==score


def test_hard_constraint_cannot_be_bought_with_large_skeleton_weight():
    original,_,model=fixture(ScoreConfig(lambda_skeleton=1e9))
    candidate=original.with_notes(n for n in original.notes if n.instrument!='kick')
    target={k:0 for k in DIFFICULTY_KEYS}
    result=model.score(original,candidate,target,rule_count=1)
    assert not result.valid and result.total is None and result.ranking_score==-math.inf
    assert any('protected_note' in reason for reason in result.reasons)


def test_tempo_and_fill_annotations_are_fixed():
    original,_,model=fixture()
    target=model.difficulty(original).normalized
    assert 'bpm_changed' in model.score(original,replace(original,bpm=240),target).reasons
    assert 'fill_annotations_changed' in model.score(original,replace(original,fill_regions=((48,96),)),target).reasons


def test_vocabulary_outside_candidates_rejected():
    original,_,model=fixture()
    bad=original.with_notes(list(original.notes)+[Note('new',3,'floor','Ghost')])
    result=model.score(original,bad,model.difficulty(original).normalized,guards=RELAXED)
    assert not result.valid and '語彙' in result.reasons[0]


def test_previous_bar_changes_rate_and_all_candidates_use_same_context():
    _,_,model=fixture()
    previous=make_bar([(95,'snare')])
    current=make_bar([(0,'snare')])
    isolated=model.difficulty(current)
    contextual=model.difficulty(current,previous)
    assert isolated.raw['s_hand']==.5
    assert contextual.raw['s_hand']==pytest.approx(48)
    assert model.score(current,current,contextual.normalized,previous=previous).difficulty_loss==0


def test_target_named_keys_ignore_mapping_order_and_exclude_rhythm():
    original,_,model=fixture()
    target=model.difficulty(original).normalized
    reversed_target=dict(reversed(list(target.items())))
    assert model.score(original,original,reversed_target).difficulty_loss==0
    with pytest.raises(ValueError):model.score(original,original,dict(target,s_rhythm=0))


@pytest.mark.parametrize('bad',[-.1,1.1,float('nan'),float('inf')])
def test_target_outside_normalized_absolute_domain_rejected(bad):
    original,_,model=fixture()
    target={k:0 for k in DIFFICULTY_KEYS};target['s_hand']=bad
    with pytest.raises(ValueError):model.score(original,original,target)


@pytest.mark.parametrize('value',[-1,1.5,True])
def test_rule_count_must_be_nonnegative_integer(value):
    original,_,model=fixture()
    with pytest.raises(ValueError):model.score(original,original,model.difficulty(original).normalized,rule_count=value)


@pytest.mark.parametrize('changes',[{'lambda_edit':-1},{'lambda_skeleton':float('nan')},
    {'difficulty_weights':(0,)*5},{'difficulty_weights':(1,)*6},{'skeleton_weights':(1,-1,1,1)}])
def test_invalid_weights_rejected(changes):
    with pytest.raises(ValueError):fixture(replace(ScoreConfig(),**changes))


def test_boundary_validation_and_equal_boundaries():
    _,_,model=fixture()
    ScoringModel({a:[1.]*6 for a in ATTRS},model.vocabulary)
    for values in ([2,1,3,4,5,6],[0,1,2,3,4,5],[1,float('nan'),3,4,5,6],[]):
        b=dict(model.boundaries);b['s_hand']=values
        with pytest.raises(ValueError):ScoringModel(b,model.vocabulary)
