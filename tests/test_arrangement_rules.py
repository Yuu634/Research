"""参照ルールの網羅性、編集の意味、骨格保護、累積制約を確認する。"""
from dataclasses import replace
from pathlib import Path
import re

import pytest

from src.arrangement import safety
from src.arrangement.examples import examples,make_bar,PARAMETERS
from src.arrangement.representation import Bar,Note,from_tokens,to_tokens
from src.arrangement.rules import REGISTRY,apply_rule
from src.arrangement.safety import Guards,edits,violations

RELAXED=Guards(max_added=None,max_deleted=None,max_edit_ratio=None)


def test_spec_covers_all_76_names():
    names=re.findall(r'^([a-z][a-z0-9_]*)\(',Path('drum_arrangement_rule_functions.txt').read_text(),re.M)
    assert len(names)==76 and len(REGISTRY)==65
    assert all(n in REGISTRY or callable(getattr(safety,n,None)) for n in names)


@pytest.mark.parametrize('name',sorted(REGISTRY))
def test_every_rule_has_deterministic_effect_and_keeps_input(name):
    changed=[]
    for bar in examples().values():
        before=repr(bar)
        result=apply_rule(bar,name,guards=RELAXED,**PARAMETERS.get(name,{}))
        assert repr(bar)==before
        assert result==apply_rule(bar,name,guards=RELAXED,**PARAMETERS.get(name,{}))
        if result.status=='applied':
            assert not violations(bar,result.after,RELAXED)
            if result.step_edits['ratio']>0:changed.append(result)
        else:assert result.after==bar
    assert changed,name


def test_hh_sixteenth_to_eighth_only_changes_hh():
    bar=make_bar([(p,'hh_closed') for p in range(0,96,6)]+[(0,'kick'),(24,'snare')])
    result=apply_rule(bar,'hh_16_to_8')
    assert result.status=='applied'
    assert [n.position for n in result.after.notes if n.instrument=='hh_closed']==list(range(0,96,12))
    assert {n for n in result.after.notes if n.instrument!='hh_closed'}=={n for n in bar.notes if n.instrument!='hh_closed'}


def test_incomplete_and_irregular_runs_are_not_thinned():
    for positions in ([0,6,12],[0,6,13,18]):
        assert apply_rule(make_bar([(p,'hh_closed') for p in positions]),'hh_16_to_8',guards=RELAXED).status=='unchanged'


def test_eighth_expansion_and_quarter_triplet_grid():
    b=examples()['hh_closed_step12']
    assert sorted(n.position for n in apply_rule(b,'hh_8_to_16',guards=RELAXED).after.notes)==list(range(0,96,6))
    b=examples()['kick_step24']
    assert sorted(n.position for n in apply_rule(b,'quarter_to_triplet_foot',guards=RELAXED).after.notes)==list(range(0,96,8))


def test_missing_ornament_fill_and_interval_do_not_guess():
    b=examples()['snare_step3']
    for name in ('remove_flams','remove_drag','remove_roll','reduce_hand_roll_density','simplify_tom_fill','add_roll'):
        result=apply_rule(b,name,guards=RELAXED)
        assert result.status=='unchanged' and 'required' in result.reason[0]


def test_flam_removal_preserves_main_and_respects_protection():
    b=examples()['flam']
    after=apply_rule(b,'remove_flams',guards=RELAXED).after
    assert len(after.notes)==1 and after.notes[0].position==24 and after.notes[0].technique=='HIT'
    guards=replace(RELAXED,locked=frozenset({'n1'}))
    assert apply_rule(b,'remove_flams',guards=guards).after==b


def test_rim_mapping_distinguishes_cross_stick_from_rimshot():
    b=make_bar([(12,'snare_rim','Normal','ACCENT')])
    assert apply_rule(b,'convert_rimshot_to_snare',guards=RELAXED).status=='unchanged'
    assert apply_rule(b,'convert_cross_stick_to_snare',guards=RELAXED).after.notes[0].instrument=='snare'


def test_seconds_interval_depends_on_bpm_and_keeps_simultaneous_hands():
    b=make_bar([(0,'snare'),(0,'hh_closed'),(6,'snare')])
    slow=apply_rule(b,'limit_hand_min_interval',t=.1,guards=RELAXED)
    fast=apply_rule(replace(b,bpm=240),'limit_hand_min_interval',t=.1,guards=RELAXED)
    assert len(slow.after.notes)==3 and len(fast.after.notes)==2


def test_collision_rejected_and_triplet_coalescing_audited():
    b=make_bar([(11,'kick'),(13,'kick'),(12,'hh_closed')])
    result=apply_rule(b,'align_kick_to_hh',guards=RELAXED)
    assert result.status=='rejected' and 'duplicate_position_instrument' in result.reason and result.after==b
    b=examples()['polyrhythm']
    result=apply_rule(b,'reduce_limb_polyrhythm',guards=RELAXED)
    assert result.status=='applied' and result.step_edits['deleted']>0


def test_three_hands_added_is_rejected():
    b=make_bar([(0,'snare'),(0,'floor'),(12,'hh_closed')])
    # 0へHHを追加する区間指定。既存の2手へ3本目を要求するため拒否。
    result=apply_rule(b,'add_hand_roll',start=0,end=1,instrument='hh_closed',guards=RELAXED)
    assert result.status=='rejected' and 'more_than_two_hands' in result.reason


def test_default_skeleton_and_fill_boundary_protection():
    b=make_bar([(0,'kick'),(12,'kick'),(24,'snare','Accent'),(48,'floor'),(72,'tom_mid')],fill_regions=((48,96),))
    g=safety.default_guards(b)
    assert {'n0','n2','n3','n4'}<=g.locked and 'n2' in g.accents
    after=apply_rule(b,'remove_offbeat_kick').after
    assert {n.id for n in after.notes}=={'n0','n2','n3','n4'}


def test_cumulative_addition_limit_cannot_reset_each_step():
    b=make_bar([(0,'snare'),(24,'snare'),(48,'snare'),(72,'snare')])
    guards=replace(RELAXED,max_added=4)
    first=apply_rule(b,'add_offbeat_kick',guards=guards)
    assert first.status=='applied' and first.cumulative_edits['added']==4
    second=apply_rule(first.after,'add_ghost_notes',original=b,guards=guards)
    assert second.status=='rejected' and 'limit_added' in second.reason


def test_all_safety_setters_and_bounds():
    b=examples()['kick_step24']
    assert safety.preserve_bpm(b).keep_bpm
    assert safety.preserve_meter(b).keep_meter
    assert safety.preserve_bar_length(b).keep_length
    assert safety.limit_added_notes(0).max_added==0
    assert safety.limit_deleted_notes(0).max_deleted==0
    assert safety.limit_edit_ratio(.2).max_edit_ratio==.2
    for fn,value in [(safety.limit_added_notes,-1),(safety.limit_deleted_notes,1.2),(safety.limit_edit_ratio,1.1)]:
        with pytest.raises(ValueError):fn(value)
    assert 'bpm_changed' in violations(b,replace(b,bpm=180),RELAXED)


def test_codec_preserves_technique_and_ghost():
    tokens=['<TEMPO_120>','<BAR>','<BEAT_1>','<POS_0>','SNARE_FLAM_Normal','<POS_3>','SNARE_HIT_Ghost','<POS_6>','SNARE_ROLL','<EOS>']
    assert to_tokens(from_tokens(tokens))==tokens
    with pytest.raises(ValueError,match='語彙'):
        to_tokens(from_tokens(tokens),vocabulary={'<EOS>'})


@pytest.mark.parametrize('kwargs',[{'r':-1},{'r':float('nan')},{'r':2}])
def test_invalid_ratio_rejected(kwargs):
    with pytest.raises(ValueError):apply_rule(examples()['ghost'],'reduce_ghost_note_ratio',**kwargs)


def test_ghost_ratio_means_removed_fraction_and_kick_ratio_remaining():
    b=make_bar([(p,'snare','Ghost') for p in range(0,96,12)])
    assert len(apply_rule(b,'reduce_ghost_note_ratio',r=.25,guards=RELAXED).after.notes)==6
    b=examples()['kick_step12']
    assert len(apply_rule(b,'reduce_kick_density',r=.25,guards=RELAXED).after.notes)==2


def test_addition_at_end_never_wraps_into_next_bar():
    b=make_bar([(95,'kick')])
    assert apply_rule(b,'add_kick_double',guards=RELAXED).after==b


def test_double_can_add_after_protected_main_hit_without_editing_it():
    b=make_bar([(0,'kick')])
    guards=replace(RELAXED,locked=frozenset({'n0'}))
    result=apply_rule(b,'add_kick_double',guards=guards)
    assert result.status=='applied'
    assert b.notes[0] in result.after.notes
    assert result.step_edits['added']==1


def test_edit_counts_substitution_once():
    b=make_bar([(12,'floor')])
    after=apply_rule(b,'floor_tom_to_snare',guards=RELAXED).after
    assert edits(b,after)==dict(added=0,deleted=0,modified=1,ratio=1)
