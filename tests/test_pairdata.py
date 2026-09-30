"""図のペア選定、難易度の向き、文脈、漏洩・原譜保持を確認する。"""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from data.pairdata.core import (
    Config,assess,bar_tokens,decreases,eligible,features,mine,pattern_hash,role,similarity,
)
from src.data.drum_difficulty_v2 import ATTRS,Bar,Hit,discretize,score_piece
from data.pairdata.build import WORKER,mine_batch,package
from data.pairdata.io import file_hash,read_pickle,write_json,write_pickle


def beat(dense=False,bpm=120):
    hits=[Hit(p,'kick') for p in (0,24,48,72)]+[Hit(p,'snare') for p in (24,72)]
    hits += [Hit(p,'hh_closed') for p in range(0,96,12 if dense else 24)]
    return Bar(bpm,tuple(sorted(hits)))


def fixture():
    bars=[beat() for _ in range(16)]
    bars[4]=beat(True)
    raw=score_piece(bars)[0]
    bounds={a:[.5,1,2,4,8,16] for a in ATTRS}
    return bars,raw,discretize(raw,bounds),bounds


def test_hihat_simplification_preserves_skeleton_and_decreases():
    bars,raw,classes,bounds=fixture()
    result,reason=assess(bars,4,8,raw,classes,bounds,Config())
    assert reason is None and result['q']>.85
    assert result['skeleton']==1
    assert np.all(np.array(result['after_classes'][:5])<=classes[4,:5])
    assert assess(bars,8,4,raw,classes,bounds,Config())[0] is None


def test_conflicting_attributes_are_not_simplification():
    assert not decreases([7,0,7,7,7],[0,1,0,0,0],[7,0,7,7,7],[0,1,0,0,0],Config())[0]
    assert not decreases([4]*5,[3]*5,[4]*5,[3,3,5,3,3],Config())[0]


def test_reconnecting_to_before_context_rejects_new_fast_kick():
    bars,_,_,bounds=fixture()
    bars[3]=Bar(120,(Hit(95,'kick'),))
    bars[4]=Bar(120,tuple(sorted(Hit(6,h.instrument,h.dynamic,h.special)
        if h.instrument=='kick' and h.position==0 else h for h in bars[4].hits)))
    raw=score_piece(bars)[0]; classes=discretize(raw,bounds)
    config=replace(Config(),rhythm_threshold=0,skeleton_threshold=0,context_threshold=0,pair_threshold=0)
    assert decreases(classes[4],classes[8],raw[4],raw[8],config)[0]
    assert assess(bars,4,8,raw,classes,bounds,config)==(None,'swapped_difficulty')


def test_reference_rhythm_does_not_control_direction():
    assert decreases([4]*5+[0],[2]*5+[7],[4]*5+[0],[2]*5+[100],Config())[0]


def test_threshold_is_strict_and_empty_patterns_are_not_similar():
    assert not decreases([1,0,0,0,0],[0]*5,[1,0,0,0,0],[0]*5,Config())[0]
    assert similarity(np.zeros((4,96)),np.zeros((4,96)),(.35,.35,.2,.1))==0


def test_tempo_phrase_and_local_roles_must_match():
    config=Config()
    assert role(4,beat(),config)==role(8,beat(),config)
    assert role(4,beat(),config)!=role(5,beat(),config)
    assert role(4,beat(),config)!=role(36,beat(),config)
    assert role(4,beat(),config)!=role(8,beat(bpm=125),config)


def test_signature_ignores_tempo_and_order_but_retains_notes():
    b=beat()
    assert pattern_hash(b)==pattern_hash(Bar(180,tuple(reversed(b.hits))))
    assert pattern_hash(b)!=pattern_hash(beat(True))


def test_quality_filters_duplicate_prefix_and_extreme_speed():
    b=beat(); duplicate=Bar(120,b.hits+(b.hits[0],))
    assert eligible(b,duplicate,np.zeros(6),Config())=='duplicate_hit_or_previous'
    assert eligible(b,None,np.array([0,0,33,0,0,0]),Config())=='rate_cap'
    assert eligible(Bar(120,()),None,np.zeros(6),Config())=='too_few_notes'


def test_mining_deduplicates_repetition_and_is_deterministic():
    bars,raw,classes,bounds=fixture()
    first,stats=mine(bars,raw,classes,bounds,Config())
    second,_=mine(bars,raw,classes,bounds,Config())
    assert first==second and len(first)==1
    assert first[0][0]==4 and stats['reject_piece_duplicate']>0


def test_last_bar_excerpt_keeps_eos_without_extra_bar():
    assert bar_tokens([0,2],[8,9,8,7,0],1,0)==[8,7,0]


def test_shifted_backbeat_reduces_skeleton_score():
    original=beat()
    changed=Bar(120,tuple(Hit(h.position-6,h.instrument,h.dynamic,h.special)
                         if h.instrument=='snare' else h for h in original.hits))
    assert similarity(features(original)[1],features(changed)[1],(.45,.45,.1))<.9


@pytest.mark.parametrize('kw',[{'local_bars':31},{'pair_threshold':1.1},{'rhythm_weight':-1},{'max_per_piece':0}])
def test_invalid_configuration_rejected(kw):
    with pytest.raises(ValueError):replace(Config(),**kw).validate()


def test_packaging_quarantines_shared_bars_and_deduplicates_pairs(tmp_path):
    root=tmp_path/'out'; work=tmp_path/'work'; meta=tmp_path/'meta'
    for p in (root,work,meta):p.mkdir()
    for name in ('bounds.json','drum_vocab.pkl'):(meta/name).write_bytes(b'fixture')
    def row(split,key,before,after):
        return dict(pair_id=before+'-'+after,before_pattern=before,after_pattern=after,
            source=dict(id=key,split=split,dataset='fixture'),q=.95,difficulty_distance=.3,
            difficulty_before={a:3 for a in ATTRS},difficulty_after={a:2 for a in ATTRS})
    groups=[('test',[row('test','001','a','b')]),
            ('val',[row('val','002','c','b'),row('val','002','d','e')]),
            ('train',[row('train','003','a','b'),row('train','003','f','g'),row('train','003','f','g')])]
    tasks=[]; records=[]
    for name,rows in groups:
        sources=[dict(id=rows[0]['source']['id'])]; records.extend(sources); tasks.append((name,sources))
        path=work/(name+'.pkl.gz')
        write_pickle(path,dict(rows=rows,pieces=[]))
        write_json(path.with_suffix('.json'),dict(ids=[s['id'] for s in sources],sha256=file_hash(path),stats={}))
    package(SimpleNamespace(output=root,work=work,source_meta=meta,shard_size=2),tasks,records,Config())
    summary=json.loads((root/'summary.json').read_text())
    assert summary['pairs']==3
    assert summary['packaging_rejections']=={'val/cross_split_bar_overlap':1,'train/exact_pair_duplicate':2}
    index=json.loads((root/'index.json').read_text())
    assert [r['pair_id'] for s in index['shards'] for r in read_pickle(root/s['path'])]==['a-b','d-e','f-g']


def test_resume_checks_content_not_only_filename(tmp_path,monkeypatch):
    monkeypatch.setitem(WORKER,'work',tmp_path)
    path=tmp_path/'batch.pkl.gz'
    write_pickle(path,{'rows':[]})
    receipt=dict(ids=['001'],sha256=file_hash(path),stats={},pairs=0)
    write_json(path.with_suffix('.json'),receipt)
    assert mine_batch(('batch',[{'id':'001'}]))==receipt
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='整合性'):
        mine_batch(('batch',[{'id':'001'}]))
