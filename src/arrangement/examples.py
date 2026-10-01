"""ルールの意味を人目で確認するための固定人工例。学習・選定用データではない。"""
from dataclasses import replace
from src.arrangement.representation import Bar, Note


def make_bar(events,**kwargs):
    return Bar(tuple(Note(f'n{i}',*event) for i,event in enumerate(events)),**kwargs).validate()


def examples():
    result={}
    for inst in ('hh_closed','snare','kick','hh_pedal','floor'):
        for step in (3,6,12,24):
            result[f'{inst}_step{step}']=make_bar([(p,inst) for p in range(0,96,step)])
    result['ghost']=make_bar([(p,'snare','Ghost' if k%2 else 'Normal') for k,p in enumerate(range(0,96,12))])
    result['hh_alternation']=make_bar([(p,'hh_closed' if k%2 else 'hh_open') for k,p in enumerate(range(0,96,12))])
    for name,tech,positions in [('flam','FLAM',[23,24]),('drag','DRAG',[22,23,24])]:
        notes=tuple(Note(f'n{k}',p,'snare','Normal' if p==24 else 'Ghost',tech if p==24 else 'HIT',name,'main' if p==24 else 'grace') for k,p in enumerate(positions))
        result[name]=Bar(notes)
    result['roll']=Bar(tuple(Note(f'n{k}',p,'snare',ornament_group='roll',ornament_role='roll') for k,p in enumerate(range(48,72,3))))
    for tech in ('RIMSHOT','XSTICK','FLAM','ROLL'):
        result['atomic_'+tech]=make_bar([(12,'snare','Normal',tech)])
    result['choke']=make_bar([(12,'crash','Normal','CHOKE')])
    result['kick_double']=make_bar([(24,'kick'),(30,'kick')])
    result['kick_triple']=make_bar([(24,'kick'),(30,'kick'),(36,'kick')])
    result['feet_alternation']=make_bar([(p,'kick' if k%2 else 'hh_pedal') for k,p in enumerate(range(0,96,6))])
    result['offbeat_kick']=make_bar([(p,'kick') for p in range(12,96,24)]+[(p,'hh_closed') for p in range(0,96,24)])
    result['near_hh']=make_bar([(p+3,'kick') for p in range(0,96,24)]+[(p,'hh_closed') for p in range(0,96,24)])
    result['interleave']=make_bar([(p,'kick' if k%2 else 'hh_closed') for k,p in enumerate(range(0,96,6))])
    result['polyrhythm']=make_bar([(p,'kick') for p in range(0,96,12)]+[(p,'hh_closed') for p in range(0,96,8)])
    result['simultaneous']=make_bar([(0,'kick'),(0,'hh_closed'),(0,'snare')])
    instruments=['snare','floor','snare','crash','snare','ride_cymbal','snare','tom_high']*2
    result['movement']=make_bar([(k*6,inst) for k,inst in enumerate(instruments)],fill_regions=((48,96),))
    result['tom_fill']=make_bar([(p,'floor' if k%2 else 'tom_low') for k,p in enumerate(range(48,96,6))],fill_regions=((48,96),))
    return result


PARAMETERS={name:dict(start=48,end=72) for name in
    ('add_roll','add_hand_roll','add_kick_triplet','add_double_bass_pattern','add_hand_foot_interleave','add_polyrhythmic_relation')}
