"""参照一覧の65編集ルール。純粋な候補生成と、安全制約付き適用を分離する。"""
from dataclasses import dataclass, replace
from collections import defaultdict
import math

from src.common.representation import Bar, Note
from src.common.safety import Guards, default_guards, edits, priority, violations
from src.data.drum_difficulty_calculator import HAND_INSTRUMENTS, FOOT_INSTRUMENTS, DISTANCE_MATRIX, TOKEN_TO_DISTANCE_KEY

HH={'hh_closed','hh_open'}
SNARE={'snare','snare_rim'}
TOMS={'tom_high','tom_mid','tom_low','floor'}
CYMBALS={'hh_closed','hh_open','ride_cymbal','ride_bell','crash','splash','china'}
REGISTRY={}


@dataclass(frozen=True)
class Context:
    protected: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Result:
    rule: str
    status: str
    before: Bar
    after: Bar
    reason: tuple[str,...]
    step_edits: dict
    cumulative_edits: dict


class NotApplicable(Exception):pass


def rule(function):
    REGISTRY[function.__name__]=function
    return function


def _ratio(r):
    if not math.isfinite(r) or not 0<=r<=1:raise ValueError('割合は0〜1')
    return r


def _positive(n):
    if type(n) is not int or n<1:raise ValueError('値は正整数')
    return n


def _interval(bar,start,end):
    if start is None or end is None:raise NotApplicable('explicit_interval_required')
    if type(start) is not int or type(end) is not int or not 0<=start<end<=bar.length:
        raise ValueError('区間は小節内の整数の半開区間')
    return start,end


def _selected(bar,instruments,ctx):
    return [n for n in bar.notes if n.instrument in instruments and n.id not in ctx.protected]


def _delete(bar,notes,ctx):
    ids={n.id for n in notes}-ctx.protected
    return bar.with_notes(n for n in bar.notes if n.id not in ids)


def _replace(bar,updates,ctx):
    return bar.with_notes(updates.get(n.id,n) if n.id not in ctx.protected else n for n in bar.notes)


def _instrument(bar,source,target,ctx):
    return _replace(bar,{n.id:replace(n,instrument=target,technique='HIT',explicit_level=True)
                         for n in bar.notes if n.instrument in source},ctx)


def _add(bar,specs):
    notes=list(bar.notes); ids={n.id for n in notes}; occupied={(n.position,n.instrument) for n in notes}; serial=0
    for pos,inst,level,tech,group,role in specs:
        if not 0<=pos<bar.length or (pos,inst) in occupied:continue
        while f'added{serial}' in ids:serial+=1
        nid=f'added{serial}'; ids.add(nid); occupied.add((pos,inst))
        notes.append(Note(nid,pos,inst,level,tech,group,role))
    return bar.with_notes(notes)


def _runs(bar,instruments,step=None,max_gap=None):
    positions=sorted({n.position for n in bar.notes if n.instrument in instruments})
    groups=[]
    for p in positions:
        gap=p-groups[-1][-1] if groups else None
        if not groups or (gap!=step if step is not None else gap>max_gap):groups.append([])
        groups[-1].append(p)
    return groups


def _thin(bar,instruments,step,ctx):
    positions={p for run in _runs(bar,instruments,step=step) if len(run)>=4 for p in run[1::2]}
    return _delete(bar,[n for n in bar.notes if n.instrument in instruments and n.position in positions],ctx)


def _expand(bar,instrument,step,ctx,parts=2):
    specs=[]
    for run in _runs(bar,{instrument},step=step):
        if len(run)<2:continue
        for pos in run:
            if pos+step not in run and pos+step!=bar.length:continue
            for k in range(1,parts):specs.append((pos+k*(step//parts),instrument,'Normal','HIT',None,'main'))
    return _add(bar,specs)


def _limit_interval(bar,instruments,t,ctx):
    if not math.isfinite(t) or t<=0:raise ValueError('tは秒単位の正数')
    chosen=[]; remove=[]
    groups=defaultdict(list)
    for n in bar.notes:
        if n.instrument in instruments:groups[n.position].append(n)
    # 同時打ちは一つの時刻として扱い、両手の同時演奏を連打と誤認しない。
    ranked=sorted(groups,key=lambda p:(any(n.id in ctx.protected for n in groups[p]),max(priority(n) for n in groups[p])),reverse=True)
    seconds_per_position=60/bar.bpm/24
    for p in ranked:
        if any(abs(p-q)*seconds_per_position<t-1e-12 for q in chosen) and not any(n.id in ctx.protected for n in groups[p]):
            remove.extend(groups[p])
        else:chosen.append(p)
    return _delete(bar,remove,ctx)


def _snap(bar,instruments,grid,ctx,max_shift,coalesce=False):
    if type(max_shift) is not int or max_shift<0:raise ValueError('max_shiftは非負のグリッド数')
    grid=sorted(set(grid)); updates={}
    if not grid:return bar
    for n in _selected(bar,instruments,ctx):
        p=min(grid,key=lambda p:(abs(p-n.position),p))
        if abs(p-n.position)<=max_shift:updates[n.id]=replace(n,position=p)
    result=_replace(bar,updates,ctx)
    if coalesce:
        groups=defaultdict(list)
        for n in result.notes:groups[n.position,n.instrument].append(n)
        remove=[]
        for members in groups.values():
            survivor=max(members,key=lambda n:(n.id in ctx.protected,priority(n)))
            remove.extend(n for n in members if n.id!=survivor.id and n.id not in ctx.protected)
        result=_delete(result,remove,ctx)
    return result


def _ornament_remove(bar,technique,ctx):
    tagged=[n for n in bar.notes if n.technique==technique]
    if not tagged:raise NotApplicable('explicit_'+technique.lower()+'_annotation_required')
    blocked={n.ornament_group for n in tagged if n.id in ctx.protected and n.ornament_group is not None}
    tagged=[n for n in tagged if n.id not in ctx.protected and n.ornament_group not in blocked]
    groups={n.ornament_group for n in tagged if n.ornament_group is not None}
    remove=[]; updates={}
    for n in bar.notes:
        if n.ornament_group in groups and n.ornament_role=='grace':remove.append(n)
        elif n in tagged:updates[n.id]=replace(n,technique='HIT',explicit_level=True,ornament_group=None,ornament_role='main')
    return _replace(_delete(bar,remove,ctx),updates,ctx)


@rule
def remove_ghost_notes(bar,ctx=Context()):return _delete(bar,[n for n in bar.notes if n.level=='Ghost'],ctx)
@rule
def reduce_ghost_note_ratio(bar,r=.5,ctx=Context()):
    notes=sorted([n for n in bar.notes if n.level=='Ghost'],key=priority)
    return _delete(bar,notes[:math.floor(len(notes)*_ratio(r))],ctx)
@rule
def convert_open_hh_to_closed(bar,ctx=Context()):return _instrument(bar,{'hh_open'},'hh_closed',ctx)
@rule
def remove_hh_open_close_alternation(bar,ctx=Context()):
    hh=sorted([n for n in bar.notes if n.instrument in HH],key=lambda n:n.position)
    if not any(a.instrument!=b.instrument for a,b in zip(hh,hh[1:])):return bar
    return convert_open_hh_to_closed(bar,ctx)
@rule
def remove_flams(bar,ctx=Context()):return _ornament_remove(bar,'FLAM',ctx)
@rule
def remove_drag(bar,ctx=Context()):return _ornament_remove(bar,'DRAG',ctx)
@rule
def remove_roll(bar,ctx=Context()):
    tagged=[n for n in bar.notes if n.technique=='ROLL' or n.ornament_role=='roll']
    if not tagged:raise NotApplicable('explicit_roll_annotation_required')
    groups=defaultdict(list)
    for n in tagged:groups[n.ornament_group or n.id].append(n)
    result=bar
    for members in groups.values():
        representative=max(members,key=lambda n:(n.id in ctx.protected,priority(n)))
        result=_delete(result,[n for n in members if n.id!=representative.id],ctx)
        result=_replace(result,{representative.id:replace(representative,technique='HIT',explicit_level=True,ornament_group=None,ornament_role='main')},ctx)
    return result
@rule
def convert_rimshot_to_snare(bar,ctx=Context()):
    return _replace(bar,{n.id:replace(n,instrument='snare',technique='HIT',explicit_level=True)
                         for n in bar.notes if n.technique=='RIMSHOT'},ctx)
@rule
def convert_cross_stick_to_snare(bar,ctx=Context()):
    return _replace(bar,{n.id:replace(n,instrument='snare',technique='HIT',explicit_level=True)
                         for n in bar.notes if n.technique=='XSTICK' or n.instrument=='snare_rim'},ctx)
@rule
def remove_choke(bar,ctx=Context()):
    return _replace(bar,{n.id:replace(n,technique='HIT',explicit_level=True) for n in bar.notes if n.technique=='CHOKE'},ctx)
@rule
def add_ghost_notes(bar,ctx=Context()):
    notes=sorted([n for n in bar.notes if n.instrument=='snare'],key=lambda n:n.position)
    return _add(bar,[((a.position+b.position)//2,'snare','Ghost','HIT',None,'main') for a,b in zip(notes,notes[1:]) if b.position-a.position>=2])
@rule
def add_open_hh(bar,ctx=Context()):
    return _replace(bar,{n.id:replace(n,instrument='hh_open') for n in bar.notes if n.instrument=='hh_closed' and n.position%24==12},ctx)
@rule
def add_flams(bar,targets=None,offset=1,ctx=Context()):
    _positive(offset)
    candidates=_selected(bar,SNARE|TOMS,ctx)
    if targets is not None:
        if not set(targets)<={n.id for n in bar.notes}:raise ValueError('targetsに未知の音符ID')
        candidates=[n for n in candidates if n.id in targets]
    result=bar
    for n in candidates:
        if n.position<offset or any(m.position==n.position-offset and m.instrument==n.instrument for m in result.notes):continue
        group='flam:'+n.id
        result=_replace(result,{n.id:replace(n,technique='FLAM',ornament_group=group)},ctx)
        result=_add(result,[(n.position-offset,n.instrument,'Ghost','HIT',group,'grace')])
    return result


def _roll(bar,start,end,instrument,step,ctx):
    start,end=_interval(bar,start,end); _positive(step)
    if instrument not in HAND_INSTRUMENTS:raise ValueError('手の楽器を指定')
    group=f'roll:{instrument}:{start}:{end}'
    updates={n.id:replace(n,ornament_group=group,ornament_role='roll') for n in bar.notes if n.instrument==instrument and start<=n.position<end}
    result=_replace(bar,updates,ctx)
    return _add(result,[(p,instrument,'Normal','HIT',group,'roll') for p in range(start,end,step)])
@rule
def add_roll(bar,start=None,end=None,instrument='snare',step=3,ctx=Context()):return _roll(bar,start,end,instrument,step,ctx)


@rule
def hh_16_to_8(bar,ctx=Context()):return _thin(bar,HH,6,ctx)
@rule
def hh_32_to_16(bar,ctx=Context()):return _thin(bar,HH,3,ctx)
@rule
def snare_16_to_8(bar,ctx=Context()):return _thin(bar,SNARE,6,ctx)
@rule
def snare_32_to_16(bar,ctx=Context()):return _thin(bar,SNARE,3,ctx)
@rule
def tom_16_to_8(bar,ctx=Context()):return _thin(bar,TOMS,6,ctx)
@rule
def reduce_hand_roll_density(bar,r=.5,ctx=Context()):
    groups=defaultdict(list)
    for n in bar.notes:
        if n.instrument in HAND_INSTRUMENTS and n.ornament_role=='roll' and n.ornament_group:groups[n.ornament_group].append(n)
    if not groups:raise NotApplicable('explicit_roll_members_required')
    result=bar
    for members in groups.values():
        keep=math.ceil(len(members)*_ratio(r))
        result=_delete(result,sorted(members,key=priority)[:len(members)-keep],ctx)
    return result
@rule
def remove_consecutive_hand_hits(bar,n=3,ctx=Context()):
    _positive(n)
    positions={p for run in _runs(bar,HAND_INSTRUMENTS,max_gap=6) for p in run[n::n+1]}
    return _delete(bar,[h for h in bar.notes if h.instrument in HAND_INSTRUMENTS and h.position in positions],ctx)
@rule
def limit_hand_min_interval(bar,t=.125,ctx=Context()):return _limit_interval(bar,HAND_INSTRUMENTS,t,ctx)
@rule
def hh_8_to_16(bar,ctx=Context()):return _expand(bar,'hh_closed',12,ctx)
@rule
def snare_8_to_16(bar,ctx=Context()):return _expand(bar,'snare',12,ctx)


def _double(bar,instruments,offset,ctx):
    _positive(offset)
    # 元の骨格音は変更しないため、保護された打撃の後への追加も候補にできる。
    return _add(bar,[(n.position+offset,n.instrument,n.level,'HIT',None,'main') for n in bar.notes if n.instrument in instruments])
@rule
def add_hand_double_stroke(bar,offset=3,ctx=Context()):return _double(bar,HAND_INSTRUMENTS,offset,ctx)
@rule
def add_hand_roll(bar,start=None,end=None,instrument='snare',step=3,ctx=Context()):return _roll(bar,start,end,instrument,step,ctx)
@rule
def quarter_to_triplet_hand(bar,instrument='hh_closed',ctx=Context()):
    if instrument not in HAND_INSTRUMENTS:raise ValueError('手の楽器を指定')
    return _expand(bar,instrument,24,ctx,parts=3)


@rule
def kick_16_to_8(bar,ctx=Context()):return _thin(bar,{'kick'},6,ctx)
@rule
def kick_32_to_16(bar,ctx=Context()):return _thin(bar,{'kick'},3,ctx)
@rule
def remove_kick_double(bar,ctx=Context()):
    positions={p for run in _runs(bar,{'kick'},max_gap=6) if len(run)==2 for p in run}
    notes=[n for n in bar.notes if n.instrument=='kick' and n.position in positions]
    remove=[]
    for run in _runs(bar,{'kick'},max_gap=6):
        if len(run)==2:remove.extend(sorted([n for n in notes if n.position in run],key=priority)[:1])
    return _delete(bar,remove,ctx)
@rule
def remove_kick_triple(bar,ctx=Context()):
    positions={run[1] for run in _runs(bar,{'kick'},max_gap=6) if len(run)==3}
    return _delete(bar,[n for n in bar.notes if n.instrument=='kick' and n.position in positions],ctx)
@rule
def reduce_kick_density(bar,r=.5,ctx=Context()):
    notes=sorted([n for n in bar.notes if n.instrument=='kick'],key=priority)
    return _delete(bar,notes[:len(notes)-math.ceil(len(notes)*_ratio(r))],ctx)
@rule
def limit_kick_min_interval(bar,t=.125,ctx=Context()):return _limit_interval(bar,{'kick'},t,ctx)
@rule
def pedal_hh_16_to_8(bar,ctx=Context()):return _thin(bar,{'hh_pedal'},6,ctx)
@rule
def remove_alternating_feet(bar,ctx=Context()):
    feet=sorted([n for n in bar.notes if n.instrument in FOOT_INSTRUMENTS],key=lambda n:n.position)
    remove=[]
    for a,b in zip(feet,feet[1:]):
        if a.instrument!=b.instrument and 0<b.position-a.position<=6:remove.append(min((a,b),key=priority))
    return _delete(bar,remove,ctx)
@rule
def kick_8_to_16(bar,ctx=Context()):return _expand(bar,'kick',12,ctx)
@rule
def add_kick_double(bar,offset=3,ctx=Context()):return _double(bar,{'kick'},offset,ctx)
@rule
def add_kick_triplet(bar,start=None,end=None,ctx=Context()):
    start,end=_interval(bar,start,end)
    if (end-start)%3:raise ValueError('3等分できる区間を指定')
    return _add(bar,[(p,'kick','Normal','HIT',None,'main') for p in range(start,end,(end-start)//3)])
@rule
def add_double_bass_pattern(bar,start=None,end=None,step=6,ctx=Context()):
    start,end=_interval(bar,start,end); _positive(step)
    return _add(bar,[(p,'kick','Normal','HIT',None,'main') for p in range(start,end,step)])
@rule
def quarter_to_triplet_foot(bar,instrument='kick',ctx=Context()):
    if instrument not in FOOT_INSTRUMENTS:raise ValueError('足の楽器を指定')
    return _expand(bar,instrument,24,ctx,parts=3)


@rule
def align_kick_to_hh(bar,max_shift=6,ctx=Context()):return _snap(bar,{'kick'},[n.position for n in bar.notes if n.instrument in HH],ctx,max_shift)
@rule
def align_kick_to_beat_grid(bar,max_shift=12,ctx=Context()):return _snap(bar,{'kick'},range(0,96,24),ctx,max_shift)
@rule
def remove_offbeat_kick(bar,ctx=Context()):return _delete(bar,[n for n in bar.notes if n.instrument=='kick' and n.position%24!=0],ctx)
@rule
def remove_syncopated_kick(bar,ctx=Context()):return align_kick_to_beat_grid(bar,ctx=ctx)
@rule
def merge_hand_foot_onsets(bar,max_shift=3,ctx=Context()):return _snap(bar,FOOT_INSTRUMENTS,[n.position for n in bar.notes if n.instrument in HAND_INSTRUMENTS],ctx,max_shift)
@rule
def remove_interleaved_hand_foot_hits(bar,ctx=Context()):
    groups=defaultdict(list)
    for n in bar.notes:groups[n.position].append(n)
    positions=sorted(groups); remove=[]
    for offset in range(len(positions)-3):
        ps=positions[offset:offset+4]
        limbs=[{n.instrument in HAND_INSTRUMENTS for n in groups[p]} for p in ps]
        if all(len(s)==1 for s in limbs) and all(limbs[k]!=limbs[k+1] and ps[k+1]-ps[k]<=6 for k in range(3)):
            remove.append(min(groups[ps[1]]+groups[ps[2]],key=priority))
    return _delete(bar,remove,ctx)
@rule
def simplify_hand_foot_pattern(bar,step=12,ctx=Context()):
    _positive(step)
    return _snap(bar,HAND_INSTRUMENTS|FOOT_INSTRUMENTS,range(0,96,step),ctx,step//2,coalesce=True)
@rule
def reduce_limb_polyrhythm(bar,ctx=Context()):
    steps=[]
    for instruments in (HAND_INSTRUMENTS,FOOT_INSTRUMENTS):
        ps=sorted({n.position for n in bar.notes if n.instrument in instruments})
        gaps={b-a for a,b in zip(ps,ps[1:])}
        if len(ps)<3 or len(gaps)!=1:raise NotApplicable('two_explicit_regular_periods_required')
        steps.append(next(iter(gaps)))
    if steps[0]==steps[1] or max(steps)%min(steps)==0:raise NotApplicable('different_non_nested_periods_required')
    return simplify_hand_foot_pattern(bar,step=12,ctx=ctx)
@rule
def remove_note_from_simultaneous_hit(bar,position=None,ctx=Context()):
    groups=defaultdict(list)
    for n in bar.notes:groups[n.position].append(n)
    for pos,notes in sorted(groups.items()):
        available=[n for n in notes if n.id not in ctx.protected]
        if len(notes)>1 and available and (position is None or pos==position):return _delete(bar,[min(available,key=priority)],ctx)
    return bar
@rule
def add_offbeat_kick(bar,ctx=Context()):return _add(bar,[(p,'kick','Normal','HIT',None,'main') for p in range(12,96,24)])
@rule
def shift_kick_off_grid(bar,offset=6,ctx=Context()):
    _positive(offset)
    return _replace(bar,{n.id:replace(n,position=n.position+offset) for n in bar.notes if n.instrument=='kick' and n.position%24==0 and n.position+offset<96},ctx)
@rule
def add_hand_foot_interleave(bar,start=None,end=None,step=6,ctx=Context()):
    start,end=_interval(bar,start,end); _positive(step)
    return _add(bar,[(p,'hh_closed' if k%2==0 else 'kick','Normal','HIT',None,'main') for k,p in enumerate(range(start,end,step))])
@rule
def add_polyrhythmic_relation(bar,start=None,end=None,ctx=Context()):
    start,end=_interval(bar,start,end)
    return _add(bar,[(p,inst,'Normal','HIT',None,'main') for inst,step in [('hh_closed',8),('kick',12)] for p in range(start,end,step)])


def _distance(a,b):return DISTANCE_MATRIX[TOKEN_TO_DISTANCE_KEY[a]][TOKEN_TO_DISTANCE_KEY[b]]


def _relocate(bar,instruments,choices,ctx,far=False,threshold=None,force=False):
    updates={}; hands=[n for n in bar.notes if n.instrument in HAND_INSTRUMENTS]
    for n in _selected(bar,instruments,ctx):
        before=[m for m in hands if m.position<n.position]; after=[m for m in hands if m.position>n.position]
        neighbors=([max(before,key=lambda m:m.position)] if before else [])+([min(after,key=lambda m:m.position)] if after else [])
        anchors=[m.instrument for m in neighbors] or ['snare']
        cost=lambda inst:sum(_distance(inst,a) for a in anchors)
        if threshold is not None and max(_distance(n.instrument,a) for a in anchors)<=threshold:continue
        target=sorted(choices,key=lambda x:((-cost(x) if far else cost(x)),x!=n.instrument,x))[0]
        if target!=n.instrument and (far or (force and n.instrument not in choices) or cost(target)<cost(n.instrument)-1e-12):
            updates[n.id]=replace(n,instrument=target,technique='HIT',explicit_level=True)
    return _replace(bar,updates,ctx)
@rule
def tom_to_nearest_tom(bar,ctx=Context()):return _relocate(bar,TOMS,TOMS,ctx)
@rule
def floor_tom_to_tom2(bar,ctx=Context()):return _instrument(bar,{'floor'},'tom_mid',ctx)
@rule
def floor_tom_to_snare(bar,ctx=Context()):return _instrument(bar,{'floor'},'snare',ctx)
@rule
def crash_to_hh(bar,ctx=Context()):return _instrument(bar,{'crash'},'hh_closed',ctx)
@rule
def ride_to_hh(bar,ctx=Context()):return _instrument(bar,{'ride_cymbal','ride_bell'},'hh_closed',ctx)
@rule
def replace_far_cymbal(bar,ctx=Context()):return _relocate(bar,CYMBALS,CYMBALS,ctx)
@rule
def reduce_tom_span(bar,ctx=Context()):return _relocate(bar,TOMS,{'tom_high','tom_mid'},ctx,force=True)
@rule
def remove_large_jump(bar,threshold=.6,ctx=Context()):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('距離閾値は0〜1')
    return _relocate(bar,HAND_INSTRUMENTS,{'snare','hh_closed','tom_mid'},ctx,threshold=threshold)
@rule
def simplify_tom_fill(bar,ctx=Context()):
    if not bar.fill_regions:raise NotApplicable('explicit_fill_region_required')
    locked={n.id for n in bar.notes if not any(a<=n.position<b for a,b in bar.fill_regions)}
    return reduce_tom_span(bar,Context(ctx.protected|frozenset(locked)))
@rule
def expand_tom_span(bar,ctx=Context()):
    notes=sorted(_selected(bar,SNARE|TOMS,ctx),key=lambda n:(n.position,n.id))
    return _replace(bar,{n.id:replace(n,instrument=('tom_high','floor')[k%2],technique='HIT',explicit_level=True) for k,n in enumerate(notes)},ctx)
@rule
def snare_to_tom_sequence(bar,ctx=Context()):
    notes=sorted(_selected(bar,SNARE,ctx),key=lambda n:(n.position,n.id))
    return _replace(bar,{n.id:replace(n,instrument=('tom_high','tom_mid','floor')[k%3],technique='HIT',explicit_level=True) for k,n in enumerate(notes) if k%2==1},ctx)
@rule
def add_large_jump(bar,ctx=Context()):return _relocate(bar,SNARE|TOMS,{'tom_high','floor','crash','ride_cymbal'},ctx,far=True)


def apply_rule(bar,name,*,original=None,guards=None,**parameters):
    """Beam候補の適用入口。拒否時は入力を返し、原譜に対する累積制限を守る。"""
    bar.validate(); original=bar if original is None else original; original.validate()
    guards=default_guards(original) if guards is None else guards
    if name not in REGISTRY:raise ValueError('未知のルール: '+name)
    if 'ctx' in parameters:raise ValueError('保護情報はguardsで指定してください')
    context=Context(guards.locked|guards.accents)
    try:proposal=REGISTRY[name](bar,ctx=context,**parameters)
    except NotApplicable as e:
        return Result(name,'unchanged',bar,bar,(str(e),),edits(bar,bar),edits(original,bar))
    errors=violations(original,proposal,guards)
    if errors:return Result(name,'rejected',bar,bar,errors,edits(bar,bar),edits(original,bar))
    changed=proposal!=bar
    return Result(name,'applied' if changed else 'unchanged',bar,proposal,
                  () if changed else ('no_eligible_change',),edits(bar,proposal),edits(original,proposal))
