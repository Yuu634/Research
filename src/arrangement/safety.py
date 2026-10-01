"""原譜を基準にする骨格保護と累積編集制限。"""
from collections import Counter
from dataclasses import dataclass, replace
import math

from src.arrangement.representation import Bar, LEVELS
from src.data.drum_difficulty_calculator import HAND_INSTRUMENTS


@dataclass(frozen=True)
class Guards:
    locked: frozenset[str] = frozenset()
    accents: frozenset[str] = frozenset()
    keep_bpm: bool = True
    keep_meter: bool = True
    keep_length: bool = True
    max_added: int | None = 8
    max_deleted: int | None = 16
    max_edit_ratio: float | None = .5


def priority(note):
    return (int(note.level=='Accent'),
            int(note.instrument in ('snare','snare_rim') and note.position in (24,72)),
            int(note.instrument=='kick' and note.position%24==0),
            int(note.position%24==0),int(note.level!='Ghost'),-note.position,note.id)


def _lock(guards,ids):return replace(guards,locked=guards.locked|frozenset(ids))


def preserve_main_kick(bar,guards=Guards(),ids=None):
    ids=set(ids) if ids is not None else {n.id for n in bar.notes if n.instrument=='kick' and n.position%24==0}
    if not ids<={n.id for n in bar.notes if n.instrument=='kick'}:raise ValueError('骨格Kick IDが不正')
    return _lock(guards,ids)


def preserve_snare_backbeat(bar,guards=Guards()):
    return _lock(guards,(n.id for n in bar.notes if n.instrument in ('snare','snare_rim') and n.position in (24,72)))


def preserve_downbeat(bar,guards=Guards()):
    ids=[]
    for p in (0,24,48,72):
        group=[n for n in bar.notes if n.position==p]
        if group:ids.append(max(group,key=priority).id)
    return _lock(guards,ids)


def preserve_accent(bar,guards=Guards()):
    return replace(guards,accents=guards.accents|frozenset(n.id for n in bar.notes if n.level=='Accent'))


def preserve_fill_boundary(bar,guards=Guards()):
    ids=[]
    for start,end in bar.fill_regions:
        notes=[n for n in bar.notes if start<=n.position<end]
        if notes:
            lo=min(n.position for n in notes); hi=max(n.position for n in notes)
            ids.extend(n.id for n in notes if n.position in (lo,hi))
    return _lock(guards,ids)


def preserve_bpm(bar,guards=Guards()):return replace(guards,keep_bpm=True)
def preserve_meter(bar,guards=Guards()):return replace(guards,keep_meter=True)
def preserve_bar_length(bar,guards=Guards()):return replace(guards,keep_length=True)


def _count(n):
    if type(n) is not int or n<0:raise ValueError('上限は非負整数')
    return n


def limit_added_notes(n,guards=Guards()):return replace(guards,max_added=_count(n))
def limit_deleted_notes(n,guards=Guards()):return replace(guards,max_deleted=_count(n))
def limit_edit_ratio(r,guards=Guards()):
    if not math.isfinite(r) or not 0<=r<=1:raise ValueError('編集率は0〜1')
    return replace(guards,max_edit_ratio=r)


def default_guards(bar):
    result=Guards()
    for function in (preserve_main_kick,preserve_snare_backbeat,preserve_downbeat,preserve_accent,preserve_fill_boundary):
        result=function(bar,result)
    return result


def edits(before,after):
    a={n.id:n for n in before.notes}; b={n.id:n for n in after.notes}
    # 注釈のみの変更は音符編集数に含めない。
    key=lambda n:(n.position,n.instrument,n.level,n.technique,n.explicit_level)
    changed=sum(key(a[k])!=key(b[k]) for k in a.keys()&b.keys())
    added=len(b.keys()-a.keys()); deleted=len(a.keys()-b.keys())
    return dict(added=added,deleted=deleted,modified=changed,
                ratio=(added+deleted+changed)/max(1,len(a)))


def violations(original,after,guards):
    after.validate(); errors=[]
    if guards.keep_bpm and original.bpm!=after.bpm:errors.append('bpm_changed')
    if guards.keep_meter and original.meter!=after.meter:errors.append('meter_changed')
    if guards.keep_length and original.length!=after.length:errors.append('length_changed')
    a={n.id:n for n in original.notes}; b={n.id:n for n in after.notes}
    if not (guards.locked|guards.accents)<=a.keys():raise ValueError('保護IDは原譜の音符を指定する')
    for key in sorted(guards.locked):
        if key not in b or (a[key].position,a[key].instrument,a[key].level,a[key].technique)!=(b[key].position,b[key].instrument,b[key].level,b[key].technique):
            errors.append('protected_note:'+key)
    for key in sorted(guards.accents):
        if key not in b or b[key].position!=a[key].position or LEVELS[b[key].level]<LEVELS[a[key].level]:
            errors.append('protected_accent:'+key)
    places=Counter((n.position,n.instrument) for n in after.notes)
    if max(places.values(),default=0)>1:errors.append('duplicate_position_instrument')
    hands=Counter(n.position for n in after.notes if n.instrument in HAND_INSTRUMENTS)
    if max(hands.values(),default=0)>2:errors.append('more_than_two_hands')
    count=edits(original,after)
    for name,value in [('added',guards.max_added),('deleted',guards.max_deleted)]:
        if value is not None and count[name]>value:errors.append('limit_'+name)
    if guards.max_edit_ratio is not None and count['ratio']>guards.max_edit_ratio+1e-12:
        errors.append('limit_edit_ratio')
    return tuple(errors)
