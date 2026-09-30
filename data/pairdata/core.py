"""小節の役割・音楽的類似度・5属性の単調減少によるペア選定。"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json

import numpy as np

from src.data.drum_difficulty_v2 import ATTRS, discretize, score_bar

VERSION = 'drum_pairdata_v1_20260930'
CONDITION_ATTRS = ATTRS[:5]


@dataclass(frozen=True)
class Config:
    local_bars: int = 32
    phrase_bars: int = 4
    min_notes: int = 4
    max_hand_rate: float = 32.
    max_foot_rate: float = 20.
    rhythm_threshold: float = .70
    skeleton_threshold: float = .90
    context_threshold: float = .70
    pair_threshold: float = .85
    difficulty_threshold: float = .20
    rhythm_weight: float = .30
    skeleton_weight: float = .50
    context_weight: float = .20
    max_per_anchor: int = 4
    max_per_piece: int = 256

    def validate(self):
        if any(v < 1 for v in (self.local_bars, self.phrase_bars, self.min_notes,
                               self.max_per_anchor, self.max_per_piece)):
            raise ValueError('件数と窓幅は正数')
        if self.local_bars % self.phrase_bars:
            raise ValueError('局所窓は仮のフレーズ長の倍数')
        for v in (self.rhythm_threshold, self.skeleton_threshold, self.context_threshold,
                  self.pair_threshold, self.difficulty_threshold):
            if not 0 <= v <= 1:
                raise ValueError('閾値は0〜1')
        weights = (self.rhythm_weight, self.skeleton_weight, self.context_weight)
        if min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError('類似度重みは非負、合計1')
        if min(self.max_hand_rate, self.max_foot_rate) <= 0:
            raise ValueError('速度上限は正数')
        return self


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=16384)
def pattern_hash(bar):
    """テンポ・トークン列順に依存しない打点＋強弱＋奏法の内容署名。"""
    return canonical_hash([(h.position, h.instrument, h.dynamic, h.special) for h in sorted(bar.hits)])


def group(instrument):
    if instrument == 'kick':
        return 0
    if instrument in ('snare', 'snare_rim'):
        return 1
    if instrument in ('hh_closed', 'hh_open', 'hh_pedal'):
        return 2
    return 3


@lru_cache(maxsize=16384)
def features(bar):
    rhythm = np.zeros((4, 96), dtype=np.float32)
    skeleton = np.zeros((3, 96), dtype=np.float32)
    toms = last_beat = 0
    for h in bar.hits:
        g = group(h.instrument)
        rhythm[g, h.position] = 1
        if g == 0:
            skeleton[0, h.position] = 2 if h.position % 24 == 0 else 1
        elif g == 1:
            skeleton[1, h.position] = 3 if h.position in (24, 72) else 1
        if h.dynamic == 1:
            skeleton[2, h.position] = 1
        toms += h.instrument in ('floor', 'tom_low', 'tom_mid', 'tom_high')
        last_beat += h.position >= 72
    # 人が付けたFillラベルではなく、タム比率と末拍への集中による代理フラグ。
    burst = last_beat >= 4 and last_beat >= 2 * max(1., (len(bar.hits)-last_beat)/3)
    fill = toms >= 2 or burst
    rhythm.setflags(write=False)
    skeleton.setflags(write=False)
    return rhythm, skeleton, bool(fill)


def similarity(a, b, weights):
    """楽器群ごとの重み付きDice。両方空の群は重みから除く。全部空なら0。"""
    total = a.sum(axis=1)+b.sum(axis=1)
    active = total > 0
    if not active.any():
        return 0.
    dice = np.divide(2*np.minimum(a,b).sum(axis=1), total,
                     out=np.zeros_like(total, dtype=float), where=active)
    w = np.asarray(weights)*active
    return float(np.dot(w,dice)/w.sum())


def context_similarity(bars, i, j):
    values = []
    for offset in (-1, 1):
        a,b = i+offset,j+offset
        if not 0 <= a < len(bars) or not 0 <= b < len(bars):
            # 曲端の役割を、通常の前後文脈と同一視しない。
            values.append(float((0 <= a < len(bars)) == (0 <= b < len(bars))))
        else:
            values.append(similarity(features(bars[a])[0],features(bars[b])[0],(.35,.35,.2,.1)))
    return float(np.mean(values))


def role(index, bar, config):
    return (index//config.local_bars, index%config.phrase_bars, features(bar)[2], bar.bpm)


def eligible(bar, previous, raw, config):
    if len(bar.hits) < config.min_notes:
        return 'too_few_notes'
    for b in (bar, previous):
        if b is not None and len({(h.position,h.instrument) for h in b.hits}) != len(b.hits):
            return 'duplicate_hit_or_previous'
    if not np.isfinite(raw).all():
        return 'nonfinite'
    if raw[2] > config.max_hand_rate or raw[3] > config.max_foot_rate:
        return 'rate_cap'
    if not features(bar)[1][:2].any():
        return 'no_kick_snare_skeleton'
    return None


def decreases(before, after, raw_before, raw_after, config):
    delta = np.asarray(before[:5],dtype=float)-np.asarray(after[:5],dtype=float)
    distance = float(np.linalg.norm(delta/7))
    valid = (np.all(delta >= 0) and distance > config.difficulty_threshold
             and np.all(np.asarray(raw_after[:5]) <= np.asarray(raw_before[:5])+1e-9))
    return bool(valid), distance


def assess(bars, i, j, raw, classes, boundaries, config):
    """iを編曲前、jを編曲後として、同じ直前小節へ接続した後まで確認。"""
    if role(i,bars[i],config) != role(j,bars[j],config):
        return None, 'role'
    if pattern_hash(bars[i]) == pattern_hash(bars[j]):
        return None, 'identical_content'
    valid, _ = decreases(classes[i],classes[j],raw[i],raw[j],config)
    if not valid:
        return None, 'original_difficulty'
    ri,si,_ = features(bars[i]); rj,sj,_ = features(bars[j])
    rhythm = similarity(ri,rj,(.35,.35,.2,.1))
    skeleton = similarity(si,sj,(.45,.45,.10))
    if rhythm < config.rhythm_threshold or skeleton < config.skeleton_threshold:
        return None, 'rhythm_or_skeleton'
    context = context_similarity(bars,i,j)
    q = config.rhythm_weight*rhythm+config.skeleton_weight*skeleton+config.context_weight*context
    if context < config.context_threshold or q <= config.pair_threshold:
        return None, 'context_or_total'
    swapped, _, flags = score_bar(bars[j], bars[i-1] if i else None)
    if flags or not np.isfinite(swapped).all():
        return None, 'swapped_invalid'
    swapped_classes = discretize(np.asarray([swapped]),boundaries)[0]
    valid,distance = decreases(classes[i],swapped_classes,raw[i],swapped,config)
    if not valid or eligible(bars[j],bars[i-1] if i else None,swapped,config):
        return None, 'swapped_difficulty'
    return dict(q=q, rhythm=rhythm, skeleton=skeleton, context=context, distance=distance,
                after_raw=list(swapped), after_classes=swapped_classes.tolist()), None


def mine(bars, raw, classes, boundaries, config):
    stats = Counter(input_bars=len(bars))
    groups = defaultdict(list)
    for i,b in enumerate(bars):
        reason = eligible(b,bars[i-1] if i else None,raw[i],config)
        if reason:
            stats['bar_reject_'+reason] += 1
        else:
            groups[role(i,b,config)].append(i)
            stats['eligible_bars'] += 1
    accepted = []
    for members in groups.values():
        for offset,i in enumerate(members):
            for j in members[offset+1:]:
                stats['candidate_pairs'] += 1
                if np.all(classes[i,:5] >= classes[j,:5]):
                    before,after = i,j
                elif np.all(classes[j,:5] >= classes[i,:5]):
                    before,after = j,i
                else:
                    stats['reject_mixed_direction'] += 1
                    continue
                result,reason = assess(bars,before,after,raw,classes,boundaries,config)
                if result is None:
                    stats['reject_'+reason] += 1
                else:
                    accepted.append((before,after,result))
    # 同じ曲の反復で件数を水増ししない。スコア同点の順位も固定。
    accepted.sort(key=lambda p:(-p[2]['q'],-p[2]['distance'],p[0],p[1]))
    chosen=[]; seen=set(); anchors=Counter()
    for before,after,result in accepted:
        signature=(pattern_hash(bars[before]),pattern_hash(bars[after]))
        if signature in seen:
            stats['reject_piece_duplicate'] += 1
        elif anchors[before] >= config.max_per_anchor or len(chosen) >= config.max_per_piece:
            stats['reject_piece_cap'] += 1
        else:
            seen.add(signature); anchors[before] += 1
            chosen.append((before,after,result))
    stats['accepted_before_global_dedup'] = len(chosen)
    return chosen,dict(stats)


def bar_tokens(starts, tokens, index, eos):
    end = starts[index+1] if index+1 < len(starts) else len(tokens)-1
    return list(map(int,tokens[starts[index]:end]))+[int(eos)]


def make_record(source, bars, starts, tokens, raw, classes, pair, eos, config):
    before,after,result=pair
    bh,ah=pattern_hash(bars[before]),pattern_hash(bars[after])
    d_before=(np.asarray(classes[before,:5],dtype=float)/7).tolist()
    d_after=(np.asarray(result['after_classes'][:5],dtype=float)/7).tolist()
    def excerpt(index):
        return bar_tokens(starts,tokens,index,eos) if 0 <= index < len(bars) else None
    return dict(pair_id=canonical_hash([bh,ah]), before_pattern=bh, after_pattern=ah,
        X_before=excerpt(before), X_after=excerpt(after), bar_positions=[0],
        d_before=d_before,d_after=d_after,delta_d=(np.asarray(d_after)-d_before).tolist(),
        raw_before=raw[before].tolist(),raw_after=result['after_raw'],
        difficulty_before=dict(zip(ATTRS,map(int,classes[before]))),
        difficulty_after=dict(zip(ATTRS,map(int,result['after_classes']))),
        original_after_raw=raw[after].tolist(),
        original_after_classes=dict(zip(ATTRS,map(int,classes[after]))),
        q=result['q'],similarities={k:result[k] for k in ('rhythm','skeleton','context')},
        difficulty_distance=result['distance'],
        source=dict(id=source['id'],dataset=source['dataset'],split=source['split'],
                    before_bar=before,after_bar=after,sha256=source['output_sha256'],
                    original_sha256=source['source_metadata']['source_sha256'],
                    license=source['source_metadata']['license']),
        context=dict(section=None,section_source='unavailable',local_block=before//config.local_bars,
            phrase_phase_proxy=before%config.phrase_bars,fill_proxy=features(bars[before])[2],
            bpm=bars[before].bpm,previous=excerpt(before-1),following=excerpt(before+1),
            original_after_previous=excerpt(after-1),original_after_following=excerpt(after+1),
            after_difficulty_context='before_previous'))
