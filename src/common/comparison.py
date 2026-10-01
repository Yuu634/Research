"""3方式を同じ原譜・目標・制約で評価する。生成失敗を分母から除外しない。"""
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.evaluation import EvaluationConfig, evaluate_result, summarize_results
from src.common.metrics import musical_edit_distance, skeleton_similarity
from src.common.safety import violations
from src.data.drum_difficulty_calculator import HAND_INSTRUMENTS, FOOT_INSTRUMENTS


def align_note_ids(original, generated):
    """内容一致を固定し、残りを最小編集数で対応。同位置・楽器、同楽器を優先する。"""
    remaining = {n.id: n for n in original.notes}
    matched = {}
    notes = list(generated.notes)
    def exact(n):
        return n.position, n.instrument, n.level, n.technique
    for i, note in enumerate(notes):
        hit = next((n for n in remaining.values() if exact(n) == exact(note)), None)
        if hit is not None:
            matched[i] = replace(note, id=hit.id, ornament_group=hit.ornament_group, ornament_role=hit.ornament_role)
            del remaining[hit.id]
    old = list(remaining.values())
    new = [i for i in range(len(notes)) if i not in matched]
    if old and new:
        def cost(a, b):
            priority = (0 if (a.position, a.instrument) == (b.position, b.instrument)
                        else 1 if a.instrument == b.instrument else 2 if a.position == b.position else 3)
            return priority + abs(a.position-b.position)/192 + .001*(a.level != b.level)
        costs = np.array([[cost(a, notes[i]) for i in new] for a in old])
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            i = new[column]
            matched[i] = replace(notes[i], id=old[row].id)
    for i, note in enumerate(notes):
        if i not in matched:
            identity = f'generated:{i}'
            while identity in {n.id for n in original.notes}:
                identity = 'new:' + identity
            matched[i] = replace(note, id=identity)
    return replace(generated, notes=tuple(matched[i] for i in range(len(notes))),
                   fill_regions=original.fill_regions).validate()


def _edge(bar, last=False):
    start = 72 if last else 0
    grid = np.zeros((24, 5), dtype=float)
    for note in bar.notes:
        if start <= note.position < start + 24:
            grid[note.position-start] = np.maximum(grid[note.position-start], [
                note.instrument == 'kick', note.instrument in ('snare', 'snare_rim'),
                note.instrument in HAND_INSTRUMENTS, note.instrument in FOOT_INSTRUMENTS, note.level == 'Accent'])
    return grid


def context_distance(prepared, candidate):
    """隣接拍の打点差が原譜からどれだけ変わったか。欠測側を平均に含めない。"""
    values = []
    for neighbor, current_last, neighbor_last in ((prepared.previous, False, True), (prepared.following, True, False)):
        if neighbor is not None:
            edge = _edge(neighbor, neighbor_last)
            before = np.abs(_edge(prepared.original, current_last)-edge).mean()
            after = np.abs(_edge(candidate, current_last)-edge).mean()
            values.append(abs(float(after-before)))
    return float(np.mean(values)) if values else None


def sequence_costs(prepared, candidate, scorer):
    """離散譜面上で計算する学習用コスト。自動微分する代理算出式を作らない。"""
    if candidate is None:
        return dict(difficulty=1., skeleton=1., context=1., edit=1., validity=1.)
    try:
        candidate = align_note_ids(prepared.original, candidate)
        d = scorer.difficulty(candidate, prepared.previous).normalized
        difficulty = sum(abs(d[k]-prepared.target[k]) for k in prepared.target)/5
        similarity, components, _ = skeleton_similarity(prepared.original, candidate)
        skeleton = 1-similarity if any(x is not None for x in components.values()) else 0.
        context = context_distance(prepared, candidate)
        invalid = bool(violations(prepared.original, candidate, prepared.guards))
        return dict(difficulty=difficulty, skeleton=skeleton, context=context or 0.,
                    edit=min(1., musical_edit_distance(prepared.original, candidate)), validity=float(invalid))
    except ValueError:
        return dict(difficulty=1., skeleton=1., context=1., edit=1., validity=1.)


def evaluate_output(prepared, candidate, scorer, *, case_id, source_id, elapsed_seconds=None,
                    rule_count=0, failure_reason=None, config=EvaluationConfig()):
    if candidate is not None:
        candidate = align_note_ids(prepared.original, candidate)
    row = evaluate_result(prepared, candidate, scorer, case_id=case_id, source_id=source_id,
                          elapsed_seconds=elapsed_seconds, rule_count=rule_count, failure_reason=failure_reason, config=config)
    row['context_distance'] = context_distance(prepared, candidate) if row['status'] == 'valid' else None
    row['bpm_preserved'] = candidate.bpm == prepared.original.bpm if candidate is not None else False
    return row


def summarize_outputs(rows):
    result = summarize_results(rows)
    context = [r['context_distance'] for r in rows if r['context_distance'] is not None]
    result['context_distance_valid'] = dict(mean=float(np.mean(context)) if context else None, n=len(context))
    result['bpm_preserved_rate'] = dict(value=sum(r['bpm_preserved'] for r in rows)/len(rows) if rows else None,
                                        denominator=len(rows))
    return result
