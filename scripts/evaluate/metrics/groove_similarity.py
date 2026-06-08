from typing import List, Set, Tuple
from .onset_utils import parse_onsets, HIHAT_INSTRUMENTS


def _iou(set_a: Set, set_b: Set) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def hihat_similarity(ref_tokens: List[str], gen_tokens: List[str]) -> float:
    """HH_CLOSED / HH_OPEN / HH_PEDAL の発音タイミング IoU。"""
    def hh_positions(tokens):
        return {
            (bar, beat, pos)
            for bar, beat, pos, inst in parse_onsets(tokens)
            if inst in HIHAT_INSTRUMENTS
        }

    return _iou(hh_positions(ref_tokens), hh_positions(gen_tokens))


def rhythm_similarity(ref_tokens: List[str], gen_tokens: List[str]) -> float:
    """ハイハット以外の発音タイミング IoU。"""
    def non_hh_positions(tokens):
        return {
            (bar, beat, pos)
            for bar, beat, pos, inst in parse_onsets(tokens)
            if inst not in HIHAT_INSTRUMENTS
        }

    return _iou(non_hh_positions(ref_tokens), non_hh_positions(gen_tokens))


def groove_similarity(
    ref_tokens: List[str],
    gen_tokens: List[str],
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """α × Rhythm Similarity + β × Hihat Similarity。"""
    rs = rhythm_similarity(ref_tokens, gen_tokens)
    hs = hihat_similarity(ref_tokens, gen_tokens)
    return alpha * rs + beta * hs
