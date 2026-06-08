from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr

from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.data.drum_difficulty_calculator import (
    compute_all_difficulty_scores,
    discretize_difficulty_score,
)

DIFFICULTY_KEYS = ('s_tech_cls', 's_indep_cls', 's_hand_cls', 's_foot_cls', 's_move_cls')
_RAW_KEYS = ('s_tech', 's_indep', 's_hand', 's_foot', 's_move')
_CLS_TO_RAW = dict(zip(DIFFICULTY_KEYS, _RAW_KEYS))


def _compute_difficulty_from_midi(
    midi_path: str,
    difficulty_bounds: dict,
    bpm: float = 120.0,
) -> Dict[str, float]:
    """MIDI → standard トークン → 全小節の平均難易度クラスを返す。"""
    tokenizer = DrumTokenizer()
    tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)

    n_bars = len(bar_positions)
    if n_bars == 0:
        return {k: 0.0 for k in DIFFICULTY_KEYS}

    bar_edges = list(bar_positions) + [len(tokens)]
    raw_scores: Dict[str, List[float]] = {k: [] for k in _RAW_KEYS}

    for i in range(n_bars):
        bar_tokens = tokens[bar_edges[i]:bar_edges[i + 1]]
        scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)
        for rk in _RAW_KEYS:
            raw_scores[rk].append(scores[rk])

    result: Dict[str, float] = {}
    for cls_key, raw_key in _CLS_TO_RAW.items():
        mean_score = float(np.mean(raw_scores[raw_key])) if raw_scores[raw_key] else 0.0
        result[cls_key] = float(discretize_difficulty_score(mean_score, difficulty_bounds[raw_key]))

    return result


def compute_spearman_metrics(
    manifest: List[Dict],
    difficulty_bounds: dict,
    bpm: float = 120.0,
) -> Dict[str, dict]:
    """
    Spearman 相関を計算する。

    戻り値:
      spearman_control_{attr}:      SpearmanCorr(指定attr, 生成譜面attr)
      spearman_indep_{attr_a}_vs_{attr_b}: SpearmanCorr(指定attr_a, 生成譜面attr_b)  a≠b
    """
    if len(manifest) < 3:
        return {}

    specified: Dict[str, List] = {k: [] for k in DIFFICULTY_KEYS}
    measured: Dict[str, List] = {k: [] for k in DIFFICULTY_KEYS}

    for entry in manifest:
        for k in DIFFICULTY_KEYS:
            specified[k].append(entry['specified_difficulty'][k])

        gen_diff = _compute_difficulty_from_midi(
            entry['generated_midi'], difficulty_bounds, bpm
        )
        for k in DIFFICULTY_KEYS:
            measured[k].append(gen_diff[k])

    results: Dict[str, dict] = {}

    # 制御評価: SpearmanCorr(指定attr_a, 生成attr_a)
    for k in DIFFICULTY_KEYS:
        stat, pval = spearmanr(specified[k], measured[k])
        results[f'spearman_control_{k}'] = {
            'correlation': float(stat),
            'pvalue': float(pval),
        }

    # 独立制御評価: SpearmanCorr(指定attr_a, 生成attr_b)  a≠b
    for ka in DIFFICULTY_KEYS:
        for kb in DIFFICULTY_KEYS:
            if ka == kb:
                continue
            stat, pval = spearmanr(specified[ka], measured[kb])
            results[f'spearman_indep_{ka}_vs_{kb}'] = {
                'correlation': float(stat),
                'pvalue': float(pval),
            }

    return results
