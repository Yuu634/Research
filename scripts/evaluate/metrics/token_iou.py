from typing import List
from .onset_utils import parse_onsets


def token_iou(ref_tokens: List[str], gen_tokens: List[str]) -> float:
    """全く同じ位置で同じ楽器が鳴っているイベントの IoU。

    イベントは (bar, beat, pos, instrument) の組。
    楽器識別は STANDARD_PITCH_MAP キーのプレフィックスマッチによる。
    """
    ref_events = set(parse_onsets(ref_tokens))
    gen_events = set(parse_onsets(gen_tokens))

    union = ref_events | gen_events
    if not union:
        return 1.0
    return len(ref_events & gen_events) / len(union)
