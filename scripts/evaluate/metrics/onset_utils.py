from typing import List, Optional, Tuple
from src.tokenizers.drum_tokenizer import STANDARD_PITCH_MAP

HIHAT_INSTRUMENTS = frozenset({'HH_CLOSED', 'HH_OPEN', 'HH_PEDAL', 'HH_HALFOPEN'})

# STANDARD_PITCH_MAP のキーを長い順に並べ、プレフィックス衝突を防ぐ
# (SNARE_RIM が SNARE より先にマッチする等)
_KNOWN_INSTRUMENTS: List[str] = sorted(
    list(STANDARD_PITCH_MAP.keys()) + ['HH_HALFOPEN'],
    key=len,
    reverse=True,
)


def extract_instrument(token: str) -> Optional[str]:
    """standardトークン文字列から楽器名を返す。構造トークンはNoneを返す。"""
    if token.startswith('<'):
        return None
    for inst in _KNOWN_INSTRUMENTS:
        if token.startswith(inst + '_') or token == inst:
            return inst
    return None


def parse_onsets(tokens: List[str]) -> List[Tuple[int, int, int, str]]:
    """standard トークン列を (bar, beat, pos, instrument) のリストに変換する。"""
    onsets: List[Tuple[int, int, int, str]] = []
    bar_idx = -1
    beat_idx = 0
    pos_idx = 0

    for token in tokens:
        if token == '<EOS>':
            break
        if token == '<BAR>':
            bar_idx += 1
            beat_idx = 0
            pos_idx = 0
        elif token.startswith('<BEAT_'):
            try:
                beat_idx = int(token[6:-1]) - 1  # 1-indexed → 0-indexed
            except ValueError:
                pass
        elif token.startswith('<POS_'):
            try:
                pos_idx = int(token[5:-1])
            except ValueError:
                pass
        elif not token.startswith('<') and bar_idx >= 0:
            instr = extract_instrument(token)
            if instr is not None:
                onsets.append((bar_idx, beat_idx, pos_idx, instr))

    return onsets
