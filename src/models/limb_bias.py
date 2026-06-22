"""
Limb-Relation Bias: ドラムの四肢構造をSelf-Attentionに反映させるバイアス機構

T5のRelative Position Biasと同じ思想で、計算量O(N²)を維持したまま
同一四肢トークン同士のAttention Logitに可学習スカラー α を加算する。

数式: Score_ij = QiKj^T / sqrt(dk) + α * M_limb_ij

Limb ID 定義:
  0: メタトークン (BAR, BEAT, TEMPO, POS, PAD など) → バイアス対象外
  1: Hand  (SNARE, TOM, HH_CLOSED, HH_OPEN, RIDE, CRASH 等)
  2: Right_Foot (KICK)
  3: Left_Foot  (HH_PEDAL)
  4: Hand + Left_Foot (HH_OPEN/CLOSED の状態切り替え時)
"""

from typing import Dict, List

import torch


# ---------------------------------------------------------------------------
# Limb ID 定数
# ---------------------------------------------------------------------------
LIMB_META       = 0
LIMB_HAND       = 1
LIMB_RFOOT      = 2
LIMB_LFOOT      = 3
LIMB_HAND_LFOOT = 4  # HH_OPEN/CLOSED 切り替え時（手と左足の同時動作）

# トークン文字列の前置詞 → Limb ID
# build_token_to_limb_map() が長いプレフィックス優先でマッチする
_PREFIX_TO_LIMB: Dict[str, int] = {
    # 明示的 LIMB トークン（長いプレフィックスを先に登録）
    "LIMB_HAND_LFOOT": LIMB_HAND_LFOOT,
    "LIMB_HAND":       LIMB_HAND,
    "LIMB_RFOOT":      LIMB_RFOOT,
    "LIMB_LFOOT":      LIMB_LFOOT,
    # Hand (1) ── 手でストロークする楽器
    "SNARE_RIM":   LIMB_HAND,
    "SNARE":       LIMB_HAND,
    "FLOOR":       LIMB_HAND,
    "TOM_LOW":     LIMB_HAND,
    "TOM_MID":     LIMB_HAND,
    "TOM_HIGH":    LIMB_HAND,
    "HH_CLOSED":   LIMB_HAND,   # 手で叩く（足はペダルで別管理）
    "HH_OPEN":     LIMB_HAND,   # 手で叩く
    "RIDE_CYMBAL": LIMB_HAND,
    "RIDE_BELL":   LIMB_HAND,
    "CRASH":       LIMB_HAND,
    "CHINA":       LIMB_HAND,
    "SPLASH":      LIMB_HAND,
    "TAMBOURINE":  LIMB_HAND,
    "COWBELL":     LIMB_HAND,
    # Right_Foot (2)
    "KICK":        LIMB_RFOOT,
    # Left_Foot (3) ── 足のみで操作するトークン
    "HH_PEDAL":    LIMB_LFOOT,
}


# ---------------------------------------------------------------------------
# 語彙からLookup Tableを構築する関数
# ---------------------------------------------------------------------------

def build_token_to_limb_map(vocab: List[str]) -> Dict[int, int]:
    """vocab（token文字列リスト）→ {token_id: limb_id} の辞書を返す。

    マッチしないトークン（BAR, BEAT, TEMPO, POS, PAD 等）は LIMB_META(0)。
    より長いプレフィックスを優先（SNARE_RIM > SNARE）。
    """
    sorted_prefixes = sorted(_PREFIX_TO_LIMB.keys(), key=len, reverse=True)
    result: Dict[int, int] = {}
    for idx, token_str in enumerate(vocab):
        limb = LIMB_META
        for prefix in sorted_prefixes:
            if token_str.startswith(prefix):
                limb = _PREFIX_TO_LIMB[prefix]
                break
        result[idx] = limb
    return result


def make_limb_lookup_table(token_to_limb: Dict[int, int], vocab_size: int) -> torch.Tensor:
    """{token_id: limb_id} の辞書から vocab_size 長の LongTensor を生成。

    model.register_buffer に渡す形式に変換する。
    """
    table = torch.zeros(vocab_size, dtype=torch.long)
    for token_id, limb_id in token_to_limb.items():
        table[token_id] = limb_id
    return table  # shape: [vocab_size]


# ---------------------------------------------------------------------------
# Limb ID テンソル生成
# ---------------------------------------------------------------------------

def build_limb_ids(
    token_ids: torch.Tensor,    # shape: [N, B]  (sequence-first)
    lookup_table: torch.Tensor, # shape: [vocab_size]  (registered buffer)
) -> torch.Tensor:
    """トークンID列 → Limb ID列。

    Returns:
        limb_ids: shape [B, N]  (batch-first)
    """
    token_ids_bf = token_ids.T                  # [N, B] → [B, N]
    return lookup_table[token_ids_bf]           # [B, N]


# ---------------------------------------------------------------------------
# M_limb マスク行列の生成
# ---------------------------------------------------------------------------

def build_limb_bias_mask(
    limb_ids: torch.Tensor,  # shape: [B, N]
    neg_val: float = 0.0,
) -> torch.Tensor:
    """同一四肢ペアを 1.0、メタトークン関与ペアを 0.0 とする [B, N, N] マスクを生成。

    Args:
        limb_ids: バッチ毎の Limb ID 列, shape [B, N]
        neg_val: 異なる四肢ペアへのバイアス値（デフォルト 0.0 = 片側強調のみ）

    Returns:
        M: shape [B, N, N], float32
    """
    ids_i = limb_ids.unsqueeze(2)   # [B, N, 1]
    ids_j = limb_ids.unsqueeze(1)   # [B, 1, N]

    # 同一 Limb かどうか
    # LIMB_HAND_LFOOT(4) は LIMB_HAND(1) および LIMB_LFOOT(3) の両方とマッチする
    same_limb = (
        (ids_i == ids_j) |
        ((ids_i == LIMB_HAND_LFOOT) & ((ids_j == LIMB_HAND) | (ids_j == LIMB_LFOOT))) |
        ((ids_j == LIMB_HAND_LFOOT) & ((ids_i == LIMB_HAND) | (ids_i == LIMB_LFOOT)))
    ).float()   # [B, N, N]

    # どちらかが LIMB_META(0) のペアを排除
    has_meta = ((ids_i == 0) | (ids_j == 0)).float()   # [B, N, N]

    # 組み立て: メタ関与→0, 同四肢→+1, 異四肢→neg_val
    if neg_val == 0.0:
        M = same_limb * (1.0 - has_meta)
    else:
        M = torch.where(
            has_meta.bool(),
            torch.zeros_like(same_limb),
            torch.where(
                same_limb.bool(),
                torch.ones_like(same_limb),
                torch.full_like(same_limb, neg_val),
            ),
        )
    return M  # [B, N, N]


# ---------------------------------------------------------------------------
# MHA 用にヘッド次元へ展開
# ---------------------------------------------------------------------------

def expand_for_mha(
    limb_bias: torch.Tensor,  # [B, N, N]
    n_heads: int,
) -> torch.Tensor:
    """nn.MultiheadAttention の attn_mask 形式 [B*n_heads, N, N] に展開する。

    PyTorch >= 1.11 の batched attn_mask 対応。
    """
    B, N, _ = limb_bias.shape
    # [B, N, N] → [B, 1, N, N] → [B, n_heads, N, N] → [B*n_heads, N, N]
    return limb_bias.unsqueeze(1).expand(-1, n_heads, -1, -1).reshape(B * n_heads, N, N)
