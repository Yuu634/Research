"""
Position-Same-Time Bias: 同じ演奏位置（BAR+BEAT+POS の組み合わせ）にある
トークン同士の Self-Attention スコアにバイアスを加える機構。

Limb-Relation Bias と直交する設計で、同時打鍵パターンの学習を促進する。

数式: Score_ij = QiKj^T / sqrt(dk) + α * M_pos_ij

  M_pos_ij = 1.0  (token_i と token_j が同一 Position-Group に属する)
           = 0.0  (それ以外、またはMETAトークン関与)

Modes (config: position_bias_mode):
  'uniform'    -- 全ペアに同一スカラー α を適用
  'per_limb'   -- HAND/RFOOT/LFOOT の 3×3 対称行列（実効 6 パラメータ/層）
  'asymmetric' -- HAND/RFOOT/LFOOT の 3×3 非対称行列（9 パラメータ/層）
                  αSNARE→KICK ≠ αKICK→SNARE に対応
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from src.musemorphose.limb_bias import LIMB_META, expand_for_mha  # noqa: F401


# ---------------------------------------------------------------------------
# Attention Bias Loss: Position（同一タイミング）モード
#   limb_bias.py の bias_loss（limb 軸）と対になる position 軸のモード。
#   同じ layer_modes リストに混在させて層ごとの役割割り当てを可能にする。
# ---------------------------------------------------------------------------
BIAS_MODE_POS_OVER_OTHER = 'pos_over_other'
BIAS_MODE_OTHER_OVER_POS = 'other_over_pos'
POS_BIAS_MODES = (BIAS_MODE_POS_OVER_OTHER, BIAS_MODE_OTHER_OVER_POS)


# ---------------------------------------------------------------------------
# ヘルパー: 前方埋め（Forward Fill）
# ---------------------------------------------------------------------------

def _forward_fill(trigger_vals: torch.Tensor, is_trigger: torch.Tensor) -> torch.Tensor:
    """
    トリガー位置での値を後続に伝播する（前方埋め）。
    cummax を使ったバッチ並列実装。

    trigger_vals: [N, B]  トリガー位置での値（非トリガーは 0）
    is_trigger:   [N, B]  bool
    Returns:      [N, B]  各位置で最後に見たトリガー値
    """
    masked = torch.where(is_trigger, trigger_vals, torch.zeros_like(trigger_vals))
    filled, _ = torch.cummax(masked, dim=0)
    return filled


# ---------------------------------------------------------------------------
# Position Group ID の構築
# ---------------------------------------------------------------------------

def build_position_ids(
    token_ids: torch.Tensor,               # [N, B]  sequence-first
    pos_token_start: int,                  # <POS_0> のトークンインデックス
    pos_token_end: int,                    # <POS_23> のトークンインデックス (inclusive)
    beat_token_start: int,                 # <BEAT_1> のトークンインデックス
    beat_token_end: int,                   # <BEAT_4> のトークンインデックス (inclusive)
    bar_token_idx: int,
    pad_token_idx: int,
    limb_lookup_table: Optional[torch.Tensor] = None,  # [vocab_size] — ノートトークン判定用
    eos_token_idx: Optional[int] = None,
) -> torch.Tensor:                         # [B, N]  long
    """
    各ノートトークンに「直前の <POS_X> トークンのシーケンスインデックス+1」を割り当てる。
    同じ POS イベントに属するトークン群が同一グループ ID を持つ。
    META トークン（BAR/BEAT/POS/TEMPO/PAD/EOS）は group_id=0（位置なし）。

    Returns:
        pos_group_ids: [B, N], long
          0     = METAトークン または POS未出現
          1..K  = 各 POS 出現ごとに固有のグループ ID（シーケンス内の位置に基づく）

    Note:
        limb_lookup_table を渡すと TEMPO トークンを正確に除外できる。
        省略した場合は BAR/BEAT/POS/PAD/EOS 以外をノートとみなすフォールバックを使用
        するが、TEMPO トークンが誤ってノート扱いされる可能性がある。
    """
    N, B = token_ids.shape
    device = token_ids.device

    is_pos = (token_ids >= pos_token_start) & (token_ids <= pos_token_end)  # [N, B]

    # POS トークンの出現位置にシーケンスインデックスを置き、非 POS 位置は 0
    seq_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, B)  # [N, B]
    pos_trigger_vals = seq_idx * is_pos.long()

    # 最後に見た POS のシーケンスインデックスを前方伝播
    pos_fill = _forward_fill(pos_trigger_vals, is_pos)  # [N, B]

    # +1 シフト: 0 を「POS 未出現」sentinel として確保
    pos_group_raw = pos_fill + 1  # [N, B]

    # ノートトークン判定
    if limb_lookup_table is not None:
        is_note = (limb_lookup_table[token_ids] != LIMB_META)  # [N, B]
    else:
        is_bar = (token_ids == bar_token_idx)
        is_beat = (token_ids >= beat_token_start) & (token_ids <= beat_token_end)
        is_pad = (token_ids == pad_token_idx)
        is_eos = ((token_ids == eos_token_idx)
                  if eos_token_idx is not None
                  else torch.zeros_like(is_bar))
        is_note = ~(is_bar | is_beat | is_pos | is_pad | is_eos)

    # META トークンは group_id=0
    pos_group_ids = pos_group_raw * is_note.long()  # [N, B]

    return pos_group_ids.T  # → [B, N]


# ---------------------------------------------------------------------------
# M_pos マスク行列の生成
# ---------------------------------------------------------------------------

def build_position_bias_mask(
    pos_group_ids: torch.Tensor,  # [B, N], long
) -> torch.Tensor:                # [B, N, N], float32
    """
    同一 Position-Group（かつ group_id != 0）のペアを 1.0、その他を 0.0 とする
    [B, N, N] マスクを生成する。build_limb_bias_mask と同一パターン。
    """
    ids_i = pos_group_ids.unsqueeze(2)    # [B, N, 1]
    ids_j = pos_group_ids.unsqueeze(1)    # [B, 1, N]

    same_pos = (ids_i == ids_j).float()                # [B, N, N]
    has_no_pos = ((ids_i == 0) | (ids_j == 0)).float() # [B, N, N]

    return same_pos * (1.0 - has_no_pos)  # [B, N, N]


# ---------------------------------------------------------------------------
# PositionBiasModule
# ---------------------------------------------------------------------------

class PositionBiasModule(nn.Module):
    """
    Position-Same-Time Bias のパラメータを保持し、各レイヤー用バイアスを計算する。

    Parameters
    ----------
    mode : str
        'uniform' | 'per_limb' | 'asymmetric'
    n_layers : int
        トランスフォーマーのレイヤー数。
    alpha_fixed : float or None
        None = 学習可能パラメータ（初期値 0.0）
        float = 固定値（勾配なし）
    per_layer : bool
        True  = レイヤーごとに独立した α/行列
        False = 全レイヤーで共有
    """

    N_NON_META = 3  # HAND(1), RFOOT(2), LFOOT(3)

    def __init__(
        self,
        mode: str,
        n_layers: int,
        alpha_fixed: Optional[float] = None,
        per_layer: bool = True,
    ):
        super().__init__()
        assert mode in ('uniform', 'per_limb', 'asymmetric'), \
            f"position_bias_mode must be 'uniform', 'per_limb', or 'asymmetric', got '{mode}'"

        self.mode = mode
        self.n_layers = n_layers
        self.per_layer = per_layer
        n_sets = n_layers if per_layer else 1

        if alpha_fixed is not None:
            # 固定値: バッファに登録（勾配なし）
            if mode == 'uniform':
                buf = torch.full((n_sets,), float(alpha_fixed))
            else:
                buf = torch.full((n_sets, self.N_NON_META, self.N_NON_META), float(alpha_fixed))
            self.register_buffer('_alpha_buf', buf)
            self.alpha_params = None
        else:
            # 学習可能パラメータ（初期値 0.0 → 学習開始時バイアス無効）
            self.register_buffer('_alpha_buf', None)
            if mode == 'uniform':
                self.alpha_params = nn.ParameterList([
                    nn.Parameter(torch.zeros(1)) for _ in range(n_sets)
                ])
            else:
                self.alpha_params = nn.ParameterList([
                    nn.Parameter(torch.zeros(self.N_NON_META, self.N_NON_META))
                    for _ in range(n_sets)
                ])

    def _get_param(self, layer_idx: int) -> torch.Tensor:
        idx = layer_idx if self.per_layer else 0
        if self.alpha_params is not None:
            return self.alpha_params[idx]
        return self._alpha_buf[idx]

    def get_bias_for_layer(
        self,
        layer_idx: int,
        pos_mask: torch.Tensor,            # [B, N, N]  build_position_bias_mask の出力
        limb_ids: Optional[torch.Tensor],  # [B, N]  per_limb/asymmetric モードで必要
    ) -> torch.Tensor:                     # [B, N, N]
        """スケール済みバイアスを返す。"""
        param = self._get_param(layer_idx)

        if self.mode == 'uniform':
            return param * pos_mask  # [1 or scalar] × [B, N, N]

        # per_limb / asymmetric: 3×3 行列を各トークンペアの limb に展開
        if limb_ids is None:
            raise ValueError("limb_ids is required for 'per_limb' and 'asymmetric' modes")

        raw = param  # [3, 3]
        A = (raw + raw.T) / 2.0 if self.mode == 'per_limb' else raw  # 対称化 or そのまま

        # limb_id: 0=META, 1=HAND, 2=RFOOT, 3=LFOOT
        # → 0-indexed for 3×3: HAND→0, RFOOT→1, LFOOT→2
        li = (limb_ids - 1).clamp(min=0)  # [B, N]  (META の 0 は clamp 後も 0、後で pos_mask=0 でゼロアウト)
        li_i = li.unsqueeze(2)  # [B, N, 1]
        li_j = li.unsqueeze(1)  # [B, 1, N]

        pair_bias = A[li_i, li_j]  # [B, N, N]  advanced indexing
        return pair_bias * pos_mask  # META ペアは pos_mask=0 でゼロアウト


# ---------------------------------------------------------------------------
# Attention Bias Loss（Position 軸）: Attn_pos / Attn_otherpos 対比を hinge で誘導
#   limb_bias.build_limb_pair_masks / compute_attn_bias_loss と同型。
# ---------------------------------------------------------------------------

def build_position_pair_masks(
    pos_group_ids: torch.Tensor,  # [B, N], long  (build_position_ids の出力)
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """pos_group_ids [B, N] → (M_pos, M_other) 各 [B, N, N] float。

    M_pos  : 同一 Position-Group の非メタペア（build_position_bias_mask と同一判定）
    M_other: 両端が非メタ かつ 異なる Position-Group のペア
    どちらかがメタ/POS未出現（group_id=0）のペアは両集合から除外される。

    Args:
        causal: True で下三角のみ有効（デコーダの因果 Attention 用）。
                掛けないと上三角ペアが平均の分母に入り値が希釈される。
    """
    ids_i = pos_group_ids.unsqueeze(2)   # [B, N, 1]
    ids_j = pos_group_ids.unsqueeze(1)   # [B, 1, N]

    has_no_pos = (ids_i == 0) | (ids_j == 0)   # [B, N, N] bool  (META/未出現)
    non_meta = ~has_no_pos
    same_pos = (ids_i == ids_j)                # [B, N, N] bool

    M_pos = same_pos & non_meta
    M_other = (~same_pos) & non_meta

    if causal:
        N = pos_group_ids.size(1)
        tril = torch.tril(
            torch.ones(N, N, dtype=torch.bool, device=pos_group_ids.device)
        )   # [N, N]
        M_pos = M_pos & tril
        M_other = M_other & tril

    return M_pos.float(), M_other.float()   # [B, N, N] each


def compute_attn_pos_bias_loss(
    attn_maps: Dict[int, torch.Tensor],  # {layer_idx: [B, N, N]}
    M_pos: torch.Tensor,                 # [B, N, N] float
    M_other: torch.Tensor,               # [B, N, N] float
    layer_modes: List[str],
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """捕捉した Attention 重みから Attn_pos / Attn_otherpos 対比の hinge 損失を計算。

    limb_bias.compute_attn_bias_loss と同型だが POS_BIAS_MODES の層のみ処理し、
    limb 系モード・'none' の層はスキップする（1 層 1 モード前提）。

        Attn_pos(l)      = Σ A_l * M_pos   / Σ M_pos
        Attn_otherpos(l) = Σ A_l * M_other / Σ M_other
        hinge_l = max(0, margin - (Attn_pos - Attn_otherpos))   # pos_over_other
                  max(0, margin - (Attn_otherpos - Attn_pos))   # other_over_pos

    Returns:
        loss    : スカラー Tensor（active 層ゼロなら 0 テンソル）
        metrics : {'attn_pos/<l>': float, 'attn_otherpos/<l>': float}（ログ用、grad 切断）
    """
    device = M_pos.device
    denom_pos = M_pos.float().sum().clamp_min(1.0)
    denom_other = M_other.float().sum().clamp_min(1.0)

    hinges: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for idx, A in attn_maps.items():
        mode = layer_modes[idx]
        if mode not in POS_BIAS_MODES:   # limb 系・'none' はスキップ
            continue
        Af = A.float()   # AMP 下の fp16 総和は精度不足 → fp32 で計算
        attn_pos = (Af * M_pos).sum() / denom_pos
        attn_other = (Af * M_other).sum() / denom_other
        metrics[f'attn_pos/{idx}'] = attn_pos.item()
        metrics[f'attn_otherpos/{idx}'] = attn_other.item()

        if mode == BIAS_MODE_POS_OVER_OTHER:
            hinge = torch.clamp(margin - (attn_pos - attn_other), min=0.0)
        else:  # BIAS_MODE_OTHER_OVER_POS
            hinge = torch.clamp(margin - (attn_other - attn_pos), min=0.0)
        hinges.append(hinge)

    if hinges:
        loss = torch.stack(hinges).mean()
    else:
        loss = torch.zeros((), device=device)
    return loss, metrics
