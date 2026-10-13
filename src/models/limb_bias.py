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

import functools
from typing import Dict, List, Optional, Tuple

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

def _same_limb_and_meta(
    limb_ids: torch.Tensor,  # shape: [B, N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """同一四肢判定 same_limb とメタ関与判定 has_meta の bool マスクを返す。

    build_limb_bias_mask / build_limb_pair_masks で共有する判定ロジック。

    Returns:
        same_limb: [B, N, N] bool  (LIMB_HAND_LFOOT は HAND/LFOOT の両方と同一扱い)
        has_meta:  [B, N, N] bool  (どちらかが LIMB_META のペア)
    """
    ids_i = limb_ids.unsqueeze(2)   # [B, N, 1]
    ids_j = limb_ids.unsqueeze(1)   # [B, 1, N]

    # 同一 Limb かどうか
    # LIMB_HAND_LFOOT(4) は LIMB_HAND(1) および LIMB_LFOOT(3) の両方とマッチする
    same_limb = (
        (ids_i == ids_j) |
        ((ids_i == LIMB_HAND_LFOOT) & ((ids_j == LIMB_HAND) | (ids_j == LIMB_LFOOT))) |
        ((ids_j == LIMB_HAND_LFOOT) & ((ids_i == LIMB_HAND) | (ids_i == LIMB_LFOOT)))
    )   # [B, N, N] bool

    # どちらかが LIMB_META(0) のペア
    has_meta = (ids_i == LIMB_META) | (ids_j == LIMB_META)   # [B, N, N] bool
    return same_limb, has_meta


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
    same_limb_b, has_meta_b = _same_limb_and_meta(limb_ids)
    same_limb = same_limb_b.float()   # [B, N, N]
    has_meta = has_meta_b.float()     # [B, N, N]

    # 組み立て: メタ関与→0, 同四肢→+1, 異四肢→neg_val
    if neg_val == 0.0:
        M = same_limb * (1.0 - has_meta)
    else:
        M = torch.where(
            has_meta_b,
            torch.zeros_like(same_limb),
            torch.where(
                same_limb_b,
                torch.ones_like(same_limb),
                torch.full_like(same_limb, neg_val),
            ),
        )
    return M  # [B, N, N]


def build_limb_pair_masks(
    limb_ids: torch.Tensor,  # shape: [B, N]
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """limb_ids [B, N] → (M_limb, M_other) 各 [B, N, N] float。

    M_limb : 非メタ同一四肢ペア（build_limb_bias_mask と同一判定）
    M_other: 非メタ異四肢ペア
    どちらかがメタトークン（PAD 含む）のペアは両集合から除外される。

    Args:
        causal: True で下三角のみ有効（デコーダの因果 Attention 用）。
                掛けないと上三角ペアが平均の分母に入り値が希釈される。
    """
    same_limb, has_meta = _same_limb_and_meta(limb_ids)   # [B, N, N] bool
    non_meta = ~has_meta
    M_limb = same_limb & non_meta
    M_other = (~same_limb) & non_meta

    if causal:
        N = limb_ids.size(1)
        tril = torch.tril(
            torch.ones(N, N, dtype=torch.bool, device=limb_ids.device)
        )   # [N, N]
        M_limb = M_limb & tril
        M_other = M_other & tril

    return M_limb.float(), M_other.float()   # [B, N, N] each


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


# ---------------------------------------------------------------------------
# Attention Bias Loss: Attn_limb / Attn_other 対比を hinge 損失で誘導する
# ---------------------------------------------------------------------------

# bias_loss の層モード
BIAS_MODE_LIMB_OVER_OTHER = 'limb_over_other'
BIAS_MODE_OTHER_OVER_LIMB = 'other_over_limb'
BIAS_MODE_NONE            = 'none'
VALID_BIAS_MODES = (BIAS_MODE_LIMB_OVER_OTHER, BIAS_MODE_OTHER_OVER_LIMB, BIAS_MODE_NONE)

# bias_loss 重みスケジューラの種類
VALID_BIAS_SCHEDS = ('constant', 'linear_warmup', 'cyclical')


class AttentionWeightCapture:
    """指定した TransformerEncoderLayer 群の self_attn.forward をパッチし、
    head 平均 Attention 重みを勾配付きで捕捉する。

    分析用 AttentionCapture（scripts/analysis/visualize_attention_maps.py）の
    勾配保持・常設パッチ版。`.detach()`/`.cpu()` せず計算グラフを保持するため、
    捕捉した重みに対する損失から QKV 重みへ勾配が流れる。

    Args:
        layers: {layer_idx: nn.TransformerEncoderLayer} … mode != 'none' の層のみ渡す

    使い方:
        cap = AttentionWeightCapture({2: layer2, 3: layer3}).attach()
        cap.enabled = True            # 捕捉 ON/OFF（False の間は need_weights を立てない）
        ... model.forward(...) ...
        maps = cap.pop_maps()         # {idx: Tensor[B, N, N]}（勾配付き）を返し内部をクリア
        cap.detach()                  # 学習終了時にパッチ解除

    注意: pop_maps()/clear() を毎バッチ必ず呼ぶこと。呼び忘れると計算グラフが
    保持され続け OOM の原因になる。
    """

    def __init__(self, layers: Dict[int, torch.nn.Module]):
        self._layers: Dict[int, torch.nn.Module] = dict(layers)
        self._originals: Dict[int, object] = {}
        self.maps: Dict[int, torch.Tensor] = {}
        self.enabled: bool = True

    def attach(self) -> 'AttentionWeightCapture':
        for idx, layer in self._layers.items():
            if idx in self._originals:
                continue
            orig = layer.self_attn.forward
            self._originals[idx] = orig
            layer.self_attn.forward = self._make_patched(orig, idx)
        return self

    def detach(self) -> None:
        for idx, layer in self._layers.items():
            if idx in self._originals:
                layer.self_attn.forward = self._originals[idx]
        self._originals = {}
        self.maps = {}

    def _make_patched(self, orig_fwd, idx: int):
        capture = self

        @functools.wraps(orig_fwd)
        def patched(*args, **kwargs):
            if not capture.enabled:
                return orig_fwd(*args, **kwargs)
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = True  # head 平均済みの重みを取得
            try:
                out_tensor, attn_weights = orig_fwd(*args, **kwargs)
            except TypeError:
                # PyTorch < 1.13 は average_attn_weights 未サポート
                kwargs.pop('average_attn_weights', None)
                out_tensor, attn_weights = orig_fwd(*args, **kwargs)
            if attn_weights is not None:
                capture.maps[idx] = attn_weights   # detach しない（勾配保持）
            return out_tensor, attn_weights

        return patched

    def pop_maps(self) -> Dict[int, torch.Tensor]:
        maps = self.maps
        self.maps = {}
        return maps

    def clear(self) -> None:
        self.maps = {}


def compute_attn_bias_loss(
    attn_maps: Dict[int, torch.Tensor],  # {layer_idx: [B, N, N]}
    M_limb: torch.Tensor,                # [B, N, N] float
    M_other: torch.Tensor,               # [B, N, N] float
    layer_modes: List[str],
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """捕捉した Attention 重みから Attn_limb / Attn_other 対比の hinge 損失を計算。

    各 active 層 l について:
        Attn_limb(l)  = Σ A_l * M_limb  / Σ M_limb
        Attn_other(l) = Σ A_l * M_other / Σ M_other
        hinge_l = max(0, margin - (Attn_limb - Attn_other))   # limb_over_other
                  max(0, margin - (Attn_other - Attn_limb))   # other_over_limb
    全体損失は active 層の hinge 平均。

    Returns:
        loss    : スカラー Tensor（active 層ゼロなら 0 テンソル）
        metrics : {'attn_limb/<l>': float, 'attn_other/<l>': float}（ログ用、grad 切断）
    """
    device = M_limb.device
    # 分母（バッチ全体の総ペア数）。0 ペアはゼロ除算回避のため 1 に clamp
    denom_limb = M_limb.float().sum().clamp_min(1.0)
    denom_other = M_other.float().sum().clamp_min(1.0)

    hinges: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for idx, A in attn_maps.items():
        mode = layer_modes[idx]
        if mode == BIAS_MODE_NONE:
            continue
        Af = A.float()   # AMP 下の fp16 総和は精度不足 → fp32 で計算
        attn_limb = (Af * M_limb).sum() / denom_limb
        attn_other = (Af * M_other).sum() / denom_other
        metrics[f'attn_limb/{idx}'] = attn_limb.item()
        metrics[f'attn_other/{idx}'] = attn_other.item()

        if mode == BIAS_MODE_LIMB_OVER_OTHER:
            hinge = torch.clamp(margin - (attn_limb - attn_other), min=0.0)
        elif mode == BIAS_MODE_OTHER_OVER_LIMB:
            hinge = torch.clamp(margin - (attn_other - attn_limb), min=0.0)
        else:
            continue
        hinges.append(hinge)

    if hinges:
        loss = torch.stack(hinges).mean()
    else:
        loss = torch.zeros((), device=device)
    return loss, metrics


def bias_loss_weight_sched(
    step: int,
    sched: str,
    max_weight: float,
    no_loss_steps: int,
    warmup_steps: int,
    cycle_steps: int,
) -> float:
    """bias_loss の重み λ を返すスケジューラ（KL β スケジューラと対をなす純関数）。

    - constant      : step < no_loss_steps → 0、以降 max_weight
    - linear_warmup : step < no_loss_steps → 0、以降 0→max_weight を warmup_steps で線形増加
    - cyclical      : beta_cyclical_sched と同型（周期前半で線形増加、後半 max_weight 維持）
    """
    if step < no_loss_steps:
        return 0.0

    if sched == 'constant':
        return float(max_weight)

    if sched == 'linear_warmup':
        if warmup_steps <= 0:
            return float(max_weight)
        frac = (step - no_loss_steps) / warmup_steps
        return float(max_weight) * min(1.0, max(0.0, frac))

    if sched == 'cyclical':
        if cycle_steps <= 0:
            return float(max_weight)
        step_in_cycle = (step - 1) % cycle_steps
        cycle_progress = step_in_cycle / cycle_steps
        if cycle_progress < 0.5:
            return float(max_weight) * cycle_progress * 2.0
        return float(max_weight)

    raise ValueError(f"Unknown bias_loss sched: {sched}. Expected one of {VALID_BIAS_SCHEDS}.")
