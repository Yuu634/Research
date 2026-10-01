"""
QK内積値とLimb-bias値の比率を算出するスクリプト

Attention logit の計算式:
    Score_ij = QK^T_ij / sqrt(d_k) + α * M_limb_ij

各層について以下を算出:
  - 同一四肢ペア (M_limb=1) における |QK^T/sqrt(d_k)| の統計量
  - 異四肢ペア (M_limb=0, 非メタ) における同統計量
  - Limb-bias の寄与量: α（同一四肢ペアにのみ加算）
  - 比率: α / mean|QK|_same  → bias が QK 内積に対してどれだけ大きいか

Decoder は Causal マスク（下三角）内のペアのみを対象とする。

出力:
  {output_dir}/qk_limbbias_ratio.csv    - 全層の統計量 CSV
  {output_dir}/qk_limbbias_encoder.png  - Encoder の比較グラフ
  {output_dir}/qk_limbbias_decoder.png  - Decoder の比較グラフ

使用方法:
  python -m scripts.analysis.qk_limb_bias_ratio \\
    --model_dir trained_model/checkpoints_drum_standard-lakh1-kl14 \\
    --output_dir output/qk_limb_ratio/modelX \\
    [--checkpoint best_params.pt] \\
    [--sample_idx 0] \\
    [--max_dec_seqlen 512] \\
    [--device cpu]
"""

import argparse
import csv
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.drum_dataloader import DrumTransformerDataset
from src.musemorphose.model import MuseMorphose
from src.musemorphose.limb_bias import (
    build_token_to_limb_map, build_limb_ids, build_limb_bias_mask,
    make_limb_lookup_table,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 設定読み込み・モデル構築（visualize_attention_maps.py と同一ロジック）
# ---------------------------------------------------------------------------

def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'config.yaml が見つかりません: {config_path}')
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_bias_params(mconf: dict, dset: DrumTransformerDataset) -> dict:
    use_limb_bias_encoder = bool(mconf.get('use_limb_bias_encoder', False))
    use_limb_bias_decoder = bool(mconf.get('use_limb_bias_decoder', False))
    alpha_limb_fixed = mconf.get('limb_bias_alpha')
    if alpha_limb_fixed is not None:
        alpha_limb_fixed = float(alpha_limb_fixed)

    token_to_limb = None
    if use_limb_bias_encoder or use_limb_bias_decoder:
        idx2token = getattr(dset, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(dset.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            use_limb_bias_encoder = use_limb_bias_decoder = False

    use_position_bias_encoder = bool(mconf.get('use_position_bias_encoder', False))
    use_position_bias_decoder = bool(mconf.get('use_position_bias_decoder', False))
    position_bias_mode = mconf.get('position_bias_mode', 'uniform')
    alpha_pos_fixed = mconf.get('position_bias_alpha')
    if alpha_pos_fixed is not None:
        alpha_pos_fixed = float(alpha_pos_fixed)
    position_bias_per_layer = bool(mconf.get('position_bias_per_layer', True))

    pos_token_ids = None
    if use_position_bias_encoder or use_position_bias_decoder:
        t2i = getattr(dset, 'token2idx', None)
        if t2i is not None:
            pos_keys  = sorted([k for k in t2i if k.startswith('<POS_')],  key=lambda k: t2i[k])
            beat_keys = sorted([k for k in t2i if k.startswith('<BEAT_')], key=lambda k: t2i[k])
            if pos_keys and beat_keys:
                pos_token_ids = {
                    'pos_start':  t2i[pos_keys[0]],
                    'pos_end':    t2i[pos_keys[-1]],
                    'beat_start': t2i[beat_keys[0]],
                    'beat_end':   t2i[beat_keys[-1]],
                    'pad':        t2i.get('<PAD>', 0),
                    'eos':        t2i.get('<EOS>', 1),
                }
            else:
                use_position_bias_encoder = use_position_bias_decoder = False
        else:
            use_position_bias_encoder = use_position_bias_decoder = False

    if (use_position_bias_encoder or use_position_bias_decoder) and token_to_limb is None:
        idx2token = getattr(dset, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(dset.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            use_position_bias_encoder = use_position_bias_decoder = False
            pos_token_ids = None

    return dict(
        use_limb_bias_encoder=use_limb_bias_encoder,
        use_limb_bias_decoder=use_limb_bias_decoder,
        token_to_limb=token_to_limb,
        alpha_limb_fixed=alpha_limb_fixed,
        use_position_bias_encoder=use_position_bias_encoder,
        use_position_bias_decoder=use_position_bias_decoder,
        position_bias_mode=position_bias_mode,
        alpha_pos_fixed=alpha_pos_fixed,
        position_bias_per_layer=position_bias_per_layer,
        pos_token_ids=pos_token_ids,
        bias_schedule_encoder=mconf.get('bias_schedule_encoder', 'both'),
        bias_schedule_decoder=mconf.get('bias_schedule_decoder', 'both'),
        bias_schedule_encoder_custom=mconf.get('bias_schedule_encoder_custom'),
        bias_schedule_decoder_custom=mconf.get('bias_schedule_decoder_custom'),
    )


def _build_model(mconf: dict, dset: DrumTransformerDataset,
                 tokenization_method: str, bias_params: dict,
                 use_difficulty: bool) -> MuseMorphose:
    cond_mode = mconf.get('cond_mode', 'in-attn')
    if cond_mode not in ('in-attn', 'pre-attn'):
        cond_mode = 'in-attn'
    bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
    if bar_anchor_mode not in ('first_token', 'bar_token'):
        bar_anchor_mode = 'first_token'
    bar_token_idx = getattr(dset, 'bar_token', None)
    if bar_anchor_mode == 'bar_token' and bar_token_idx is None:
        bar_anchor_mode = 'first_token'

    n_s_tech_cls  = getattr(dset, 'n_s_tech_cls',  mconf.get('n_s_tech_cls',  9))
    n_s_indep_cls = getattr(dset, 'n_s_indep_cls', mconf.get('n_s_indep_cls', 9))
    n_s_hand_cls  = getattr(dset, 'n_s_hand_cls',  mconf.get('n_s_hand_cls',  9))
    n_s_foot_cls  = getattr(dset, 'n_s_foot_cls',  mconf.get('n_s_foot_cls',  9))
    n_s_move_cls  = getattr(dset, 'n_s_move_cls',  mconf.get('n_s_move_cls',  9))

    model = MuseMorphose(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
        d_polyph_emb=mconf.get('d_polyph_emb', 0),
        d_rfreq_emb=mconf.get('d_rfreq_emb', 0),
        d_s_tech_emb=mconf.get('d_s_tech_emb', 32),
        d_s_indep_emb=mconf.get('d_s_indep_emb', 32),
        d_s_hand_emb=mconf.get('d_s_hand_emb', 32),
        d_s_foot_emb=mconf.get('d_s_foot_emb', 32),
        d_s_move_emb=mconf.get('d_s_move_emb', 32),
        n_s_tech_cls=n_s_tech_cls,
        n_s_indep_cls=n_s_indep_cls,
        n_s_hand_cls=n_s_hand_cls,
        n_s_foot_cls=n_s_foot_cls,
        n_s_move_cls=n_s_move_cls,
        use_difficulty=use_difficulty,
        use_attr_cls=True,
        tokenization_method=tokenization_method,
        cond_mode=cond_mode,
        bar_anchor_mode=bar_anchor_mode,
        bar_token_idx=bar_token_idx,
        use_hierarchical_pe=bool(mconf.get('use_hierarchical_pe', False)),
        hier_pe_bar_vocab_size=max(1, int(mconf.get('hier_pe_bar_vocab_size', 512))),
        hier_pe_beat_vocab_size=max(1, int(mconf.get('hier_pe_beat_vocab_size', 8))),
        hier_pe_pos_vocab_size=max(1, int(mconf.get('hier_pe_pos_vocab_size', 32))),
        hier_pe_use_struct_bias=bool(mconf.get('hier_pe_use_struct_bias', True)),
        **bias_params,
    )
    return model


# ---------------------------------------------------------------------------
# 層入力の捕捉（forward pre-hook）
# ---------------------------------------------------------------------------

class LayerInputCapture:
    """TransformerEncoderLayer への入力隠れ状態を pre-hook で捕捉する。"""

    def __init__(self, layers: List[nn.TransformerEncoderLayer]):
        self._layers = layers
        self._handles: list = []
        self.hidden_states: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        for i, layer in enumerate(self._layers):
            def _make_hook(idx):
                def hook(module, args):
                    # args[0]: src = [T, B, d_model]  (batch_first=False のデフォルト)
                    self.hidden_states[idx] = args[0].detach().float().cpu()
                return hook
            handle = layer.register_forward_pre_hook(_make_hook(i))
            self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# QK^T / sqrt(d_k) の手動計算
# ---------------------------------------------------------------------------

def _compute_qk_scores(layer: nn.TransformerEncoderLayer,
                       x: torch.Tensor) -> torch.Tensor:
    """
    層への入力隠れ状態 x から QK^T / sqrt(d_head) を計算する。

    Args:
        layer: nn.TransformerEncoderLayer (batch_first=False, norm_first=False を想定)
        x:     [T, B, d_model]  (CPU, float32)

    Returns:
        scores: [B, n_head, T, T]  (CPU, float32)
    """
    mha = layer.self_attn
    T, B, d_model = x.shape
    n_head = mha.num_heads
    d_head = d_model // n_head

    # Pre-LN モデルの場合は layer.norm1 を事前適用
    if getattr(layer, 'norm_first', False):
        dev = next(layer.parameters()).device
        x = layer.norm1(x.to(dev)).detach().float().cpu()

    if mha.in_proj_weight is not None:
        # 標準実装: W_Q, W_K, W_V が in_proj_weight に結合されている
        # in_proj_weight: [3*d_model, d_model]
        w = mha.in_proj_weight.detach().float().cpu()
        b = mha.in_proj_bias.detach().float().cpu() if mha.in_proj_bias is not None else None
        qkv = F.linear(x.reshape(T * B, d_model), w, b)   # [T*B, 3*d_model]
        q, k, _ = qkv.chunk(3, dim=-1)                     # each [T*B, d_model]
    else:
        # 個別射影 (kdim != embed_dim 等の場合)
        wq = mha.q_proj_weight.detach().float().cpu()
        wk = mha.k_proj_weight.detach().float().cpu()
        x_flat = x.reshape(T * B, d_model)
        q = F.linear(x_flat, wq)
        k = F.linear(x_flat, wk)

    # [T*B, d_model] → [B, n_head, T, d_head]
    q = q.reshape(T, B, n_head, d_head).permute(1, 2, 0, 3)
    k = k.reshape(T, B, n_head, d_head).permute(1, 2, 0, 3)

    # QK^T / sqrt(d_head): [B, n_head, T, T]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    return scores


# ---------------------------------------------------------------------------
# Limb バイアスマスクの生成
# ---------------------------------------------------------------------------

def _build_M_limb(token_ids: torch.Tensor,
                  limb_table: torch.Tensor) -> torch.Tensor:
    """
    Args:
        token_ids: [T, B]  (sequence-first, long)
        limb_table: [vocab_size]

    Returns:
        M_limb: [B, T, T]  (0 or 1)
    """
    limb_ids = build_limb_ids(token_ids, limb_table)  # [B, T]
    return build_limb_bias_mask(limb_ids)              # [B, T, T]


# ---------------------------------------------------------------------------
# 統計量の算出
# ---------------------------------------------------------------------------

def _compute_stats(
    qk_scores: torch.Tensor,   # [B, n_head, T, T]
    M_limb: torch.Tensor,      # [B, T, T]  (0 or 1)
    alpha: Optional[float],
    is_decoder: bool = False,
) -> dict:
    """
    QK 内積値と Limb-bias 値を比較する統計量を算出する。

    Decoder では Causal マスク（下三角, i >= j）のみを有効ペアとする。
    同一四肢ペア (M_limb=1) と 非同一ペア (M_limb=0) を分けて集計する。

    Returns:
        dict with keys:
          n_same_pairs, n_other_pairs,
          qk_same_mean, qk_same_std, qk_same_p25, qk_same_p50, qk_same_p75,
          qk_other_mean, qk_other_std,
          qk_all_mean, qk_all_std,
          alpha, ratio_vs_same, ratio_vs_all
    """
    B, n_head, T, _ = qk_scores.shape

    # Decoder: 上三角（未来）は無効
    if is_decoder:
        valid_mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)  # [1, T, T]
    else:
        valid_mask = torch.ones(1, T, T, dtype=torch.bool)

    M_same = M_limb.bool()                   # [B, T, T]
    mask_same  = M_same  & valid_mask         # 同一四肢 & 有効
    mask_other = (~M_same) & valid_mask       # 非同一（メタ含む）& 有効

    # ヘッドを平均した絶対値: [B, T, T]
    qk_abs = qk_scores.abs().mean(dim=1)

    def _collect(mask: torch.Tensor) -> np.ndarray:
        # mask: [B, T, T] → 選択された要素の 1D 配列
        expanded = mask.expand_as(qk_abs)
        return qk_abs[expanded].numpy()

    vals_same  = _collect(mask_same)
    vals_other = _collect(mask_other)
    vals_all   = _collect(valid_mask.expand(B, -1, -1))

    def _stat(arr: np.ndarray) -> dict:
        if arr.size == 0:
            nan = float('nan')
            return dict(mean=nan, std=nan, p25=nan, p50=nan, p75=nan)
        return dict(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
        )

    s_same  = _stat(vals_same)
    s_other = _stat(vals_other)
    s_all   = _stat(vals_all)

    def _ratio(a: Optional[float], denom: float) -> float:
        if a is None or denom == 0.0 or np.isnan(denom):
            return float('nan')
        return float(a) / denom

    return {
        'n_same_pairs':  int(vals_same.size),
        'n_other_pairs': int(vals_other.size),
        'qk_same_mean': s_same['mean'],
        'qk_same_std':  s_same['std'],
        'qk_same_p25':  s_same['p25'],
        'qk_same_p50':  s_same['p50'],
        'qk_same_p75':  s_same['p75'],
        'qk_other_mean': s_other['mean'],
        'qk_other_std':  s_other['std'],
        'qk_all_mean':  s_all['mean'],
        'qk_all_std':   s_all['std'],
        'alpha':         float(alpha) if alpha is not None else float('nan'),
        'ratio_vs_same': _ratio(alpha, s_same['mean']),
        'ratio_vs_all':  _ratio(alpha, s_all['mean']),
    }


# ---------------------------------------------------------------------------
# Alpha 値の取得
# ---------------------------------------------------------------------------

def _get_alpha(model: MuseMorphose, component: str, layer_idx: int) -> Optional[float]:
    """指定層の alpha_limb 値をモデルから取得する。無効なら None。"""
    obj = model.encoder if component == 'encoder' else model.decoder
    if not getattr(obj, 'use_limb_bias', False):
        return None
    fixed = getattr(obj, 'alpha_limb_fixed', None)
    if fixed is not None:
        return float(fixed.item())
    alpha_list = getattr(obj, 'alpha_limb', None)
    if alpha_list is not None and layer_idx < len(alpha_list):
        return float(alpha_list[layer_idx].item())
    return None


# ---------------------------------------------------------------------------
# CSV 出力
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'component', 'layer',
    'n_same_pairs', 'n_other_pairs',
    'qk_same_mean', 'qk_same_std', 'qk_same_p25', 'qk_same_p50', 'qk_same_p75',
    'qk_other_mean', 'qk_other_std',
    'qk_all_mean', 'qk_all_std',
    'alpha', 'ratio_vs_same', 'ratio_vs_all',
]


def _save_csv(rows: List[dict], output_dir: str) -> None:
    path = os.path.join(output_dir, 'qk_limbbias_ratio.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'[info] CSV 保存: {path}')


# ---------------------------------------------------------------------------
# グラフ出力
# ---------------------------------------------------------------------------

def _plot_component(rows: List[dict], component: str, output_dir: str) -> None:
    """1コンポーネント分の QK vs Limb-bias 比較グラフを保存する。"""
    comp_rows = [r for r in rows if r['component'] == component]
    if not comp_rows:
        return

    layers     = [r['layer']          for r in comp_rows]
    qk_same    = [r['qk_same_mean']   for r in comp_rows]
    qk_other   = [r['qk_other_mean']  for r in comp_rows]
    qk_same_p25 = [r['qk_same_p25']  for r in comp_rows]
    qk_same_p75 = [r['qk_same_p75']  for r in comp_rows]
    alphas     = [r['alpha']          for r in comp_rows]
    ratio_same = [r['ratio_vs_same']  for r in comp_rows]

    has_alpha = any(not np.isnan(a) for a in alphas)
    x = np.arange(len(layers))

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(layers) * 1.2), 9))

    # ── Top panel: compare mean |QK| against α ──────────────────────────
    ax = axes[0]
    w = 0.28
    bars_same  = ax.bar(x - w, qk_same,  w, label='mean |QK| (same-limb)',  color='steelblue')
    bars_other = ax.bar(x,     qk_other, w, label='mean |QK| (other, incl. meta)', color='cornflowerblue', alpha=0.7)

    # Show interquartile range as error bars
    yerr_low  = [max(0.0, m - p25) for m, p25 in zip(qk_same, qk_same_p25)]
    yerr_high = [max(0.0, p75 - m) for m, p75 in zip(qk_same, qk_same_p75)]
    ax.errorbar(x - w, qk_same, yerr=[yerr_low, yerr_high],
                fmt='none', ecolor='navy', elinewidth=1.5, capsize=4)

    if has_alpha:
        ax.plot(x + w * 0.5, alphas, 'r^--', markersize=9, linewidth=2.0,
                label='α (limb bias contribution)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylabel('Absolute attention logit', fontsize=11)
    ax.set_title(
        f'{component}: mean |QK^T/√d_k| vs α (limb bias)\n'
        'Error bars = IQR (P25-P75) of same-limb pairs',
        fontsize=12,
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # ── Bottom panel: ratio α / mean|QK|_same ──────────────────────────
    ax2 = axes[1]
    if has_alpha:
        valid_ratio = [r if not np.isnan(r) else 0.0 for r in ratio_same]
        colors = ['tomato' if r >= 1.0 else 'steelblue' for r in valid_ratio]
        ax2.bar(x, valid_ratio, color=colors,
                label='α / mean|QK| (same-limb)')
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
                    label='ratio = 1 (α and QK equal magnitude)')
        ax2.set_ylabel('Ratio  α / mean|QK|_same', fontsize=11)
        ax2.set_title(
            f'{component}: relative strength of limb-bias α vs QK dot product on same-limb pairs\n'
            'Red bars (ratio >= 1) -> bias dominates the QK dot product',
            fontsize=12,
        )
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Limb bias disabled (no α)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title(f'{component}: no ratio plot (α unavailable)', fontsize=12)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f'qk_limbbias_{component.lower()}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[info] グラフ保存: {out_path}')


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def run(args) -> None:
    model_dir      = args.model_dir
    output_dir     = args.output_dir
    ckpt_name      = args.checkpoint
    sample_idx     = args.sample_idx
    device         = args.device
    max_dec_seqlen = args.max_dec_seqlen

    os.makedirs(output_dir, exist_ok=True)

    # ── 設定ロード ──────────────────────────────────────────────────────
    config = _load_config(model_dir)
    dconf  = config['data']
    mconf  = config['model']
    tokenization_method = dconf.get('tokenization_method', 'standard')
    use_difficulty      = bool(mconf.get('use_difficulty', False))

    print(f'[info] model_dir          : {model_dir}')
    print(f'[info] tokenization_method: {tokenization_method}')
    print(f'[info] use_difficulty     : {use_difficulty}')

    # ── データセット ────────────────────────────────────────────────────
    val_split_path = dconf.get('val_split')
    pieces = None
    if val_split_path and os.path.isfile(val_split_path):
        with open(val_split_path, 'rb') as f:
            all_pieces = pickle.load(f)
        pieces = all_pieces[:max(1, sample_idx + 1)]
        print(f'[info] val split: {val_split_path}  ({len(pieces)} pieces を使用)')
    else:
        print('[warning] val_split が見つかりません。全データから読み込みます。')

    dset = DrumTransformerDataset(
        data_dir=dconf['data_dir'],
        vocab_path=dconf['vocab_path'],
        model_enc_seqlen=dconf.get('enc_seqlen', 128),
        model_dec_seqlen=dconf.get('dec_seqlen', 1280),
        model_max_bars=dconf.get('max_bars', 16),
        pieces=pieces,
        pad_to_same=True,
        appoint_st_bar=0,
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method,
    )
    print(f'[info] vocab_size: {dset.vocab_size}  dataset size: {len(dset)}')

    loader = DataLoader(dset, batch_size=1, shuffle=False)
    if sample_idx < 0 or sample_idx >= len(dset):
        raise ValueError(f'sample_idx が範囲外です: {sample_idx} (件数: {len(dset)})')
    batch = next(batch for i, batch in enumerate(loader) if i == sample_idx)

    # ── チェックポイントロード ───────────────────────────────────────────
    ckpt_path = os.path.join(model_dir, 'params', ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'チェックポイントが見つかりません: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    _cls_key_map = {
        's_tech_attr_emb.emb_lookup.weight':  'n_s_tech_cls',
        's_indep_attr_emb.emb_lookup.weight': 'n_s_indep_cls',
        's_hand_attr_emb.emb_lookup.weight':  'n_s_hand_cls',
        's_foot_attr_emb.emb_lookup.weight':  'n_s_foot_cls',
        's_move_attr_emb.emb_lookup.weight':  'n_s_move_cls',
    }
    for sd_key, attr in _cls_key_map.items():
        if sd_key in state_dict:
            ckpt_cls = state_dict[sd_key].shape[0]
            if getattr(dset, attr, None) != ckpt_cls:
                print(f'[info] {attr}: dataset={getattr(dset, attr)} → checkpoint={ckpt_cls} に上書き')
                setattr(dset, attr, ckpt_cls)

    # ── モデル構築 ──────────────────────────────────────────────────────
    bias_params = _resolve_bias_params(mconf, dset)
    model = _build_model(mconf, dset, tokenization_method, bias_params, use_difficulty)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f'[info] チェックポイントをロード: {ckpt_path}')

    # ── バッチをデバイスへ ───────────────────────────────────────────────
    # enc_input shape (batch): (B, n_bars, enc_seqlen)
    # permute(2, 0, 1) → (enc_seqlen, B, n_bars)
    enc_inp_3d      = batch['enc_input'].permute(2, 0, 1)               # (enc_seqlen, B, n_bars)
    enc_inp         = enc_inp_3d.to(device)
    dec_inp         = batch['dec_input'].permute(1, 0).to(device)       # (dec_seqlen, B)
    dec_inp_bar_pos = batch['bar_pos'].to(device)
    padding_mask    = batch['enc_padding_mask'].to(device)

    # Limb 算出用のトークン ID (CPU, long)
    enc_seqlen = enc_inp_3d.shape[0]
    enc_tokens_flat = enc_inp_3d.reshape(enc_seqlen, -1).long().cpu()  # (enc_seqlen, B*n_bars)
    dec_tokens      = batch['dec_input'].permute(1, 0).long().cpu()    # (dec_seqlen, B)

    s_tech_cls  = batch['s_tech_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_indep_cls = batch['s_indep_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_hand_cls  = batch['s_hand_cls_seq'].permute(1, 0).to(device)  if use_difficulty else None
    s_foot_cls  = batch['s_foot_cls_seq'].permute(1, 0).to(device)  if use_difficulty else None
    s_move_cls  = batch['s_move_cls_seq'].permute(1, 0).to(device)  if use_difficulty else None

    # ── レイヤーリスト ───────────────────────────────────────────────────
    if hasattr(model.encoder, 'encoder_layers'):
        enc_layers = list(model.encoder.encoder_layers)
    else:
        enc_layers = list(model.encoder.tr_encoder.layers)
    dec_layers = list(model.decoder.decoder_layers)
    print(f'[info] Encoder layers: {len(enc_layers)}  Decoder layers: {len(dec_layers)}')

    # ── Forward + 隠れ状態捕捉 ──────────────────────────────────────────
    with LayerInputCapture(enc_layers) as enc_cap, \
         LayerInputCapture(dec_layers) as dec_cap:
        with torch.no_grad():
            model(
                enc_inp, dec_inp, dec_inp_bar_pos,
                s_tech_cls=s_tech_cls,
                s_indep_cls=s_indep_cls,
                s_hand_cls=s_hand_cls,
                s_foot_cls=s_foot_cls,
                s_move_cls=s_move_cls,
                padding_mask=padding_mask,
            )

    # ── Limb テーブル構築 ────────────────────────────────────────────────
    if hasattr(model, 'token_to_limb_table'):
        limb_table = model.token_to_limb_table.cpu()
        print('[info] token_to_limb_table をモデルから取得')
    else:
        idx2token = getattr(dset, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(dset.vocab_size)]
            limb_table = make_limb_lookup_table(
                build_token_to_limb_map(vocab_list), dset.vocab_size
            )
            print('[info] token_to_limb_table を語彙から再構築')
        else:
            limb_table = None
            print('[warning] limb テーブルを構築できません。M_limb は全て 0 になります。')

    # ── 各層の統計量算出 ─────────────────────────────────────────────────
    rows: List[dict] = []

    for component, layers, token_ids, cap, is_dec in [
        ('Encoder', enc_layers, enc_tokens_flat, enc_cap, False),
        ('Decoder', dec_layers, dec_tokens,       dec_cap, True),
    ]:
        # decoder は長大なため --max_dec_seqlen で切り詰め
        if is_dec and max_dec_seqlen is not None:
            token_ids = token_ids[:max_dec_seqlen]

        if limb_table is not None:
            M_limb = _build_M_limb(token_ids, limb_table)  # [B_eff, T, T]
        else:
            T_len = token_ids.shape[0]
            B_eff = token_ids.shape[1]
            M_limb = torch.zeros(B_eff, T_len, T_len)

        for layer_idx, layer in enumerate(layers):
            if layer_idx not in cap.hidden_states:
                print(f'[warning] {component} Layer {layer_idx} の隠れ状態が捕捉されませんでした')
                continue

            x = cap.hidden_states[layer_idx]   # [T, B, d_model]
            if is_dec and max_dec_seqlen is not None:
                x = x[:max_dec_seqlen]

            qk = _compute_qk_scores(layer, x)  # [B, n_head, T, T]
            alpha = _get_alpha(model, component.lower(), layer_idx)
            stats = _compute_stats(qk, M_limb, alpha, is_decoder=is_dec)

            row = {'component': component, 'layer': layer_idx, **stats}
            rows.append(row)

            alpha_str = f'{alpha:.6f}' if alpha is not None else 'N/A'
            ratio_str = (f'{stats["ratio_vs_same"]:.4f}'
                         if not np.isnan(stats['ratio_vs_same']) else 'N/A')
            print(
                f'  [{component} L{layer_idx:02d}]'
                f'  mean|QK|_same={stats["qk_same_mean"]:.5f}'
                f'  mean|QK|_other={stats["qk_other_mean"]:.5f}'
                f'  α={alpha_str}'
                f'  ratio={ratio_str}'
                f'  (n_same={stats["n_same_pairs"]})'
            )

    # ── 出力 ────────────────────────────────────────────────────────────
    _save_csv(rows, output_dir)
    _plot_component(rows, 'Encoder', output_dir)
    _plot_component(rows, 'Decoder', output_dir)

    print(f'\n[info] 完了。結果は {output_dir} に保存されました。')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Attention の QK 内積値と Limb-bias 値の比率を算出する'
    )
    parser.add_argument('--model_dir', required=True,
                        help='モデルディレクトリ（config.yaml と params/ を含む）')
    parser.add_argument('--output_dir', default='output/qk_limb_ratio',
                        help='出力先ディレクトリ（デフォルト: output/qk_limb_ratio）')
    parser.add_argument('--checkpoint', default='best_params.pt',
                        help='使用するチェックポイントファイル名（デフォルト: best_params.pt）')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='使用するサンプルのインデックス（デフォルト: 0）')
    parser.add_argument('--max_dec_seqlen', type=int, default=512,
                        help='Decoder 解析に使う最大系列長（メモリ節約用。デフォルト: 512）')
    parser.add_argument('--device', default='cpu',
                        help='デバイス（cpu / cuda、デフォルト: cpu）')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
