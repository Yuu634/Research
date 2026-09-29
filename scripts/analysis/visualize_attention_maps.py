"""
各層のAttentionMapを図で保存するスクリプト

モデルの forward を1回実行し、Encoder・Decoder 各層の Self-Attention 重みを
フックで捕捉してヒートマップとして保存する。

出力:
  {output_dir}/encoder_layer{i:02d}_avg.png     : 層 i のヘッド平均注意マップ
  {output_dir}/encoder_layer{i:02d}_perhead.png : 層 i のヘッド別注意マップ
  {output_dir}/decoder_layer{i:02d}_avg.png     : 同上 (Decoder)
  {output_dir}/decoder_layer{i:02d}_perhead.png : 同上 (Decoder)
  {output_dir}/encoder_summary.png              : 全 Encoder 層サマリー
  {output_dir}/decoder_summary.png              : 全 Decoder 層サマリー

使用方法:
  python -m scripts.analysis.visualize_attention_maps \\
    --model_dir trained_model/checkpoints_drum_standard-lakh1-kl14 \\
    --output_dir output/attention_maps/model8 \\
    [--checkpoint best_params.pt] \\
    [--sample_idx 0] \\
    [--device cpu] \\
    [--no_perhead]
"""

import argparse
import functools
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.drum_dataloader import DrumTransformerDataset
from src.models.musemorphose import MuseMorphose
from src.models.limb_bias import build_token_to_limb_map
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------

def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'config.yaml が見つかりません: {config_path}')
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# モデル構築（evaluate.py と同一ロジック）
# ---------------------------------------------------------------------------

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
# Attention フック
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    TransformerEncoderLayer の self_attn.forward をパッチし、
    need_weights=True で注意重みを捕捉するコンテキストマネージャ。

    maps[layer_idx]: numpy array, shape (B, n_heads, T, T) or (B, T, T)
    """

    def __init__(self, layers: List[nn.TransformerEncoderLayer]):
        self._layers = layers
        self._originals: Dict[int, object] = {}
        self.maps: Dict[int, np.ndarray] = {}

    def __enter__(self):
        for i, layer in enumerate(self._layers):
            orig = layer.self_attn.forward
            self._originals[i] = orig
            capture = self
            idx = i

            def _make_patched(orig_fwd, _idx):
                @functools.wraps(orig_fwd)
                def patched(*args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False  # PyTorch >= 1.13: ヘッドごとの重みを取得
                    try:
                        out_tensor, attn_weights = orig_fwd(*args, **kwargs)
                    except TypeError:
                        # PyTorch < 1.13 は average_attn_weights 未サポート
                        del kwargs['average_attn_weights']
                        out_tensor, attn_weights = orig_fwd(*args, **kwargs)
                    if attn_weights is not None:
                        capture.maps[_idx] = attn_weights.detach().float().cpu().numpy()
                    return out_tensor, attn_weights
                return patched

            layer.self_attn.forward = _make_patched(orig, idx)
        return self

    def __exit__(self, *_):
        for i, layer in enumerate(self._layers):
            if i in self._originals:
                layer.self_attn.forward = self._originals[i]


def _get_layer_lists(model: MuseMorphose):
    """Encoder・Decoder のレイヤーリストを返す。"""
    if hasattr(model.encoder, 'encoder_layers'):
        enc_layers = list(model.encoder.encoder_layers)
    else:
        enc_layers = list(model.encoder.tr_encoder.layers)
    dec_layers = list(model.decoder.decoder_layers)
    return enc_layers, dec_layers


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def _attn_to_2d(attn: np.ndarray) -> np.ndarray:
    """(B?, n_heads?, T, T) → (T, T)  バッチ・ヘッドを平均。
    全パディングバーの NaN (softmax(-inf,...,-inf)) を nanmean で無視する。"""
    while attn.ndim > 2:
        attn = np.nanmean(attn, axis=0)
    return attn


def _robust_vlim(mat: np.ndarray, q_low: float = 1.0, q_high: float = 99.0):
    """外れ値・NaN に頑健な vmin/vmax をパーセンタイルから計算する。
    全パディングバーの NaN が残存しても安全に処理する。"""
    flat = mat.ravel()
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(valid, q_low))
    vmax = float(np.percentile(valid, q_high))
    vmin = max(0.0, vmin)
    if vmax <= vmin:
        vmax = float(valid.max())
    if vmax <= vmin:  # 全要素が同一値の場合
        vmax = vmin + 1e-6
    return vmin, vmax


def _plot_layer_avg(attn: np.ndarray, layer_idx: int, component: str,
                    output_dir: str) -> None:
    """ヘッド・バッチ平均の (T, T) ヒートマップを1枚保存。"""
    mat = _attn_to_2d(attn)
    T = mat.shape[0]

    vmin, vmax = _robust_vlim(mat)
    fig_size = max(5, min(T // 8, 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(mat, aspect='auto', origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'{component} Layer {layer_idx}  (avg over heads & batch)', fontsize=12)
    ax.set_xlabel('Key position', fontsize=10)
    ax.set_ylabel('Query position', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f'{component.lower()}_layer{layer_idx:02d}_avg.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[info] 保存: {out_path}')


def _plot_layer_perhead(attn: np.ndarray, layer_idx: int, component: str,
                        output_dir: str) -> None:
    """ヘッドごとのサブプロットを保存。attn shape: (B, n_heads, T, T) or (n_heads, T, T)。"""
    # バッチを平均して (n_heads, T, T) にする
    if attn.ndim == 4:
        attn = attn.mean(axis=0)
    if attn.ndim == 2:
        attn = attn[np.newaxis]   # 擬似的にヘッド1扱い

    n_heads = attn.shape[0]
    ncols = min(4, n_heads)
    nrows = (n_heads + ncols - 1) // ncols

    T = attn.shape[-1]
    cell = max(3, min(T // 16, 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows),
                              squeeze=False)

    for h in range(n_heads):
        r, c = divmod(h, ncols)
        ax = axes[r][c]
        vmin_h, vmax_h = _robust_vlim(attn[h])
        im = ax.imshow(attn[h], aspect='auto', origin='upper', cmap='viridis', vmin=vmin_h, vmax=vmax_h)
        ax.set_title(f'Head {h}', fontsize=9)
        ax.set_xlabel('Key', fontsize=7)
        ax.set_ylabel('Query', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(n_heads, nrows * ncols):
        r, c = divmod(h, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'{component} Layer {layer_idx}  — Per-head attention (avg over batch)',
                 fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(output_dir,
                            f'{component.lower()}_layer{layer_idx:02d}_perhead.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[info] 保存: {out_path}')


def _plot_summary(all_attn: Dict[int, np.ndarray], n_layers: int,
                  component: str, output_dir: str) -> None:
    """全層のヘッド平均マップをまとめたサマリー図を保存。"""
    ncols = min(4, n_layers)
    nrows = (n_layers + ncols - 1) // ncols

    T_max = max((_attn_to_2d(v).shape[0] for v in all_attn.values()), default=1)
    cell = max(3, min(T_max // 16, 6))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(cell * ncols, cell * nrows),
                              squeeze=False)

    all_mats = [_attn_to_2d(v) for v in all_attn.values()]
    if all_mats:
        combined = np.concatenate([m.ravel() for m in all_mats])
        valid = combined[np.isfinite(combined)]
        if valid.size > 0:
            vmin_g = max(0.0, float(np.percentile(valid, 1)))
            vmax_g = float(np.percentile(valid, 99))
            if vmax_g <= vmin_g:
                vmax_g = float(valid.max())
            if vmax_g <= vmin_g:
                vmax_g = vmin_g + 1e-6
        else:
            vmin_g, vmax_g = 0.0, 1.0
    else:
        vmin_g, vmax_g = 0.0, 1.0

    for i in range(n_layers):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i not in all_attn:
            ax.set_visible(False)
            continue
        mat = _attn_to_2d(all_attn[i])
        im = ax.imshow(mat, aspect='auto', origin='upper', cmap='viridis',
                       vmin=vmin_g, vmax=vmax_g)
        ax.set_title(f'Layer {i}', fontsize=10)
        ax.set_xlabel('Key', fontsize=8)
        ax.set_ylabel('Query', fontsize=8)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_layers, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'{component}  Attention Maps  (avg heads & batch)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(output_dir, f'{component.lower()}_summary.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[info] サマリー図を保存: {out_path}')


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def run(args) -> None:
    model_dir   = args.model_dir
    output_dir  = args.output_dir
    ckpt_name   = args.checkpoint
    sample_idx  = args.sample_idx
    device      = args.device
    save_perhead = not args.no_perhead

    os.makedirs(output_dir, exist_ok=True)

    # ---- 設定ロード ----
    config = _load_config(model_dir)
    dconf  = config['data']
    mconf  = config['model']
    tokenization_method = dconf.get('tokenization_method', 'standard')
    use_difficulty      = bool(mconf.get('use_difficulty', False))

    print(f'[info] model_dir          : {model_dir}')
    print(f'[info] tokenization_method: {tokenization_method}')
    print(f'[info] use_difficulty     : {use_difficulty}')

    # ---- データセット（語彙取得 + サンプル取得）----
    val_split_path = dconf.get('val_split')
    pieces = None
    if val_split_path and os.path.isfile(val_split_path):
        with open(val_split_path, 'rb') as f:
            all_pieces = pickle.load(f)
        pieces = all_pieces[:max(1, sample_idx + 1)]
        print(f'[info] val split: {val_split_path}  ({len(pieces)} pieces を使用)')
    else:
        print('[warning] val_split が見つかりません。全データセットから読み込みます。')

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

    # ---- チェックポイントを先読みしてクラス数をデータセットに反映 ----
    ckpt_path = os.path.join(model_dir, 'params', ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'チェックポイントが見つかりません: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # チェックポイントの埋め込み重みからクラス数を読み取り、データセットの値を上書き
    # （バリデーション用の小さいデータでは全クラスが現れないため不一致が起きる）
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
            dset_cls = getattr(dset, attr, None)
            if dset_cls != ckpt_cls:
                print(f'[info] {attr}: dataset={dset_cls} → checkpoint={ckpt_cls} に上書き')
                setattr(dset, attr, ckpt_cls)

    # ---- モデル構築 ----
    bias_params = _resolve_bias_params(mconf, dset)
    model = _build_model(mconf, dset, tokenization_method, bias_params, use_difficulty)

    model.load_state_dict(state_dict)
    print(f'[info] チェックポイントをロード: {ckpt_path}')

    model = model.to(device).eval()

    # ---- バッチをデバイスへ ----
    enc_inp          = batch['enc_input'].permute(2, 0, 1).to(device)
    dec_inp          = batch['dec_input'].permute(1, 0).to(device)
    dec_inp_bar_pos  = batch['bar_pos'].to(device)
    padding_mask     = batch['enc_padding_mask'].to(device)

    s_tech_cls  = batch['s_tech_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_indep_cls = batch['s_indep_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_hand_cls  = batch['s_hand_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_foot_cls  = batch['s_foot_cls_seq'].permute(1, 0).to(device) if use_difficulty else None
    s_move_cls  = batch['s_move_cls_seq'].permute(1, 0).to(device) if use_difficulty else None

    # ---- レイヤーリスト取得 ----
    enc_layers, dec_layers = _get_layer_lists(model)
    print(f'[info] Encoder layers: {len(enc_layers)}  Decoder layers: {len(dec_layers)}')

    # ---- forward + 注意重み捕捉 ----
    with AttentionCapture(enc_layers) as enc_cap, \
         AttentionCapture(dec_layers) as dec_cap:
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

    enc_maps = enc_cap.maps
    dec_maps = dec_cap.maps

    print(f'[info] 捕捉した Encoder 注意マップ数: {len(enc_maps)}')
    print(f'[info] 捕捉した Decoder 注意マップ数: {len(dec_maps)}')

    # ---- 描画・保存 ----
    for component, maps, n_layers in [
        ('Encoder', enc_maps, len(enc_layers)),
        ('Decoder', dec_maps, len(dec_layers)),
    ]:
        for layer_idx in range(n_layers):
            if layer_idx not in maps:
                print(f'[warning] {component} Layer {layer_idx} の注意重みが捕捉されませんでした')
                continue
            attn = maps[layer_idx]
            shape_str = 'x'.join(str(d) for d in attn.shape)
            print(f'[info] {component} Layer {layer_idx}  attn shape: {shape_str}')

            _plot_layer_avg(attn, layer_idx, component, output_dir)
            if save_perhead:
                _plot_layer_perhead(attn, layer_idx, component, output_dir)

        _plot_summary(maps, n_layers, component, output_dir)

    print(f'\n[info] 完了。図は {output_dir} に保存されました。')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='各層のAttentionMapをヒートマップとして保存する'
    )
    parser.add_argument('--model_dir', required=True,
                        help='チェックポイントディレクトリ（config.yaml と params/ を含む）')
    parser.add_argument('--output_dir', default='output/attention_maps',
                        help='図の保存先ディレクトリ（デフォルト: output/attention_maps）')
    parser.add_argument('--checkpoint', default='best_params.pt',
                        help='使用するチェックポイントファイル名（デフォルト: best_params.pt）')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='使用するサンプルのインデックス（デフォルト: 0）')
    parser.add_argument('--device', default='cpu',
                        help='デバイス（cpu / cuda, デフォルト: cpu）')
    parser.add_argument('--no_perhead', action='store_true',
                        help='指定するとヘッド別図を省略してヘッド平均図のみ保存')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
