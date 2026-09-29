"""
指定したモデルディレクトリの params/ 以下にある全チェックポイントから
encoder.position_bias_module および decoder.position_bias_module の値を
抽出してCSVと折れ線グラフに保存する。

モード別のパラメータ形状:
  uniform    : スカラー (1,) × n_layers
  per_limb   : 3×3 対称行列 (3,3) × n_layers
  asymmetric : 3×3 非対称行列 (3,3) × n_layers

使用方法:
  python -m scripts.analysis.inspect_position_bias \
    --model_dir trained_model/checkpoints_drum_standard-lakh1-kl14 \
    --output_dir output/position_bias/model2
"""

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


_STEP_RE = re.compile(r'step_(\d+)')

# per_limb / asymmetric モードの limb ラベル
_LIMB_LABELS = ['HAND', 'RFOOT', 'LFOOT']


def _parse_step(filename: str) -> int:
    m = _STEP_RE.search(filename)
    return int(m.group(1)) if m else -1


def _extract_position_bias(state_dict: dict, prefix: str) -> list[torch.Tensor] | None:
    """
    state_dict から position_bias_module のパラメータを取得する。

    固定値バッファ ({prefix}.position_bias_module._alpha_buf) を優先し、
    なければ可学習 ParameterList ({prefix}.position_bias_module.alpha_params.{i}) を使用。

    Returns:
        list of tensors (one per layer set):
          uniform    -> each tensor shape (1,)
          per_limb / asymmetric -> each tensor shape (3, 3)
        None: position_bias_module が見つからない場合
    """
    buf_key = f'{prefix}.position_bias_module._alpha_buf'
    if buf_key in state_dict and state_dict[buf_key] is not None:
        buf = state_dict[buf_key]  # (n_sets,) or (n_sets, 3, 3)
        return [buf[i] for i in range(buf.shape[0])]

    param_prefix = f'{prefix}.position_bias_module.alpha_params.'
    params: dict[int, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(param_prefix):
            idx = int(key[len(param_prefix):])
            params[idx] = tensor
    if not params:
        return None
    return [params[i] for i in sorted(params)]


def _is_matrix_mode(layers: list[torch.Tensor]) -> bool:
    return layers[0].ndim >= 2


def _csv_columns(prefix: str, n_sets: int, is_matrix: bool) -> list[str]:
    cols = []
    for i in range(n_sets):
        if is_matrix:
            for r in range(3):
                for c in range(3):
                    cols.append(f'{prefix}_layer{i}_r{r}c{c}')
        else:
            cols.append(f'{prefix}_layer{i}')
    return cols


def _layer_values_to_dict(prefix: str, layers: list[torch.Tensor], is_matrix: bool) -> dict:
    row = {}
    for i, t in enumerate(layers):
        if is_matrix:
            mat = t.detach().cpu().numpy()
            for r in range(3):
                for c in range(3):
                    row[f'{prefix}_layer{i}_r{r}c{c}'] = float(mat[r, c])
        else:
            row[f'{prefix}_layer{i}'] = float(t.detach().cpu().item())
    return row


def inspect(model_dir: str, output_dir: str) -> None:
    params_dir = os.path.join(model_dir, 'params')
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(f'params ディレクトリが見つかりません: {params_dir}')

    _EXCLUDE = {'best_params.pt', 'final_params.pt'}
    ckpt_files = sorted(
        [f for f in os.listdir(params_dir)
         if f.endswith('.pt') and f not in _EXCLUDE],
        key=lambda f: (_parse_step(f), f),
    )
    if not ckpt_files:
        ckpt_files = sorted(f for f in _EXCLUDE if os.path.isfile(os.path.join(params_dir, f)))
    if not ckpt_files:
        raise FileNotFoundError(f'チェックポイント (.pt) が見つかりません: {params_dir}')

    os.makedirs(output_dir, exist_ok=True)

    rows: list[dict] = []
    enc_n_sets = dec_n_sets = 0
    enc_is_matrix = dec_is_matrix = False

    for fname in ckpt_files:
        path = os.path.join(params_dir, fname)
        state_dict = torch.load(path, map_location='cpu', weights_only=True)

        enc_layers = _extract_position_bias(state_dict, 'encoder')
        dec_layers = _extract_position_bias(state_dict, 'decoder')

        step = _parse_step(fname)
        row: dict = {'checkpoint': fname, 'step': step}

        if enc_layers:
            enc_n_sets = max(enc_n_sets, len(enc_layers))
            enc_is_matrix = _is_matrix_mode(enc_layers)
            row.update(_layer_values_to_dict('enc', enc_layers, enc_is_matrix))

        if dec_layers:
            dec_n_sets = max(dec_n_sets, len(dec_layers))
            dec_is_matrix = _is_matrix_mode(dec_layers)
            row.update(_layer_values_to_dict('dec', dec_layers, dec_is_matrix))

        rows.append(row)

        has_enc = '✓' if enc_layers else '–'
        has_dec = '✓' if dec_layers else '–'
        if enc_layers and not enc_is_matrix:
            enc_str = ', '.join(f'{v:.6f}' for v in
                                [float(t.item()) for t in enc_layers])
        elif enc_layers:
            enc_str = f'[{len(enc_layers)} × 3×3 matrix]'
        else:
            enc_str = 'N/A'
        if dec_layers and not dec_is_matrix:
            dec_str = ', '.join(f'{v:.6f}' for v in
                                [float(t.item()) for t in dec_layers])
        elif dec_layers:
            dec_str = f'[{len(dec_layers)} × 3×3 matrix]'
        else:
            dec_str = 'N/A'
        print(f'[{fname}]  step={step}  enc[{has_enc}]={enc_str}  dec[{has_dec}]={dec_str}')

    # CSV
    fieldnames = ['checkpoint', 'step']
    fieldnames += _csv_columns('enc', enc_n_sets, enc_is_matrix)
    fieldnames += _csv_columns('dec', dec_n_sets, dec_is_matrix)

    csv_path = os.path.join(output_dir, 'position_bias.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n[info] 結果を保存しました: {csv_path}')
    print(f'[info] チェックポイント数: {len(rows)}'
          f'  enc_sets={enc_n_sets}  dec_sets={dec_n_sets}'
          f'  matrix_mode: enc={enc_is_matrix}  dec={dec_is_matrix}')

    _plot(rows, enc_n_sets, dec_n_sets, enc_is_matrix, dec_is_matrix, output_dir)


def _plot(rows: list[dict],
          enc_n_sets: int, dec_n_sets: int,
          enc_is_matrix: bool, dec_is_matrix: bool,
          output_dir: str) -> None:
    steps = [r['step'] for r in rows]
    tab10 = plt.cm.tab10.colors

    for side, n_sets, is_matrix, title_prefix, fname_base in [
        ('enc', enc_n_sets, enc_is_matrix, 'Encoder', 'position_bias_encoder'),
        ('dec', dec_n_sets, dec_is_matrix, 'Decoder', 'position_bias_decoder'),
    ]:
        if n_sets == 0:
            continue

        if not is_matrix:
            _plot_uniform(rows, steps, side, n_sets, title_prefix,
                          os.path.join(output_dir, f'{fname_base}.png'), tab10)
        else:
            _plot_matrix(rows, steps, side, n_sets, title_prefix,
                         os.path.join(output_dir, f'{fname_base}.png'), tab10)


def _plot_uniform(rows, steps, side, n_sets, title_prefix, fig_path, tab10):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(n_sets):
        key = f'{side}_layer{i}'
        values = [r.get(key) for r in rows]
        if any(v is not None for v in values):
            ax.plot(steps, values,
                    color=tab10[i % len(tab10)], linestyle='-', marker='o',
                    markersize=4, linewidth=2.0, label=f'Layer {i}')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('position_bias α', fontsize=12)
    ax.set_title(f'{title_prefix} position_bias (uniform) per layer over training steps',
                 fontsize=13)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'[info] グラフを保存しました: {fig_path}')


def _plot_matrix(rows, steps, side, n_sets, title_prefix, fig_path, tab10):
    """3×3 のサブプロット: (r, c) ごとに全レイヤーの推移を描画する。"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    fig.suptitle(f'{title_prefix} position_bias matrix per element over training steps',
                 fontsize=13)

    for r in range(3):
        for c in range(3):
            ax = axes[r][c]
            ax.set_title(f'{_LIMB_LABELS[r]} → {_LIMB_LABELS[c]}', fontsize=9)
            for i in range(n_sets):
                key = f'{side}_layer{i}_r{r}c{c}'
                values = [row.get(key) for row in rows]
                if any(v is not None for v in values):
                    ax.plot(steps, values,
                            color=tab10[i % len(tab10)], linestyle='-', marker='s',
                            markersize=3, linewidth=1.5, label=f'Layer {i}')
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel('α', fontsize=9)
            if r == 2:
                ax.set_xlabel('Step', fontsize=9)

    # 凡例は右上サブプロットにまとめる
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=8,
                   bbox_to_anchor=(1.0, 1.0), ncol=1)

    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'[info] グラフを保存しました: {fig_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='全チェックポイントから position_bias の値を抽出して CSV とグラフに保存する'
    )
    parser.add_argument('--model_dir', required=True,
                        help='モデルのルートディレクトリ（params/ サブディレクトリを含む）')
    parser.add_argument('--output_dir', default='output/position_bias',
                        help='CSV の保存先ディレクトリ（デフォルト: output/position_bias）')
    args = parser.parse_args()

    inspect(args.model_dir, args.output_dir)


if __name__ == '__main__':
    main()
