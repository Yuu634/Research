"""
指定したモデルディレクトリの params/ 以下にある全チェックポイントから
encoder.alpha_limb および decoder.alpha_limb の値を抽出してCSVに保存する。

使用方法:
  python -m scripts.analysis.inspect_alpha_limb \
    --model_dir /path/to/run_dir \
    --output_dir output/alpha_limb
"""

import argparse
import csv
import os
import re

import torch


_STEP_RE = re.compile(r'step_(\d+)')


def _parse_step(filename: str) -> int:
    """ファイル名からステップ数を取得する。見つからなければ -1。"""
    m = _STEP_RE.search(filename)
    return int(m.group(1)) if m else -1


def _extract_alpha(state_dict: dict, prefix: str) -> list[float] | None:
    """
    state_dict から `{prefix}.alpha_limb.{i}` を全レイヤー分取得して返す。
    1つも見つからなければ None。
    """
    values = {}
    for key, tensor in state_dict.items():
        if key.startswith(f'{prefix}.alpha_limb.'):
            idx = int(key.split('.')[-1])
            values[idx] = tensor.item()
    if not values:
        return None
    return [values[i] for i in sorted(values)]


def inspect(model_dir: str, output_dir: str) -> None:
    params_dir = os.path.join(model_dir, 'params')
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(f'params ディレクトリが見つかりません: {params_dir}')

    ckpt_files = sorted(
        [f for f in os.listdir(params_dir) if f.endswith('.pt')],
        key=lambda f: (_parse_step(f), f),
    )
    if not ckpt_files:
        raise FileNotFoundError(f'チェックポイント (.pt) が見つかりません: {params_dir}')

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'alpha_limb.csv')

    rows: list[dict] = []
    enc_n_layers = dec_n_layers = 0

    for fname in ckpt_files:
        path = os.path.join(params_dir, fname)
        state_dict = torch.load(path, map_location='cpu', weights_only=True)

        enc_alphas = _extract_alpha(state_dict, 'encoder')
        dec_alphas = _extract_alpha(state_dict, 'decoder')

        if enc_alphas:
            enc_n_layers = max(enc_n_layers, len(enc_alphas))
        if dec_alphas:
            dec_n_layers = max(dec_n_layers, len(dec_alphas))

        step = _parse_step(fname)
        row = {'checkpoint': fname, 'step': step}
        if enc_alphas:
            for i, v in enumerate(enc_alphas):
                row[f'enc_alpha_{i}'] = v
        if dec_alphas:
            for i, v in enumerate(dec_alphas):
                row[f'dec_alpha_{i}'] = v

        rows.append(row)

        has_enc = '✓' if enc_alphas else '–'
        has_dec = '✓' if dec_alphas else '–'
        enc_str = ', '.join(f'{v:.6f}' for v in enc_alphas) if enc_alphas else 'N/A'
        dec_str = ', '.join(f'{v:.6f}' for v in dec_alphas) if dec_alphas else 'N/A'
        print(f'[{fname}]  step={step}  enc[{has_enc}]={enc_str}  dec[{has_dec}]={dec_str}')

    # CSV のカラム順を決定
    fieldnames = ['checkpoint', 'step']
    fieldnames += [f'enc_alpha_{i}' for i in range(enc_n_layers)]
    fieldnames += [f'dec_alpha_{i}' for i in range(dec_n_layers)]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n[info] 結果を保存しました: {csv_path}')
    print(f'[info] チェックポイント数: {len(rows)}'
          f'  enc_layers={enc_n_layers}  dec_layers={dec_n_layers}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='全チェックポイントから alpha_limb の値を抽出して CSV に保存する'
    )
    parser.add_argument('--model_dir', required=True,
                        help='モデルのルートディレクトリ（params/ サブディレクトリを含む）')
    parser.add_argument('--output_dir', default='output/alpha_limb',
                        help='CSV の保存先ディレクトリ（デフォルト: output/alpha_limb）')
    args = parser.parse_args()

    inspect(args.model_dir, args.output_dir)


if __name__ == '__main__':
    main()
