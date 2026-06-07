"""
trained_model/ 以下のすべての checkpoint ディレクトリを走査し、
valloss.txt から train_loss と valid_loss の曲線を 1 枚のグラフにまとめて保存する。

Usage:
    python -m src.utils.plot_loss_curves [--trained_model_dir PATH]
"""

import re
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse_valloss_file(file_path: Path):
    """valloss.txt を解析して train / val の RC・KL 損失を返す。

    Returns:
        steps: list[int]
        train_rc, train_kl, val_rc, val_kl: list[float]
    """
    steps, train_rc, train_kl, val_rc, val_kl = [], [], [], [], []

    pattern = re.compile(
        r'\[step (\d+)\]\s+RC:\s*([\d.]+)\s*\|\s*KL:\s*([\d.\-]+)'
        r'\s*\|\s*\[val\]\s*\|\s*RC:\s*([\d.]+)\s*\|\s*KL:\s*([\d.\-]+)'
    )

    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                train_rc.append(float(m.group(2)))
                train_kl.append(float(m.group(3)))
                val_rc.append(float(m.group(4)))
                val_kl.append(float(m.group(5)))

    return steps, train_rc, train_kl, val_rc, val_kl


def plot_and_save(ckpt_dir: Path):
    valloss_path = ckpt_dir / 'valloss.txt'
    if not valloss_path.exists():
        print(f"  スキップ: valloss.txt が見つかりません ({ckpt_dir.name})")
        return

    steps, train_rc, train_kl, val_rc, val_kl = parse_valloss_file(valloss_path)
    if not steps:
        print(f"  スキップ: 有効なデータがありません ({ckpt_dir.name})")
        return

    train_total = [rc + kl for rc, kl in zip(train_rc, train_kl)]
    val_total   = [rc + kl for rc, kl in zip(val_rc,   val_kl)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(ckpt_dir.name, fontsize=14)

    # --- Total loss ---
    axes[0].plot(steps, train_total, label='train total', color='#1f77b4', linewidth=1.5)
    axes[0].plot(steps, val_total,   label='val total',   color='#ff7f0e', linewidth=1.5, linestyle='--')
    axes[0].set_title('Total Loss (RC + KL)')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- RC loss ---
    axes[1].plot(steps, train_rc, label='train RC', color='#2ca02c', linewidth=1.5)
    axes[1].plot(steps, val_rc,   label='val RC',   color='#d62728', linewidth=1.5, linestyle='--')
    axes[1].set_title('Reconstruction Loss (RC)')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- KL loss ---
    axes[2].plot(steps, train_kl, label='train KL', color='#9467bd', linewidth=1.5)
    axes[2].plot(steps, val_kl,   label='val KL',   color='#8c564b', linewidth=1.5, linestyle='--')
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = ckpt_dir / 'loss_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='全 checkpoint の train/val loss 曲線を生成する')
    parser.add_argument(
        '--trained_model_dir', '-d',
        type=str,
        default=str(Path(__file__).parents[2] / 'trained_model'),
        help='trained_model ディレクトリのパス (デフォルト: リポジトリルートの trained_model/)',
    )
    args = parser.parse_args()

    trained_model_dir = Path(args.trained_model_dir)
    if not trained_model_dir.exists():
        print(f"エラー: ディレクトリが見つかりません: {trained_model_dir}")
        return

    ckpt_dirs = sorted([
        d for d in trained_model_dir.iterdir()
        if d.is_dir() and d.name.startswith('checkpoints')
    ])

    if not ckpt_dirs:
        print(f"checkpoint ディレクトリが見つかりませんでした: {trained_model_dir}")
        return

    print(f"{len(ckpt_dirs)} 個の checkpoint ディレクトリを処理します\n")
    for ckpt_dir in ckpt_dirs:
        print(f"[{ckpt_dir.name}]")
        plot_and_save(ckpt_dir)

    print("\n完了")


if __name__ == '__main__':
    main()