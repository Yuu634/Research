"""保存済みJSON Linesから損失曲線・探索スコアを再生成する（再学習なし）。"""
import argparse
from pathlib import Path

from src.common.loss_logging import plot_loss_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained-model-dir', '--trained_model_dir', '-d', type=Path,
                        default=Path('trained_model'), help='実験ディレクトリを再帰走査するルート')
    args = parser.parse_args()
    if not args.trained_model_dir.is_dir():
        parser.error(f'ディレクトリが見つかりません: {args.trained_model_dir}')
    directories = sorted({p.parent for name in ('train_loss.log', 'eval_loss.log')
                          for p in args.trained_model_dir.rglob(name)})
    if not directories:
        print('損失ログが見つかりませんでした')
    for directory in directories:
        plot_loss_curves(directory)
        print(f'保存: {directory / "loss_curve.png"}', flush=True)
    # 基準YAMLの旧checkpointでも従来の再描画を継続できるようにする。
    legacy = sorted({p.parent for p in args.trained_model_dir.rglob('valloss.txt')} - set(directories))
    if legacy:
        from src.utils.plot_loss_curves import plot_and_save
        for directory in legacy:
            plot_and_save(directory)


if __name__ == '__main__':
    main()
