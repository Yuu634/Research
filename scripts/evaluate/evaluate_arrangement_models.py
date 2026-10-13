"""指定した実験・方式の保存済みモデルを再学習せずCPUでtest/itest評価する。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('musemorphose', 'beamsearch', 'beam_muse'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scope', choices=('selected', 'all'), default='selected')
    parser.add_argument('--splits', nargs='+', choices=('test', 'itest'), default=['test', 'itest'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1:
        parser.error('workersとbatch-sizeは正整数')
    # CLIのhelpはモデルや重い依存を読み込まず表示する。
    import torch
    from data.pairdata.io import file_hash, write_json
    from scripts.evaluate.refresh_test_metrics import refresh_beam, refresh_neural
    try:
        args.root = args.run.parent
        args.variant = None if args.model == 'beamsearch' else args.model
        if args.model == 'beamsearch':
            if not (args.run/'run_manifest.json').is_file():
                raise ValueError('BeamSearchの実行記録がありません')
        else:
            selection = json.loads((args.run/'selection.json').read_text())
            if selection['selected_on'] != 'val' or args.model not in selection['winners']:
                raise ValueError('指定方式のval選定記録がありません')
            directory = args.run/selection['winners'][args.model]['name']
            completed = json.loads((directory/'completed.json').read_text())
            if not (directory/'best.pt').is_file() or completed['trial']['variant'] != args.model:
                raise ValueError('指定方式の学習完了記録・選定重みがありません')
            checksum = file_hash(directory/'best.pt')
            if (checksum != completed['checkpoint_sha256']
                    or checksum != selection['winners'][args.model]['checkpoint_sha256']):
                raise ValueError('val選定記録・学習完了記録と重みが不一致です')
        status_path = args.root/'test_evaluation_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        torch.set_num_threads(1)
        (refresh_beam if args.model == 'beamsearch' else refresh_neural)(args.run, args, status)
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
