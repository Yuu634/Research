"""既存pairdataに自己再構成ペアを追加し、別データセットとして全件検証する。"""
import argparse
import json
from pathlib import Path

from data.pairdata.reconstruction import SPLITS, build_dataset, validate_dataset
from data.pairdata.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/drum_pairdata_v1'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/drum_pairdata_v1+reconst'))
    parser.add_argument('--reconstruction-splits', nargs='+', choices=SPLITS, default=['train'])
    parser.add_argument('--shard-size', type=int, default=4096)
    parser.add_argument('--validate-only', action='store_true', help='既存出力を元pairdataと照合して再検証')
    args = parser.parse_args()
    try:
        if args.validate_only:
            report = validate_dataset(args.output, args.input)
            write_json(args.output / 'validation.json', report)
        else:
            report = build_dataset(args.input, args.output, args.reconstruction_splits, args.shard_size)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
