"""
E-GMD データセットをsplit別に保存する。

クリーニング条件:
  - time_signature == "4-4" のみ (groove_extract.py の is_valid と同一)

使用方法:
  python3 cleaning_groove_full.py --groovefull_dir data/raw/groove_full --output_dir data/filtered

出力先: <output_dir>/groove_full/{train,validation,test}/
"""

import argparse
import csv
import io
import os
import shutil

import mido


def parse_args():
    parser = argparse.ArgumentParser(description="E-GMD MIDI 分割 & 保存")
    parser.add_argument(
        "--groovefull_dir",
        default="data/raw/e-gmd",
        help="e-gmd ルートディレクトリ (e-gmd-v1.0.0.csv を含む)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/filtered",
        help="保存先ディレクトリ (デフォルト: data/filtered)",
    )
    return parser.parse_args()


def is_valid(midi_bytes: bytes) -> bool:
    """クリーニング条件をすべて満たす場合 True を返す。"""
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    except Exception:
        return False

    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                if msg.numerator != 4 or msg.denominator != 4:
                    return False

    return True


def main():
    args = parse_args()

    csv_path = os.path.join(args.groovefull_dir, "e-gmd-v1.0.0.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    stats = {"total": 0, "skipped_ts": 0, "skipped_invalid": 0, "skipped_missing": 0, "saved": 0}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        stats["total"] += 1

        if row["time_signature"] != "4-4":
            stats["skipped_ts"] += 1
            continue

        src_path = os.path.join(args.groovefull_dir, row["midi_filename"])
        if not os.path.exists(src_path):
            print(f"  [WARN] not found: {src_path}")
            stats["skipped_missing"] += 1
            continue

        with open(src_path, "rb") as mf:
            midi_bytes = mf.read()

        if not is_valid(midi_bytes):
            stats["skipped_invalid"] += 1
            continue

        split = row["split"]
        out_dir = os.path.join(args.output_dir, "groove_full", split)
        os.makedirs(out_dir, exist_ok=True)

        filename = os.path.basename(row["midi_filename"])
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as out_f:
            out_f.write(midi_bytes)
        stats["saved"] += 1

    print(f"\n完了: {stats['saved']} / {stats['total']} 件を保存")
    print(f"  スキップ (4/4以外): {stats['skipped_ts']} 件")
    print(f"  スキップ (is_valid失敗): {stats['skipped_invalid']} 件")
    print(f"  スキップ (ファイル未発見): {stats['skipped_missing']} 件")


if __name__ == "__main__":
    main()
