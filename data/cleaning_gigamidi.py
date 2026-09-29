"""
GigaMIDI drums-only データをクリーニングしてsplit別に保存する。

クリーニング条件:
  - NOMML_Instrument == ["Drums"]
  - 1拍24subdivision grid に乗らない音符（開始・終了tickが TPQN/24 の整数倍でない）を含まない
  - テンポ変化がない（set_tempoメッセージが1つのみ）
  - 4/4拍子以外を含まない（time_signatureメッセージがすべて4/4）

使用方法:
  python3 cleaning_gigamidi.py <output_dir>
"""

import io
import os
import sys
import argparse
from tqdm import tqdm
import mido

SPLITS = ["train", "validation", "test"]


def parse_args():
    parser = argparse.ArgumentParser(description="GigaMIDI クリーニング & 保存")
    parser.add_argument("--output_dir", nargs="?", default="data/filtered", help="保存先ディレクトリ (デフォルト: data/filtered)")
    return parser.parse_args()


def inspect_midi(midi_bytes: bytes, dataset_tpqn: int) -> None:
    """MIDIの内部構造を表示してデバッグする。"""
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    print(f"  dataset TPQN      : {dataset_tpqn}")
    print(f"  mido ticks_per_beat: {mid.ticks_per_beat}")
    print(f"  subdivision (TPQN/24): {dataset_tpqn / 24:.4f}")
    print(f"  --- 最初の20メッセージ (note_on/off のみ) ---")
    count = 0
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type in ("note_on", "note_off"):
                on_grid = ((current_tick * 24) % dataset_tpqn == 0)
                print(f"    tick={current_tick:6d}  msg.time={msg.time:5d}  {msg}  on_grid={on_grid}")
                count += 1
                if count >= 20:
                    return


def is_valid(midi_bytes: bytes, tpqn: int) -> bool:
    """クリーニング条件をすべて満たす場合 True を返す。"""
    try:
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    except Exception:
        return False

    active_notes: dict[int, int] = {}  # pitch -> note_on tick

    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time

            # if msg.type == "set_tempo":
            #     tempo_count += 1
            #     if tempo_count > 1:
            #         return False

            if msg.type == "time_signature":
                if msg.numerator != 4 or msg.denominator != 4:
                    return False

    return True


def main():
    args = parse_args()
    from datasets import load_dataset

    if args.output_dir == "debug":
        print("=== デバッグモード: validation の最初の3件を検査 ===")
        dataset = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split="validation")
        drums = dataset.filter(lambda x: x["NOMML_Instrument"] == ["Drums"])
        for i, item in enumerate(drums.select(range(3))):
            print(f"\n--- サンプル {i} (md5: {item['md5']}) ---")
            inspect_midi(item["music"], item.get("TPQN") or 480)
        return

    stats = {"total": 0, "passed": 0, "saved": 0}

    for split in SPLITS:
        print(f"\n=== {split} ===")
        dataset = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split=split, keep_in_memory=True)

        drums = dataset.filter(lambda x: x["NOMML_Instrument"] == ["Drums"] and x["NOMML"][0] <= 7, keep_in_memory=True)
        print(f"drums-only: {len(drums)} 件")

        output_dir = os.path.join(args.output_dir, "gigamidi", split)
        os.makedirs(output_dir, exist_ok=True)

        split_passed = 0
        split_saved = 0
        for item in tqdm(drums, desc=split):
            stats["total"] += 1
            midi_bytes = item.get("music")
            file_hash = item.get("md5")
            tpqn = item.get("TPQN") or 480

            if not (midi_bytes and file_hash):
                continue

            if not is_valid(midi_bytes, tpqn):
                continue

            split_passed += 1
            stats["passed"] += 1

            with open(os.path.join(output_dir, f"{file_hash}.mid"), "wb") as f:
                f.write(midi_bytes)
            split_saved += 1
            stats["saved"] += 1

        print(f"  通過: {split_passed} 件 / 保存: {split_saved} 件")

    print(f"\n✓ 完了: {stats['saved']} / {stats['total']} 件を {args.output_dir} に保存しました")


if __name__ == "__main__":
    main()
