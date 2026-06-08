import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="data/raw/gigamidi")
args = parser.parse_args()

SPLITS = ["train", "validation", "test"]
OUTPUT_BASE = args.output_dir

total_saved = 0
for split in SPLITS:
    dataset = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split=split, cache_dir="/mnt/kiso-qnap5/obara/hf_cache", keep_in_memory=True)
    filtered = dataset.filter(lambda x: x["NOMML_Instrument"] == ["Drums"], keep_in_memory=True)
    print(f"{split}: {len(dataset)} -> {len(filtered)} 件")

    output_dir = os.path.join(OUTPUT_BASE, split)
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for item in tqdm(filtered, desc=split):
        file_hash = item.get("md5")
        midi_bytes = item.get("music")
        if file_hash and midi_bytes:
            with open(os.path.join(output_dir, f"{file_hash}.mid"), "wb") as f:
                f.write(midi_bytes)
            saved += 1
    total_saved += saved

print(f"\n✓ 完了: 合計 {total_saved} ファイルを {OUTPUT_BASE} に保存しました")