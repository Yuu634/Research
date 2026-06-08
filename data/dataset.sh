#!/bin/bash

# 作成するデータセット指定
create_target=(groove_full) #(groove groove_full slakh2100 gigamidi)

if [[ " ${create_target[*]} " == *"slakh2100"* ]]; then
    echo "Creating Slakh2100 dataset..."
    # LakhMIDIDatasetダウンロード
    if [ ! -f data/lmd_full.tar.gz ]; then
        wget -P data http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
    fi
    if [ ! -d data/raw/lmd_full ]; then
        tar -xzf data/lmd_full.tar.gz -C data/raw
    fi

    # LakhMIDIからslakhMIDIを抽出
    if [ ! -d data/raw/slakh2100 ]; then
        python data/extract_slakh_from_lmd.py \
            --lmd_dir data/raw/lmd_full \
            --output data/raw/slakh2100
    fi
    if [ ! -d data/filtered/slakh2100 ]; then
        python data/cleaning_slakh.py \
            --dataset_dir data/raw/slakh2100 \
            --output_dir  data/filtered/slakh2100
    fi
fi

if [[ " ${create_target[*]} " == *"gigamidi"* ]]; then
    echo "Creating GigaMIDI dataset..."
    # GigaMIDIダウンロード
    if [ ! -d data/raw/gigamidi ]; then
        python data/download_gigamidi.py \
            --output_dir data/raw/gigamidi
    fi

    if [ ! -d data/filtered/gigamidi ]; then
        python data/cleaning_gigamidi.py \
            --output_dir  data/filtered
    fi
fi

if [[ " ${create_target[*]} " == *"groove_full"* ]]; then
    echo "Creating Groove Full dataset..."
    if [ ! -d data/filtered/groove_full ]; then
        python data/cleaning_groove_full.py \
            --groovefull_dir data/raw/e-gmd \
            --output_dir data/filtered
    fi
fi

if [[ " ${create_target[*]} " == *"groove"* ]]; then
    echo "Creating Groove MIDI dataset..."
    # Groove MIDI Datasetをダウンロードしておく
    if [ ! -d data/filtered/groove ]; then
        python data/groove_extract.py \
            --groove_zip data/groove-v1.0.0.zip \
            --output_dir data/filtered
    fi
fi