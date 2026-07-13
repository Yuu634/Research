#!/bin/bash
echo "ドラム譜面難易度修正モデルの学習"

# 変数
CONFIG_PATH1="config/drum_config_standard-lakhaug1-biasloss15.yaml" # 学習設定ファイルパス
#CONFIG_PATH2="config/drum_config_standard-lakhaug1-biasloss11.yaml" # 学習設定ファイルパス

# ドラム譜面難易度修正モデルの学習
python -m scripts.train.train_drum --config ${CONFIG_PATH1}

#python -m scripts.train.train_drum --config ${CONFIG_PATH2}