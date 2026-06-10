echo "ドラム譜面難易度修正モデルの学習"

# 変数
CONFIG_PATH="config/drum_config_standard-model15.yaml" # 学習設定ファイルパス


# ドラム譜面難易度修正モデルの学習
python -m scripts.train.train_drum --config ${CONFIG_PATH}