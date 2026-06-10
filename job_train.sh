#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gca50130

# 作業ディレクトリへ移動
cd /home/acg17149ap/Drum_MuseMorphose

# Environment Modules の初期化
source /etc/profile.d/modules.sh

# Environment Modules の設定
module load cuda/13.0

# Python 仮想環境のアクティベート
source /home/acg17149ap/anaconda3/bin/activate
conda activate musemorphose

mkdir -p logs

python -m scripts.train.train_drum --config config/drum_config_standard-model15.yaml  > logs/model15.log  2>&1

wait
echo "すべての学習が完了しました。"
