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

checkpoints=(
    "checkpoints_drum_standard-model8-inattn"
    "checkpoints_drum_standard-model9-inattn"
    "checkpoints_drum_standard-model10-inattn"
    "checkpoints_drum_standard-model11-inattn"
    "checkpoints_drum_standard-model12-inattn"
    "checkpoints_drum_standard-model13-inattn"
    "checkpoints_drum_standard-model14-inattn"
    "checkpoints_drum_standard-model15-inattn"
)

for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.evaluate.evaluate \
        --input_midi_dir data/test_dataset \
        --model_path trained_model/${checkpoint}/params/best_params.pt \
        --difficulty_labels_file <(echo '[
            {"s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"s_tech_cls": 2, "s_indep_cls": 2, "s_hand_cls": 2, "s_foot_cls": 2, "s_move_cls": 2},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 6, "s_indep_cls": 6, "s_hand_cls": 6, "s_foot_cls": 6, "s_move_cls": 6},
            {"s_tech_cls": 7, "s_indep_cls": 7, "s_hand_cls": 7, "s_foot_cls": 7, "s_move_cls": 7},
            {"s_tech_cls": 0, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 7, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 0, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 7, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 0, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 7, "s_foot_cls": 4, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 0, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 7, "s_move_cls": 4},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 0},
            {"s_tech_cls": 4, "s_indep_cls": 4, "s_hand_cls": 4, "s_foot_cls": 4, "s_move_cls": 7}
            ]') \
        --output_dir output/eval/${checkpoint}
done

wait
echo "すべての学習が完了しました。"
