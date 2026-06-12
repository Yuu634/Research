echo "Limb-bias分析結果"

checkpoints=(
    "checkpoints_drum_standard-model8-inattn"
    "checkpoints_drum_standard-model9-inattn"
    "checkpoints_drum_standard-model10-inattn"
    "checkpoints_drum_standard-model11-inattn"
    # "checkpoints_drum_standard-model12-inattn"
    # "checkpoints_drum_standard-model13-inattn"
    # "checkpoints_drum_standard-model14-inattn"
    # "checkpoints_drum_standard-model15-inattn"
)

for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.analysis.inspect_alpha_limb \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/alpha_limb/${checkpoint}
    python -m scripts.analysis.inspect_position_bias \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/position_bias/${checkpoint}
done
