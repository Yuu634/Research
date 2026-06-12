echo "Limb-bias分析結果"

checkpoints=(
    "checkpoints_drum_standard-model16"
    "checkpoints_drum_standard-model17"
    "checkpoints_drum_standard-model18"
)

for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.analysis.inspect_alpha_limb \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/alpha_limb/${checkpoint}
    python -m scripts.analysis.inspect_position_bias \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/position_bias/${checkpoint}
done
