echo "Limb-bias分析結果"

checkpoints=(
    "checkpoints_drum_standard-model1"
    "checkpoints_drum_standard-model2"
    "checkpoints_drum_standard-model3"
    "checkpoints_drum_standard-model4"
    "checkpoints_drum_standard-model5"
    "checkpoints_drum_standard-model6"
)

for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.analysis.inspect_alpha_limb \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/alpha_limb/${checkpoint}
done
