
checkpoints=(
    "checkpoints_drum_standard-model1"
    "checkpoints_drum_standard-model2"
    "checkpoints_drum_standard-model3"
    "checkpoints_drum_standard-model4"
    "checkpoints_drum_standard-model5"
    "checkpoints_drum_standard-model6"
    "checkpoints_drum_standard-model7"
)

# echo "Limb-bias分析結果"
# for checkpoint in "${checkpoints[@]}"; do
#     python -m scripts.analysis.inspect_alpha_limb \
#         --model_dir trained_model/${checkpoint} \
#         --output_dir output/alpha_limb/${checkpoint}
#     python -m scripts.analysis.inspect_position_bias \
#         --model_dir trained_model/${checkpoint} \
#         --output_dir output/position_bias/${checkpoint}
# done

echo "AttentionMap分析結果"
# Validデータをforwardに通しAttentionMapを可視化
for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.analysis.visualize_attention_maps \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/attention_maps/${checkpoint}
done
