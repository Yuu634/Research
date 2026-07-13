
checkpoints=(
    "checkpoints_drum_standard-lakhaug1-biasloss1"
    "checkpoints_drum_standard-lakhaug1-biasloss2"
    "checkpoints_drum_standard-lakhaug1-biasloss3"
    "checkpoints_drum_standard-lakhaug1-kl12"
    "checkpoints_drum_standard-lakhaug1-kl13"
    "checkpoints_drum_standard-lakhaug1-kl14"
    "checkpoints_drum_standard-lakhaug1-kl15"
    "checkpoints_drum_standard-lakhaug1-kl16"
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

# echo "AttentionMap分析結果"
# for checkpoint in "${checkpoints[@]}"; do
#     python -m scripts.analysis.visualize_attention_maps \
#         --model_dir trained_model/${checkpoint} \
#         --output_dir output/attention_maps/${checkpoint} \
#         --sample_idx 5
# done

echo "Bias比率分析結果"
for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.analysis.qk_limb_bias_ratio \
        --model_dir trained_model/${checkpoint} \
        --output_dir output/qk_limb_ratio/${checkpoint}
done
