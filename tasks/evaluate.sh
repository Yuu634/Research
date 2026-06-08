echo "性能評価"

checkpoints=(
    "checkpoints_drum_standard-model1"
    "checkpoints_drum_standard-model2"
    "checkpoints_drum_standard-model3"
    "checkpoints_drum_standard-model4"
    "checkpoints_drum_standard-model5"
    "checkpoints_drum_standard-model6"
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
