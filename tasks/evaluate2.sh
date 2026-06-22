echo "性能評価"

checkpoints=(
    "checkpoints_drum_standard-model1"
)

# difficulty_mode: global_shift, upper_cap, full_assign
for checkpoint in "${checkpoints[@]}"; do
    python -m scripts.evaluate.evaluate \
        --input_midi_dir data/test_dataset \
        --model_path trained_model/${checkpoint}/params/best_params.pt \
        --difficulty_bounds data/metadata_new/difficulty_bounds_standard.pkl \
        --difficulty_labels_file <(echo '[
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": -2, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": -1, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 1, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 2, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": -2, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": -1, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 1, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 2, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": -2, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": -1, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 1, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 2, "s_foot_cls": 0, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": -2, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": -1, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 1, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 2, "s_move_cls": 0},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": -2},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": -1},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 1},
            {"difficulty_mode": "global_shift", "s_tech_cls": 0, "s_indep_cls": 0, "s_hand_cls": 0, "s_foot_cls": 0, "s_move_cls": 2}
            ]') \
        --output_dir output/eval-1attr-shift/${checkpoint}
        #--no_limb_bias_decoder \
        # --no_limb_bias_encoder \
        # --no_position_bias_encoder \
        # --no_position_bias_decoder
done
