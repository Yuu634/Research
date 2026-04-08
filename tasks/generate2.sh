echo "ドラム譜面生成（難易度レベル指定）"

# 入力譜面の難易度をシフトして生成
# python -m scripts.generate.generate_drum_from_input \\
#     --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \\
#     --input_midi input_drum.mid \\
#     --output_midi output_drum.mid \\
#     --difficulty_mode global_shift \\
#     --difficulty_shift 0 0 2 1 0

# 入力譜面の難易度を上限値で制限して生成
# python -m scripts.generate.generate_drum_from_input \\
#     --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \\
#     --input_midi input_drum.mid \\
#     --output_midi output_drum.mid \\
#     --difficulty_mode upper_cap \\
#     --difficulty_cap 3 3 4 4 3

# 入力譜面の難易度を全小節分完全に指定して生成
# python -m scripts.generate.generate_drum_from_input \\
#     --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \\
#     --input_midi input_drum.mid \\
#     --output_midi output_drum.mid \\
#     --difficulty_mode full_assign \\
#     --difficulty_labels_file difficulty_labels.json