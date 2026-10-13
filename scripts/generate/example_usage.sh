#!/bin/bash
# ドラム譜面生成スクリプトの使用例

# 基本設定
MODEL_PATH="trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt"
INPUT_MIDI="input_drum.mid"
OUTPUT_DIR="generated_outputs"

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_DIR"

echo "==================================================="
echo "MuseMorphose ドラム譜面生成デモ"
echo "==================================================="

# 1. オリジナルと同じ難易度で生成
echo ""
echo "[1] オリジナルと同じ難易度で生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_same_difficulty.mid" \
  --difficulty_shift 0 0 0 0 0

# 2. 手のスピードを上げて生成
echo ""
echo "[2] 手のスピードを+2にして生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_faster_hands.mid" \
  --difficulty_shift 0 0 2 0 0

# 3. 足のスピードを上げて生成
echo ""
echo "[3] 足のスピードを+2にして生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_faster_feet.mid" \
  --difficulty_shift 0 0 0 2 0

# 4. 全体的に難易度を上げて生成
echo ""
echo "[4] 全体的に難易度を上げて生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_harder_overall.mid" \
  --difficulty_shift 1 1 2 2 1

# 5. 特殊奏法と独立性を上げて生成
echo ""
echo "[5] 特殊奏法と独立性を上げて生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_technical.mid" \
  --difficulty_shift 2 2 0 0 0

# 6. テンポを変更して生成
echo ""
echo "[6] テンポを140 BPMにして生成..."
python scripts/generate/generate_drum_from_input.py \
  --model_path "$MODEL_PATH" \
  --input_midi "$INPUT_MIDI" \
  --output_midi "$OUTPUT_DIR/output_bpm140.mid" \
  --difficulty_shift 0 0 0 0 0 \
  --bpm 140

echo ""
echo "==================================================="
echo "生成完了！"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "==================================================="
