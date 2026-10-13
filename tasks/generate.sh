echo "ドラム譜面生成"
# 引数にドラム譜面を入力できるように要修正

# generate_drum_difficulty.py
python generate_drum_difficulty.py \
    --config configs/train_drum_difficulty.yaml \
    --ckpt ckpts/model.pt \
    --output output \
    --n_pieces 5 \
    --n_samples_per_piece 3

# generate_drum_cp.py
python generate_drum_cp.py \
    --config configs/train_drum_cp.yaml \
    --ckpt ckpts/model.pt \
    --output output \
    --n_pieces 5 \
    --n_samples_per_piece 3