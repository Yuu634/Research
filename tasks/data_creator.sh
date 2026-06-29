for TOKENIZATION_METHOD in "standard-lakh"; do
    echo "=== トークン化手法: ${TOKENIZATION_METHOD} ==="
    
    echo "=== 変数設定 ==="
    OPEN_DATASET_DIR="data/raw/e-gmd"           # ドラム譜面の公開データセットディレクトリ
    LAKH_DATASET_DIR="data/filtered/lakhmidi"
    DRUM_DATASET_DIR="data/processed_new/drum_dataset_${TOKENIZATION_METHOD}" # 難易度ラベル付与前データセットパス
    DRUM_VOCAB_PATH="data/metadata_new/drum_vocab_${TOKENIZATION_METHOD}.pkl" # ドラム譜面の語彙ファイルパス
    DIFFICULTY_BOUNDS_PATH="data/metadata_new/difficulty_bounds_${TOKENIZATION_METHOD}.pkl" # 難易度の境界値を保存するファイルパス
    DIFFICULTY_DATASET_DIR="data/processed_new/drum_dataset_${TOKENIZATION_METHOD}_with_difficulty" # 難易度ラベル付与後データセットパス
    #DEFAULT_BPM=120 # BPMのデフォルト値

    echo "=== MIDIデータセットからpklデータセットを作成 ==="
    python -m scripts.preprocess.prepare_drum_dataset \
        --midi_dir ${OPEN_DATASET_DIR} ${LAKH_DATASET_DIR} \
        --output_dir ${DRUM_DATASET_DIR} \
        --vocab_path ${DRUM_VOCAB_PATH} \
        --tokenization_method "standard" \
        --file_extension .midi .mid \
        --train_ratio 0.9 \
        #--use_limb_tokens \

    echo "=== 各難易度項目の境界値を計算 ==="
    python -m src.utils.compute_difficulty_bounds \
        ${DRUM_DATASET_DIR} \
        ${DRUM_VOCAB_PATH} \
        ${DIFFICULTY_BOUNDS_PATH} \
        #${DEFAULT_BPM}

    echo "=== ドラム譜面データセットの難易度ラベル付与 ==="
    python -m scripts.preprocess.prepare_drum_dataset_with_difficulty \
        ${DRUM_DATASET_DIR} \
        ${DRUM_VOCAB_PATH} \
        ${DIFFICULTY_BOUNDS_PATH} \
        ${DIFFICULTY_DATASET_DIR} \
        #${DEFAULT_BPM}

done