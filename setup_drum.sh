#!/bin/bash

# ドラム譜MuseMorphose クイックスタートスクリプト

echo "================================"
echo "ドラム譜MuseMorphose セットアップ"
echo "================================"

# 1. 依存関係のインストール
echo ""
echo "[1/5] 依存関係をインストール中..."
pip install -r requirements_drum.txt

# 2. テスト: トークナイザーの動作確認
echo ""
echo "[2/5] トークナイザーのテスト..."
python drum_tokenizer.py

if [ $? -ne 0 ]; then
    echo "エラー: トークナイザーのテストに失敗しました"
    exit 1
fi

# 3. 語彙ファイルの生成
echo ""
echo "[3/5] 語彙ファイルを生成中..."
python -c "
from drum_tokenizer import DrumTokenizer
tokenizer = DrumTokenizer()
tokenizer.save_vocab('./drum_vocab.pkl')
print(f'Vocabulary size: {tokenizer.vocab_size}')
print('Vocabulary saved to: ./drum_vocab.pkl')
"

if [ $? -ne 0 ]; then
    echo "エラー: 語彙ファイルの生成に失敗しました"
    exit 1
fi

# 4. 設定ファイルの確認
echo ""
echo "[4/5] 設定ファイルを確認中..."
if [ ! -f "config/drum_config.yaml" ]; then
    echo "エラー: config/drum_config.yaml が見つかりません"
    exit 1
fi
echo "設定ファイル: config/drum_config.yaml"

# 5. 次のステップを表示
echo ""
echo "[5/5] セットアップ完了!"
echo ""
echo "================================"
echo "次のステップ"
echo "================================"
echo ""
echo "1. ドラムMIDIファイルを準備してください"
echo ""
echo "2. データセットを準備します:"
echo "   python prepare_drum_dataset.py \\"
echo "       --midi_dir /path/to/your/drum/midis \\"
echo "       --output_dir ./drum_dataset \\"
echo "       --vocab_path ./drum_vocab.pkl"
echo ""
echo "3. config/drum_config.yaml を編集して、パスやパラメータを調整します"
echo ""
echo "4. 学習を開始します:"
echo "   python train_drum.py --config config/drum_config.yaml"
echo ""
echo "詳細は README_drum.md を参照してください"
echo "================================"
