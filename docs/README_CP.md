# MuseMorphose CPモードガイド

このドキュメントでは、ドラム学習および生成向けに実装した四肢ベースCPモードについて説明します。

## 概要

CPモードでは、以下の複合イベント表現を導入しています。

- CPイベント: [Position, Hand1, Hand2, Right_Foot, Left_Foot]
- 構造イベント: <TEMPO_xxx>, <BAR>, <BEAT_x>, <EOS>

重要ポイント:

- 手スロットは利き手非依存で、右手/左手固定ではなく Hand1/Hand2 を使います。
- HH_PEDAL は Left_Foot に割り当てます。
- ハイハット打撃は Hand1/Hand2 のどちらにも割り当て可能です。
- CP要素トークンは instrument+technique+velocity の情報を保持します。

## トークナイズ方式の切り替え

設定キー:

- data.tokenization_method: standard または cp_limb_v1

この切り替えに対応しているファイル:

- train_drum.py
- prepare_drum_dataset.py
- drum_dataloader.py

## CP埋め込み設計

CPモードでは、モデル内でCP専用埋め込みを使用します。

1. CPの各要素を独立に埋め込む
   - Position embedding
   - Hand1 embedding
   - Hand2 embedding
   - Right_Foot embedding
   - Left_Foot embedding
2. 5つの埋め込みをConcatする
3. 線形層でモデル次元へ投影する

関連設定キー:

- model.d_cp_pos_emb
- model.d_cp_limb_emb

## CP出力ヘッド

CPモードでは、各時刻で以下の多頭出力を予測します。

- event_type head
- structural head
- cp_pos head
- cp_hand1 head
- cp_hand2 head
- cp_right_foot head
- cp_left_foot head

損失は event_type に応じてマスクして計算します。

- event_type: 有効時刻すべて
- structural: structural時刻のみ
- CP要素ヘッド: CP時刻のみ

ヘッドごとの損失重み設定キー:

- model.cp_loss_weights

## データ準備

### 1) データセットの作成

例:

python3 prepare_drum_dataset.py \
  --midi_dir dataset \
  --output_dir ./drum_dataset_remi \
  --vocab_path ./drum_vocab_remi.pkl \
  --file_extension .midi \
  --tokenization_method remi

python3 prepare_drum_dataset.py \
  --midi_dir dataset \
  --output_dir ./drum_dataset_standard \
  --vocab_path ./drum_vocab.pkl \
  --file_extension .midi \
  --tokenization_method standard

python3 prepare_drum_dataset.py \
  --midi_dir dataset \
  --output_dir ./drum_dataset_cp \
  --vocab_path ./drum_vocab_cp.pkl \
  --file_extension .midi \
  --tokenization_method cp_limb_v1

### 2) 難易度閾値の算出

例:

python3 compute_difficulty_bounds.py drum_dataset_remi drum_vocab_remi.pkl difficulty_bounds_remi.pkl 120

python3 compute_difficulty_bounds.py drum_dataset_standard drum_vocab.pkl difficulty_bounds_standard.pkl 120

python3 compute_difficulty_bounds.py drum_dataset_cp drum_vocab_cp.pkl difficulty_bounds_cp.pkl 120

### 3) 難易度ラベルの付与

例:

python3 prepare_drum_dataset_with_difficulty.py drum_dataset_remi drum_vocab_remi.pkl difficulty_bounds_remi.pkl drum_dataset_remi_with_difficulty

python3 prepare_drum_dataset_with_difficulty.py drum_dataset_standard drum_vocab.pkl difficulty_bounds_standard.pkl drum_dataset_standard_with_difficulty

python3 prepare_drum_dataset_with_difficulty.py drum_dataset_cp drum_vocab_cp.pkl difficulty_bounds_cp.pkl drum_dataset_cp_with_difficulty

## 学習（CP）

configで以下を設定してください。

- data.tokenization_method: cp_limb_v1
- data.vocab_path: CP語彙pickleのパス
- data.data_dir: CPデータセットのディレクトリ

実行:

CUDA_VISIBLE_DEVICES=2 python3 train_drum.py --config config/drum_config_remi.yaml
CUDA_VISIBLE_DEVICES=1 python3 train_drum.py --config config/drum_config_standard.yaml
CUDA_VISIBLE_DEVICES=0 python3 train_drum.py --config config/drum_config_cp.yaml

注意:

- 既存のstandardモードも引き続き利用可能です。
- チェックポイント互換性は、tokenization mode とヘッド構成が一致している必要があります。

## CP専用生成スクリプト

新規スクリプト:

- generate_drum_cp.py

使い方:

python3 generate_drum_cp.py <config> <ckpt> <output_dir> <n_pieces> <n_samples_per_piece>

例:

python3 generate_drum_cp.py \
  configs/train_drum_difficulty.yaml \
  checkpoints_drum_gpu/params/step_50000-RC_0.123-KL_-0.000-model.pt \
  ./outputs_cp \
  3 \
  2

出力ファイル:

- *.mid : 生成MIDI
- *.txt : 生成されたCPイベントログ

## CPからMIDIへの変換ヘルパー

drum_to_midi.py に以下の関数を追加しています。

- cp_events_to_tokens(cp_data, idx2struct_token, idx2limb_token)
- cp_data_to_midi(cp_data, idx2struct_token, idx2limb_token, output_path, bpm=None)

これらは、CPイベント系列を既存トークン形式へ復元し、その後MIDIへ変換します。

## 現在の実装範囲と制約

- CP学習経路は実装済みです。
- CP生成スクリプトは実装済みです。
- 既存の generate_drum_difficulty.py は standard向け運用を維持し、cp_limb_v1 モードを明示的に拒否します。

## 最小設定例（CP）

model:
  d_cp_pos_emb: 64
  d_cp_limb_emb: 64
  cp_loss_weights:
    event_type: 1.0
    structural: 1.0
    cp_pos: 1.0
    cp_hand1: 1.0
    cp_hand2: 1.0
    cp_right_foot: 1.0
    cp_left_foot: 1.0

data:
  tokenization_method: cp_limb_v1
  vocab_path: ./drum_vocab_cp.pkl
  data_dir: ./drum_dataset_cp
