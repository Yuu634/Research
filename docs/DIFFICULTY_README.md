# ドラム譜難易度制御モデル - 使用方法

MuseMorphoseをベースにした、ドラム譜の5つの難易度指標を制御できる音楽生成モデルです。

## 難易度指標

1. **S_tech** (特殊奏法の頻度): ゴースト、リムショット、チョークなどの頻度
2. **S_indep** (手足の独立性): 手と足が独立して動く割合
3. **S_hand** (手の連打スピード): 手の最大打撃密度 (notes/sec)
4. **S_foot** (足の連打スピード): 足の最大打撃密度 (notes/sec)
5. **S_move** (打点間移動速度): 手の楽器間移動の最大速度

各指標は8クラス (0~7) に離散化され、独立に制御可能です。

---

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install torch numpy miditoolkit pyyaml tqdm
```

### 2. ドラム譜データの準備

MIDIファイルをトークン化して`drum_dataset/`に配置します（詳細は既存のドキュメント参照）。

---

## データセット準備の手順

### ステップ1: 難易度の境界値を計算

全データセットから各難易度指標の分布を解析し、8分位点を計算します。

```bash
python compute_difficulty_bounds.py drum_dataset drum_vocab.pkl difficulty_bounds.pkl 120
```

**引数**:
- `drum_dataset`: トークン化されたデータディレクトリ
- `drum_vocab.pkl`: 語彙ファイル
- `difficulty_bounds.pkl`: 出力ファイル（境界値）
- `120`: デフォルトBPM（オプション）

**出力例**:
```
s_tech bounds: [0.05, 0.12, 0.18, 0.25, 0.32, 0.42, 0.55]
s_indep bounds: [0.15, 0.28, 0.42, 0.56, 0.68, 0.78, 0.88]
...
```

### ステップ2: データセットに難易度クラスを付加

各小節の難易度を計算し、新しいデータセットを作成します。

```bash
python prepare_drum_dataset_with_difficulty.py \
    drum_dataset \
    drum_vocab.pkl \
    difficulty_bounds.pkl \
    drum_dataset_with_difficulty \
    120
```

**引数**:
- `drum_dataset`: 元のデータディレクトリ
- `drum_vocab.pkl`: 語彙ファイル
- `difficulty_bounds.pkl`: 境界値ファイル（ステップ1で作成）
- `drum_dataset_with_difficulty`: 出力ディレクトリ
- `120`: デフォルトBPM（オプション）

**処理内容**:
- 各pklファイルを読み込み
- 各小節の5つの難易度指標を計算
- 境界値で離散化（0~7のクラス）
- `(bar_pos, tokens, difficulty_classes)` の形式で保存

---

## モデルの訓練

### 基本的な訓練

```bash
python train_drum.py --config configs/train_drum_difficulty.yaml
```

### 設定ファイルのカスタマイズ

`configs/train_drum_difficulty.yaml` を編集して以下を調整できます:

- **データパス**: `data.data_dir`, `data.vocab_path`
- **モデルサイズ**: `model.enc_d_model`, `model.dec_d_model`
- **難易度埋め込み**: `model.d_s_tech_emb` など（デフォルト32次元）
- **訓練設定**: `training.max_lr`, `training.batch_size`

### 重要な設定項目

```yaml
model:
  use_difficulty: true  # 難易度属性を使用（必須）

data:
  data_dir: './drum_dataset_with_difficulty'  # 難易度付きデータセット
  difficulty_bounds_path: './difficulty_bounds.pkl'  # 境界値
```

---

## 生成（難易度制御）

生成スクリプトでは、潜在表現を抽出した後、5つの難易度クラスを指定して新しいドラム譜を生成します。

### 基本的な使用例

```python
import torch
from model.musemorphose import MuseMorphose
from drum_dataloader import DrumTransformerDataset

# モデルをロード
model = MuseMorphose(
    # ... パラメータ ...
    use_difficulty=True
)
model.load_state_dict(torch.load('ckpt.pt'))
model.eval()

# データセットから潜在表現を抽出
# ... (詳細は既存のgenerate.pyを参照)

# 難易度クラスを指定（0~7の範囲）
s_tech_cls = torch.tensor([3, 4, 5, 6])  # 4小節分
s_indep_cls = torch.tensor([2, 3, 4, 5])
s_hand_cls = torch.tensor([4, 5, 6, 7])
s_foot_cls = torch.tensor([2, 2, 3, 3])
s_move_cls = torch.tensor([3, 4, 4, 5])

# 生成
logits = model.generate(
    dec_input,
    latent_emb,
    s_tech_cls=s_tech_cls,
    s_indep_cls=s_indep_cls,
    s_hand_cls=s_hand_cls,
    s_foot_cls=s_foot_cls,
    s_move_cls=s_move_cls
)
```

### 難易度のランダムシフト

元の曲の難易度を±3クラス程度シフトさせることで、多様なバリエーションを生成できます。

```python
import numpy as np

# 元の難易度クラス（データセットから取得）
original_s_tech = data['s_tech_cls']  # shape: (n_bars,)

# ランダムにシフト
shift = np.random.randint(-3, 4)  # -3 ~ +3
shifted_s_tech = (original_s_tech + shift).clamp(0, 7)  # 0~7の範囲に制限
```

---

## ファイル構成

```
MuseMorphose/
├── drum_difficulty_calculator.py     # 難易度計算モジュール
├── compute_difficulty_bounds.py      # 境界値計算スクリプト
├── prepare_drum_dataset_with_difficulty.py  # データセット準備
├── model/
│   └── musemorphose.py               # モデル（5属性対応）
├── drum_dataloader.py                # データローダー（難易度対応）
├── train_drum.py                     # 訓練スクリプト（既存）
├── configs/
│   └── train_drum_difficulty.yaml    # 設定ファイル
└── DESIGN_DRUM_DIFFICULTY.md         # 詳細設計書
```

---

## トラブルシューティング

### データセット準備時のエラー

**エラー**: `No drum track found in MIDI file`
- MIDIファイルにドラムトラックが含まれているか確認

**エラー**: `Unknown data format`
- 既存のデータが正しくトークン化されているか確認
- `(bar_pos, tokens)` のタプル形式になっているか確認

### 訓練時のエラー

**エラー**: `KeyError: 's_tech_cls_seq'`
- データローダーで `use_difficulty=True` を設定しているか確認
- データセットが難易度クラスを含んでいるか確認

**エラー**: CUDA out of memory
- `batch_size` を小さくする（例: 8 → 4）
- `model.enc_d_model` を小さくする（例: 512 → 256）

### 生成時のエラー

**エラー**: モデルが難易度クラスを受け付けない
- モデル初期化時に `use_difficulty=True` を設定
- `use_attr_cls=True` も同時に設定されているか確認

---

## 検証・評価

生成されたドラム譜の難易度を検証するスクリプト:

```python
from drum_difficulty_calculator import compute_all_difficulty_scores

# 生成されたトークン列
generated_tokens = [...]  # トークンのリスト

# 小節ごとに区切って難易度を計算
bar_tokens = generated_tokens[bar_start:bar_end]
scores = compute_all_difficulty_scores(bar_tokens, bpm=120)

print(f"S_tech: {scores['s_tech']:.3f}")
print(f"S_indep: {scores['s_indep']:.3f}")
print(f"S_hand: {scores['s_hand']:.3f} notes/sec")
print(f"S_foot: {scores['s_foot']:.3f} notes/sec")
print(f"S_move: {scores['s_move']:.3f}")
```

---

## 詳細情報

- **設計書**: `DESIGN_DRUM_DIFFICULTY.md` を参照
- **難易度計算の詳細**: `drum_difficulty_calculator.py` のdocstringを参照
- **モデルアーキテクチャ**: `model/musemorphose.py` のコメントを参照

---

## 今後の拡張

1. **BPM情報の統合**: データセットに各曲のBPMを含める
2. **連続値制御**: 離散クラスではなく連続値で制御
3. **複合難易度**: 5指標を組み合わせた総合難易度の計算
4. **リアルタイム生成**: より高速な生成アルゴリズム

---

## ライセンス・引用

MuseMorphoseをベースにした実装です。元論文:

```
@inproceedings{musemorphose,
  title={MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with One Transformer VAE},
  author={Wu, Shih-Lun and Yang, Yi-Hsuan},
  booktitle={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2023}
}
```

---

## お問い合わせ

問題が発生した場合は、issueを作成するか、設計書を参照してください。
