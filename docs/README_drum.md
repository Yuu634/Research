# ドラム譜MuseMorphose: ドラム譜専用トークン表記法による学習

このプロジェクトは、ドラム譜特有のトークン表記法を用いてMuseMorphoseモデルを0から学習するための実装です。

## 概要

従来のREMI表現の代わりに、ドラム演奏の技法（チョーク、リムショット、フラム等）を明示的に表現する独自のトークン表記法を使用します。これにより、モデルが演奏難易度や奏法の複雑さを直接学習できるようになります。

## トークン表記法の特徴

1. **Duration（音価）の廃止**: 音の長さは表現せず、「いつ発音したか」のみを記録
2. **トークンの結合**: `[楽器]_[技法]_[ベロシティ]` の形式で1つのトークンとして扱う
3. **構造トークン**: `<BAR>`, `<BEAT_1>`～`<BEAT_4>`, `<POS_0>`～`<POS_23>`
4. **演奏技法の明示**: RIMSHOT, XSTICK, FLAM, ROLL, CHOKE等を独立したトークンとして表現
5. **テンポトークン（5 BPM刻み）**: `<TEMPO_120>`, `<TEMPO_125>` 等、各小節のテンポを明示的に記録

## ファイル構成

```
MuseMorphose/
├── drum_tokenizer.py           # ドラム譜トークナイザー
├── drum_to_midi.py             # トークン列→MIDIファイル変換
├── prepare_drum_dataset.py     # データセット準備スクリプト
├── drum_dataloader.py          # ドラム譜用データローダー
├── train_drum.py               # 学習スクリプト
├── config/
│   └── drum_config.yaml        # 学習設定ファイル
└── README_drum.md              # このファイル
```

## 使用方法

### クイックスタート（GPU - Blackwell対応）

**Blackwell GPU (NVIDIA RTX PRO 6000, sm_120)を使用する場合**:

```bash
# 1. PyTorchをインストール（Blackwell対応）
bash install_pytorch_blackwell.sh

# 2. データセットを準備
python prepare_drum_dataset.py \
    --midi_dir /path/to/your/drum/midis \
    --output_dir ./drum_prepare \
    --vocab_path ./drum_vocab.pkl

# 3. GPU学習を開始
bash start_gpu_training.sh
```

詳細なセットアップガイド: **`BLACKWELL_GPU_SETUP.md`**

### クイックスタート（CPU）

最も簡単な方法は、CPUで学習を開始することです：

```bash
# データセットを準備
python prepare_drum_dataset.py \
    --midi_dir /path/to/your/drum/midis \
    --output_dir ./drum_prepare \
    --vocab_path ./drum_vocab.pkl

# 学習を開始（CPU）
bash quickstart_cpu.sh
```

または、手動で実行：

```bash
python train_drum.py --config config/drum_config.yaml --device cpu
```

### GPU学習（一般的なGPU）

PyTorchが既にインストールされている場合：

```bash
# GPU用設定ファイルを使用
python train_drum.py --config config/drum_config_gpu.yaml

# または、コマンドライン引数で指定
python train_drum.py --config config/drum_config.yaml --device cuda
```

### 1. データセットの準備

ドラムMIDIファイルをトークン化してデータセットを作成します。

```bash
python prepare_drum_dataset.py \
    --midi_dir /path/to/your/drum/midis \
    --output_dir ./drum_dataset \
    --vocab_path ./drum_vocab.pkl \
    --train_ratio 0.9 \
    --seed 42
```

オプション:
- `--midi_dir`: ドラムMIDIファイルが格納されているディレクトリ
- `--output_dir`: トークン化されたデータの保存先ディレクトリ
- `--vocab_path`: 語彙ファイルの保存先パス
- `--train_ratio`: 訓練データの割合（デフォルト: 0.9）
- `--seed`: ランダムシード（デフォルト: 42）
- `--file_extension`: MIDIファイルの拡張子（デフォルト: .mid）

### 2. 設定ファイルの編集

`config/drum_config.yaml` を編集して、モデルと学習のパラメータを設定します。

主な設定項目:
- **データ設定**: データディレクトリ、バッチサイズ、系列長など
- **モデル設定**: エンコーダー/デコーダーのレイヤー数、次元数など
- **学習設定**: 学習率、KLベータ、チェックポイント間隔など

### 3. モデルの学習

設定ファイルを指定して学習を開始します。

```bash
python train_drum.py --config config/drum_config.yaml
```

学習の進捗は以下に保存されます:
- `checkpoints_drum/log.txt`: 学習ログ
- `checkpoints_drum/valloss.txt`: 検証損失
- `checkpoints_drum/params/`: モデルパラメータ
- `checkpoints_drum/optim/`: オプティマイザー状態

### 4. 学習の再開

学習を中断した場合、設定ファイルで以下を指定して再開できます:

```yaml
model:
  pretrained_params_path: './checkpoints_drum/params/step_10000-RC_2.345-KL_0.123-model.pt'
  pretrained_optim_path: './checkpoints_drum/optim/step_10000-RC_2.345-KL_0.123-optim.pt'

training:
  trained_steps: 10000  # 再開するステップ数
```

## トークン語彙の例

### 構造トークン
- `<BAR>`: 小節の先頭
- `<BEAT_1>`, `<BEAT_2>`, `<BEAT_3>`, `<BEAT_4>`: 拍の切り替わり
- `<POS_0>`～`<POS_23>`: 1拍を24分割した発音タイミング

### ドラム打撃トークン
- **スネア**: `SNARE_HIT_Ghost`, `SNARE_HIT_Normal`, `SNARE_HIT_Accent`
- **リムショット**: `SNARE_RIMSHOT_Normal`, `SNARE_RIMSHOT_Accent`
- **クロススティック**: `SNARE_XSTICK_Normal`, `SNARE_XSTICK_Accent`
- **フラム**: `SNARE_FLAM_Normal`, `SNARE_FLAM_Accent`
- **キック**: `KICK_HIT_Normal`, `KICK_HIT_Accent`
- **ハイハット**: `HH_CLOSED_HIT_Ghost`, `HH_OPEN_HIT_Normal`, `HH_PEDAL`
- **ライド**: `RIDE_BOW_HIT_Normal`, `RIDE_BELL_HIT_Accent`
- **クラッシュ**: `CRASH_HIT_Accent`, `CRASH_CHOKE`

## トークン列の例

```
<BAR>
<BEAT_1> <POS_0> KICK_HIT_Normal HH_CLOSED_HIT_Normal
<POS_6> HH_CLOSED_HIT_Ghost
<POS_12> SNARE_HIT_Normal HH_CLOSED_HIT_Normal
<POS_18> HH_CLOSED_HIT_Ghost
<BEAT_2> <POS_0> KICK_HIT_Accent CRASH_HIT_Accent
<POS_12> CRASH_CHOKE
<EOS>
```

この例では:
1. 小節の1拍目にキックとクローズドハイハットを同時に演奏
2. 少し後にゴーストノートのハイハット
3. 2拍目にスネアとハイハット
4. さらに後にゴーストノートのハイハット
5. 2小節目の1拍目にアクセント付きキックとクラッシュ
6. その後クラッシュをチョーク（手で掴んで音を止める）

## モデルアーキテクチャ

MuseMorphoseはVariational Autoencoder (VAE)ベースのTransformerモデルです:

- **エンコーダー**: 小節ごとの音楽情報を潜在空間にエンコード
- **デコーダー**: 潜在変数から音楽系列を生成
- **潜在空間**: 演奏難易度などの属性を調整可能

ドラム譜版では、ポリフォニーやリズム頻度などの属性埋め込みは使用していません（`cond_mode: 'none'`）。

## GPU最適化と Mixed Precision Training

### Mixed Precision Training (AMP)

GPUメモリを節約し、学習を高速化するために、Mixed Precision Training（自動混合精度）を使用できます。

**有効化方法**:
```yaml
# config/drum_config.yaml または config/drum_config_gpu.yaml
training:
  device: 'cuda'
  use_amp: true  # Mixed Precision Trainingを有効化
```

**メリット**:
- GPUメモリ使用量を約30-50%削減
- 学習速度が約1.5-2倍向上（GPU次第）
- 数値の安定性を維持

### GPUメモリ別の推奨設定

**24GB VRAM（例: RTX 4090, RTX 6000）**:
```yaml
data:
  batch_size: 8
  dec_seqlen: 1280
  enc_seqlen: 128
training:
  use_amp: true
```

**12GB VRAM（例: RTX 3080, RTX 4070）**:
```yaml
data:
  batch_size: 4
  dec_seqlen: 1280
  enc_seqlen: 128
training:
  use_amp: true
```

**8GB VRAM（例: RTX 3060）**:
```yaml
data:
  batch_size: 2
  dec_seqlen: 640
  enc_seqlen: 64
model:
  enc_d_model: 256
  enc_d_ff: 1024
  dec_d_model: 256
  dec_d_ff: 1024
training:
  use_amp: true
```

詳細は **`BLACKWELL_GPU_SETUP.md`** を参照してください。

## トラブルシューティング

### CUDAエラー: no kernel image is available

**エラーメッセージ:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因:**
使用しているGPUのCUDA capability（例: sm_120 for Blackwell世代）がインストールされているPyTorchでサポートされていません。

**解決方法:**

1. **CPUで実行する（推奨）:**
   ```yaml
   # config/drum_config.yaml
   training:
     device: 'cpu'
   ```
   または
   ```bash
   python train_drum.py --config config/drum_config.yaml --device cpu
   ```

2. **PyTorchを最新版にアップグレード:**
   ```bash
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

詳細は `CUDA_TROUBLESHOOTING.md` を参照してください。

### MIDIファイルが見つからない

```
Found 0 MIDI files
```

- `--midi_dir` のパスが正しいか確認してください
- MIDIファイルの拡張子が `.mid` でない場合は `--file_extension` を指定してください

### ドラムトラックが見つからない

```
ValueError: No drum track found in MIDI file
```

- MIDIファイルがドラムトラック（is_drum=True）を含んでいるか確認してください
- 一般的なDAWでドラムパートを録音する際、チャンネル10をドラムチャンネルに設定してください

### メモリ不足

```
RuntimeError: CUDA out of memory
```

- `config/drum_config.yaml` の `batch_size` を小さくしてください
- `dec_seqlen` や `enc_seqlen` を小さくしてください
- モデルの次元数（`d_model`, `d_ff`）を小さくしてください

### 学習が進まない

- 学習率が適切か確認してください（`max_lr`, `min_lr`）
- KLベータのスケジューリングを調整してください（`kl_max_beta`, `kl_cycle_steps`）
- データセットのサイズが十分か確認してください

## テンポトークン機能に関するFAQ

### テンポトークンとは？

テンポトークン（`<TEMPO_XXX>`形式）は、ドラム譜のテンポを**5 BPM刻みで量子化**して記録するトークンです。
各小節の開始時に自動的に追加されます。

詳細: [TEMPO_TOKEN_PKL_GUIDE.md](TEMPO_TOKEN_PKL_GUIDE.md)

### テンポトークンの量子化ルール

5 BPM単位で量子化されます：

```
入力BPM      →  量子化後      →  テンポトークン
120-124      →  120          →  <TEMPO_120>
125-129      →  125          →  <TEMPO_125>
130-134      →  130          →  <TEMPO_130>
```

**例**:
- 122 BPM の MIDI → `<TEMPO_120>` (120-124の範囲に対応)
- 127 BPM の MIDI → `<TEMPO_125>` (125-129の範囲に対応)

### テンポトークンはいつ追加されるか？

`prepare_drum_dataset.py` でデータセットを準備する際に、**自動的に各小節の開始時にテンポトークンが追加**されます。

```bash
python prepare_drum_dataset.py \
    --midi_dir /path/to/your/drum/midis \
    --output_dir ./drum_prepare \
    --vocab_path ./drum_vocab.pkl
```

実行後のpklファイルには既にテンポトークンが含まれています。

### テンポトークンが正しく含まれているか確認する方法

[verify_tempo_token_in_pkl.py](verify_tempo_token_in_pkl.py) で検証できます：

```bash
python verify_tempo_token_in_pkl.py <midi_file> <output_pkl>

# 例
python verify_tempo_token_in_pkl.py test_token/sample_midis/simple_beat.mid test_output.pkl
```

出力例：
```
✓ ALL CHECKS PASSED - Tempo tokens are correctly integrated!
```

### 語彙サイズの増加

テンポトークンの追加により、語彙が以下のように増加します：

| 項目 | 数量 |
|------|------|
| テンポトークン | 37個 (60, 65, 70, ..., 240 BPM) |
| その他トークン | ~200個 |
| **合計語彙サイズ** | **~250個** |

### 旧形式のpklファイルは使用できるか？

テンポトークンなしの旧形式のpklファイルも読み込み可能ですが、テンポ情報が含まれません。
新しい学習には、テンポトークンを含むpklファイルの使用を推奨します。

### テンポが変わるMIDIには対応しているか？

はい、テンポ変更に対応しています。各小節で現在のテンポを自動検出し、テンポトークンを更新します：

```
original MIDI:
  - 0 tick: 120 BPM (小節1-2)
  - 1000 tick: 130 BPM (小節3以降)

tokenized output:
  <TEMPO_120> <BAR> ... (小節1)
  <TEMPO_120> <BAR> ... (小節2)
  <TEMPO_130> <BAR> ... (小節3)
```

## 生成方法（今後の実装）

学習済みモデルを使って新しいドラムパターンを生成する機能は今後実装予定です。

基本的な流れ:
1. 潜在変数をサンプリング（または属性から生成）
2. デコーダーでトークン列を生成
3. トークン列をMIDIファイルに変換（`drum_to_midi.py`を使用）

## 参考文献

- [MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with Just One Transformer VAE](https://arxiv.org/abs/2105.04090)
- [REMI: A Revolution in Music Representation for MIDI](https://dl.acm.org/doi/10.1145/3394171.3413671)

## ライセンス

このプロジェクトは元のMuseMorphoseプロジェクトと同じライセンスに従います。

## 問い合わせ

問題が発生した場合は、GitHubのIssuesに報告してください。
