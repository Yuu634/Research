# ドラム譜面生成スクリプト - 入力MIDIから生成

`generate_drum_from_input.py` は、入力されたドラム譜MIDIファイルから学習済みモデルを用いて、指定した演奏難易度レベルに沿った新しいドラム譜を生成するスクリプトです。

## 機能

1. **学習済みモデルのロード**: チェックポイントディレクトリの `config.yaml` からトークン設計手法を自動取得
2. **入力譜面データの読み込み**: ドラム譜MIDIファイルをトークン列に変換
3. **演奏難易度レベル算出**: 各小節の5つの難易度指標（特殊奏法、独立性、手スピード、足スピード、移動速度）を自動計算
4. **難易度制御生成**: 指定した難易度シフトに基づいて新しいドラム譜を生成
5. **MIDI保存**: 生成された譜面を指定パスにMIDIファイルとして保存

## 必須引数

- `--model_path`: 学習済みモデルのチェックポイントファイルパス（.ptファイル）
- `--input_midi`: 入力ドラム譜MIDIファイルのパス
- `--output_midi`: 出力MIDIファイルの保存先パス

## オプション引数

- `--difficulty_shift`: 難易度レベルのシフト値（5つの整数値）
  - 形式: `s_tech s_indep s_hand s_foot s_move`
  - デフォルト: `0 0 0 0 0`（難易度変更なし）
  - 例: `0 0 2 1 0`（手のスピードを+2、足のスピードを+1）
  
- `--nucleus_p`: Nucleus samplingのp値（デフォルト: 0.9）
- `--temperature`: サンプリング温度（デフォルト: 1.2）
- `--use_latent_sampling`: 潜在表現抽出時にサンプリングを使用
- `--latent_sampling_var`: 潜在サンプリングの分散（デフォルト: 0.1）
- `--bpm`: テンポ（BPM）（デフォルト: 120）

## 使用例

### 基本的な使用方法（難易度変更なし）

```bash
python scripts/generate/generate_drum_from_input.py \
  --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \
  --input_midi input_drum.mid \
  --output_midi output_drum.mid
```

### 難易度を変更して生成

```bash
# 手のスピードを+2、足のスピードを+1にして生成
python scripts/generate/generate_drum_from_input.py \
  --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \
  --input_midi input_drum.mid \
  --output_midi output_drum_harder.mid \
  --difficulty_shift 0 0 2 1 0
```

### テンポとサンプリングパラメータを指定

```bash
python scripts/generate/generate_drum_from_input.py \
  --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \
  --input_midi input_drum.mid \
  --output_midi output_drum.mid \
  --bpm 140 \
  --temperature 1.0 \
  --nucleus_p 0.95 \
  --use_latent_sampling \
  --latent_sampling_var 0.2
```

## 処理フロー

1. **モデルロード**
   - 指定されたチェックポイントからモデルをロード
   - `config.yaml`からトークン設計手法を取得
   - 語彙ファイルをロード

2. **入力MIDI処理**
   - 入力MIDIファイルを読み込み
   - トークン列に変換
   - 小節位置を特定

3. **難易度計算**
   - 各小節について以下の5つの難易度スコアを計算:
     - **S_tech**: 特殊奏法の頻度
     - **S_indep**: 手足の独立性
     - **S_hand**: 手の最大連打スピード
     - **S_foot**: 足の最大連打スピード
     - **S_move**: 打点間の最大移動速度
   - `data/metadata/difficulty_bounds.pkl`の境界値を用いて8クラス（0-7）に離散化

4. **難易度シフト適用**
   - 指定されたシフト値を各小節の難易度レベルに適用
   - クラス範囲（0-7）にクリッピング

5. **生成**
   - 入力トークンから潜在表現を抽出
   - ターゲット難易度レベルを条件として生成
   - Nucleus samplingとtemperature制御を使用

6. **MIDI保存**
   - 生成されたトークン列をMIDI形式に変換
   - 指定されたパスに保存

## 難易度指標の説明

### S_tech（特殊奏法の頻度）
- ゴーストノート、リムショット、フラム、ロールなどの特殊奏法の割合
- 高いほど技術的に複雑

### S_indep（手足の独立性）
- 手と足が異なるタイミングで演奏される割合
- 高いほど手足の独立した動きが必要

### S_hand（手の連打スピード）
- 1秒あたりの最大手打撃数
- 高いほど速い手の動きが必要

### S_foot（足の連打スピード）
- 1秒あたりの最大足打撃数
- 高いほど速い足の動きが必要

### S_move（打点間移動速度）
- 連続する打点間の最大移動速度
- 高いほど大きな腕の移動が必要

## 注意事項

1. **トークン設計手法**: 現在、標準トークン化方式（`standard`）のみをサポート。CP形式（`cp_limb_v1`）は未対応。

2. **必要なファイル**:
   - 学習済みモデルチェックポイント（.ptファイル）
   - チェックポイントディレクトリ内の`config.yaml`
   - 語彙ファイル（`config.yaml`で指定されたパス）
   - 難易度境界値ファイル（`data/metadata/difficulty_bounds.pkl`）

3. **入力MIDI要件**:
   - ドラム譜であること（is_drum=Trueのトラックを含む）
   - 小節が明確に区切られていること

4. **GPU使用**: デフォルトでCUDAデバイスを使用（`config.yaml`の設定に従う）

## トラブルシューティング

### エラー: "Model checkpoint not found"
- `--model_path`で指定したファイルパスが正しいか確認してください

### エラー: "config.yaml not found"
- モデルチェックポイントが正しいディレクトリ構造にあるか確認
- 通常: `checkpoints_xxx/params/best_params.pt`

### エラー: "Vocabulary file not found"
- `config.yaml`内の`vocab_path`が正しいか確認
- 語彙ファイルが存在するか確認

### エラー: "Difficulty bounds file not found"
- `data/metadata/difficulty_bounds.pkl`が存在するか確認
- 必要に応じて`src/utils/compute_difficulty_bounds.py`を実行して生成

### エラー: "No bars found in input MIDI"
- 入力MIDIファイルにドラムトラックが含まれているか確認
- MIDIファイルが破損していないか確認

## 出力

スクリプトは以下の情報を出力します：

- 入力MIDIの小節数
- 各小節の元の難易度レベル
- シフト後のターゲット難易度レベル
- 潜在表現の形状
- 生成プロセスの進捗
- 生成時間
- 平均エントロピー

生成されたMIDIファイルは、指定した`--output_midi`パスに保存されます。
