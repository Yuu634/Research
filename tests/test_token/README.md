# ドラム専用トークン検証テスト

このディレクトリには、ドラム専用トークン設計が正しく機能するかを検証するテストコードが含まれています。

## 概要

以下の3つの主要なテストを実行できます:

1. **サンプルMIDI生成** - テスト用のドラムMIDIファイルを生成
2. **往復変換テスト** - MIDI → トークン(pkl) → MIDIの変換を行い、元のMIDIと一致するか検証
  - 対応方式: `standard` / `remi` / `cp_limb_v1`
3. **トークン内容検証** - pklファイル内のトークン列が元のMIDI譜面と正しく対応しているか検証

## ディレクトリ構造

```
test_token/
├── README.md                      # このファイル
├── sample_midi_generator.py       # テスト用MIDIファイル生成
├── test_round_trip.py             # 往復変換テスト
├── test_token_verification.py     # トークン内容検証
├── run_all_tests.py               # すべてのテストを実行
├── sample_midis/                  # 生成されたサンプルMIDI
└── output/                        # テスト結果の出力先
    ├── *_tokens.pkl               # トークン列データ
    └── *_reconstructed.mid        # 復元されたMIDIファイル
```

## 使い方

### 1. すべてのテストを一度に実行

最も簡単な方法は、`run_all_tests.py`を使用することです:

```bash
cd /mnt/kiso-qnap5/obara/MuseMorphose
python3 tests/test_token/run_all_tests.py
```

これにより以下が自動的に実行されます:
1. サンプルMIDIファイルの生成
2. 各MIDIファイルの往復変換テスト（`standard` / `remi` / `cp_limb_v1`）
3. トークン内容の検証

### 2. 個別にテストを実行

#### ステップ1: サンプルMIDIファイルを生成

```bash
python3 tests/test_token/sample_midi_generator.py --output_dir tests/test_token/sample_midis
```

生成されるファイル:
- `simple_beat.mid` - シンプルな8ビートパターン（4小節）
- `complex_beat.mid` - 複雑なパターン（ゴーストノート、アクセント、フラム、チョークを含む）

#### ステップ2: 往復変換テストを実行

```bash
python3 tests/test_token/test_round_trip.py --midi_dir tests/test_token/sample_midis --output_dir tests/test_token/output/remi --tokenization_method remi

python3 tests/test_token/test_round_trip.py --midi_dir tests/test_token/sample_midis --output_dir tests/test_token/output/standard --tokenization_method standard

python3 tests/test_token/test_round_trip.py --midi_dir tests/test_token/sample_midis --output_dir tests/test_token/output/cp --tokenization_method cp_limb_v1
```

または単一のファイルをテスト:

```bash
python3 tests/test_token/test_round_trip.py --single_file tests/test_token/sample_midis/simple_beat.mid --output_dir tests/test_token/output --tokenization_method remi
```

3方式を一括で実行:

```bash
python3 tests/test_token/test_round_trip.py --midi_dir tests/test_token/sample_midis --output_dir tests/test_token/output --all_methods
```

このテストは以下を検証します:
- MIDIファイルをトークン列に変換できるか
- トークン列をpklファイルとして保存/読み込みできるか
- トークン列からMIDIファイルに復元できるか
- 復元されたMIDIが元のMIDIと一致するか（ノート数、タイミング、ベロシティ）

#### ステップ3: トークン内容を検証

```bash
python3 tests/test_token/test_token_verification.py --pkl_path tests/test_token/output/simple_beat_tokens.pkl --midi_path tests/test_token/sample_midis/simple_beat.mid
```

このテストは以下を検証します:
- pklファイル内のトークン列が正しい形式か
- 各トークンが元のMIDI譜面の音符と正しく対応しているか
- トークンの統計情報（各ドラムの出現回数など）

## トークン仕様

### 構造トークン

- `<PAD>` - パディング
- `<EOS>` - シーケンス終了
- `<BAR>` - 小節区切り
- `<BEAT_1>` ~ `<BEAT_4>` - 拍（1-4拍目）
- `<POS_0>` ~ `<POS_23>` - 位置（1拍を24分割）

### ドラムトークン

形式: `{楽器}_{技法}_{ベロシティレベル}`

#### ベロシティレベル
- `Ghost` - ゴーストノート（ベロシティ < 40）
- `Normal` - 通常の打撃（40 ≤ ベロシティ < 100）
- `Accent` - アクセント（ベロシティ ≥ 100）

#### 楽器の例
- `KICK_HIT_Normal` - 通常のキック
- `SNARE_HIT_Accent` - アクセント付きスネア
- `HH_CLOSED_HIT_Ghost` - ゴーストノートのクローズドハイハット
- `CRASH_HIT_Accent` - アクセント付きクラッシュ
- `SNARE_FLAM_Normal` - スネアのフラム（装飾音）
- `CRASH_CHOKE` - クラッシュチョーク

## テスト結果の見方

### 往復変換テスト

成功の基準:
- ✓ 一致したノート数が元のノート数と同じ
- ✓ タイミング誤差が許容範囲内（20 ticks以内）
- ✓ 欠損ノートや余分なノートがない

典型的な出力:
```
ステップ6: 元のMIDIファイルと復元したMIDIファイルを比較
  元のノート数: 42
  復元後のノート数: 42

  比較結果:
    一致したノート: 42/42
    タイミング誤差: 0件
    ベロシティ誤差: 3件
      平均誤差: 5.00

  ✓ 往復変換成功: 元のMIDIファイルとほぼ一致
```

### トークン内容検証

成功の基準:
- ✓ 一致率が90%以上
- ✓ トークン数とMIDIイベント数がほぼ同じ

典型的な出力:
```
ステップ3: トークン列とMIDIイベントを照合
  ✓ 照合完了

  照合結果:
    MIDIイベント総数: 42
    トークンイベント総数: 42
    一致したイベント: 40
    一致率: 95.2%

  ✓ 検証成功: トークン列は元のMIDI譜面とよく一致しています
```

## トラブルシューティング

### 依存関係エラー

必要なパッケージがインストールされているか確認:

```bash
pip install miditoolkit numpy
```

### MIDIファイルが見つからない

まずサンプルMIDIファイルを生成してください:

```bash
python3 tests/test_token/sample_midi_generator.py
```

### トークン不一致

一部のドラム音（特殊な奏法）が正しくマッピングされていない可能性があります。
`drum_tokenizer.py`の`DRUM_NOTE_MAP`を確認してください。

## カスタムMIDIファイルでのテスト

自分のMIDIファイルでテストすることもできます:

```bash
# 往復変換テスト
python3 tests/test_token/test_round_trip.py --single_file /path/to/your/drum.mid --output_dir tests/test_token/output

# トークン内容検証
python3 tests/test_token/test_token_verification.py --pkl_path tests/test_token/output/drum_tokens.pkl --midi_path /path/to/your/drum.mid
```

**注意**: テストするMIDIファイルには以下の要件があります:
- ドラムトラック（is_drum=True）が含まれている
- General MIDI (GM) ドラムマップに準拠している
- 4/4拍子

## 期待される結果

正しく実装されている場合:
- すべての往復変換テストが成功（✓）
- トークン内容検証の一致率が90%以上
- 生成されたpklファイルが正しい形式

不一致が発生する可能性がある箇所:
- ベロシティの微妙な変化（閾値付近）
- 特殊な奏法（フラム、ロールなど）の解釈
- 極端に短い音価（チョーク）の検出

## 出力ファイル

### pklファイルの内容

```python
{
    'tokens': ['<BAR>', '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', ...],
    'indices': [2, 3, 7, 41, ...],  # トークンのインデックス
    'bar_positions': [0, 15, 30, ...],  # 各小節の開始位置
    'vocab': {0: '<PAD>', 1: '<EOS>', ...}  # 語彙辞書
}
```

### 復元されたMIDIファイル

往復変換によって生成されたMIDIファイルで、元のMIDIとほぼ同じ内容を持つべきです。
DAW（Digital Audio Workstation）で開いて視覚的に確認することもできます。

## ライセンス

元のMuseMorphoseプロジェクトのライセンスに従います。
