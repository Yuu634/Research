# Technology Stack

## Architecture

設定駆動のバッチ型パイプライン構成です。`config/*.yaml` を単一の真実源として、`scripts.preprocess` → `scripts.train` → `scripts.generate` の順で処理を進めます。モデル本体は `src/models`、データ変換は `src/data`/`src/tokenizers` に分離されています。

## Core Technologies

- **Language**: Python 3 系
- **Framework**: PyTorch ベースの Transformer VAE
- **Runtime**: ローカル Python 実行（`python -m ...`）

## Key Libraries

- **PyTorch**: 学習・推論・AMP（混合精度）
- **NumPy / SciPy**: 前処理と数値計算
- **PyYAML**: 設定ファイル管理
- **miditoolkit**: MIDI 入出力
- **tqdm**: 前処理進捗表示

## Development Standards

### Type Safety
厳密な静的型付けより、テンソル形状と系列長の整合性を実行時に保証する方針です。新規コードでは、入出力テンソルの shape 前提を関数境界で明確化します。

### Code Quality
- 実験パラメータは YAML に寄せ、コード内ハードコードを減らす。
- 共有ロジックは `src/`、実行エントリは `scripts/` に置く。
- 既存の legacy import（`sys.path` 追加）と package import（`from src...`）が混在するため、新規コードは package import を優先する。

### Testing
`tests/` にスモークテストと検証スクリプトを配置し、特にトークナイズ往復・データ整合性・CUDA/環境依存の確認を重視します。

## Development Environment

### Required Tools
- Python 実行環境
- `pip` による依存導入
- 任意で CUDA 対応 GPU（学習高速化）

### Common Commands
```bash
# Dependencies
pip install -r requirements.txt
pip install -r requirements_drum.txt

# Preprocess (example)
python -m scripts.preprocess.prepare_drum_dataset --help

# Train (example)
python -m scripts.train.train_drum --config config/drum_config.yaml

# Generate (example)
python -m scripts.generate.generate_drum_cp --help
```

## Key Technical Decisions

- **Config-first 運用**: 実験設定は YAML に集約し、再現性を担保。
- **Pickle 中間表現**: MIDI 生データを直接学習せず、`pkl` 化して反復学習を高速化。
- **複数トークン化モード**: `tokenization_method` で standard / `cp_limb_v1` を切り替え。
- **段階的近代化**: 旧実装を保持しつつ、`src` パッケージ化と `python -m` 実行へ移行。

---
_Document standards and patterns, not every dependency_
