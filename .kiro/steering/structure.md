# Project Structure

## Organization Philosophy

**レイヤ分離 + パイプライン分離** を基本にしています。コアロジックは `src/` に集約し、前処理/学習/生成の実行フローは `scripts/` と `tasks/` で組み立てる設計です。新しい実験は「設定追加 + scripts エントリ追加」で拡張するのが標準パターンです。

## Directory Patterns

### Core Domain Modules
**Location**: `/src/`  
**Purpose**: モデル、データローダー、トークナイザー、ユーティリティなど再利用可能な中核実装。  
**Example**: `src/models/musemorphose.py` に学習/生成の共通モデルを実装。

### Pipeline Entry Points
**Location**: `/scripts/{preprocess,train,generate}/`  
**Purpose**: CLI 引数を受け取り、`src/` の機能を組み合わせて処理を実行。  
**Example**: `scripts/train/train_drum.py` が YAML を読み、DataLoader・Optimizer・学習ループを起動。

### Configuration as Source of Truth
**Location**: `/config/`  
**Purpose**: データパス、モデル構成、学習ハイパーパラメータ、トークン化モードを定義。  
**Example**: `config/drum_config.yaml` の `data.tokenization_method` で処理系を切り替える。

### Operational Task Scripts
**Location**: `/tasks/`  
**Purpose**: よく使う実行手順の定型化（前処理・学習・生成）。  
**Example**: `tasks/train.sh` で `python -m scripts.train.train_drum` を実行。

### Validation and Smoke Tests
**Location**: `/tests/`  
**Purpose**: 依存確認、トークナイズ整合性、変換処理、学習関連の回帰確認。  
**Example**: `tests/test_drum_system.py` で import と基本動作を一括チェック。

## Naming Conventions

- **Files**: `snake_case.py` を基本とする。
- **Modules**: ドメインごとに小さく分割（`models`, `data`, `tokenizers`, `utils`）。
- **Functions**: 処理内容がわかる動詞中心命名（`prepare_*`, `train_*`, `generate_*`）。

## Import Organization

```python
# Preferred (package import)
from src.models.musemorphose import MuseMorphose
from src.data.drum_dataloader import DrumTransformerDataset

# Legacy style still exists in some scripts
# (avoid adding new code in this style)
# import sys
# sys.path.append('./model')
```

**Path Aliases**:
- 公式エイリアスは未定義。`python -m scripts...` 実行で package import を解決する運用。

## Code Organization Principles

- `src/` は再利用可能ロジックに限定し、I/O や CLI は `scripts/` 側で扱う。
- 新しい tokenization / conditioning は「設定キー追加 + データ層 + モデル層」の順で拡張する。
- 既存パターンに沿う変更（新設定・新スクリプト追加）を優先し、既存フローを壊す全面改修は避ける。

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
