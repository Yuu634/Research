# Research & Design Decisions

## Summary
- **Feature**: `training-extension-drum-difficulty-conditioning`
- **Discovery Scope**: Extension
- **Key Findings**:
  - 既存の `scripts/train/train_drum.py` は mode 検証、difficulty 連携、cp 分岐、階層PE入力をすでに受け取れる構造であり、拡張は境界ごとの契約明確化が主眼になる。
  - `src/data/drum_dataloader.py` は standard/remi/cp_limb_v1 で共通サンプルキーを返しつつ cp 専用キーを追加する optional 拡張方式を採用しており、後方互換を維持しやすい。
  - `src/models/musemorphose.py` は潜在条件結合、cp 6系統loss（structural除外）、Barアンカー、階層PEフォールバックを保持しているため、設計では学習系フローと失敗条件の規約化を重視する。

## Research Log

### 学習エントリポイントの拡張余地
- **Context**: Requirement 1, 4, 6 の実現方式を定義するため、実行境界を確認。
- **Sources Consulted**:
  - `scripts/train/train_drum.py`
  - `src/utils/tokenization_mode.py`
- **Findings**:
  - 起動時に `validate_tokenization_method` と `ensure_runtime_mode_compatibility` でモード整合をチェックしている。
  - モデル初期化時に `tokenization_method`、`use_difficulty`、`bar_anchor_mode`、`use_hierarchical_pe` などを一括注入する。
  - cp モード時は `compute_cp_loss`、それ以外は `compute_loss` を使用している。
- **Implications**:
  - 設計上の主コンポーネントは「起動検証」「データ整列」「条件結合」「損失選択」に分離する。

### データ供給スキーマと小節難易度整列
- **Context**: Requirement 2, 3, 4 を満たす入力契約を確定するため。
- **Sources Consulted**:
  - `src/data/drum_dataloader.py`
- **Findings**:
  - 返却辞書は `enc_input/dec_input/dec_target/bar_pos` に加え、階層PE用 `enc_*idx`, `dec_*idx` を optional で含む。
  - `use_difficulty=True` 時は bar-level difficulty を token-level列 `s_*_cls_seq` へ展開する。
  - cp モードでは `cp_*_input/target` を追加しつつ `dec_input/dec_target` 互換キーを維持している。
- **Implications**:
  - 設計では Data Service の契約を「共通最小キー + モード拡張キー」で定義し、不整合時エラー規約を明示する。

### モデル条件結合と損失分岐
- **Context**: Requirement 3, 4, 5 のアーキテクチャ実現性を検証するため。
- **Sources Consulted**:
  - `src/models/musemorphose.py`
- **Findings**:
  - `_build_dec_seg_emb` が bar latent を時系列へブロードキャストする。
  - `_build_conditional_seg_emb` が difficulty 5属性を decoder 条件へ連結する。
  - cp モードでは 6系統 logits (`event_type`, `cp_pos`, `cp_hand1`, `cp_hand2`, `cp_right_foot`, `cp_left_foot`) を出力し、`compute_cp_loss` で mask 付き損失を計算する（structural除外）。
- **Implications**:
  - 設計では cp と standard/remi の差分を「入出力ヘッドと損失関数」に局所化する。

### 回帰保護とテスト戦略
- **Context**: Requirement 7 の検証観点を固めるため。
- **Sources Consulted**:
  - `tests/test_musemorphose_hierarchical_and_compat.py`
  - `tests/test_tokenization_mode_validation.py`
- **Findings**:
  - Barアンカー分岐、階層PEフォールバック、cp backward compatibility の単体検証が存在する。
  - mode 検証の失敗ケースがテスト化されている。
- **Implications**:
  - 設計のテスト戦略では 1-step 学習、互換モード、shape不整合エラーの3系統を必須にする。

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| 既存訓練スクリプト集中型 | `train_drum.py` で分岐集約し `src/` を呼び出す | 既存運用と互換、導入コストが低い | script が肥大化しやすい | 現行実装と整合
| 学習オーケストレータ分離型 | 学習制御を service 化して script を薄くする | 境界が明確、テストしやすい | リファクタ工数が増える | 将来候補
| モード別完全分離型 | standard/remi/cp を別学習経路に分離 | 実装が直感的 | 比較実験の再現性低下、重複増加 | 本仕様では不採用

## Design Decisions

### Decision: 学習骨格は単一フロー維持
- **Context**: モード比較再現性と運用手順統一が必須。
- **Alternatives Considered**:
  1. モード別完全分離
  2. 単一フローで入出力差分のみ分岐
- **Selected Approach**: 単一フローで `tokenization_method` に応じて入出力と損失のみ切り替える。
- **Rationale**: 要件 1 と 4 の両立に最も適合し、既存運用との互換が高い。
- **Trade-offs**: 分岐条件が増えるため契約管理が重要。
- **Follow-up**: 失敗時メッセージと shape 検証を標準化する。

### Decision: 小節難易度は bar-level から token-level へ展開
- **Context**: Decoder 条件入力は token 粒度が必要。
- **Alternatives Considered**:
  1. bar-level のまま参照時に動的解決
  2. DataLoader で事前展開
- **Selected Approach**: DataLoader で `s_*_cls_seq` を生成しモデルへ直接供給する。
- **Rationale**: モデル側の責務を削減し、再利用性と検証性を高められる。
- **Trade-offs**: バッチ生成時の整合チェックが必須。
- **Follow-up**: bar 数不一致時の fail-fast エラーを統一する。

### Decision: cp 損失は既存 multi-head mask 方式を継続
- **Context**: cp_limb_v1 の意味論維持と既存モデル再利用が要求される。
- **Alternatives Considered**:
  1. 単一ヘッド統合損失へ再設計
  2. 既存 `compute_cp_loss` を中心に重み調整可能化
- **Selected Approach**: 既存 6系統 loss（structural除外）を維持し `cp_loss_weights` で重み制御する。
- **Rationale**: 既存挙動との比較と回帰検証が容易。
- **Trade-offs**: ヘッド間重み設計の運用知識が必要。
- **Follow-up**: 実験ログにヘッド別損失を必須出力とする。

## Risks & Mitigations
- 難易度ラベルと小節境界の不整合で条件誤注入が起きる — Data Service で bar 数一致検証を実施し fail-fast で停止。
- cp 系列の shape 不一致で loss 計算が破綻する — 事前 shape チェックとヘッド別失敗ログを追加。
- 互換モードで拡張設定が混入する — 起動時の mode/語彙/設定検証を必須化し、無効組み合わせを拒否。

## References
- `scripts/train/train_drum.py` — 学習オーケストレーションと mode 分岐
- `src/data/drum_dataloader.py` — モード別サンプルスキーマと difficulty 展開
- `src/models/musemorphose.py` — 条件結合、cp ヘッド、損失計算
- `src/utils/tokenization_mode.py` — artifact 互換検証
- `tests/test_musemorphose_hierarchical_and_compat.py` — 互換/回帰の既存テスト基盤
- `tests/test_tokenization_mode_validation.py` — mode 検証の単体テスト
