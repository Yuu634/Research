# リサーチ & 設計判断ログ

## サマリー
- Feature: token-design-remi-standard-cp
- Discovery Scope: Extension
- Key Findings:
  - 既存コードは `tokenization_method` を前処理・学習・生成で共通利用しており、拡張はこの単一制御面を維持するのが最小リスク。
  - Position は standard で 1拍24分割の規則が確立済みで、CP と REMI も同じ分割規則へ統一する。
  - Velocity は standard を基準とし、**Ghost (<15)、Normal (<100)、Accent (>=100)** の3段階へ統一する。CP と REMI も同一規則を採用する。

## リサーチログ

### トピック: モード分岐の統合ポイント
- Context: 仕様変更を最小改修で適用できる拡張境界を特定する。
- Sources Consulted:
  - scripts/preprocess/prepare_drum_dataset.py
  - scripts/train/train_drum.py
  - scripts/generate/generate_drum_cp.py
  - src/data/drum_dataloader.py
- Findings:
  - `tokenization_method` が主要エントリポイントの共通スイッチになっている。
  - CP 生成パスは専用経路として分離され、`cp_limb_v1` 以外を拒否する。
  - DrumTransformerDataset は語彙 payload 形状と tokenization metadata で schema を切り替える。
- Implications:
  - 設計は単一制御面を維持し、モード検証仕様を共通化する必要がある。

### トピック: Position 方針の整合性
- Context: CP と REMI の Position 設計を standard と同一手法に統一する。
- Sources Consulted:
  - src/tokenizers/drum_tokenizer.py
  - src/tokenizers/drum_cp_tokenizer.py
  - tests/test_token/test_round_trip.py
- Findings:
  - standard は `POSITIONS_PER_BEAT = 24` を使い、整数量子化で Position を算出する。
  - CP も同一定数と同等の量子化式を利用できる。
  - REMI も standard と同一の Position 設計（1拍24分割）へ統一する方針とする。
  - CP は非ノートイベント向けに `pos_pad_value = 24` を予約している。
- Implications:
  - standard / CP / REMI 間で差分のない共有 Position 契約を定義できる。

### トピック: Velocity 方針の整合性
- Context: standard、CP、REMI を同一の3段階 Velocity 規則に統一する。
- Sources Consulted:
  - src/tokenizers/drum_tokenizer.py
  - src/tokenizers/drum_cp_tokenizer.py
  - scripts/preprocess/prepare_drum_dataset.py
- Findings:
  - 新しい canonical 閾値は **Ghost (<15)、Normal (<100)、Accent (>=100)**。
  - CP は standard の速度分類ロジックを再利用する設計にできる。
  - REMI は現状の細粒度速度表現から 3 段階表現へ移行が必要。
- Implications:
  - REMI には Velocity 方針変更と成果物互換の取り扱い設計が必要。

### トピック: CP ハイハット Left_Foot 状態遷移型設計
- Context: 難易度調整モデルにおいて、ペダル操作が学習信号として機能する仕様を検討する。
- Sources Consulted:
  - src/tokenizers/drum_cp_tokenizer.py (lines 291-350: hh_left_foot_pressed 状態機械)
  - tests/test_cp_tokenizer_schema.py (lines 86-110: HH_OPEN/CLOSED 推論テスト)
- Findings:
  - 毎フレーム記述方式では冗長性が高くモデル学習効率が低い。
  - 状態遷移型（DOWN/UP）では難易度指標計算が直接的になる。
  - SNARE/KICK などのアクション駆動設計と意味論を統一できる。
- Implications:
  - Left_Foot は HH_OPEN/HH_CLOSED 遷移時のみ DOWN/UP トークンを記述し、それ以外は `<NONE>` となる。
  - デコード側で状態機械を実装し、LEFT_FOOT_PRESSED フラグを HH_PEDAL_HIT_Down/Up で更新する。

### トピック: ドラム楽器語彙拡張（タンバリン）
- Context: CP トークン設計において基本ドラムキット以外の楽器対応が必要。
- Sources Consulted:
  - src/tokenizers/drum_tokenizer.py (DRUM_NOTE_MAP)
  - tests/test_token/test_round_trip.py (PITCH_TOLERANCE_MAP)
- Findings:
  - タンバリン（Pitch 54）は既に DRUM_NOTE_MAP で認識されている。
  - 手で演奏される楽器は Hand1/Hand2 スロットで統一可能。
  - ベロシティ3段階規則をタンバリンにも適用可能（Ghost/Normal/Accent）。
- Implications:
  - タンバリンを Hand スロット対応楽器として明示化する。
  - 難易度計算時にタンバリン出現頻度を複雑度指標に含める。
  - 既存テストで PITCH_TOLERANCE_MAP の整合性を検証する。

## アーキテクチャパターン評価

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| A. 既存分岐維持 + 共有ポリシー追加 | 既存 tokenization_method 分岐を残し、Velocity/Position を共有契約化 | 移行リスクが低く後方互換を保ちやすい | 共通検証が弱いと将来的なドリフトが再発 | 現要件と最も整合 |
| B. プラグインレジストリ化 | tokenizer の動的登録へ再設計 | 長期拡張性が高い | リファクタ工数と導入リスクが高い | 将来検討 |
| C. モード別パイプライン完全分離 | モードごとに処理系を分離 | 分離度が高い | 重複実装と保守コスト増 | 今回は過剰 |

## 設計判断

### Decision: tokenization_method を単一制御面として維持
- Context: 既存運用は1つのモードキーを前提に構築されている。
- Alternatives Considered:
  1. レイヤごとに別モードキーを導入する。
  2. 既存キーを維持し、検証を統一する。
- Selected Approach: tokenization_method を維持し、検証仕様を共通化する。
- Rationale: 既存設定との互換を保ち、運用手順の破壊を避けるため。
- Trade-offs: 柔軟性は限定されるが、再現性と安全性が高い。
- Follow-up: preprocess/train/generate でモード検証契約を共通化する。

### Decision: Velocity は3段階規則を全モードで統一
- Context: standard、CP、REMI の速度意味論を一致させる必要がある。
- Alternatives Considered:
  1. REMI の細粒度速度表現を維持する。
  2. 全モードを standard 3段階規則へ統一する。
- Selected Approach: **Ghost (<15)、Normal (<100)、Accent (>=100)** を standard/CP/REMI の共通規則とする。
- Rationale: モード横断の比較可能性と解釈一貫性を確保するため。
- Trade-offs: REMI は速度解像度を犠牲にする。
- Follow-up: REMI の成果物互換と移行ポリシーを設計書で明示する。

### Decision: CP と REMI の Position は standard 量子化規則へ固定
- Context: CP と REMI の Position 設計を standard と同一にする要件がある。
- Alternatives Considered:
  1. CP/REMI それぞれで独自の Position 粒度を許容する。
  2. standard の量子化規則と index 範囲に固定する。
- Selected Approach: 1拍24分割と同一量子化式を CP と REMI の両方に適用する。
- Rationale: モード間で時間軸の解釈差を発生させないため。
- Trade-offs: CP/REMI 独自の Position 最適化余地は減る。
- Follow-up: Position 範囲/PAD の schema 検証を必須化し、REMI 側にも同等検証を適用する。

### Decision: CP ハイハット Left_Foot は状態遷移時のみトークン化
- Context: 難易度調整モデルにおいて、ペダル操作が頻度・精度として学習される必要がある。
- Alternatives Considered:
  1. HH_CLOSED 状態継続中は毎フレーム `HH_PEDAL_HIT_Normal` を記述する（冗長型）。
  2. HH_OPEN/HH_CLOSED の状態遷移時のみ DOWN/UP トークンを割り当てる（遷移型）。
- Selected Approach: **状態遷移型**を採用。HH_OPEN → HH_CLOSED 時に `HH_PEDAL_HIT_Down`、HH_CLOSED → HH_OPEN 時に `HH_PEDAL_HIT_Up`、状態変化なし時は `<NONE>` を割り当てる。
- Rationale: 
  1. セマンティクス正確性：\"HIT\" はアクション（踏み込み/放す動作）を意味するべき。
  2. モデル学習効率：トークン量を削減し、難易度指標に直結するペダルアクション検出が容易。
  3. スキーマ一貫性：SNARE/KICK と同じく\"アクション駆動\"の設計になる。
- Trade-offs: Left_Foot 状態を推断するため、デコード側で簡易な状態機械の実装が必要。
- Follow-up: decoder（cp_data_to_midi）で HH_PEDAL_HIT_Down/Up イベント時に LEFT_FOOT_PRESSED 状態を更新する仕様を定義。

### Decision: CP ドラム楽器語彙にタンバリンを追加
- Context: 難易度調整モデルの表現力を向上させるため、基本ドラムキット以外の副楽器への対応が必要。
- Alternatives Considered:
  1. 基本ドラムキット（KICK, SNARE, TOM, CRASH, RIDE）のみに限定する。
  2. タンバリン・コウベル・チャイナシンバルなどの副楽器を追加対応する。
- Selected Approach: タンバリン（TAMBOURINE, Pitch 54）および副楽器を Hand1/Hand2 スロットで符号化可能にする。
- Rationale:
  1. 実演奏の多様性：ドラムパターンにおけるタンバリン活用が一般的。
  2. 難易度指標拡張：楽器種別の複雑性を難易度計算に反映可能。
  3. 既存スキーマ互換：Hand1/Hand2 スロットは既に多楽器対応設計になっている。
- Trade-offs: 語彙拡張に伴い DRUM_NOTE_MAP の更新と既存テスト互換性検証が必要。
- Follow-up: タンバリンをベロシティ3段階規則の適用対象に明示し、物理的制約（手のみ演奏可能）をスキーマに反映する。

### Decision: 後方互換は standard 優先で維持
- Context: standard は既にテスト済み資産と運用フローが存在する。
- Alternatives Considered:
  1. 全成果物を破壊的に再生成する。
  2. standard/cp を安定維持し、REMI のみ互換層を定義する。
- Selected Approach: standard/cp は互換維持、REMI は方針更新を明示して段階移行する。
- Rationale: 既存運用停止リスクを抑えつつ要件適合を実現するため。
- Trade-offs: 一時的に REMI 読み込み互換処理が必要。
- Follow-up: タスクフェーズで廃止期限と移行完了条件を定義する。

## リスクと緩和策
- Risk: 既存 REMI 成果物が新 Velocity 規則と意味不整合になる。
  - Mitigation: mode/version 識別子を保持し、読み込み時に互換判定を実施する。
- Risk: Position 規則が tokenizer と dataloader 間でドリフトする。
  - Mitigation: 共通検証フックと CI テストを追加し、範囲/PAD を固定検証する。
- Risk: REMI の既存 Position 表現との互換が揺らぐ。
  - Mitigation: REMI 成果物に mode/version 識別を付与し、移行期間は変換・検証ルールを明示する。
- Risk: エントリスクリプトごとにエラー表現が不一致になる。
  - Mitigation: モード検証エラーの共通フォーマットを定義し再利用する。

## 参照
- src/tokenizers/drum_tokenizer.py
- src/tokenizers/drum_cp_tokenizer.py
- scripts/preprocess/prepare_drum_dataset.py
- src/data/drum_dataloader.py
- scripts/train/train_drum.py
- scripts/generate/generate_drum_cp.py
- tests/test_token/run_all_tests.py
