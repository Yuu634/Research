# Requirements Document

## Introduction
本仕様は、MuseMorphose のドラム難易度調整学習を標準化し、standard/remi と cp_limb_v1 を同一の訓練運用で扱えるようにすることを目的とする。特に小節単位の難易度条件を潜在表現へ結合する訓練経路を明確化し、cp_limb_v1 では既存の model-extension-multimode-hierarchical-pe 構造を再利用して実験再現性と後方互換を維持する。

## Requirements

### Requirement 1: トークン化モード別学習フローの統一
**Objective:** As a モデル開発者, I want standard/remi/cp_limb_v1 を同じ学習運用で切り替えたい, so that モード比較実験を同一手順で再現できる

#### Acceptance Criteria
1. When 学習ジョブ開始時に tokenization_method が指定されるとき, the MuseMorphose Training Service shall 指定モードの学習経路のみを有効化する
2. If 未対応または不正な tokenization_method が指定されたとき, the MuseMorphose Training Service shall 利用可能モードを含むエラーを返して実行を停止する
3. While 単一ジョブが実行中である間, the MuseMorphose Training Service shall モードの入力表現を同一バッチ内で混在させない
4. The MuseMorphose Training Service shall standard/remi/cp_limb_v1 のいずれでも Encoder -> latent -> Decoder の学習骨格を維持する

### Requirement 2: 小節難易度ラベルの条件入力
**Objective:** As a 難易度制御研究者, I want 小節ごとの難易度ラベルを確実に学習へ渡したい, so that 小節単位で難易度制御可能な生成を実現できる

#### Acceptance Criteria
1. When 学習バッチが構築されるとき, the MuseMorphose Data Service shall 各小節に対応する難易度ラベル列をモデル入力として提供する
2. If 難易度ラベル数と小節数が一致しないとき, the MuseMorphose Data Service shall 不整合内容を明示したエラーを返して処理を停止する
3. While 難易度条件学習が有効な間, the MuseMorphose Training Service shall 小節境界に基づき難易度ラベルを一意に対応付ける
4. The MuseMorphose Data Service shall standard/remi/cp_limb_v1 の全モードで比較可能な小節粒度の難易度情報を供給する

### Requirement 3: standard remi における潜在結合学習
**Objective:** As a モデル設計者, I want Encoder 潜在と小節難易度ラベルを結合して学習したい, so that 条件付き再構成性能を向上できる

#### Acceptance Criteria
1. When tokenization_method が standard または remi のとき, the MuseMorphose Training Service shall Encoder 由来の小節潜在表現と対応難易度ラベルを結合して Decoder 条件を構成する
2. While standard または remi の自己回帰再構成を行う間, the MuseMorphose Training Service shall 小節対応を保持した条件情報を各小節へ適用する
3. If 難易度条件が無効化されているとき, the MuseMorphose Training Service shall 互換モードとして既存条件なし再構成挙動へフォールバックする
4. The MuseMorphose Training Service shall 条件結合を有効化しても既存の再構成学習処理を維持する

### Requirement 4: cp_limb_v1 学習構造の再利用
**Objective:** As a 実装担当者, I want cp_limb_v1 の既存拡張モデル構造をそのまま学習系で利用したい, so that 新旧実装差分を最小化できる

#### Acceptance Criteria
1. When tokenization_method が cp_limb_v1 のとき, the MuseMorphose Training Service shall model-extension-multimode-hierarchical-pe で定義された入力統合と出力予測構造を利用して学習する
2. While cp_limb_v1 学習が進行している間, the MuseMorphose Training Service shall standard/remi と同一の学習ループ手順で実行する
3. If cp_limb_v1 用設定が欠落しているとき, the MuseMorphose Training Service shall 欠落キーを示すエラーを返して開始を拒否する
4. The MuseMorphose Training Service shall cp_limb_v1 の条件付け経路で小節難易度ラベルを standard/remi と同じ意味論で扱う

### Requirement 5: 学習損失と最適化の整合
**Objective:** As a 品質保証担当者, I want 各モードで損失計算の整合を保ちたい, so that 学習結果を公平に比較できる

#### Acceptance Criteria
1. When standard または remi の学習ステップが実行されるとき, the MuseMorphose Training Service shall 再構成損失と潜在正則化項を計算して最適化を実行する
2. When cp_limb_v1 の学習ステップが実行されるとき, the MuseMorphose Training Service shall cp_limb_v1 の出力仕様に整合する損失計算を実行する
3. If 損失計算に必要な出力形状が不整合のとき, the MuseMorphose Training Service shall 形状情報を含むエラーで当該ステップを失敗として記録する
4. The MuseMorphose Training Service shall モード別の学習損失を同一ログ経路で追跡可能にする

### Requirement 6: 設定運用とスクリプト互換
**Objective:** As a 運用担当者, I want 既存の学習スクリプト運用を維持したまま拡張を有効化したい, so that 段階的な導入を安全に進められる

#### Acceptance Criteria
1. When 既存引数で学習スクリプトが起動されるとき, the MuseMorphose Pipeline Service shall 追加設定が未指定でも互換モードで実行できる
2. Where 難易度条件学習を有効化する機能が含まれる, the MuseMorphose Pipeline Service shall 設定ファイルで有効無効を切り替え可能にする
3. If 設定値の組み合わせが無効であるとき, the MuseMorphose Pipeline Service shall 修正指針を含む検証エラーを返す
4. The MuseMorphose Pipeline Service shall モード名と語彙設定の不一致を起動前に検証する

### Requirement 7: 検証可能性と回帰保護
**Objective:** As a 研究運用者, I want 拡張挙動と互換挙動の双方を検証したい, so that 導入後の品質を定量的に保証できる

#### Acceptance Criteria
1. When 要件検証を実施するとき, the MuseMorphose Test Service shall standard/remi/cp_limb_v1 の各モードで少なくとも1ステップ学習が完走することを確認できる
2. While 互換モード検証を実行している間, the MuseMorphose Test Service shall 追加条件無効時に既存学習フローが継続動作することを確認する
3. If 回帰検証で既存挙動との差分が閾値を超えたとき, the MuseMorphose Test Service shall 差分指標を報告して検証を失敗扱いにする
4. The MuseMorphose Test Service shall モード別に要件受け入れ結果を再現可能な形式で記録する
