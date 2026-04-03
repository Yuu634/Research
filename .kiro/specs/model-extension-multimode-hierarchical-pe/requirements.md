# Requirements Document

## Introduction
本仕様は、MuseMorphose のモデル構造を tokenization_method ごとに安全に拡張し、standard/remi と cp_limb_v1 を同一の学習・推論フレーム上で運用可能にすることを目的とする。特に cp_limb_v1 では 6 次元 Compound Word 入力、Bar 起点の小節潜在変数、難易度ドメイン条件付け、並列 6 ヘッド予測を明確に定義し、既存 checkpoint との後方互換を維持する。

## Requirements

### Requirement 1: モード別入力表現の統一フロー
**Objective:** As a モデル開発者, I want standard/remi/cp_limb_v1 を同一学習フローで処理したい, so that モード比較実験を同一コードパスで再現できる

#### Acceptance Criteria
1. When tokenization_method が standard または remi のとき, the MuseMorphose Model Service shall 既存単語列入力を用いた Encoder -> VAE latent -> Decoder フローで学習と推論を実行する
2. When tokenization_method が cp_limb_v1 のとき, the MuseMorphose Model Service shall 6 次元複合入力を用いた同一フローで学習と推論を実行する
3. While 単一ジョブが実行中である間, the MuseMorphose Model Service shall モード別入力表現を混在させない
4. The MuseMorphose Model Service shall 入力表現の差分を embedding 層と出力 head に局所化し、潜在変数経路の意味論を共通化する

### Requirement 2: cp_limb_v1 の Compound Embedding 統合
**Objective:** As a モデル設計者, I want cp_limb_v1 の 6 属性を独立埋め込みで統合したい, so that 四肢情報と時間情報を同時に表現できる

#### Acceptance Criteria
1. When cp_limb_v1 の各タイムステップが入力されるとき, the MuseMorphose Model Service shall [Type, Position, Hand1, Hand2, Right_Foot, Left_Foot] の各属性を独立 Embedding でベクトル化する
2. The MuseMorphose Model Service shall 6 属性埋め込みを結合後、線形射影で d_model 次元へ変換する
3. If 任意属性が PAD/NA のとき, the MuseMorphose Model Service shall 他属性と同一の統合処理経路で扱い、次元欠落を発生させない

### Requirement 3: 小節潜在アンカーの明確化
**Objective:** As a 生成制御担当者, I want 小節潜在の抽出アンカーを明示したい, so that 既存互換を保ちながら Bar 起点の制御精度を高められる

#### Acceptance Criteria
1. When Encoder 出力が計算されるとき, the MuseMorphose Model Service shall 各小節に対して 1 つの潜在変数 z_bar を導出する
2. Where Bar アンカー抽出が有効であり小節内に <Bar> 位置が存在するとき, the MuseMorphose Model Service shall <Bar> 位置の hidden state を z_bar 導出に使用する
3. If Bar アンカー抽出が無効または <Bar> 位置を一意に特定できないとき, the MuseMorphose Model Service shall 既存互換として小節先頭トークン位置の hidden state から z_bar を導出する
4. While 小節境界が連続する系列を扱う間, the MuseMorphose Model Service shall 各小節に対して対応する z_bar を一意に関連付ける

### Requirement 4: 小節条件付けベクトルの構築
**Objective:** As a 難易度条件生成の利用者, I want z_bar と 5 種難易度ドメインを結合したい, so that 小節ごとに制御可能な条件情報を decoder へ与えられる

#### Acceptance Criteria
1. When decoder 条件付けを構築するとき, the MuseMorphose Model Service shall z_bar と 5 つの難易度ドメイン embedding を結合して C_bar を生成する
2. The MuseMorphose Model Service shall 既存の dec_seg_emb 経路を維持し、条件付け経路の後方互換を損なわない
3. If 難易度ドメインが無効化された設定のとき, the MuseMorphose Model Service shall 既存互換の条件付け挙動へフォールバックする

### Requirement 5: Decoder In-Attention 条件注入
**Objective:** As a モデル設計者, I want 小節条件 C_bar を decoder へ直接注入したい, so that 自己回帰生成中に小節スタイルと難易度制約を維持できる

#### Acceptance Criteria
1. When Decoder の各層が masked self-attention を計算するとき, the MuseMorphose Model Service shall 該当小節の C_bar を同小節タイムステップへ注入する
2. While 自己回帰生成が進行している間, the MuseMorphose Model Service shall タイムステップが属する小節に応じた条件付けを維持する
3. The MuseMorphose Model Service shall 条件注入によって自己回帰マスク規則を破壊しない

### Requirement 6: cp_limb_v1 の並列 6 ヘッド出力
**Objective:** As a 研究者, I want cp_limb_v1 で 6 属性を同時予測したい, so that 属性独立性を維持して次トークン分布を推定できる

#### Acceptance Criteria
1. When cp_limb_v1 の decoder 出力が得られたとき, the MuseMorphose Model Service shall Type/Position/Hand1/Hand2/Right_Foot/Left_Foot の各属性に対応する独立 linear head を適用する
2. The MuseMorphose Model Service shall 各 head について次時刻トークン分布を算出可能な logits を返す
3. While cp_limb_v1 以外のモードが選択されている間, the MuseMorphose Model Service shall 既存単語列 head の挙動を維持する

### Requirement 7: 階層 Positional Encoding の導入とフォールバック
**Objective:** As a 実装担当者, I want bar/beat/pos を用いた階層 PE を optional で使いたい, so that 既存モデル互換を保ったまま時間構造表現を強化できる

#### Acceptance Criteria
1. When use_hierarchical_pe=true かつ階層インデックスが提供されるとき, the MuseMorphose Model Service shall PE_bar + PE_beat + PE_pos を Encoder/Decoder 入力へ加算する
2. If 階層インデックスが未提供または use_hierarchical_pe=false のとき, the MuseMorphose Model Service shall 既存の絶対位置 PE へ自動フォールバックする
3. The MuseMorphose Model Service shall cp_limb_v1 を含む全モードで同一の PE 選択ルールを適用する
4. The MuseMorphose Model Service shall 既存設定ファイルが無変更でも起動可能なデフォルト値を提供する

### Requirement 8: データ供給と生成時状態管理
**Objective:** As a パイプライン開発者, I want 学習時と生成時で階層インデックスを整合させたい, so that forward と generate の挙動を一致させられる

#### Acceptance Criteria
1. When standard/remi データローダがバッチを返すとき, the MuseMorphose Data Service shall bar_idx/beat_idx/pos_idx/is_struct を model 入力として返却可能にする
2. When cp_limb_v1 データローダがバッチを返すとき, the MuseMorphose Data Service shall 既存 CP 系列に加えて同等の階層インデックスを返却可能にする
3. When generate が自己回帰で進行するとき, the MuseMorphose Generation Service shall 生成トークンに応じて bar_idx/beat_idx/pos_idx を逐次更新する
4. If 階層情報を算出できない互換モードのとき, the MuseMorphose Generation Service shall 絶対位置 PE 経路で生成を継続する

### Requirement 9: 後方互換と検証可能性
**Objective:** As a 運用担当者, I want 既存 checkpoint と既存スクリプトを壊したくない, so that 拡張導入を段階的に実施できる

#### Acceptance Criteria
1. When use_hierarchical_pe=false のとき, the MuseMorphose Model Service shall 既存 loss 推移と大きな乖離なく学習を実行できる
2. The MuseMorphose Pipeline Service shall 既存 train/generate スクリプトの引数互換を維持する
3. Where 新規設定が導入される, the MuseMorphose Pipeline Service shall use_hierarchical_pe と語彙サイズ設定にデフォルト値を定義する
4. The MuseMorphose Test Service shall 単体・結合・回帰テストで拡張挙動と互換挙動の双方を検証可能にする
