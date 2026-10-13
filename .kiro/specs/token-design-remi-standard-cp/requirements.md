# Requirements Document

## Introduction
本仕様は、MuseMorphose のドラム向けトークン設計を拡張し、REMI・ドラム standard（既存維持）・四肢ベース CP（固定長6次元）を同一パイプライン上で整合的に扱える状態を定義する。目的は、既存資産との互換性を維持しながら、表現力と実験再現性を向上させることである。

## Requirements

### Requirement 1: トークン化モードの共存と選択
**Objective:** As a 音楽生成研究者, I want トークン化方式を実験単位で明示選択できること, so that 比較実験を再現可能に実行できる

#### Acceptance Criteria
1. When ユーザーが前処理・学習・生成の実行時に tokenization_method を指定したとき, the MuseMorphose Tokenization Service shall 指定モードに対応するトークン化仕様のみを適用する
2. If 未対応または不正な tokenization_method が指定されたとき, the MuseMorphose Tokenization Service shall 実行を停止し、利用可能なモード名を含むエラーを提示する
3. While 単一ジョブが実行中である間, the MuseMorphose Tokenization Service shall 複数モードの語彙形式を同一データ系列内に混在させない
4. Where 既存データセットを再利用する機能が含まれる, the MuseMorphose Tokenization Service shall データセット作成時モードと実行時モードの一致を検証する
5. The MuseMorphose Tokenization Service shall 少なくとも standard・remi・cp_limb_v1 を識別可能なモードとして扱う

### Requirement 2: ドラム向け REMI 表現の導入
**Objective:** As a データエンジニア, I want ドラム演奏データに対して REMI を適用できること, so that ドラム用トークン設計の比較実験を統一的に実行できる

#### Acceptance Criteria
1. When ドラム MIDI を remi モードで処理したとき, the MuseMorphose Tokenization Service shall REMI 互換の時間・打点・構造イベント体系でトークン列を生成する
2. If 入力ノートのベロシティ値が与えられたとき, the MuseMorphose Tokenization Service shall standard モードと同一の3段階クラス化規則でベロシティを正規化して表現する
3. While remi モードが有効である間, the MuseMorphose Tokenization Service shall ベロシティ表現において細粒度の連続値トークン化を必須としない
4. Where 入力に非ドラム楽器トラックが含まれる, the MuseMorphose Tokenization Service shall 非ドラム楽器を処理対象外として一貫した挙動で扱う
5. The MuseMorphose Tokenization Service shall remi モードの適用範囲をドラム演奏データに限定する

### Requirement 3: ドラム standard 設計の後方互換維持
**Objective:** As a 既存運用担当者, I want standard モードの挙動が維持されること, so that 既存の学習済み資産とテスト資産を継続利用できる

#### Acceptance Criteria
1. When standard モードで既存データを再トークン化したとき, the MuseMorphose Tokenization Service shall 既存仕様と互換なトークン意味論を維持する
2. If 既存 standard データセットまたは語彙ファイルが入力されたとき, the MuseMorphose Tokenization Service shall 追加の移行作業なしで読込可能とする
3. While standard モードが有効である間, the MuseMorphose Tokenization Service shall remi および cp_limb_v1 専用必須フィールドを要求しない
4. Where 既存の standard テストスイートが含まれる, the MuseMorphose Tokenization Service shall 既存テスト期待値との整合性を維持する
5. The MuseMorphose Tokenization Service shall standard モードを互換維持対象として明示する

### Requirement 4: 四肢ベース CP の固定長6次元表現
**Objective:** As a モデル設計者, I want 四肢CPを固定長6次元で扱えること, so that 時間構造と手足コンビネーションを同時学習できる

#### Acceptance Criteria
1. When cp_limb_v1 モードで任意のタイムステップを符号化するとき, the MuseMorphose Tokenization Service shall 各ステップを [Type, Position, Hand1, Hand2, Right_Foot, Left_Foot] の固定長6次元として表現する
2. If Type が <Bar> または <Beat_n> であるとき, the MuseMorphose Tokenization Service shall Position と四肢次元を <NA> として扱う
3. While Type が <Note> である間, the MuseMorphose Tokenization Service shall Position を standard モードと同一の拍内分割・インデックス規則で解釈し、四肢次元を発音有無を含む肢別トークンとして解釈する
4. Where 小節境界情報を利用する機能が含まれる, the MuseMorphose Tokenization Service shall <Bar> イベントを小節アンカーとして提供する
5. The MuseMorphose Tokenization Service shall cp_limb_v1 の全イベントで次元数を常に6に維持する
6. When cp_limb_v1 モードで <Note> イベントのベロシティを符号化するとき, the MuseMorphose Tokenization Service shall standard モードと同一の3段階クラス化規則を適用する
7. When cp_limb_v1 モードで Position 値を符号化するとき, the MuseMorphose Tokenization Service shall standard モードと同一の Position 設計手法を適用する

### Requirement 5: 四肢意味論と演奏規則の整合
**Objective:** As a ドラム演奏モデリング担当者, I want 四肢トークンの意味が一貫していること, so that 生成結果を演奏解釈可能に保てる

#### Acceptance Criteria
1. When 手で演奏される打点を符号化するとき, the MuseMorphose Tokenization Service shall Hand1/Hand2 次元に対応トークンを割り当てる
2. When ハイハットの状態遷移が HH_OPEN → HH_CLOSED へ遷移するとき, the MuseMorphose Tokenization Service shall Left_Foot に "HH_PEDAL_HIT_Down" トークンを割り当てる
3. When ハイハットの状態遷移が HH_CLOSED → HH_OPEN へ遷移するとき, the MuseMorphose Tokenization Service shall Left_Foot に "HH_PEDAL_HIT_Up" トークンを割り当てる
4. While ハイハット状態に遷移がない間, the MuseMorphose Tokenization Service shall Left_Foot を "<NONE>" として表現する
5. Where 両足バスドラムの推奨規則が適用される, the MuseMorphose Tokenization Service shall 右足から左足への順序規則を保持して表現する
6. The MuseMorphose Tokenization Service shall Hand1・Hand2・Right_Foot・Left_Foot の各次元を意味的に独立したスロットとして定義する

### Requirement 6: 検証可能性と成果物整合性
**Objective:** As a 品質保証担当者, I want 各モードの出力仕様を検証可能にしたい, so that 拡張後の品質を客観的に確認できる

#### Acceptance Criteria
1. When データセット成果物が生成されたとき, the MuseMorphose Tokenization Service shall トークン化モードと語彙解決に必要な識別情報を取得可能な状態にする
2. If 学習または生成時に語彙・モード不一致が検出されたとき, the MuseMorphose Tokenization Service shall 原因が識別できるエラー情報を返して処理を停止する
3. While 受け入れ検証を実施している間, the MuseMorphose Tokenization Service shall モード別に同一入力に対するスキーマ検証結果を再現可能に出力する
4. Where remi または cp_limb_v1 が選択されていない, the MuseMorphose Tokenization Service shall 非選択モードに起因する追加要求を standard フローへ波及させない
5. The MuseMorphose Tokenization Service shall モードごとの期待スキーマを満たすか判定できる検証基準を提供する

### Requirement 7: ドラム楽器語彙の拡張と統一
**Objective:** As a 難易度調整モデル開発者, I want ドラム楽器の種類を拡張できること, so that より豊かなドラム演奏パターンを難易度指標として学習できる

#### Acceptance Criteria
1. When cp_limb_v1 モードでドラム楽器を符号化するとき, the MuseMorphose Tokenization Service shall 基本ドラムキット（KICK, SNARE, TOM, CRASH, RIDE）に加えてタンバリン・コウベル・チャイナシンバルなどの副楽器をサポートする
2. If タンバリン（TAMBOURINE, Pitch 54）が入力されたとき, the MuseMorphose Tokenization Service shall 手（Hand1 または Hand2）で演奏されるスロットに対応トークンを割り当てる
3. While タンバリンが演奏されている間, the MuseMorphose Tokenization Service shall 通常ドラム楽器と同一のベロシティ3段階クラス化規則（Ghost <15, Normal <100, Accent ≥100）を適用する
4. Where 楽器ごとの演奏部位（手vs足）が物理的に決定している場合, the MuseMorphose Tokenization Service shall その制約を意味論に反映する（例：タンバリン・コウベルは手のみ）
5. When cp_limb_v1 モードで新規楽器を追加するとき, the MuseMorphose Tokenization Service shall 難易度指標計算（演奏の複雑度・技術難度）への影響を検証基準に含める
6. The MuseMorphose Tokenization Service shall 楽器拡張に伴う語彙更新を DRUM_NOTE_MAP に反映し、既存テストスイートとの互換性を維持する
