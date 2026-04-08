# Requirements Document

## Introduction
本仕様は、入力ドラム譜面を小節単位の身体的難易度ラベルで編集し、指定難易度へ再生成する拡張を定義する。対象モードは standard、remi、cp_limb_v1 の3方式であり、長尺譜面に対しても config の max_bars に基づくウィンドウ生成で安定動作させる。あわせて、出力保存の堅牢化、語彙整合検証、複数小節入力で1小節しか生成されない既知バグの解消を受け入れ条件に含める。

## Requirements

### Requirement 1: 入出力契約の明確化
**Objective:** As a 譜面編集ユーザー, I want 入力MIDIと小節難易度ラベルを与えて難易度編集済みMIDIを得たい, so that 目標難易度に沿った譜面を再生成できる

#### Acceptance Criteria
1. When 生成コマンドに入力譜面パスと難易度指定が与えられるとき, the Generation Pipeline Service shall 編集後の譜面をMIDIとして出力する
2. If 入力譜面パスが存在しないとき, the Generation Pipeline Service shall 失敗理由を示して処理を停止する
3. While 小節難易度ラベルが適用される間, the Generation Pipeline Service shall 小節境界とラベル境界を一対一で維持する
4. The Generation Pipeline Service shall 入出力契約を standard/remi/cp_limb_v1 で共通化する

### Requirement 2: 難易度指定方式の3モード対応
**Objective:** As a 編曲担当者, I want 全小節指定・全体シフト・上限値設定の3方式で難易度を指定したい, so that 編集意図に応じた運用ができる

#### Acceptance Criteria
1. When 全小節指定モードが選択されたとき, the Difficulty Specification Service shall 小節数と同数の難易度配列をそのまま適用する
2. When 全体シフトモードが選択されたとき, the Difficulty Specification Service shall 入力譜面の難易度ラベルを全小節一律でシフトした難易度配列を適用する
3. When 上限値設定モードが選択されたとき, the Difficulty Specification Service shall 上限を超える小節ラベルを上限値へクリップする
4. If 指定方式と必要引数の組み合わせが不正なとき, the Difficulty Specification Service shall 修正可能なエラーメッセージを返して停止する
5. The Difficulty Specification Service shall 生成前に最終小節ラベル列を決定し、下流で再解釈を発生させない

### Requirement 3: max_bars ベースの長尺生成
**Objective:** As a 研究運用者, I want max_bars 単位で長尺譜面を分割生成したい, so that 入力小節数が多くても難易度編集を継続できる

#### Acceptance Criteria
1. When 入力小節数が max_bars を超えるとき, the Sliding Window Generation Service shall 指定した小節数のスライド幅・max_bars小節のウィンドウ幅でスライディングウィンドウを行い順次生成する
2. While ウィンドウ境界をまたいで生成する間, the Sliding Window Generation Service shall 小節順序を保持して出力へ連結する
3. If 最終ウィンドウが max_bars 未満のとき, the Sliding Window Generation Service shall 残小節数に合わせて生成して欠落を起こさない
4. The Sliding Window Generation Service shall ウィンドウ処理後の総生成小節数が入力小節数と一致することを保証する

### Requirement 4: 複数小節入力で1小節しか生成されない不具合の解消
**Objective:** As a モデル利用者, I want 複数小節入力時に全小節が再生成されることを保証したい, so that 実用上の編集結果を得られる

#### Acceptance Criteria
1. When 複数小節譜面を入力するとき, the Decoder Control Service shall 少なくとも入力小節数分の生成ステップを実行する
2. If 生成終了条件が1小節目で成立する異常系が発生するとき, the Decoder Control Service shall 早期終了を抑止し不足小節を補完生成する
3. While max_bars 分割生成中である間, the Decoder Control Service shall 各ウィンドウで target_bars を正しく設定する
4. The Decoder Control Service shall 生成結果の bar count を検証し、不足時は失敗として報告する

### Requirement 5: 語彙整合検証
**Objective:** As a 品質管理者, I want 学習済みモデル語彙と入力譜面語彙の整合を生成前に検証したい, so that 語彙不一致による破損生成を防げる

#### Acceptance Criteria
1. When 生成開始前にモデルとトークナイザがロードされるとき, the Vocabulary Compatibility Guard shall 語彙サイズおよび主要special tokenの一致を検証する
2. If 語彙サイズが不一致のとき, the Vocabulary Compatibility Guard shall 差分情報を含むエラーを返して処理を停止する
3. While tokenization_method が cp_limb_v1 のとき, the Vocabulary Compatibility Guard shall multi-head語彙（event/type/pos/limb）の整合を検証する
4. The Vocabulary Compatibility Guard shall 検証結果をログに記録して再現性を確保する

### Requirement 6: 出力保存先ディレクトリの自動作成
**Objective:** As a CLI利用者, I want 保存先ディレクトリが未作成でも出力できるようにしたい, so that 実行前準備の手間を減らせる

#### Acceptance Criteria
1. When 出力パスの親ディレクトリが存在しないとき, the Output Writer Service shall 必要ディレクトリを再帰作成する
2. If ディレクトリ作成に失敗したとき, the Output Writer Service shall 権限などの原因を含むエラーを返す
3. While 出力保存を行う間, the Output Writer Service shall 既存ファイル上書きポリシーを明示的に適用する
4. The Output Writer Service shall MIDI とデバッグテキストの双方を同一プレフィックスで保存する

### Requirement 7: tokenization_method 3方式対応
**Objective:** As a 開発者, I want standard/remi/cp_limb_v1 を単一生成フローで扱いたい, so that モードごとの分岐バグを減らせる

#### Acceptance Criteria
1. When tokenization_method が standard のとき, the Mode Routing Service shall standard tokenizer と生成経路を選択する
2. When tokenization_method が remi のとき, the Mode Routing Service shall remi tokenizer と生成経路を選択する
3. When tokenization_method が cp_limb_v1 のとき, the Mode Routing Service shall CP multi-head 生成経路を選択する
4. If 未対応モードが指定されたとき, the Mode Routing Service shall 利用可能モード一覧を含むエラーで fail-fast する
5. The Mode Routing Service shall 全モードで共通の難易度指定インターフェースを維持する
