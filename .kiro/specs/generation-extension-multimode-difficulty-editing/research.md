# Research and Design Notes

## Summary
- Feature: generation-extension-multimode-difficulty-editing
- Scope: 入力譜面を小節難易度で編集する生成拡張
- Key Findings:
  - 既存生成は mode ごとに経路分離されており、共通CLI契約が不足しやすい。
  - 長尺生成はトークン長基準よりも小節基準の窓分割が安定する。
  - 既知バグ（複数小節入力でも1小節生成）は target_bars の管理と終了条件で再発しやすい。
  - 語彙整合を生成前に検証しないと mode不一致時に後段で壊れた出力が発生する。

## Investigation Log

### Topic: 既存生成経路のモード分岐
- Sources:
  - scripts/generate/generate_drum_from_input.py
  - scripts/generate/generate_drum_cp.py
- Findings:
  - cp_limb_v1 は専用経路で multi-head 入出力を扱う。
  - standard/remi は別 tokenizer 契約を持つため、共通CLIで mode routing が必要。
- Implications:
  - mode解決を先に固定し、その後に adapter を注入する設計が妥当。

### Topic: max_bars とスライディング境界
- Sources:
  - generate_drum_cp.py の生成ループ
  - drum_dataloader.py の小節境界情報
- Findings:
  - トークン長ベース切り捨ては小節途中を跨ぎやすく、編集意図を損なう。
  - 小節単位で window を切ると難易度ラベルとの整合が維持しやすい。
- Implications:
  - WindowPlanner は max_bars をハード制約とし、bar境界のみで分割する。

### Topic: 語彙整合検証の必要性
- Sources:
  - vocab payload 契約（standard/remi と cp_limb_v1 の差）
  - 既存 checkpoint ロード処理
- Findings:
  - cp_limb_v1 は event/type/pos/limb の複合語彙契約で、単純な vocab_size 比較だけでは不十分。
  - special token index のずれは実行時例外や意味破壊を引き起こす。
- Implications:
  - mode別の VocabularyCompatibilityGuard を preflight 必須とする。

## Decisions

### Decision 1: 難易度指定解決を独立サービス化
- Chosen: DifficultySpecificationService
- Why:
  - 3方式を1箇所で検証・解決し、生成側の分岐爆発を防ぐ。

### Decision 2: max_bars で小節単位窓分割
- Chosen: WindowPlanner + WindowGenerationOrchestrator
- Why:
  - 長尺対応と小節ラベル整合を同時に満たす最小変更。

### Decision 3: 早期EOS抑止ガード導入
- Chosen: DecoderTerminationGuard
- Why:
  - 1小節生成バグの再発防止に直接効く。

### Decision 4: mode別語彙ガードを生成前に強制
- Chosen: VocabularyCompatibilityGuard
- Why:
  - 不一致を前段で止め、壊れた生成を回避する。

## Risks and Mitigation
- Risk: 窓ごとの接続で構造トークンが重複する。
  - Mitigation: WindowResultAssembler で窓境界の先頭/末尾構造トークンを正規化する。
- Risk: global_shift がラベル範囲を超える。
  - Mitigation: 難易度範囲を事前定義しクリップする。
- Risk: mode追加時に routing 分岐漏れ。
  - Mitigation: 許可モード集合を単一定義で管理しテストで監視する。
