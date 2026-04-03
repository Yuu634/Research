# Research Notes

## Inputs Considered
- .kiro/draft_model.md のモデル構造要件と実装フェーズ案。
- 既存 spec で採用されている Requirements / Design / Tasks の文書構造。
- 既存 MuseMorphose パイプラインの tokenization_method 分岐と後方互換優先方針。

## Key Decisions
1. 拡張方針は「モード共通フロー + 入出力差分の局所化」を採用する。
2. 階層 PE は optional 機能として導入し、デフォルト無効で互換性を優先する。
3. cp_limb_v1 のモデル差分は 6 属性入力統合と 6 並列 head のみを明示し、既存 latent 経路は維持する。
4. 実装順序は model コア（PE selector/embedding）を先行し、tokenizer/dataloader/generate の受け渡しを後続にする。

## Open Questions
1. generate で <BEAT_n> を明示トークンとして持たないモードの beat 推定規則をどこまで厳密化するか。
2. cp_limb_v1 の 6 head loss 重みを均一にするか、Type を主タスクとして重み付けするか。
3. 階層 PE の is_struct 補正を additive bias とするか、専用埋め込みとして独立加算するか。

## Suggested Validation Metrics
- 1-step train 完走率（3 モード）。
- use_hierarchical_pe=false 時の baseline loss 乖離率。
- cp_limb_v1 での head 別 token accuracy（Type/Position/Hand1/Hand2/Right_Foot/Left_Foot）。
