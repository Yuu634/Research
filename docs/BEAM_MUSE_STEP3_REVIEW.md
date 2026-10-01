# ステップ3：相対／絶対の難易度条件

5属性の順序は既存v2の`s_tech, s_indep, s_hand, s_foot, s_move`。
各属性は8クラスを7で割った0〜1の数値を用いる。

| `difficulty_mode` | Decoder/Encoderへ渡す値 | 範囲 |
|---|---|---|
| `relative`（既定） | `d_target - D(X_input)` | −1〜1 |
| `absolute` | `d_target` | 0〜1 |

どちらも`D(X_input)`の5値を連結し、計10次元で埋め込む。
相対値は比率や絶対値ではなく符号付き差分であり、負値をクリップしない。
`X_input`はハイブリッドではBeamSearch後の中間譜面、単体では原譜。
難易度算出には原譜の前小節を使い、既存ペア・評価の文脈を維持する。

入力JSONの`target.mode=absolute/delta`はユーザーの目標指定方法であり、
この設定は正規化した絶対目標をモデルへどう表現するかという独立の選択。
目標が範囲外ならエラーにし、自動クリップしない。設定はcheckpointに保存し、
学習後に推論だけのモード変更は行わない。

確認：`tests/test_beam_muse_model.py`で負・ゼロ・正の差分、絶対値、範囲外を検証。
