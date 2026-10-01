# ステップ2：Decoderへの条件入力

ユーザーが選択した既存の`in-attn`を拡張する。
原譜の骨格776次元と難易度10次元を、それぞれLinear＋GELUで埋め込む。
対象小節のVAE潜在`z`、骨格埋め込み、難易度埋め込み、前後小節の文脈埋め込みを連結し、
既存`VAETransformerDecoder.seg_emb_proj`でモデル次元へ投影する。
各層の因果Self-Attentionの直前に加算する。骨格専用cross-attentionは追加しない。
比較用に既存の`pre-attn`も設定で選択可能。

Encoderは前・対象・後を別々の小節として処理し、各BAR位置から表現を取得する。
対象の入力はハイブリッドで`X_rule`、単体比較で`X_before`。
骨格・難易度条件は対象Encoderにも加算し、`q(z|X_input,S,d)`を構成する。
前後文脈の平均潜在と存在フラグから文脈条件を作る。文脈の欠測はゼロ化する。
学習では対象潜在をサンプリング、評価・生成では既定で平均を使用する。

Decoderは対象小節の全トークン列を因果的に生成する。
既存擬似ペアが1小節単位であるため、曲全体の複数小節同時生成は本実装の対象外。
前後小節は入力文脈として使い、教師譜面をEncoderや条件へ渡さない。
PADは語彙の実際のIDを使用し、Encoderのpaddingをmask、再構成損失のpaddingを除外する。

基準YAML・既存`MuseMorphose`は変更しない。新規学習モデルは別クラス・checkpoint形式。
`src/musemorphose.MuseMorphose`から旧クラスを引き続き利用できる。
新しい3方式比較のニューラル単体は同じVAE部品と連続難易度条件を使い、
骨格条件とBeamSearchを外す。既存学習済み重みとの比較と混同しない。

確認：`python -m pytest -q tests/test_beam_muse_model.py`。
骨格・難易度で出力が変化すること、条件層までの逆伝播、因果性、PAD不変性、
checkpoint復元を小規模モデルで確認する。
