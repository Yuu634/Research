# モデル構造
現状のレポジトリ内のMuseMorphoseモデルアーキテクチャを以下の要件を満たすように修正

- トークン設計がREMI or standardの場合、MuseMorphoseモデルと同様の学習フローで学習・推論が行えるようにモデル設計

- トークン設計が四肢ベースのCPの場合、以下のモデル構造で実装する
1. 入力層（Compound Word の埋め込みと統合）
各タイムステップの6次元複合単語トークン [Type, Position, Hand1, Hand2, Right_Foot, Left_Foot] を受け取ります。
• 各トークンをそれぞれ対応する独立したEmbedding層でベクトル化します（$E_{Type}$, $E_{Position}$ など）。
• これら6つのベクトルを結合（Concatenate）し、線形層を通じてTransformerの隠れ層次元（$d_{model}$）に射影します。$$X = W \cdot [E_{Type} \oplus E_{Position} \oplus E_{Hand1} \oplus E_{Hand2} \oplus E_{Right\_Foot} \oplus E_{Left\_Foot}] + b$$
2. エンコーダ（Encoder）セクション【小節単位のスタイル学習】
統合された楽曲全体の系列データ（$X$）を Encoder に入力します。
• <Bar> トークンの活用: 楽曲全体を単一のベクトルに圧縮するのではなく、系列内で Type 属性が <Bar> を示しているタイムステップの隠れ状態（Hidden State）を利用します。
• 潜在変数の抽出: 各小節の <Bar> トークン位置の出力ベクトルから、その小節固有のドラムパターンや微細なニュアンスを抽出し、小節単位の潜在変数 $z_{bar}$ に圧縮・マッピングします。
3. 条件付けベクトルの構築【ドメイン情報の統合】
Decoderの生成を制御するための、小節ごとのきめ細やかな条件付け情報を構築します。
• Encoderから得られた該当小節の潜在変数 $z_{bar}$ と、5つの身体難易度ドメイン情報の埋め込みベクトルを結合します。
• これにより、小節単位の最終的な条件付けベクトル $C_{bar}$ を生成します。$$C_{bar} = z_{bar} \oplus E_{domain\_1} \oplus E_{domain\_2} \oplus E_{domain\_3} \oplus E_{domain\_4} \oplus E_{domain\_5}$$
4. デコーダ（Decoder）セクション【In-Attentionによる自己回帰生成】
• In-Attention機構の適用: Decoderの各Transformerレイヤーにおいて、通常の自己回帰（Masked Self-Attention）の処理中、ステップ3で構築した $C_{bar}$ を該当小節のすべてのタイムステップに対して直接注入します。
• これにより、Decoderは「その小節の持つスタイルと、指定された難易度ドメイン条件」の強い制約を受けながら、次のタイムステップの情報を表現する隠れ状態（Hidden State）を出力します。
5. 出力層（6つの独立した線形層による並列予測）
Decoderの最終出力（隠れ状態）を、6つの独立した線形層（Prediction Heads）に分岐させます。
各線形層が、次のタイムステップにおける6つの属性それぞれのトークン確率分布（Softmax）を予測します。
• Linear_Type(Decoder_Output) $\rightarrow$ 次の Type トークン（マスタースイッチ）
• Linear_Position(Decoder_Output) $\rightarrow$ 次の Position トークン（拍内の相対位置）
• Linear_Hand1(Decoder_Output) $\rightarrow$ 次の Hand1 トークン
• Linear_Hand2(Decoder_Output) $\rightarrow$ 次の Hand2 トークン
• Linear_Right_Foot(Decoder_Output) $\rightarrow$ 次の Right_Foot トークン
• Linear_Left_Foot(Decoder_Output) $\rightarrow$ 次の Left_Foot トークン

## 設計計画（実装フェーズ）

### 0. 設計方針
- 既存の `tokenization_method` 分岐を維持し、`standard` / `remi` は単語列入力、`cp_limb_v1` は6次元複合入力として扱う。
- 学習フロー（Encoder -> VAE latent -> Decoder）は全モードで共通化し、入力表現と出力ヘッドだけをモード差分にする。
- 既存checkpoint互換を守るため、階層PEはフラグで有効化し、既定値は現行PE（絶対位置PE）を維持する。

### 1. 対象コンポーネントと責務
1. `src/models/transformer_helpers.py`
	- `HierarchicalDrumPositionalEncoding` を追加。
	- 入力: `bar_idx`, `beat_idx`, `pos_idx`, `is_struct`。
	- 出力: $PE_{bar} + PE_{beat} + PE_{pos}$（必要に応じて構造イベント補正）。

2. `src/models/musemorphose.py`
	- `forward` / `generate` に階層PE用入力を追加（optional）。
	- `standard` / `remi` は `token_emb + hierarchical_pe`、`cp_limb_v1` は `cp_emb + hierarchical_pe` を選択可能にする。
	- 既存呼び出し互換のため、階層情報未指定時は現行 `PositionalEncoding` に自動フォールバック。

3. `src/data/drum_dataloader.py`
	- `standard` / `remi` 向けに各時刻の `bar_idx`, `beat_idx`, `pos_idx`, `is_struct` をバッチとして返す。
	- `cp_limb_v1` では既存の構造/CP系列に加えて同様の階層インデックスを返せるよう統一I/Fを定義。

4. `src/tokenizers/drum_tokenizer.py` / `src/tokenizers/drum_cp_tokenizer.py`
	- 既存トークン列生成時に階層インデックスを同時生成可能にする。
	- `remi` / `standard` で Bar, Beat, Pos を一貫定義（`Beat=1..4`, `Pos=0..23`）し、モデル入力で再利用する。

### 2. forward改修設計
1. Encoder入力
	- 現状: `enc_inp = emb_dropout(token_emb) + pe`。
	- 改修後: `enc_inp = emb_dropout(token_emb) + pe_abs_or_hier`。
	- `pe_abs_or_hier` は以下で決定:
	  - 階層インデックスあり: 階層PE。
	  - なし: 現行絶対PE。

2. Decoder入力
	- 現状: `dec_inp = emb_dropout(token_emb) + pe`。
	- 改修後: `dec_inp = emb_dropout(token_emb) + pe_abs_or_hier`。
	- `cp_limb_v1` 分岐でも同一のPE選択ロジックを再利用し、モード差分を最小化する。

3. 条件付けベクトル
	- 既存の `dec_seg_emb` 構築と5種難易度結合は維持。
	- 追加仕様は「PEの高度化」のみとし、latent/attribute経路の意味論は変更しない。

### 3. 生成（generate）時の設計
- 自己回帰生成中に `bar_idx`, `beat_idx`, `pos_idx` を逐次更新する状態管理器を導入。
- `<BAR>` 生成で `bar_idx += 1`, `beat_idx=1`, `pos_idx=0` に更新。
- `<BEAT_n>` 生成で `beat_idx=n`, `pos_idx=0` に更新。
- 通常ノート生成で `pos_idx` を進め、拍境界を跨ぐ場合は `beat_idx` を更新。
- 階層情報を計算できない互換モードでは現行PEへフォールバック。

### 4. 互換性と移行
- 追加設定:
  - `use_hierarchical_pe: bool = false`
  - `hier_pe_bar_vocab_size`, `hier_pe_beat_vocab_size`, `hier_pe_pos_vocab_size`
- 既存設定ファイルは無変更でも起動可能（default false）。
- 学習済みモデル再利用時は `use_hierarchical_pe=false` を既定とし、破壊的変更を回避。

### 5. テスト計画
1. 単体テスト
	- 階層PEのshape/範囲/加算結果検証。
	- `forward` のフォールバック分岐（hier有無）検証。

2. 結合テスト
	- `standard`, `remi`, `cp_limb_v1` すべてで1step学習が完走すること。
	- dataloader出力とmodel入力のshape整合を検証。

3. 回帰テスト
	- `use_hierarchical_pe=false` で既存loss推移と大きな乖離がないこと。
	- 既存生成スクリプトが引数追加後も後方互換で実行できること。

### 6. 実装順序（推奨）
1. `transformer_helpers.py` に階層PEを追加。
2. `musemorphose.py` へPE選択ロジック（互換フォールバック付き）を追加。
3. tokenizer/dataloaderで階層インデックスを出力。
4. train/generateスクリプトへ新規テンソル受け渡しを追加。
5. テスト整備と既存回帰確認。