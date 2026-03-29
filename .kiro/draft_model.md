# モデル構造
現状のレポジトリ内のMuseMorphoseモデルアーキテクチャを以下の要件を満たすように修正

- トークン設計がREMI or ドラム譜特化型設計の場合、MuseMorphoseモデルと同様の学習フローで学習・推論が行えるようにモデル設計

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