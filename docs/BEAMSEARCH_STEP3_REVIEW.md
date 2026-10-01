# BeamSearch実装：ステップ3・入力内容の確認

作成日：2026-10-01。ブランチ：`linear-06/beamsearch`。
ステップ2のコミット`04ba76a`後、ユーザーの続行指示を受けて実装した。
**状態：入力変換・検証を実装済み。ユーザーの続行指示を受け、[ステップ4の評価指標](BEAMSEARCH_STEP4_REVIEW.md)へ進んだ。**
探索本体・学習・モデル比較・選定・pushは未実施。

## 今回の入力仕様

図の入力を「元の1小節」「目標難易度」「文脈・保護情報」に分ける。
目標は5属性の名前付き辞書とし、**絶対値か差分かを必ず指定する**。

| 入力項目 | 必須 | 内容・用途 |
|---|---|---|
| `schema_version` | 必須 | 整数`1` |
| `original` | 必須 | 編集する単一小節。4/4・96位置のstandard文字列トークンと明示注釈 |
| `previous` | 必須 | 元譜の直前小節。ない場合は明示的に`null`。難易度の境界計算に使用 |
| `target.mode` | 必須 | `absolute`＝到達目標、`delta`＝元譜からの増減 |
| `target.values` | 必須 | 承認済みv2の5属性をすべて指定 |
| `following` | 任意 | 元譜の直後小節。保存のみで、現スコアには使用しない |
| `context` | 任意 | `section`・`phrase`を文字列または`null`で指定。保存のみ |
| `extra_locked_ids` | 任意 | 既定の骨格保護へ追加する音符ID。既定保護の解除は行わない |

入力の不足・未知のキー・JSONキーの重複・不正な数値を検出する。
譜面からBPMを読み、別のBPM引数は設けない。前後小節はそれぞれ自身のBPMを保持する。
他トラック、楽曲全体の連続編集、MIDI直接入力はこのAPIの対象外。
MIDIは既存の前処理でstandard化し、対象の1小節を切り出して渡す。

### 5属性と目標の解釈

| キー | 属性 |
|---|---|
| `s_tech` | 奏法・強弱制御 |
| `s_indep` | 手足のリズム独立性 |
| `s_hand` | 手の連打速度 |
| `s_foot` | 足の連打速度 |
| `s_move` | 打点間の移動距離 |

図の配列順には依存しない。`s_rhythm`は参考値で、目標には含めない。
v2のクラス0〜7を7で割った値を使用し、境界値は再推定しない。

- `absolute`：各値が0〜1の目標そのもの。例：`s_hand=0.4`を目指す。
- `delta`：原譜の正規化難易度に加算する量。例：元譜が0.6なら`-0.2`で目標0.4。
  差分は相対割合ではなく、正規化尺度上の加算値。
- 原譜の難易度は**指定された同じ前小節**を使って再計算する。
- 加算後の目標が0〜1を外れる入力はエラー。クリップや符号の自動解釈はしない。
- 整数・小数のみを受け付け、数値文字列・真偽値・NaN・Infinityは拒否する。
- 各属性の増減を混在させることは可能。ただし、保護・編集上限・利用可能ルールによる
  目標の達成可能性はこの段階では判定しない。

探索スコアの内部APIへは、常に変換後の0〜1の絶対目標を渡す。
0.2などクラス刻みにない目標も受け付けるが、誤差0になるとは限らない。

### 譜面・保護・注釈

各小節は次の形式。`tokens`は`TEMPO, BAR, …, EOS`を持つ単一小節である。

```json
{
  "tokens": ["<TEMPO_120>", "<BAR>", "<BEAT_1>", "<POS_0>",
             "KICK_HIT_Normal", "HH_CLOSED_HIT_Normal", "<BEAT_2>", "<POS_0>",
             "SNARE_HIT_Normal", "<EOS>"],
  "meter": [4, 4],
  "length": 96,
  "fill_regions": [],
  "note_annotations": {}
}
```

- `meter`・`length`も明示必須。96位置は1拍24位置、小節全体96位置。
- 語彙はv2メタデータのstandard語彙を使用。数値IDは先にその語彙で復号する。
- 音符IDはトークンの打撃出現順に`n0, n1, …`と付与され、確認レポートで一覧化する。
  追加保護はその入力のIDを指定する。トークン並べ替え後のIDは再確認が必要。
- 拍・位置指定の逆行、重複指定、打撃のない位置ブロック、同位置・同楽器の重複、
  3手以上の同時打撃、語彙で表現できない内容、難易度算出不可の譜面を拒否する。
- 空小節は`TEMPO, BAR, EOS`で表現できる。空小節と`previous=null`は区別する。
- `fill_regions`は`[[開始, 終了], …]`の半開区間。0以上96以下で重複不可。
  区間内の最初・最後の打撃を保護する。未指定のFillを推定しない。
- 装飾音の明示注釈例：`"note_annotations":{"n1":{"ornament_group":"drag1","ornament_role":"grace"}}`。
  役割は`main/grace/roll`。奏法トークンは保持し、近接打撃だけからグループを推測しない。
- 既定保護はステップ1を継続する。拍頭Kick、第2・第4拍Snare、各拍の代表打撃、
  Accent、明示Fill境界を保護。追加8音・削除16音・原譜比編集率0.5を上限とする。

`previous=null`は前小節なしで計算する明示指定であり、欠損を自動補完しない。
曲中の小節では実際の前小節を渡す。後続の連続編集では、確定済み前小節を渡す必要がある。
後小節の難易度が変わる影響や接続の自然さは、現時点の採点に含まれない。

## 実装と確認例

- 入力API：`src/common/inputs.py`の`prepare_input(payload, model)`
- CLI：`scripts/evaluate/review_arrangement_inputs.py`
- テスト：`tests/test_arrangement_inputs.py`
- [絶対値と差分の比較HTML](../output/beamsearch_step3/index.html)
- [絶対値の入力例](../output/beamsearch_step3/absolute_request.json)
- [差分の入力例](../output/beamsearch_step3/delta_request.json)

```bash
# 変更前の譜面、等価な2種類の目標、保護音・スコア内訳を生成
python -m scripts.evaluate.review_arrangement_inputs --examples

# 入力例を編集して確認（探索はまだ実行しない）
python -m scripts.evaluate.review_arrangement_inputs \
  --input output/beamsearch_step3/delta_request.json \
  --output output/beamsearch_step3/readback

# 既存の擬似ペアから16件を入力へ変換
python -m scripts.evaluate.review_arrangement_inputs \
  --pairdata data/processed/drum_pairdata_v1 --split train --limit 16 \
  --output output/beamsearch_step3/train
```

レポートには元譜の生スコア・クラス・正規化値、指定値・変換後目標、
前後小節の有無、全音符ID、保護ID、無編集候補のスコア、入力・境界・語彙・実装のハッシュを保存する。
生成物は`output/beamsearch_step3/`に保存し、Gitには含めない。

### ペアデータとの接続

`request_from_pair`はv1ペアデータの次の項目を使う。

| ペア側 | 探索入力側 |
|---|---|
| `X_before` | `original` |
| `d_after` | 5属性名に対応させた`absolute`目標 |
| `context.previous` | `previous` |
| `context.following` | `following` |
| `context.section` | `context.section` |

`X_after`、候補の元の前後小節、類似度・信頼度を探索入力には渡さない。
`d_after`はユーザー指定目標を模した条件値として使用する。
この設計で将来評価できるのは「ペアの目標難易度を与えた条件付き編曲」であり、
利用者の希望難易度そのものを自動予測する性能ではない。

ペアの候補難易度が元譜の前小節で計算済みであることを確認する。
Fill代理特徴・小節周期位置を、明示的なFill区間や楽曲構造の正解に読み替えない。
v1には4/4・96位置の前処理済みデータとして接続し、元MIDIの拍子をこの変換器で再解析しない。

## 検証結果

- 新規30テスト：目標変換、文脈依存、保護・注釈、入力拒否、教師譜面の非使用などを確認。
- 既存のルール・探索スコア・v2・ペアデータのテストも併せて177件成功。
- `--help`、例生成、生成JSONの再読込、範囲外目標のCLI拒否を確認。
- train/val/test/itest各16件、計64ペアの変換・入力検証が成功。
  元譜の再計算値と保存ラベルが一致し、入力目標も保存された`d_after`と一致。
  この照合で重み・閾値・モデルを調整していない。
- 既存ローダーの既知の2件の失敗は変更対象外で、今回のテスト対象には含めていない。

## 今回確認していただきたい事項

1. 4/4・96位置の単一小節を対象とし、原譜の前小節を明示して渡すこと。
2. 目標は名前付き5属性とし、`absolute`または`delta`を明示すること。
3. 後小節・曲区間情報は保存のみ、Fill・装飾音は明示注釈のみを使用すること。

確認後、ステップ4で目標難易度への到達・骨格保持・編集量などの評価指標を実装する。
