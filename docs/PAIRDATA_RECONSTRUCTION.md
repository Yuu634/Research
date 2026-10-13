# 編曲ペア＋自己再構成データ

元データ`data/processed/drum_pairdata_v1`を変更せず、
`data/processed/drum_pairdata_v1+reconst`にtrainの自己再構成ペアを追加する。
val・test・itestは元の圧縮シャードをそのままコピーする。

## 追加規約

- 既存trainペアの編曲前小節を対象とする。未採用の曲・小節や編曲後小節からは追加しない。
- 同一データセット・曲ID・編曲前小節番号は1件にまとめる。
  同じ打点でも曲・小節が異なる場合は文脈が異なり得るので別件として保持する。
- `X_after = X_before`、`d_after = d_before`、`delta_d = [0, 0, 0, 0, 0]`。
- 元の前後文脈を維持する。6属性の生スコア・クラスと元文脈でのafter情報もbeforeの値にする。
- `source.after_bar`は`source.before_bar`に合わせる。
  `original_after_previous/following`もbeforeの文脈へ置き換える。
- 再構成ペアだけに`task_type="reconstruction"`と`parent_pair_id`を追加する。
  既存編曲ペアのフィールド・IDは変更しない。識別時は
  `pair.get('task_type', 'arrangement')`を使う。
- 再構成IDは`reconst:`と元小節の識別情報のハッシュで作り、編曲IDと区別する。
- 再構成の`q`と類似度は1、難易度距離は0。元の難→易の採用閾値は再構成へ適用しない。

これは前後の小節を参照して対象1小節を再構成するペアデータであり、従来モデルの
曲全体の自己再構成データ形式へ変換する処理ではない。

## 作成と検証

```bash
python -m scripts.preprocess.add_pairdata_reconstruction \
  --input data/processed/drum_pairdata_v1 \
  --output data/processed/drum_pairdata_v1+reconst \
  --reconstruction-splits train

python -m scripts.preprocess.add_pairdata_reconstruction \
  --input data/processed/drum_pairdata_v1 \
  --output data/processed/drum_pairdata_v1+reconst \
  --validate-only
```

既定値は上記と同じ。既存出力を上書きしない。作成途中に中断した場合は別の出力先を使う。
成功時に`validation.json`を保存する。それ以前の出力は学習に使用しない。
従来の`data.pairdata.validate`は難→易専用なので、追加データには上記CLIを使用する。

元データの検証記録と索引・全シャード・語彙・境界・難易度式のハッシュを確認する。
追加後は全レコードを元データから再導出して照合し、再構成の難易度は前小節付きで再計算する。
ペアIDの一意性、曲と現在小節の内容署名の分割間重複も確認する。
元データの原譜照合結果を継承し、原譜ファイルの全再読込までは行わない。
前後文脈の共有や近似重複については元pairdataの制約を引き継ぐ。

## 作成件数

| 分割 | 編曲 | 再構成 | 合計 |
|---|---:|---:|---:|
| train | 38,004 | 30,532 | 68,536 |
| val | 5,429 | 0 | 5,429 |
| test | 6,713 | 0 | 6,713 |
| itest | 76 | 0 | 76 |
| 合計 | 50,222 | 30,532 | 80,754 |

これはキャッシュ作成前の件数。既存の学習用キャッシュではトークン上限などにより
共通入力から除外する例があり、採用後の件数はキャッシュmanifestで確認する。

## 学習への入力

### MuseMorphose単体だけを学習する場合

```bash
python -m scripts.train.tune_beam_muse \
  --phase all \
  --config config/musemorphose_candidates.json \
  --output output/musemorphose_pair_reconst_v1 \
  --device cuda
```

このconfigは`drum_pairdata_v1+reconst`を使い、単体の学習率2候補
（0.0002、0.0001）を各10epoch学習する。原譜と絶対目標難易度を入力し、
骨格の直接条件入力は使わない。trainで編曲と自己再構成を混ぜ、valの編曲性能で
候補を選び、test/itestで選定済み単体モデルを評価する。
`select`だけの実行は事前に`prepare`を完了しておく。

単体用の新規キャッシュではBeamSearch探索を行わず、`intermediate`欄に原譜を保持する。
manifestの`identity.intermediate_mode="original"`で識別し、ハイブリッドへの流用は拒否する。
`data.beam_model`は語彙・難易度境界の互換性と共通採点設定の取得に使う。
単体のみの評価レポートに、原譜をBeamSearch結果として掲載することはない。

既存の中間譜面付き共通キャッシュを`--cache`で指定し、`--phase select`から
単体学習を実行することもできる。その場合もEncoder入力は原譜。
新規単体キャッシュは中間譜面のトークン上限を検査しないため、3方式の公平な比較では
共通キャッシュで同じ適格入力集合を使う。

出力の`selection.json`に単体の最良候補、各候補の`best.pt`に重みを保存する。
`scripts.evaluate.report_beam_muse --run <出力先>`は単体のみの実験にも対応する。
これは比較用の連続難易度条件付きモデルであり、従来の`train_drum.py`用YAMLとは別経路。

単体実行対応時は、実データからtrain編曲2件＋再構成2件、val/test/itest各2件を抽出し、
幅32・1層・1epochの2候補で`--phase all`の準備・学習・選定・評価・HTML出力を検証した。
新旧checkpointの復元と、従来の3候補比較で全重みが変更前と一致することも確認した。
回帰テストは376件成功・同じ既存5件失敗。小規模実験の適格率は0%であり、本規模の性能評価ではない。

### 単体とハイブリッドを比較する場合

```bash
python -m scripts.train.tune_beam_muse \
  --phase prepare \
  --config config/beam_muse_candidates.json \
  --output output/beam_muse_pair_reconst_v1

python -m scripts.train.tune_beam_muse \
  --phase select \
  --config config/beam_muse_candidates.json \
  --output output/beam_muse_pair_reconst_v1 --device cuda

python -m scripts.train.tune_beam_muse \
  --phase test \
  --config config/beam_muse_candidates.json \
  --output output/beam_muse_pair_reconst_v1 --device cuda
```

新しいキャッシュ・実験出力先を使う。既定configの`data.pairdata`には再構成付きデータ、
`data.beam_model`には選定済みBeamSearchを指定してある。語彙・境界はpairdata内から読む。
通常ペアとの比較はconfigを分けるか、CLIの`--pairdata`で上書きする。
既存ローダーは自己再構成ペアも通常の教師ペアとして読み、train内をシャッフルする。
今回の追加はデータ単位の混合であり、損失係数・タスク別サンプラー・事前学習段階は追加しない。
キャッシュ除外前はtrainの約44.5%が再構成。固定の1:1混合ではない。
再構成と編曲は同じ再構成・KL・系列補助損失の対象になる。

val・test・itestは編曲のみなので、選定と最終評価は編曲性能を測る。
再構成性能を独立に評価する集合や指標は今回追加していない。
元の編曲教師と保護制約の不整合を解消する処理も含まない。

## 実施した検証

- 全80,754件の保持・生成規約を照合し、追加30,532件の難易度再計算に成功。
- 元データの全シャードのハッシュが変わっていないことを確認。
  val・test・itestはコピー先も元と同じSHA-256。
- 再構成の実データ4件で既存`prepare_example`を通過。
  BeamSearchの編集履歴が空、現在難易度と目標が一致することを確認。
- 単体・ハイブリッド各々の小規模モデルで損失計算・逆伝播が成功。
  本規模学習や性能改善の確認は行っていない。
- 回帰テストは変更前352件成功・5件失敗、変更後358件成功・同じ5件失敗。
  新規6件は元ペア保持、評価シャード保持、文脈とラベル、破損・改変拒否を検証。
- 追加CLIの`--help`が成功。

既存5件の失敗はCPのTambourine保持、ローダー難易度整合2件、既存生成のBAR数、
階層PEの形状であり、今回の追加による失敗はない。
