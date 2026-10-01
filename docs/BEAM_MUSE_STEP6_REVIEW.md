# ステップ6：ハイパーパラメータ比較・実行方法

作成日：2026-10-01。ブランチ：`linear-07/rule-and-musemorphose`。
**実装と小規模な接続検証を完了。本規模学習・性能比較・最良ハイブリッドの確定は未実施。**

## 段階別の確認資料

1. [骨格特徴とディレクトリ構成](BEAM_MUSE_STEP1_REVIEW.md)
2. [Decoderへのin-attn条件入力](BEAM_MUSE_STEP2_REVIEW.md)
3. [相対／絶対難易度条件](BEAM_MUSE_STEP3_REVIEW.md)
4. [再構成・KL・系列補助損失](BEAM_MUSE_STEP4_REVIEW.md)
5. [3方式の共通評価](BEAM_MUSE_STEP5_REVIEW.md)
6. 本資料：実行・比較・検証結果

## 比較設計

`config/beam_muse_candidates.json`の共通設定にtrialごとの`model/loss/training`を上書きする。
未知の設定キーや不正な値は拒否する。探索候補を追加する場合は一意の名前を付ける。

| 候補 | 方式 | 難易度条件 | 学習率 |
|---|---|---|---:|
| muse_absolute_lr2 | ニューラル単体 | 絶対 | 0.0002 |
| muse_absolute_lr1 | ニューラル単体 | 絶対 | 0.0001 |
| hybrid_relative_lr2 | ハイブリッド | 相対 | 0.0002 |
| hybrid_relative_lr1 | ハイブリッド | 相対 | 0.0001 |
| hybrid_absolute_lr2 | ハイブリッド | 絶対 | 0.0002 |
| hybrid_absolute_lr1 | ハイブリッド | 絶対 | 0.0001 |

全候補で同じ損失と評価基準を使用する。単体は原譜、ハイブリッドは中間譜面をEncoderへ渡す。
単体では骨格の直接条件を無効にし、骨格損失は共通に残す。
したがってこの単体比較は新規学習する連続条件付きVAEであり、既存学習済みMuseMorphoseの
再現評価ではない。既存モデルと重みを使うCLIは保持している。

Encoder/Decoder各6層・8head・幅512・FFN2048・潜在512という基準YAMLの主要寸法を引き継ぐ。
新設定では条件幅128、10epoch、batch 8、seed 42を初期値とする。
中間譜面が追加音を含み得るため1小節512トークンを共通上限とし、上限超過は切り詰めず除外する。
基準YAMLの既存学習設定自体は変更しない。
次元・層数・条件幅・dropout・難易度表現・条件入力方式・損失係数・学習率・KL係数・
seed・補助損失の頻度と件数をJSONで比較できる。

## 実行

リポジトリ直下で実行する。成果物はGit対象外の`output/`以下へ保存する。

```bash
# 小規模な接続確認。CPUでも実行可能。本評価とは出力先を分ける
bash tasks/beam_muse.sh smoke output/beam_muse_step6_smoke --device cpu --smoke-limit 4

# 全入力の共通検証、選定済みBeamSearchによる中間譜面キャッシュ作成
bash tasks/beam_muse.sh prepare output/beam_muse_full

# trainで学習、valでepoch・候補を選定して固定
bash tasks/beam_muse.sh select output/beam_muse_full --device cuda

# 固定済みの2ニューラルモデルとBeamSearchをtest/itestで比較
bash tasks/beam_muse.sh test output/beam_muse_full --device cuda

# 設定・重み・選定順位を照合し、HTMLの比較表を作成
python -m scripts.evaluate.report_beam_muse --run output/beam_muse_full

# selection.jsonに記録された候補名のcheckpointを指定して編曲
python -m scripts.generate.arrange_beam_muse \
  --input output/beamsearch_step3/absolute_request.json \
  --checkpoint output/beam_muse_full/hybrid_relative_lr2/best.pt \
  --output output/beam_muse_full/inference
```

最後の`hybrid_relative_lr2`はパスの例であり、選定結果を示さない。
入力形式は前タスクのJSON契約を継続し、相対／絶対条件のモードはcheckpointから復元する。
生成結果はJSON・打点位置図・MIDIで確認できる。`status=invalid/failed`もそのまま表示する。
MIDIは奏法専用音源のアーティキュレーションを再現するものではない。

`--phase all`で全段階を連続実行できるが、人手確認には上記の個別実行を使う。
`--config`で候補設定、`--cache`で共通キャッシュ、`--threads`でCPUスレッド数を指定する。
本評価には入力件数を切り詰めるオプションを設けていない。
`--smoke-limit`はsmokeにだけ適用し、smokeのキャッシュ・選定結果を本評価へ流用しない。

## 再開と記録

- 原データシャード・語彙・境界・BeamSearch選定設定・関連実装のハッシュを検証する。
- キャッシュには共通除外理由・分割ごとの件数・出力ハッシュを記録する。
- 曲の分割重複とペアID重複を拒否し、教師譜面の難易度も元の前小節で再検証する。
- `run.json`に全候補、キャッシュ・コードのハッシュ、device・PyTorch・CUDA・スレッド数を保存する。
- 各trialに`history.json`、`best.pt`、`val_rows.json`、`completed.json`を保存する。
- 同じ条件で完了済みのtrialはハッシュ確認後に再利用する。途中で中断したtrialはseedから再実行する。
  optimizerや途中の乱数状態を復元するステップ単位の再開は実装していない。
- キャッシュ作成は全分割の完了を単位とする。作成途中で中断した場合は再作成する。
- コード・設定・データ・実行環境が変わった場合は別の実験出力先が必要。
  推論時も学習時と同じ実装・語彙・境界・BeamSearch設定を検査する。
  ディレクトリ移行で検証済みの旧成果物は[移行記録](SOURCE_LAYOUT.md)の対応表で読み込む。
  移行後の学習は新しい出力先を指定し、過去の実験記録へ新しい結果を混在させない。
- 入力の採点・文法制約・自己回帰samplingには追加計算が必要で、基準モデルと同じ学習速度は想定しない。

## 今回の検証範囲

train先頭4件・val先頭4件を共通に使い、幅32・1層・1epochの3候補で動作確認した。
各候補は2回の更新と2件の系列補助損失計算を実施。
再構成・KL・勾配は有限で、学習→val選定→checkpoint復元→推論まで通過した。
smokeの3候補は単体・相対ハイブリッド・絶対ハイブリッド。
同じコマンドで再開し、3つのcheckpointの更新時刻が変わらず再利用されることも確認した。

**smokeのニューラル出力はval全4件で制約違反となり、同時達成率・適格率は0%。**
これは学習経路の接続確認であり、BeamSearch単体より良い性能を示す結果ではない。
この重みを完成モデルや本評価の最良モデルとして配布しない。
全6候補の本規模学習、test/itestの実データ評価、信頼区間・複数seed・人手評価は未実施。

- [smoke比較表](../output/beam_muse_step6_smoke/review.html)
- [smoke推論の原譜・中間譜面・出力とMIDI](../output/beam_muse_step6_smoke/inference/index.html)

回帰テストでは相対値の符号、条件層への勾配、因果性、padding、教師入力の分離、
音符IDの対応、生成文法、失敗の分母、checkpoint復元・trial再開を検証する。
合成例を用いた選定→holdout評価も実行し、val選定時にtest/itestを読まず、
holdout評価で重みを変更しないことを確認した。

全体回帰の既存失敗は変更前`a0a3330`でも確認した：
CPのTambourine保持1件、ローダー難易度整合2件、既存生成のBAR数1件、階層PEの形状1件。
この5件は今回の変更対象に含めていない。
変更前は322件成功・5件失敗、変更後は340件成功・同じ5件失敗。新規18件はすべて成功。
新規3 CLIの`--help`とBashラッパーの構文確認も成功した。

検証コマンド：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
python -m scripts.train.tune_beam_muse --help
python -m scripts.generate.arrange_beam_muse --help
python -m scripts.evaluate.report_beam_muse --help
```

## ステップ7

上記の実装・候補設定・テスト・確認資料を`linear-07/rule-and-musemorphose`へpushする。
データ・重み・学習ログ・生成譜面・キャッシュはコミット対象外。
