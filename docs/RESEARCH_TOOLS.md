# モデル改良のための研究用ツール

基準モデルは `config/drum_config_standard-lakh1-kl14.yaml` です。
新しいモデルは設定を別名で作成し、チェックポイントと出力先も分けて比較してください。
以下はリポジトリ直下から実行します。

## 追加依存

```bash
pip install -r requirements_research.txt
```

`pandas` は学習損失比較、`pretty_midi` はデータ拡張、`datasets` は GigaMIDI 取得に使用します。
通常の学習・生成の依存は `requirements.txt` にあります。

## Attention と Bias の分析

```bash
bash tasks/analysis.sh trained_model/checkpoints_drum_standard-lakh1-kl14 trained_model/my_variant
```

各モデルの `config.yaml`、`params/best_params.pt`、設定が参照する語彙・検証データが必要です。
学習経過の Bias 値はステップ別チェックポイントから取得し、ステップ別がない場合は
`best_params.pt`・`final_params.pt` を確認します。その場合、学習ステップ不明は `-1` として記録します。

| スクリプト | 用途 |
| --- | --- |
| `scripts/analysis/visualize_attention_maps.py` | Encoder・Decoder の各層・各ヘッドの Attention を可視化 |
| `scripts/analysis/inspect_alpha_limb.py` | 四肢バイアス係数を抽出し、学習経過を CSV・図で確認 |
| `scripts/analysis/inspect_position_bias.py` | 位置バイアス係数を抽出し、学習経過を CSV・図で確認 |
| `scripts/analysis/qk_limb_bias_ratio.py` | QK スコアと四肢バイアスの寄与を比較 |

基準モデルでは Bias が無効なので、係数や寄与比が空欄・NaN になる場合があります。
単体実行では `python -m scripts.analysis.visualize_attention_maps --help` などで引数を確認できます。
分析対象のアーキテクチャを変更した場合は、各スクリプトのモデル構築・フックも合わせて確認してください。

## 学習・評価の比較

### test / itest の属性制御・発音類似度

ニューラル方式の `test_comparison.json` / `itest_comparison.json`、BeamSearch単体の
`test/summaries.json` / `itest/summaries.json` に、方式ごとに以下の項目を保存します。
`eval_loss.log` の各評価行の `evaluation` にも同じ集計を記録します。
共通の評価処理を使うためvalにも出力しますが、従来の候補・エポック選定基準は変更しません。

| JSONキー | 定義と保存内容 |
| --- | --- |
| `attribute_spearman` | 5属性それぞれの指定目標と生成難易度の順位相関。`coefficient`、有効件数 `n`、算出不能理由 `reason` |
| `attribute_label_distance` | 属性ごとの `abs(7 × 正規化目標 − 生成ラベル)` の平均 `mean` と有効小節数 `n` |
| `label_distance` | 各小節で5属性のラベル距離を等重みで平均し、さらに小節間で平均した `mean` と有効小節数 `n` |
| `rhythm_similarity` | ハイハット3種以外の `(楽器, 発音位置)` 集合を原譜と比較したF1の小節平均 `mean` と有効小節数 `n` |
| `hihat_similarity` | オープン・クローズ・ペダルハイハットを統合した発音位置集合を原譜と比較したF1の小節平均 `mean` と有効小節数 `n` |

発音F1は `2 × 共通イベント数 / (原譜のイベント数 + 生成譜面のイベント数)`。
96位置グリッド上の完全一致を使い、無音位置・強弱・奏法・音符IDは一致数に含めません。
両方の集合が空なら `null` として平均から除外し、片方だけ空なら0です。
入力ごとの評価結果の `metrics` にも `rhythm_similarity` / `hihat_similarity` を保存します。
既存のBeamSearchバッチは原譜と保存済み候補から類似度を補完して集計し、バッチ本体を変更しません。

Spearmanは同順位を平均順位として扱い、2件未満なら `insufficient_samples`、
目標が一定なら `constant_target`、生成値が一定なら `constant_generated` を理由として係数を `null` にします。
属性を混ぜて1本の相関を求めることはしません。相対指定も解決後の絶対目標を比較します。
ラベル尺度は0〜7で、離散ラベルに一致しない連続目標も丸めずに距離を算出します。
いずれも適格出力が対象です。失敗・制約違反は既存の `status_counts` / `valid_rate` に残り、
新指標の平均に成功値として混ぜません。旧結果で類似度が未記録なら欠測値として扱います。

既存の学習結果に追加する場合は、学習時と同じ設定・device・出力先で
`python -m scripts.train.tune_beam_muse --phase test ...` を実行します。
BeamSearch単体は `python -m scripts.evaluate.compare_arrangement_models --phase test ...` です。
検証済みの実装更新だけを互換表で許可し、重み・選定結果・学習時の実行記録は保持します。
ニューラル方式は再生成・再評価が必要です。保存済みJSONがコード変更だけで自動更新されることはありません。

保存済みの全チェックポイント・全BeamSearch候補をCPUでまとめて再評価する場合は、次を実行します。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.evaluate.refresh_test_metrics --root trained_model --scope all --workers 16
```

ニューラル方式は各モデルの `test_summary.json` / `itest_summary.json` と詳細行を保存し、
実験直下の `test_all_models.json` / `itest_all_models.json` に全候補をまとめます。
選定済みモデルについては従来の `test_comparison.json` / `itest_comparison.json` も更新します。
BeamSearchは全候補と比較用2方式を評価し、各モデルのsummaryと従来の `test/summaries.json` 等を更新します。
CPU再評価の実行環境は各モデルの `test_evaluation.json` 等に記録し、学習時の実行記録とは分けます。
`evaluation_cache/` の完了バッチを再利用して中断後に再開できます。
`trained_model/test_evaluation_status.json` に完了した出力先と、重み未保存などの評価不可理由を記録します。
全候補のtest結果は事後比較用で、valによる選定結果を変更しません。

### invalidを含む追加評価

`scripts/evaluate/evaluate_including_invalid.py` は、編曲制約に違反した出力も含めて
難易度のSpearman相関・ラベル距離・発音類似度を測定する研究用CLIです。
既存の `output/eval-result/*_summary.json` に収録された全モデルを対象にします。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.evaluate.evaluate_including_invalid --workers 16
```

保存先は既定で `output/eval-result`。`test_including_invalid_summary.csv` と
`itest_including_invalid_summary.csv` に全出力の指標、`*_valid_invalid_comparison.csv` に
`all_outputs`・`valid_only`・`invalid_only` の比較を保存します。JSONには同じ3集計と
従来の制約込み成功率も残します。モデル別の詳細行・生成譜面と、実装・入力・重みの
ハッシュを持つ再開用キャッシュも保存します。従来のCSV・JSONは更新しません。

ニューラル方式は保存済みチェックポイントから同じCPU条件でgreedy再生成し、
全件について計測時間以外の従来評価との一致を確認します。BeamSearchは保存済みの
全件validの評価を再利用します。現在のCLIはinvalidを含むBeamSearch保存結果を検出すると
停止するため、その場合は保存候補からの追加測定を実装してから使用してください。

元の `status` と `valid_rate` は保持します。難易度は制約判定を通さず、同じ境界・前小節で
算出します。無出力・算出不能は誤差や相関に値を捏造せず、`unmeasurable_n` に記録します。
`target_success_ignoring_validity` は適格性と独立に全5属性が許容範囲内かを判定し、
その集計対象の全件を分母にします。算出不能は不成功です。
発音集合が原譜・出力とも空の場合は従来どおり欠測なので、各類似度の `n` も確認してください。

### 追加実験を含む評価結果の統合

`scripts/evaluate/collect_arrangement_evaluations.py` は追加実験の全チェックポイントについて、
1回のCPU greedy生成から通常評価とinvalid込み評価を同時に保存します。
既定では `trained_model/beam-muse_v1` の6モデルを追加し、旧 `output/eval-result` の10モデルを
詳細行から再集計・検証して再利用し、合計16モデルを `output/eval_result` にまとめます。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.evaluate.collect_arrangement_evaluations --workers 16
```

test／itestごとに `*_summary.csv`（validのみの指標）、`*_including_invalid_summary.csv`（全出力）、
`*_valid_invalid_comparison.csv`（3集合の内訳）、`*_selected_comparison.csv`（val選定済みモデル）を保存します。
通常のsummaryでは `n`・status件数・valid_rateは全入力、指標ごとの `*_n` はその指標の測定件数です。
3集合比較CSVの `n` は各集合の件数です。JSONに完全な集計、モデル別ファイルに生成譜面・詳細行、
manifestに実装・重み・入力のハッシュと実行環境を記録し、バッチ単位で再開できます。
原譜・前後文脈・目標・制約が旧実験と一致することも検証します。

同じ候補名でも実験ディレクトリが異なるモデルは別IDで掲載します。単体・複合それぞれの
val選定を保持し、testによる再選定はしません。追加実験で保存済みのGPU評価は元の場所に残し、
今回のCPU評価は新しい出力先へ保存します。再構成損失の再計算はこのCLIの対象外です。

ルールベースBeamSearchのステップ1は[編曲ルールの確認資料](BEAMSEARCH_STEP1_REVIEW.md)を参照。
`src/common/representation.py`は音符・奏法の保持、`src/beamsearch/rules.py`は65編集関数、
`src/common/safety.py`は骨格保護・累積編集制限、`src/common/examples.py`は固定人工例を担当する。
以下で変更前後図と試聴用MIDIを生成する。探索・モデル選定は後述のステップ5のCLIを使う。

```bash
python -m scripts.evaluate.review_arrangement_rules
```

出力は`output/beamsearch_step1/`。実運用の候補には`apply_rule`で安全制約を適用する。

ステップ2の`src/beamsearch/scoring.py`は難易度一致・骨格保持・ルール数・編集量の採点器。
難易度計算は`src/common/difficulty.py`、骨格類似度と編集距離は`src/common/metrics.py`を共用する。
[探索スコアの確認資料](BEAMSEARCH_STEP2_REVIEW.md)に式・係数・固定例の内訳を記録した。
`python -m scripts.evaluate.review_arrangement_scoring`で`output/beamsearch_step2/`へHTML・JSONを生成する。
係数は確認用の初期値。選定済み設定とBeamSearch本体は後述のステップ5を参照。

ステップ3の`src/common/inputs.py`は小節・前後文脈・目標・保護情報の検証と入力変換を担当する。
[入力内容の確認資料](BEAMSEARCH_STEP3_REVIEW.md)に絶対値／差分の指定方法を記載した。
`python -m scripts.evaluate.review_arrangement_inputs --examples`で入力例と確認HTMLを生成する。
`--input`でJSON、`--pairdata`で生成済みペアを検証できる。探索は実行しない。

ステップ4の`src/common/evaluation.py`は固定基準で目標到達・骨格保持・編集量を評価し、
無出力・違反も含む成功率と曲単位マクロ平均を計算する。
[評価指標の確認資料](BEAMSEARCH_STEP4_REVIEW.md)に定義・閾値・分母・選定方針案を記載した。
`python -m scripts.evaluate.review_arrangement_metrics --examples`で人工例のHTML・JSON・CSVを生成する。
`--pairdata`で無編集と固定ルール列を診断できる。BeamSearch・モデル選定にはステップ5のCLIを使う。

ステップ5の`src/beamsearch/beam_search.py`は原譜の制約を保つBeamSearch本体。
[比較実験・選定の確認資料](BEAMSEARCH_STEP5_REVIEW.md)に探索仕様と結果を記載する。
`config/beamsearch_candidates.json`で探索幅・深さ・係数を比較し、
`config/beamsearch_selected.json`を推論用の選定設定として使う。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.evaluate.compare_arrangement_models --phase all --workers 24
python -m scripts.evaluate.report_arrangement_comparison --export-model config/beamsearch_selected.json
python -m scripts.generate.arrange_beam_search --input output/beamsearch_step3/absolute_request.json
```

比較CLIはval選定後にtest/itestを評価し、入力・設定・実装が一致する場合だけバッチ結果を再利用する。
再検証CLIは選定モデルの全履歴を再生し、譜面・指標の一致を検査する。
推論CLIはJSON・打点位置図・試聴用MIDIを出力する。人手による編曲品質評価は未実施。

ルールベース×MuseMorphoseの実装は[段階別の確認資料と実行方法](BEAM_MUSE_STEP6_REVIEW.md)を参照。
`src/musemorphose`・`src/beamsearch`・`src/beam_muse`に実装本体を配置し、共通部品は`src/common`へまとめる。
`src/common/skeleton.py`は原譜の骨格抽出、`src/common/conditioning.py`は符号付き相対／絶対条件、
`src/common/losses.py`は再構成・KL・系列方策勾配、`src/common/comparison.py`は3方式の比較評価を担当する。
旧パス・キャッシュ・重み・実装ハッシュの扱いは[ディレクトリ移行記録](SOURCE_LAYOUT.md)を参照。

```bash
bash tasks/beam_muse.sh smoke output/beam_muse_layout_smoke --device cpu
python -m scripts.evaluate.report_beam_muse --run output/beam_muse_layout_smoke
```

`scripts/train/tune_beam_muse.py`は中間譜面準備・train学習・val選定・固定後のtest/itest評価を分離する。
`config/beam_muse_candidates.json`でモデル・損失・学習パラメータを比較する。
`scripts/generate/arrange_beam_muse.py`は新形式のcheckpointから1小節を編曲する。
本規模学習は未実施であり、smokeの重みを性能検証済みモデルとして扱わない。

難易度属性v2は[承認済み算出式](ATTRIBUTE_FORMULA_REVIEW.md)と[実装・ラベリング結果](ATTRIBUTE_V2_RESULTS.md)を参照。
旧式・基準YAML・既存チェックポイントは維持し、新ラベルを別ディレクトリへ生成する。

### tasksの3方式対応

`scripts/tasks.py` は `train.sh` 以外の6タスクから共用する実行先選択CLIです。
先頭の方式名または `--model` で `musemorphose` / `beamsearch` / `beam_muse` / `all` を指定します。
旧MIDI生成・旧評価・Bias/Attention解析・旧データ準備は `legacy` として保持し、
これら4タスクでは方式省略時も以前の引数を受け付けます。
新モデルの生成入力は既存の1小節JSON形式で、MIDI全曲の旧生成処理とは別です。

| タスク | 新モデルに対する処理 |
|---|---|
| `beam_muse.sh` | ニューラルのprepare/select/test、BeamSearchのselect/test。allは探索設定を先に選定 |
| `generate.sh` | val選定重みと学習時メタデータを照合し、1小節JSONから生成。allは出力を方式別に分離 |
| `evaluate.sh` | 保存済みモデルをCPUでtest/itest再評価。scopeはselected/all、混在実験でも指定方式だけを対象 |
| `analysis.sh` | ニューラルの選定・評価・重み統計、BeamSearchの履歴再生・ルール使用状況・比較表 |
| `data_creator.sh` | 共通のv2ラベル・ペア準備とtrain限定の自己再構成ペア追加。allでもデータを重複作成しない |
| `plot_loss.sh` | 指定方式の実験ログを再描画。方式省略時はtrained_model全体 |

`scripts/evaluate/evaluate_arrangement_models.py` は実験先を明示して再評価し、
`refresh_test_metrics.py` のキャッシュ・照合・集約処理を再利用します。
`scripts/analysis/analyze_arrangement_models.py` は新ニューラルモデルの学習完了記録・
実装ハッシュ・重みを検証し、JSON/HTMLと構成要素別RMS画像を保存します。
RMSには保存された浮動小数のバッファも含み、性能指標やAttention強度とは解釈しません。
BeamSearchには既存の `report_arrangement_comparison.py` を使い、旧解析ツールも削除しません。
各CLIの `--help` とタスクの `--dry-run` が利用できます。
用例・既定の実験先は [README](../README.md#新しい3モデルをtasksから実行する) を参照してください。

| スクリプト／モジュール | 用途 |
|---|---|
| `src/data/drum_difficulty_v2.py` | 秒単位の打撃負荷、手足非同期、2手の移動DP、LZ反復性 |
| `scripts/preprocess/label_drum_difficulty_v2.py` | 生スコア算出、train限定の境界推定、pkl付与、全件検証 |
| `scripts/evaluate/evaluate_rhythm_attribute_v2.py` | Zenodo 40刺激の知覚的複雑さとの相関診断 |

ラベリングは既存依存で実行できる。Zenodo診断だけは追加依存rdataを使う。
以下の`--no-deps`はNumPy・pandas・xarray・typing_extensionsが既にある環境向け。
新規環境では`--no-deps`を外して依存もインストールする。

```bash
python -m scripts.preprocess.label_drum_difficulty_v2 --help
python -m pip install --no-deps --target output/revise_attribute_v2/deps -r requirements_attribute_v2.txt
python -m scripts.evaluate.evaluate_rhythm_attribute_v2 --help
```

6番目のs_rhythmは検証用の反復性ラベル。現行5属性モデルの条件には追加しない。
v2クラスはv1と意味が異なるため、既存チェックポイントの条件値として置き換えない。

```bash
python -m src.utils.plot_loss --log trained_model/baseline/log.txt trained_model/variant/log.txt --output output/loss_comparison
python -m src.utils.plot_valloss --log trained_model/baseline/valloss.txt trained_model/variant/valloss.txt --output output/val_comparison
python -m scripts.analysis.plot_loss_curves --trained-model-dir trained_model
python -m scripts.analysis.collect_eval_results --checkpoints baseline variant --eval-dir output/eval --output output/eval/comparison.csv
bash tasks/evaluate_ensemble.sh output/eval/ensemble trained_model/baseline/params/best_params.pt trained_model/variant/params/best_params.pt
```

`scripts/analysis/plot_loss_curves.py`（`bash tasks/plot_loss.sh`）は、各実験の
`train_loss.log` / `eval_loss.log` を再帰的に読み、`loss_curve.png` を再生成します。
学習時にも各epochのチェックポイント選定後に同じ形式で自動保存します。
横軸は共通の学習step、パネルは再構成・KL・合計（再構成 + β × KL）です。
βは記録値を表示し、補助方策損失は合計に含めません。KL_rawが異なる場合は点線で併記します。
最良位置は `completed.json` の採用epochと `history.json` のstepから取得し、
損失最小のepochへ選び直しません。test / itestは採用チェックポイント上の独立した点です。
旧形式の検証ログは同じstepを平均し、履歴があればトークン加重の再構成損失を優先します。
集約済み評価ログがある場合はそちらを優先します。
探索スコアは指標ごとに分けた `beamsearch_scores.png`、符号付き補助方策損失は
stepごとの寄与の和を `auxiliary_loss_curve.png` に保存します。
勾配学習のないBeamSearch単体の `loss_curve.png` は学習損失がない旨を表示します。
再描画は元ログ・設定・重みを変更しません。
旧チェックポイントの `valloss.txt` は従来どおり `loss_curves.png` に描画します。
旧CLIの `python -m src.utils.plot_loss_curves` も保持しています。

アンサンブル評価の入力・境界値・評価条件は環境変数 `INPUT_MIDI_DIR`、`DIFFICULTY_BOUNDS`、
`DIFFICULTY_LABELS` で指定できます。各モデルの語彙・トークン方式の互換性が必要です。
`src/utils/regression_validation.py` は指標の変化率を閾値と比較し、JSON の回帰レポートを作る API です。
用例は `tests/test_regression_validation_reporting.py` を参照してください。

## データ拡張・追加データの前処理

難易度v2付き譜面からの擬似編曲ペアは、[data/pairdata](../data/pairdata/README.md)に実装と結果をまとめた。
`core.py`が小節の類似度と難→易の選択、`build.py`が大量生成と重複除外、
`validate.py`が全ペアの再計算、`io.py`が学習用シャードの読み込みを担当する。
ユーザーの配置指定により、このデータ作成CLIは以下のpackage形式で実行する。

```bash
python -m data.pairdata.build --workers 24
python -m data.pairdata.validate --workers 24
```

元の曲分割を保持し、現在小節の内容が分割をまたぐペアを除外する。
文脈の区間名は不明として扱い、代理特徴を用いた擬似ペアであることを明示する。
出力は `data/processed/drum_pairdata_v1/`、再開用バッチは `data/work/drum_pairdata_v1/`。

```bash
python -m data.augment_drum_midi --input_dir data/filtered/lakhmidi --output_dir data/filtered/lakh_augment --augment_hihat --augment_location --seed 42
python -m data.cleaning_groove_full --help
python -m data.download_gigamidi --help
python -m data.cleaning_gigamidi --help
```

GigaMIDI は実行時に外部データセットを取得します。ダウンロード用のキャッシュ先は
`data.download_gigamidi` の `--cache_dir` で指定できます。
新しいデータの語彙と難易度境界値は、基準モデル用とは別の場所に作成してください。

## トークン・テンポの手動検証

```bash
python tests/test_token/run_all_tests.py
python tests/verify_tempo_token_in_pkl.py tests/test_token/sample_midis/simple_beat.mid output/tempo_check.pkl
```

トークン検証はサンプル MIDI を生成し、standard・REMI・CP の往復変換と内容の比較を行います。
`tests/test_token/sample_midis/` と `tests/test_token/output/` は再生成可能なため Git 管理対象外です。
独自 MIDI は `test_round_trip.py` の `--single_file`、`--midi_dir` で指定できます。
これらは手動診断用 CLI で、自動回帰テストは `python -m pytest -q tests` で実行します。
復元時のサンプル2曲の診断では、standard・CP の往復比較で音符の差分、REMI の
トークン内容検証で一致率不足を検出しました。診断コマンドは終了コード1を返します。
テンポ検証は成功しています。今回はモデル・トークナイザー本体の変更は行っていません。
追加データの外部ダウンロードと、実際の学習済み重みでの分析は未実行です。

## 再構成ペアの追加

`scripts/train/tune_beam_muse.py --config config/musemorphose_candidates.json`で、
再構成付きペアを使ったMuseMorphose単体の学習・選定・評価も実行できます。
単体のみの新規キャッシュ作成ではBeamSearch探索を省略します。

`scripts/preprocess/add_pairdata_reconstruction.py`は、既存の編曲ペアを保持したまま、
trainの元小節を使った自己再構成ペアを別データセットへ追加します。
val・test・itestは既定では変更しません。詳細は[作成規約と学習方法](PAIRDATA_RECONSTRUCTION.md)を参照してください。

```bash
python -m scripts.preprocess.add_pairdata_reconstruction --help
python -m scripts.preprocess.add_pairdata_reconstruction --reconstruction-splits train
python -m scripts.preprocess.add_pairdata_reconstruction --validate-only
```

## 設計資料

- [四肢別 Attention 損失](DESIGN_attn_bias_loss.md)
- [位置別 Attention 損失](DESIGN_attn_pos_bias_loss.md)
- [難易度計算](DIFFICULTY_README.md)

資料中の旧実験名や実装当時の制約は履歴として扱い、現行コードと照合してください。
