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

```bash
python -m src.utils.plot_loss --log trained_model/baseline/log.txt trained_model/variant/log.txt --output output/loss_comparison
python -m src.utils.plot_valloss --log trained_model/baseline/valloss.txt trained_model/variant/valloss.txt --output output/val_comparison
python -m scripts.analysis.collect_eval_results --checkpoints baseline variant --eval-dir output/eval --output output/eval/comparison.csv
bash tasks/evaluate_ensemble.sh output/eval/ensemble trained_model/baseline/params/best_params.pt trained_model/variant/params/best_params.pt
```

アンサンブル評価の入力・境界値・評価条件は環境変数 `INPUT_MIDI_DIR`、`DIFFICULTY_BOUNDS`、
`DIFFICULTY_LABELS` で指定できます。各モデルの語彙・トークン方式の互換性が必要です。
`src/utils/regression_validation.py` は指標の変化率を閾値と比較し、JSON の回帰レポートを作る API です。
用例は `tests/test_regression_validation_reporting.py` を参照してください。

## データ拡張・追加データの前処理

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

## 設計資料

- [四肢別 Attention 損失](DESIGN_attn_bias_loss.md)
- [位置別 Attention 損失](DESIGN_attn_pos_bias_loss.md)
- [難易度計算](DIFFICULTY_README.md)

資料中の旧実験名や実装当時の制約は履歴として扱い、現行コードと照合してください。
