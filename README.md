# Drum MuseMorphose

ドラム MIDI の演奏難易度を条件として学習・生成する Transformer VAE です。
YUU-11 の整理では、ユーザー指定の最良モデル設定
[`drum_config_standard-lakh1-kl14.yaml`](config/drum_config_standard-lakh1-kl14.yaml)
を基準にしています。設定値は変更していません。

## 環境

Python 3.12 を推奨します。リポジトリ直下で実行してください。

```bash
pip install -r requirements.txt
# または
conda env create -f environment.yml
conda activate musemorphose
```

学習設定は CUDA を使用します。実行環境の CUDA に適合する PyTorch を使用してください。
CPU での学習は `bash tasks/train.sh --device cpu` で指定できます。
依存ファイルは直接依存を列挙したもので、過去の学習環境を固定したロックファイルではありません。

## 学習・評価データの収集

Lakh MIDI、Groove MIDI、Groove2Groove を採用しています。
原データ、メタデータ、ドラム抽出 MIDI、トークン pkl を `data/` の別ディレクトリに保存します。

```bash
python -m scripts.preprocess.download_drum_sources
python -m scripts.preprocess.build_drum_collection --workers 16
python -m scripts.preprocess.validate_drum_collection --workers 16 --sample-per-group 16
```

出典、変換方針、公式分割、除外理由、実測件数は [収集記録](docs/DATA_COLLECTION.md) を参照してください。
この収集工程の pkl は難易度ラベル付与前の形式です。基準モデルの学習に使う前に別途ラベルを付与します。

## 基準モデル用の前処理

E-GMD を `data/raw/e-gmd/`、展開済み Lakh MIDI を `data/raw/lmd_full/` に用意します。

```bash
bash data/dataset.sh
bash tasks/data_creator.sh
```

`data/dataset.sh` は Lakh のドラムを `data/filtered/lakhmidi/` に抽出します。
`tasks/data_creator.sh` は E-GMD と抽出済み Lakh を standard 方式でトークン化し、
難易度境界値を計算して、小節ごとの難易度ラベルを付与します。
入力ディレクトリは `bash tasks/data_creator.sh MIDI_DIR [MIDI_DIR ...]` で変更できます。

主な出力は以下のとおりです。

- 語彙: `data/metadata_new/drum_vocab_standard-lakh.pkl`
- 難易度境界値: `data/metadata_new/difficulty_bounds_standard-lakh.pkl`
- 学習データ: `data/processed_new/drum_dataset_standard-lakh_with_difficulty/`

既存チェックポイントで推論する場合は、その学習時の語彙・難易度境界値を使用してください。
データ、語彙、境界値、モデル重みは Git に含まれません。

## 学習

```bash
bash tasks/train.sh
```

基準設定は standard 方式、難易度条件付けあり、`in-attn`、`bar_token` アンカーです。
Limb Bias・Position Bias・階層位置埋め込みは無効です。
出力先は `trained_model/checkpoints_drum_standard-lakh1-kl14/` で、
学習設定のコピー `config.yaml`、`params/best_params.pt`、最適化状態、学習ログを保存します。

## 入力 MIDI から生成

```bash
bash tasks/generate.sh input.mid output/generated.mid \
    --difficulty_mode global_shift --difficulty_shift 0 0 2 1 0
```

難易度の順番は技術・独立性・手・足・移動です。
`global_shift`、`upper_cap`、`full_assign` を使用できます。
詳しい引数は `python -m scripts.generate.generate_drum_from_input --help` で確認してください。
生成はチェックポイントの親ディレクトリにある `config.yaml` を読み込みます。

## 評価・可視化

```bash
bash tasks/evaluate.sh --input_midi_dir data/test_dataset
bash tasks/plot_loss.sh
python -m scripts.analysis.collect_eval_results \
    --checkpoints checkpoints_drum_standard-lakh1-kl14 \
    --output output/eval/results_table.csv
```

評価条件は [`config/evaluation_difficulty_labels.json`](config/evaluation_difficulty_labels.json) にあります。
評価・生成スクリプトの追加引数で入力、出力、モデル、難易度境界値を上書きできます。

## 新しい3モデルをtasksから実行する

`train.sh` 以外のタスクは、先頭に `musemorphose`（単体）、`beamsearch`（単体）、
`beam_muse`（複合）、`all`（3方式）を指定できます。`--model` での指定も可能です。
既定の実験先は順に `trained_model/musemorphose_v1`、`trained_model/beamsearch_v1`、
`trained_model/beam-muse_v1`。単一方式は `--run`、3方式は `--root` で変更できます。

```bash
# 方式を指定して学習／探索設定の選定。BeamSearchには勾配学習がない。
bash tasks/beam_muse.sh musemorphose --phase all --device cuda:0
bash tasks/beam_muse.sh beamsearch --phase all
bash tasks/beam_muse.sh beam_muse --phase all --device cuda:0

# 新しい3方式の実験。BeamSearch選定→単体学習→複合学習の順で実行。
bash tasks/beam_muse.sh all --root output/new_three_model_run --phase all

# v2ラベル付き譜面から共通pairdataを作成（allでも作成は1回）。
bash tasks/data_creator.sh all --stage pairs

# 新モデルは文脈・難易度目標・保護情報を持つ1小節JSONを入力する。
python -m scripts.evaluate.review_arrangement_inputs --examples
bash tasks/generate.sh all output/beamsearch_step3/absolute_request.json output/three_model_generation

# 保存済みのval選定モデルをCPUでtest/itest評価。再学習・再選定はしない。
bash tasks/evaluate.sh all --scope selected --workers 8
bash tasks/analysis.sh all
bash tasks/plot_loss.sh all
```

`--scope all` で指定方式の全候補を評価・解析できます。複合実験に含まれる単体候補は、
`--model musemorphose --run trained_model/beam-muse_v1` で指定します。
生成は選定記録から重みを選び、保存済みキャッシュから語彙・境界・BeamSearch設定を取得します。
`--checkpoint` で候補を直接指定する場合も、学習完了記録の方式とハッシュを照合します。
`all` の生成結果は方式ごとの出力ディレクトリに分けます。

新ニューラルモデルの解析は選定結果・評価・構成要素別の保存重み統計、
BeamSearchの解析は探索設定・ルール使用状況・履歴再生による評価値の検証です。
旧基準モデルのBias・Attention解析は `bash tasks/analysis.sh legacy MODEL_DIR` で実行できます。
以前のMIDI生成、評価、データ準備の引数は維持し、方式名を省略すると旧タスクの動作になります。
`beam_muse.sh` は方式省略時に複合実験、`plot_loss.sh` は全実験を対象にします。

データ準備の `--stage all` は、収集済み原データのトークン化・検証、v2ラベル付与、
ペア作成、trainのみの自己再構成ペア追加を順に実行します。原データのダウンロードは別工程です。
`--stage validate` は既存ペアを検証し、モデル重みには触れません。
各タスクの `--help` で引数を確認でき、`--dry-run` で実行コマンドだけを表示できます。

## 構成と整理方針

- `src/`: モデル、トークナイザー、データローダー、難易度計算、MIDI 変換。
- `src/beamsearch/`: 編曲ルール・BeamSearch・探索スコア。
- `src/musemorphose/`: VAE・Encoder/Decoder・Bias・ニューラル生成。
- `src/beam_muse/`: BeamSearchとMuseMorphoseの接続・比較実験。
- `src/common/`: 譜面表現・入力検証・保護制約・条件・損失・評価。
- `scripts/`: 前処理・学習・生成・評価と評価結果の集計。
- `tasks/`: 基準モデルを使う実行手順。
- `tests/`: 現行共通部品の回帰テスト。

`.claude`、`.github`、`.kiro`、`logs`、旧ピアノ用処理、比較実験用設定、
環境固有のジョブ、保存済み出力と Python キャッシュを削除しました。
今後の改良で再利用する分析・データ拡張・トークン検証・回帰比較のスクリプトは保持します。
[研究用ツール一覧と実行方法](docs/RESEARCH_TOOLS.md) を参照してください。
[ディレクトリ移行と既存成果物の扱い](docs/SOURCE_LAYOUT.md)に、旧パスとの対応と検証結果を記載しています。

CP・REMI・Bias の共通実装と、評価から参照されるアンサンブル処理は依存関係維持のため残しています。
モデル本体とトークン ID の定義は変更していません。

## 検証

```bash
pip install pytest
python -m pytest -q tests
```

整理前から以下の5件が失敗しています。今回の整理でモデルや生成アルゴリズムの修正は行っていません。

- `test_cp_tokenizer_schema.py`: 複数手音符でのタンバリン保持（1件）。
- `test_drum_dataloader_difficulty_alignment.py`: 難易度配列の長さ・不一致検出（2件）。
- `test_generate_drum_from_input_extension.py`: 早期 EOS に対する小節数保証（1件）。
- `test_musemorphose_hierarchical_and_compat.py`: 階層位置埋め込みの形状（1件）。

## 元実装と引用

本リポジトリは Shih-Lun Wu、Yi-Hsuan Yang による
「MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE」
の実装を基にしたドラム拡張です。元のライセンスは [LICENSE](LICENSE) を参照してください。

```
@article{wu2023musemorphose,
    title={{MuseMorphose}: Full-Song and Fine-Grained Piano Music Style Transfer with One {Transformer VAE}},
    author={Shih-Lun Wu and Yi-Hsuan Yang},
    year={2023},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
}
```
