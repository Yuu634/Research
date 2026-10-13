# Drum MuseMorphose：演奏難易度を調整するドラム譜生成

**元のリズムの特徴を保ちながら、演奏者に合わせて難易度を調整するドラム自動編曲**を目指す研究です。
ピアノ向け音楽生成モデル MuseMorphose をドラムに拡張し、入力MIDIから「手の連打を減らす」「足の負荷を下げる」といった条件に沿う譜面を生成します。

## 手法

Transformer VAE（変分オートエンコーダ）で入力譜面を小節ごとの潜在表現に変換し、指定した難易度を条件にトークン列を生成してMIDIへ戻します。

```mermaid
flowchart LR
    A[入力ドラムMIDI] --> B[トークン化]
    B --> C[Encoder：小節の潜在表現]
    C --> D[Decoder：条件付き生成]
    E[小節ごとの目標難易度] --> D
    D --> F[生成ドラムMIDI]
```

ドラムの表現には、小節・拍・打点位置・テンポと、楽器・奏法・強弱のトークンを使います。
難易度は次の5属性をそれぞれ8クラス（0〜7）に分けて指定します。

| 属性 | 表す内容 |
|---|---|
| 技術 `s_tech` | 特殊奏法の頻度 |
| 独立性 `s_indep` | 手足が異なるタイミングで動く割合 |
| 手 `s_hand` | 手の最大打撃密度 |
| 足 `s_foot` | 足の最大打撃密度 |
| 移動 `s_move` | 楽器間の移動速度 |

難易度は譜面から算出する指標です。指定どおりの生成や、実際の演奏のしやすさを保証するものではありません。

## 研究で比較する要素

- **表現と条件付け**：standard・REMI・CPのトークン表現、小節ごとの難易度指定。
- **Attentionの改良**：手足の関係と同時打点の関係を扱うバイアス・補助損失、その層別適用。
- **生成と評価**：複数モデルのアンサンブル推論、難易度制御と原譜のリズム保持の評価。

評価では、指定難易度と生成譜面の難易度の相関、他属性への影響、Groove Similarity、Token IoUを確認します。
基準設定は [drum_config_standard-lakh1-kl14.yaml](config/drum_config_standard-lakh1-kl14.yaml) です。
standard表現と5属性の条件付けを使い、Limb Bias・Position Bias・階層位置埋め込みは無効です。

## 実行方法

リポジトリ直下で実行します。以下の複数行コマンドはBash形式です。
Python環境にドラム用の依存を導入し、CUDAを使う場合は環境に適合するPyTorchを用意してください。

```bash
python -m pip install -r requirements_drum.txt
```

**データ準備**：MIDIのトークン化 → 難易度境界の計算 → 小節ラベルの付与を行い、基準YAMLのデータ・語彙パスに配置します。
処理例は [tasks/data_creator.sh](tasks/data_creator.sh) にあります。既定では拡張データ用の `standard-lakhaug` に出力するため、入力・出力先を使用する設定に合わせてください。

**学習**：基準設定を明示して実行します。

```bash
python -m scripts.train.train_drum --config config/drum_config_standard-lakh1-kl14.yaml
```

**生成**：学習済み重み、その実験の `config.yaml`、語彙、難易度境界ファイルが必要です。
このブランチの生成CLIは境界を `data/metadata/difficulty_bounds_standard.pkl` から読むため、学習時と同じ境界ファイルをその場所に用意してください。

```bash
# 技術・独立性・手・足・移動の順に指定。手を1クラス下げる目標の例
python -m scripts.generate.generate_drum_from_input \
    --model_path trained_model/checkpoints_drum_standard-lakh1-kl14/params/best_params.pt \
    --input_midi input.mid --output_midi output/generated.mid \
    --difficulty_mode global_shift --difficulty_shift 0 0 -1 0 0
```

難易度は一括増減（`global_shift`）、上限指定（`upper_cap`）、小節ごとの指定（`full_assign`）に対応します。
引数の詳細は各CLIの `--help` で確認できます。

## コードと資料

| 場所 | 内容 |
|---|---|
| `src/models/` | VAE・Transformer・Attentionバイアス |
| `src/data/`・`src/tokenizers/` | データローダー・難易度計算・トークン化 |
| `scripts/` | 前処理・学習・生成・評価・解析のCLI |
| `config/`・`tasks/` | 実験設定と実行スクリプト |
| `tests/` | トークン変換・データ整合性・モデルの検証 |

詳細は [ドラム表現](docs/README_drum.md)、[難易度指標](docs/DIFFICULTY_README.md)、[四肢別Attention損失](docs/DESIGN_attn_bias_loss.md)、[位置別Attention損失](docs/DESIGN_attn_pos_bias_loss.md) を参照してください。
資料には旧パスの実行例も含まれるため、実行時は `python -m scripts...` と各CLIの `--help` を使用してください。

## 元実装と引用

本研究は Shih-Lun Wu、Yi-Hsuan Yang による MuseMorphose の実装を基にしたドラム拡張です。
ライセンスは [LICENSE](LICENSE) を参照してください。

```bibtex
@article{wu2023musemorphose,
    title={{MuseMorphose}: Full-Song and Fine-Grained Piano Music Style Transfer with One {Transformer VAE}},
    author={Shih-Lun Wu and Yi-Hsuan Yang},
    year={2023},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
}
```
