# 3モデルのディレクトリ構成と移行記録

`linear-07/rule-and-musemorphose`上で、実装の入口と本体を同じ場所へ整理した。
旧`src/arrangement`・`src/models`は削除し、CLI・分析ツール・テストの参照を更新した。
基準YAML、探索ルール、層・パラメータ名、難易度・損失・評価の算出式は維持する。

```text
src/
├── beamsearch/        # beam_search・rules・scoring・selected_model
├── musemorphose/      # model・conditional・Transformer・Bias・generation
├── beam_muse/        # model・data・experiment
├── common/           # 譜面・条件・難易度・制約・損失・評価・互換読込
├── data/
├── tokenizers/
└── utils/
```

`common`はモデル実装をimportせず、必要な採点器は引数で受け取る。
`musemorphose`は`beam_muse`に依存せず、ハイブリッドが単体モデルと探索を組み合わせる。
CP・REMI・Limb/Position Biasと分析・比較用ツールも移動先で保持する。

## 主な移動対応

| 旧配置 | 現在の配置 |
|---|---|
| `src/arrangement/beam_search.py`, `rules.py`, `selected_model.py` | `src/beamsearch/`の同名ファイル |
| `src/arrangement/scoring.py` | 探索スコアは`src/beamsearch/scoring.py`、難易度は`src/common/difficulty.py`、指標は`src/common/metrics.py` |
| `src/arrangement/representation.py`, `inputs.py`, `safety.py`, `evaluation.py`, `examples.py` | `src/common/`の同名ファイル |
| `src/models/musemorphose.py` | `src/musemorphose/model.py` |
| `src/models/`のTransformer・Bias部品 | `src/musemorphose/`の同名ファイル |
| `src/beam_muse/features.py` | `src/common/skeleton.py` |
| `src/beam_muse/conditioning.py`, `losses.py` | `src/common/`の同名ファイル |
| `src/beam_muse/evaluation.py` | `src/common/comparison.py` |
| `src/beam_muse/generation.py` | `src/musemorphose/generation.py` |

旧importの別名パッケージは残さない。外部のノートブック等は表に従ってimportを更新する。

## 保存済み成果物

**重み**：state_dictの層名・形状を変更していない。standard・REMI・CPの旧モデルと
連続条件付きモデルの保存重みを、新配置のクラスへ`strict=True`で読み込める。
コードを含むモデル全体のpickleを一般的に互換化する仕組みではない。

**pickleキャッシュ**：旧`Bar`・`Note`・`PreparedInput`等の参照は、
`data.pairdata.io.read_pickle`が`common.serialization`の対応表を通して復元する。
元ファイルを書き換えないため、既存のファイルハッシュも維持する。
この読込経路は従来どおり信頼できるローカル成果物を対象とする。

**実装ハッシュ**：`config/source_layout_migration.json`に、移動前コミット
`b2a3af332480604f1098c7b8b0e40f8e3b33bdef`の実装ハッシュ集合と移動後の全ハッシュ集合を記録する。
旧記録と対応表の移動前集合、現在のコードと移動後集合が、ともに完全一致した場合だけ
旧キャッシュ・ハイブリッドcheckpoint・結果レポートの読込を許容する。
任意のコード変更や一部ファイルの欠落を許容するものではなく、検査を無効化するオプションも設けない。
過去の実験manifestやcheckpoint内部の来歴は書き換えない。

**学習・比較の再開**：移行をまたぐ実験出力先への追記は行わない。
旧中間譜面キャッシュを読み込んで新しい実験出力先に学習することはできる。
同じ新配置で完了したtrialの再利用は従来どおり可能。

```bash
# 旧4件キャッシュを再利用し、新しい出力先で移行後の学習経路を確認
python -m scripts.train.tune_beam_muse --phase smoke --device cpu \
  --cache output/beam_muse_step6_smoke/cache \
  --output output/source_layout_verification/final_smoke --smoke-limit 4

# 既存BeamSearch結果を、元ファイルを変更せず別の場所へ再検証出力
python -m scripts.evaluate.report_arrangement_comparison \
  --output output/source_layout_verification/beamsearch_review
```

## 検証

- 移動前は340件成功・既存5件失敗。移動後は追加12件を含む352件成功・同じ5件失敗。
- 新規テストで依存方向、旧pickleのクラス・実体復元、移行ハッシュの完全照合と変更拒否を検証。
- standard・REMI・CP・連続条件付きの4経路で、移動前の重みを厳密ロードし、
  保存入力に対する出力が要素単位で完全一致。
- 保存済みBeamSearchのval 5,406件・test 6,697件・itest 76件、計12,179件で履歴・評価値が一致。
- 旧ハイブリッドcheckpointからの推論に成功。移行前と同じ制約違反の出力として評価した。
- 影響する16個のCLIの`--help`が成功。
- 単体・相対ハイブリッド・絶対ハイブリッドの小規模学習を移行後に再実行し、
  各checkpointの全テンソルが移行前の同条件学習結果と完全一致。

今回の作業は配置と互換読込の整理であり、教師譜面と保護条件の不整合の解消や
本規模の学習・性能比較は含まない。
