# 学習・評価データ収集

## 採用

2026-09-29、ユーザーが Lakh MIDI、Groove MIDI、Groove2Groove の3種類を採用した。
比較は [サーベイ](DATASET_SURVEY.md)、Groove2Groove の内容は [詳細調査](GROOVE2GROOVE_SURVEY.md) を参照。

## 保存先

```text
data/
├── raw/
│   ├── lakh/              # 公式アーカイブ、lmd_full、元ファイル名対応表
│   ├── groove/            # 公式MIDI-onlyアーカイブ、groove/info.csv等
│   └── groove2groove/     # 公式アーカイブ、raw/fixed MIDI・コード譜・スタイル表
├── extracted/drum_collection_v1/<dataset>/<ID>.mid
├── tokens/drum_collection_v1/<ID>.pkl
└── metadata/drum_collection_v1/
    ├── policy.json        # 出典・ライセンス・変換方針
    ├── manifest.jsonl     # 入力1件につき1行、原本と成果物を関連付け
    ├── summary.json       # 件数・除外理由・分割の集計
    ├── validation.json    # 成果物の全件検証
    ├── drum_vocab.pkl
    ├── train_split.pkl / val_split.pkl / test_split.pkl / itest_split.pkl
    └── <dataset>_<split>_split.pkl
```

データ本体・大規模メタデータ・pkl はローカル保存し、既存の Git 除外設定を維持する。
GitHub には処理スクリプト、テスト、調査・採用記録、再現方法、集計結果を保存する。

## 再現コマンド

リポジトリ直下で実行する。取得には curl、処理には requirements.txt の Python 依存を使用する。
Linux のプロセス並列とファイル単位タイムアウトを利用する。
Lakh の変換は検証済みアーカイブから直接読み出すため、NAS 上の18万件のファイル走査を必要としない。

```bash
python -m scripts.preprocess.download_drum_sources
python -m scripts.preprocess.build_drum_collection --workers 16
python -m scripts.preprocess.validate_drum_collection --workers 16 --sample-per-group 16
```

ダウンロードは `.part` から再開し、チェックサム確認後に正式名へ変更する。
Groove は公式 SHA-256、Groove2Groove は公式 MD5、Lakh は MusPy の配布定義の SHA-256 と照合する。
出典 URL・ライセンス・実測 SHA-256 は `data/raw/<dataset>/acquisition.json` に記録する。
取得・処理済みの結果は再利用する。処理方針を変える場合は別の `--collection` 名を使う。

## 抽出方針

- MIDI チャンネル10（0始まりでは9）の全トラックを対象とする。トラック名だけでは判定しない。
- 複数ドラムトラックの音符を抽出 MIDI に保持し、テンポ、拍子、強弱、位置、チャンネルの制御イベントを残す。
- Groove2Groove は fixed 版を対象に先頭2小節（8四分音符）のカウントインを除く。境界前のテンポ等は開始時点へ引き継ぐ。
- Lakh/Groove の先頭無音や途中の休符を詰めない。元ファイルは raw にそのまま残す。
- ピッチ制限はトークン作成時に行う。コンガ等を含む抽出 MIDI は保存する。
- ドラムがないファイル、壊れた MIDI、非同期 type 2、非 PPQ 形式は理由付きで記録する。
- 標準チャンネル以外へ独自仕様でドラムを割り当てた MIDI の推定検出は行わない。

## トークン化

`config/drum_config_standard-lakh1-kl14.yaml` の語彙と互換の standard 形式を利用する。
モデル本体・基準 YAML・既存トークナイザーは変更しない。
pkl は既存データローダーで扱える `(bar_positions, token_indices)` の2要素タプル。
ファイル名は数値 ID とし、語彙と分割リストを token ディレクトリから分離する。

取得データの TPB を勝手に推定し直さず、フラムと推測した隣接音符を統合しない専用アダプター
`standard_declared_tpb_no_flam_v1` を使う。宣言 TPB に基づいて既存の24分割位置・ベロシティ分類・テンポ量子化を適用する。
既存語彙に Ghost がある楽器では弱打を保持し、未対応の強弱だけ Normal に対応付ける。
メモリ上のノートから直接変換して、作業用 MIDI の再保存による同一ピッチの重なりの変化を避ける。
テンポは既存語彙の60〜240 BPM・5刻みに変換されるため、元 MIDI のテンポは別途残す。
音長0の打撃が最後の小節境界にあっても、開始時刻に基づいて小節を確保し、音符を保持する。
これは旧データの前処理を完全再現するプロファイルではない。

Groove の電子ドラム固有ピッチは[公式対応表](https://magenta.withgoogle.com/datasets/groove)を基に変換する。
22→42、26→46、58→43 に加え、タム、スネアリム、シンバルエッジの対応をコードとメタデータに明示する。
既存語彙が区別できる HH ペダルとライドベルは維持する。GM の Lakh/Groove2Groove にはこの変換を適用しない。

4/4以外を含むファイルは抽出 MIDI のみ保存する。拍子メタ情報がない場合は MIDI の既定4/4を使う。
未対応ピッチ、完全なノートとして解析できないイベント、変換で落ちた音符は入力ごとに記録する。
処理量が極端な入力（50万ノート超・4,096小節超）や120秒を超える処理も除外理由を残す。

**この Issue で作る pkl には難易度ラベルを付与しない。** 人手の正解ラベルは付属せず、
計算式とクラス境界の選定は収集と別の工程である。読み込み確認では `use_difficulty=False` を使用する。
既存の難易度条件付き学習を始める前に、確定 train のみで境界値を決めてラベルを付与する必要がある。

## 分割と重複

Groove/Groove2Groove は公式分割を保持する。Groove2Groove の itest は独立した追加評価用リストとし、学習に混ぜない。
Lakh はトークン列の SHA-256 に基づく再現可能な80/10/10の分割を作る（実件数が厳密にこの割合になるとは限らない）。

完全一致するトークン列は同一グループとし、分割リストへ代表1件だけを掲載する。
公式分割があるグループでは公式側を代表に選ぶ。複数の公式分割に跨る完全一致グループは、双方を分割リストから除外する。
原本・抽出 MIDI・トークンは除外後も保存し、`split_exclusion` と `duplicate_of` を記録する。

この検査は**ファイル全体のトークン完全一致**が対象。別編曲、部分一致、速度違い、長さ違い、類似曲の非重複は保証しない。
Lakh の楽曲・アーティスト単位の厳密な独立評価や、旧学習集合（E-GMD 等）との照合は追加の研究設計が必要。
データセット別の分割リストも保存するため、合成データが多すぎる場合はサンプリング比率を調整できる。
学習・評価では対応する `metadata/drum_collection_v1/*_split.pkl` を読み込み、データローダーの `pieces` に指定する。

## 検証

`validate_drum_collection` は全入力の記録、全出力ファイル一覧、生成時に記録した対応音符数、
公式分割の維持、分割間の完全一致トークン重複がないことを全件確認する。

ファイル内容の再読み出しでは、抽出 MIDI の SHA-256 と pkl の語彙範囲・小節開始・EOS・PAD不在・対応音符数・内容ハッシュを確認する。
`--sample-per-group 16` はデータセット・分割・処理結果ごとに ID の SHA-256 が小さい16件を選ぶ。
`--include-ids` で修正対象などの必須確認IDを追加できる。`--sample-per-group` を省略すると全ファイルを再読み出す。

今回の環境ではNASの1ファイル読み出しが中央値約2.6秒だったため、最終確認は全件の一覧・メタデータ・分割確認と、
層別サンプル＋末尾音符を修正した186件の再読み出しで実施した。**全ファイル内容の再読み出し検証を完了したとは扱わない。**
実際に確認したID・件数・範囲を `validation.json` に保存する。
処理対象外ファイルは正常成果物として数えず、manifest と summary の理由を確認する。

関連テスト:

```bash
python -m pytest -q \
  tests/test_build_drum_collection.py \
  tests/test_survey_groove2groove.py \
  tests/test_tokenization_policy.py \
  tests/test_tokenization_mode_validation.py
```

取得元のコードとデータのライセンスは別に管理する。
Lakh/Groove は CC BY 4.0、Groove2Groove は CC BY-NC 4.0。元の出典情報と引用情報を保持する。

## 実測結果（2026-09-29）

公式配布物を取得して全189,044件を処理した。以下は小節断片数ではなく、元MIDI単位のファイル数。

| データセット | 入力 | 抽出 MIDI | トークン pkl |
| --- | ---: | ---: | ---: |
| Lakh MIDI | 178,561 | 118,255 | 95,345 |
| Groove MIDI | 1,150 | 1,150 | 1,138 |
| Groove2Groove | 9,333 | 9,006 | 8,994 |

Lakh の入力178,561件は、チェックサム検証済みアーカイブと公式ファイル名メタデータの実件数。概要ページ記載の176,581件とは差がある。

完全一致するトークン列を重複除外した分割リスト:

| データセット | train | val | test | itest |
| --- | ---: | ---: | ---: | ---: |
| Lakh MIDI | 58,765 | 7,422 | 7,348 | 0 |
| Groove MIDI | 886 | 122 | 129 | 0 |
| Groove2Groove | 5,512 | 1,114 | 1,192 | 1,112 |

主な対象外・除外理由:

- Lakh: ドラムなし56,225件、解析不可または非対応形式4,081件。抽出できた中で、非4/4は21,931件、対応ピッチなし946件、完全なノートなし23件、処理量上限超過10件をトークン化対象外とした。
- Groove: 非4/4の12件をトークン化対象外とした。
- Groove2Groove: 本編ドラムなし327件、対応ピッチなし12件。
- 完全一致トークンの重複除外は Lakh 21,810件、Groove 1件、Groove2Groove 64件。原本・抽出 MIDI・pkl は保持した。

全件集計で末尾の小節境界にある音長0の打撃の欠落を検出し、小節数の計算を修正した。
該当186ファイルを再処理して399音符を復元し、解析済みの対応音符について追加欠落は0件となった。
境界音符、再分割、Unicode改行を含むメタデータの再開処理を含め、関連テスト33件が通過した。
原 MIDI に含まれる語彙外音符や、未完結ノートイベントの扱いは manifest の各カウントで区別する。

最終検証は全189,044件のメタデータ、128,411 MIDI / 105,477 pkl のファイル一覧、
全分割リストを確認した。再読み出し対象508レコード中、成果物がある428 MIDIと346 pklを内容検証し、
修正186件もすべて通過した。全件の生成時記録で対応音符の追加欠落は0、分割間の完全一致トークン重複は0。
近似重複と全ファイル内容の再読み出しは検証範囲に含めていない。
3データセットそれぞれのサンプルを既存データローダーでも読み込み、encoder `(16, 128)` / decoder `(1280,)` の入力を確認した。

今回の再読み出し検証コマンド（修正記録は今回の実行に付属）:

```bash
python -m scripts.preprocess.validate_drum_collection --workers 16 \
  --sample-per-group 16 \
  --include-ids data/metadata/drum_collection_v1/boundary_note_repair.json
```

## 難易度ラベルの追加（2026-09-30）

この収集結果を入力として、[承認済みの属性v2](ATTRIBUTE_FORMULA_REVIEW.md)による
ラベリングを別ディレクトリ `data/processed/drum_collection_difficulty_v2/` へ実施した。
元のトークン・分割は保持し、移動モデルの適用外となる曲を除外した分割リストは
`data/metadata/drum_collection_difficulty_v2/` に置く。
利用時は[実施結果と除外の内訳](ATTRIBUTE_V2_RESULTS.md)を確認すること。
