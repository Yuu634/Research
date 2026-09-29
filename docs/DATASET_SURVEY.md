# 学習・評価用ドラム MIDI データセットの調査

調査日: 2026-09-29。対象は MuseMorphose によるドラム譜の難易度制御と、Beam Search を含む生成方式の比較です。
2026-09-29、ユーザーが Lakh MIDI + Groove MIDI + Groove2Groove を採用しました。収集手順は [DATA_COLLECTION.md](DATA_COLLECTION.md) を参照してください。

## 比較表

| 候補 | 内容・規模 | 本タスクとの適合性 | 分割・重複で確認する点 | 公開元の利用条件 |
| --- | --- | --- | --- | --- |
| [Lakh MIDI v0.1](https://colinraffel.com/projects/lmd/) | マルチトラック MIDI 176,581件。全件にドラムがあるわけではない | 長い楽曲と多様な編曲からドラムを抽出する主学習候補 | 破損 MIDI を含む。MD5 重複除去だけでは別編曲・類似曲の漏洩を防げない | CC BY 4.0 として配布。公開元・論文を引用し、各ファイルの出典情報を保持 |
| [Groove MIDI v1.0.0](https://magenta.withgoogle.com/datasets/groove) | 10人の演奏者、1,150 MIDI、約13.6時間の人のドラム演奏 | 演奏の自然さ、速度・強弱、beat/fill の比較に適する。第一候補 | 公式 train/validation/test を保持。独自 Roland ピッチと GM の対応に注意 | CC BY 4.0 |
| [Expanded Groove MIDI v1.0.0](https://magenta.withgoogle.com/datasets/e-gmd) | GMD 由来の1,059系列を43音色で再収録、合計45,537系列 | 既存モデルの再現には有用。MIDI の多様性増加とは区別する | GMD と演奏が重複。公式は記号データのみなら GMD を推奨。両者を独立にランダム分割しない | CC BY 4.0 |
| [Slakh2100](https://github.com/ethman/slakh-generation) | Lakh 由来の2,100楽曲、パート別 MIDI と合成音源 | トラック情報を活用した抽出やマルチトラック比較に有用 | Lakh と独立した評価データとして扱わない。元 MIDI の対応とリリースごとの重複除去を確認 | データは CC BY 4.0、生成コードは MIT |
| [GigaMIDI v2.0.0](https://huggingface.co/datasets/Metacreation/GigaMIDI/blob/main/README.md) | 公開カード記載で2,136,218 MIDI。表情・楽器情報を持つ大規模集合 | 追加学習候補。ドラム抽出・品質選別・処理コストの検証が必要 | Lakh など既存集合を包含。出典横断の重複確認が必要 | CC BY-NC 4.0、アクセス申請・利用条件への同意が必要 |
| [Groove2Groove](https://groove2groove.telecom-paris.fr/) | 1,476基本スタイル×A/Bの合成伴奏。fixed 全9,333ファイル（train 5,733） | スタイル多様性を補う候補。難易度対応ペアではない。追加パーカッションの扱いが必要 | スタイル単位の公式分割を保持。raw/fixed・同一コード進行の重複に注意 | CC BY-NC 4.0（Zenodo APIで確認） |

規模は原データ全体の値です。ドラム抽出・品質確認後の利用可能件数は実測します。
上表の適合性・優先順位は、公開情報と現行ソースを照合した判断です。

## 選定案

当初の第一候補は **Lakh MIDI + Groove MIDI** です。
追加候補 **Lakh MIDI + Groove MIDI + Groove2Groove** の詳細は [Groove2Groove 調査](GROOVE2GROOVE_SURVEY.md) を参照してください。
Groove2Groove は合成伴奏のスタイル多様性を補いますが、人の演奏・難易度正解ラベルの代替にはなりません。その後、ユーザーが3種類の採用を決定しました。
Lakh で楽曲規模の多様性を確保し、Groove で人の演奏を補います。
Groove は公式の分割を維持し、学習には train、モデル選択には validation、最終評価には test を使用します。
Lakh の分割方法は出典・楽曲グループ・ドラム内容の重複を確認して決めます。

最良モデルの既存学習データには E-GMD/Lakh 系が使われているため、新たに分割した test が
その既存モデルにとって未学習であるとは限りません。新規モデルを公平に比較する場合は、
確定した分割で基準モデルも再学習するか、旧学習集合との非重複を確認します。

本サーベイのデータセットには、本研究の5項目の演奏難易度を人手で評価した正解ラベルが
揃っているとは確認できていません。現在の計算式による難易度と、人手による評価は区別します。

## 抽出と変換の方針

- ドラムトラックを抽出した MIDI は、タイミング、強弱、テンポ、拍子をできるだけ保存する。
- 抽出時の音符削除と、モデル入力への量子化・ピッチ変換を分離する。
- 複数のドラムトラックを持つ楽曲は、元トラックの情報をメタデータに残す。
- 現行 standard トークナイザーは4/4と24分割位置を前提とするため、対象外拍子・未対応ピッチ・量子化の影響を記録する。
- Groove の電子ドラム固有ピッチは、元 MIDI を保持したうえで明示的なマッピングを適用する。
- 難易度の境界値を新たに計算する場合は train のみから求め、validation/test には同じ境界値を適用する。

既存の `data/cleaning_lakhmidi.py` はグリッド外音符・一部小節を除外し小節を詰めるため、
原データを保存する抽出とは別の研究用クリーニング処理として保持します。

## 保存構成

```text
data/
├── raw/<dataset>/                  # 入手した原データ
├── metadata/<collection>/          # 出典・分割・採否・語彙・統計
├── extracted/<collection>/         # ドラムのみを抽出した MIDI
└── tokens/<collection>/            # モデル入力のトークン列 pkl
```

MIDI・pkl・大規模メタデータはローカルのデータ領域に保存し、GitHub には処理スクリプト、
調査・選定記録、設定、集計結果と再現手順を push します。
現在の `.gitignore` / `AGENTS.md` に従い、実データ・モデル重みはコミットしません。

## この環境の確認結果

採用した3データセットを、この作業ツリーの `data/raw/` に公式配布元から取得し、チェックサムを確認した。
Lakh の検証済みアーカイブと公式ファイル名対応表は178,561件。概要ページ記載の176,581件との差を記録し、実配布物の件数を使用する。
Groove は1,150件のMIDI-only版、Groove2Groove は全配布物を保存し処理には fixed 9,333件を使用する。
保存先はNAS。実測の抽出・トークン件数と検証結果は [収集手順](DATA_COLLECTION.md) に記録する。

## 進捗

- [x] 候補のサーベイ
- [x] 人手によるデータセット選定（Lakh MIDI + Groove MIDI + Groove2Groove）
- [x] 採用データの取得・完全性確認
- [x] ドラム抽出・メタデータ保存・トークン化
- [x] 件数・分割・出典の検証
- [x] `linear-02/data-create` へ最終成果を push
