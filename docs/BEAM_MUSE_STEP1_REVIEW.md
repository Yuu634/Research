# ステップ1：骨格特徴と3モデルの構成

`src/musemorphose`は単体モデル、`src/beamsearch`はルール探索、
`src/beam_muse`は両者を接続するモデルの入口とする。
提案された`beam-muse`は通常のPython import構文に使えないため、`beam_muse`とした。
初期実装では旧ディレクトリを保持していたが、追加の整理指示により実装本体を移動した。
共通の骨格取得は`src/common/skeleton.py`に配置する。
現在の構造と旧成果物の読み込み方法は[移行記録](SOURCE_LAYOUT.md)を参照。

骨格は常に`X_before`から取得し、教師譜面やBeamSearchの出力から取得しない。
96位置×8チャネルにKick、Snare、主要Kick（拍頭）、バックビートSnare（2・4拍目）、
Accent、Fill区間、Fill内の最初・最後の打点を格納する。
Snareは`snare_rim`も含む。空のFillに開始・終了打点は作らない。

Fill既知フラグ・曲区間既知フラグと6種類の曲区間one-hotを加え、計776次元とする。
曲区間は`intro/verse/chorus/bridge/outro/other`。未知の注釈文字列はotherとし、
注釈なしは全ゼロ＋既知フラグ0とする。実データのFill代理値やフレーズ位相から
Fill・Aメロ・サビなどを推定しない。明示的な「Fillなし」は`fill_known=True`で区別できる。

確認：`python -m pytest -q tests/test_beam_muse_features.py`。
位置・主要打点・注釈・欠測と既知の空集合の区別を検証する。
