# Implementation Tasks

- [x] 1. 生成CLIの入出力契約を拡張する
- [x] 1.1 入力MIDIパスと難易度指定パラメータのバリデーションを追加する
  - 必須引数不足・ファイル不在を実行前に検出して失敗理由を返す。
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 1.2 mode共通の出力契約を定義する
  - standard/remi/cp_limb_v1 で同じCLI引数体系を維持する。
  - _Requirements: 1.4, 7.5_

- [x] 2. 難易度指定3方式の解決ロジックを実装する
- [x] 2.1 全小節指定方式を実装する
  - 入力配列長と小節数の一致を検証する。
  - _Requirements: 2.1, 2.4_

- [x] 2.2 全体シフト方式を実装する
  - 元ラベルへオフセットを適用し、ラベル範囲を安全にクリップする。
  - _Requirements: 2.2, 2.5_

- [x] 2.3 上限値設定方式を実装する
  - cap を超える小節のみ上限へ丸める。
  - _Requirements: 2.3, 2.5_

- [x] 3. (P) mode routing と tokenizer adapter を統一する
- [x] 3.1 standard/remi/cp_limb_v1 の経路選択を共通化する
  - 未対応 mode は利用可能一覧付きで fail-fast する。
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 3.2 mode別 adapter の入出力契約を固定化する
  - 難易度指定の受け口を3モードで統一する。
  - _Requirements: 1.4, 7.5_

- [x] 4. max_bars + stride ベースの小節スライディング生成を実装する
- [x] 4.1 WindowPlanner を追加し max_bars 幅・window_stride_bars 移動量で小節分割する
  - `window_stride_bars <= max_bars` と `window_stride_bars > 0` を検証し、不正値は fail-fast する。
  - 最終窓の不足小節を許容し、順序付き窓列を作成する。
  - _Requirements: 3.1, 3.3_

- [x] 4.2 WindowGenerationOrchestrator を実装する
  - 各窓で target_bars=window_bar_count を設定して生成する。
  - _Requirements: 3.2, 3.4, 4.1, 4.3_

- [x] 4.3 WindowResultAssembler を実装する
  - stride による重複小節領域をマージし、bar順序と総小節数を維持する。
  - 重複解決方針（先行窓優先または後続窓優先）を固定化する。
  - _Requirements: 3.2, 3.4, 4.4_

- [x] 5. 1小節生成バグを修正する
- [x] 5.1 DecoderTerminationGuard を導入する
  - EOS発生時も target_bars 未達なら生成継続する。
  - _Requirements: 4.2, 4.3_

- [x] 5.2 生成結果の小節数検証を追加する
  - 入力 bar 数と一致しない結果は失敗として扱う。
  - _Requirements: 4.4_

- [x] 6. 語彙整合ガードを実装する
- [x] 6.1 standard/remi の語彙サイズ・special token検証を追加する
  - 不一致時は expected/actual をログ出力して停止する。
  - _Requirements: 5.1, 5.2, 5.4_

- [x] 6.2 cp_limb_v1 の multi-head語彙整合検証を追加する
  - event/type/pos/limb と pad index を検証する。
  - _Requirements: 5.3, 5.4_

- [x] 7. 出力保存処理を堅牢化する
- [x] 7.1 保存先親ディレクトリの自動作成を追加する
  - ディレクトリ未存在時に再帰作成し、失敗時は原因を返す。
  - _Requirements: 6.1, 6.2_

- [x] 7.2 MIDI とデバッグテキストの同時保存契約を整備する
  - 同一プレフィックス命名で成果物を保存する。
  - _Requirements: 6.3, 6.4_

- [x] 8. 検証と回帰テストを追加する
- [x] 8.1 (P) 難易度指定3方式の単体テストを追加する
  - full_assign / global_shift / upper_cap の出力を検証する。
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 8.2 (P) max_bars + stride 分割生成の統合テストを追加する
  - 長尺入力で window_stride_bars を変えても全小節が生成されることを検証する。
  - 最終窓短小節ケースと重複窓マージ後の総小節一致を検証する。
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.4_

- [x] 8.3 (P) 1小節生成バグの回帰テストを追加する
  - 複数小節入力で出力小節数が一致することを検証する。
  - _Requirements: 4.2, 4.4_

- [x] 8.4 (P) mode別語彙整合テストを追加する
  - standard/remi/cp_limb_v1 の検証失敗ケースを確認する。
  - _Requirements: 5.1, 5.2, 5.3, 7.1, 7.2, 7.3_
