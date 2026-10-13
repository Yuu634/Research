# BeamSearchの編曲ルール一覧

確認日：2026-10-10。

現在の実装には **65種類の編集ルール** と **11項目の骨格・安全制約** がある。
標準探索の `standard_v1` は、65種類に引数違いの14アクションを加えた **79アクション** を使う。
この文書では、関数名だけでは分かりにくい適用条件・既定引数・実際の編集内容をまとめる。

実装の根拠は [編集ルール](../src/beamsearch/rules.py)、
[探索と標準アクション](../src/beamsearch/beam_search.py)、
[骨格・安全制約](../src/common/safety.py)。
[当初の参照一覧](../drum_arrangement_rule_functions.txt)より、現在の実装を優先して記載した。

## 共通の読み方

対象はstandard表現の4/4・1小節。位置は0〜95の整数で、1拍は24位置、小節長は96位置。
拍頭は `0, 24, 48, 72`、8分間隔は12、3連符の間隔は8、16分間隔は6、32分間隔は3となる。
`start, end` は半開区間 `[start, end)` を表し、終点を含まない。

表の引数は `bar` と内部の保護情報 `ctx` を省略した関数の既定値。
`step`・`offset`・`max_shift` は位置グリッド数、`t` は秒、`r` は0〜1の割合である。
1位置の長さは `60 / BPM / 24` 秒なので、120 BPMでは約20.83 msになる。

| 表中の表記 | 対象 |
| --- | --- |
| HH | `hh_closed`, `hh_open`。ペダルHHは含まない |
| Snare群 | `snare`, `snare_rim` |
| Tom群 | `tom_high`, `tom_mid`, `tom_low`, `floor` |
| シンバル群 | `hh_closed`, `hh_open`, `ride_cymbal`, `ride_bell`, `crash`, `splash`, `china` |
| 手の楽器 | `HAND_INSTRUMENTS` に属する楽器。Snare・Tom・シンバル等 |
| 足の楽器 | `kick`, `hh_pedal` |

分類は編集意図による。例えば音を削除すると、連打速度だけでなく独立性や移動距離も変わり得る。
「簡略化」「追加」といった名前は、実測難易度の増減を保証しない。
探索では編集後の5属性を再計算して採点する。

共通の編集動作は次のとおり。

- **保護音**：`apply_rule` は `locked` と `accents` の音符を編集対象から外す。削除・移動・置換の候補数が減るため、指定割合や間隔が必ず実現するわけではない。
- **追加**：同位置・同楽器に既存音があれば追加しない。小節外への追加もしない。既存音を移動して空きを作る処理は行わない。
- **間引き**：`*_16_to_8`・`*_32_to_16` は対象楽器群の異なる発音位置をまとめ、指定間隔が4打点以上続く区間で2・4・6……番目の位置を削除対象にする。
- **細分化**：`*_8_to_16`・`quarter_to_triplet_*` は、対象の単一楽器で等間隔の打点が2つ以上続く区間に追加する。次の打点がある区間と、終点がちょうど小節末になる区間を細分化する。
- **位置合わせ**：最寄り位置が `max_shift` 以内なら移動し、同距離なら早い位置を選ぶ。移動・置換で同位置・同楽器が重複すると原則拒否される。`simplify_hand_foot_pattern` は例外で、代表音を残して統合する。
- **優先順位**：Accent、2・4拍目のSnare、拍頭Kick、拍頭の音、非Ghostを順に優先する。同条件では早い位置などで決定的に順位付けする。

## 1. 特殊奏法・強弱・HH開閉：14種類

| No. | ルール名 | 既定引数 | 編集内容・適用条件 |
| --- | --- | --- | --- |
| 1 | `remove_ghost_notes` | なし | 楽器を問わず、強弱が `Ghost` の音を削除する。 |
| 2 | `reduce_ghost_note_ratio` | `r=0.5` | Ghost音数をNとして、低優先度から `floor(N*r)` 音を削除候補にする。**rは削除割合**。 |
| 3 | `convert_open_hh_to_closed` | なし | Open HHを同位置・同強弱のClosed HHへ置換し、奏法を通常の `HIT` にする。 |
| 4 | `remove_hh_open_close_alternation` | なし | 時刻順のHHにOpen／Closedの切り替わりが1つでもあれば、Open HHをClosed HHへ統一する。 |
| 5 | `remove_flams` | なし | 明示された `FLAM` を通常打撃へ変更し、対応する装飾グループの副打 `grace` を削除する。対象注釈がなければ適用しない。 |
| 6 | `remove_drag` | なし | 明示された `DRAG` を通常打撃へ変更し、対応する副打を削除する。対象注釈がなければ適用しない。 |
| 7 | `remove_roll` | なし | 奏法 `ROLL` または役割 `roll` の音をグループ別にまとめ、保護・優先順位に基づく代表音を通常打撃へ変更し、他を削除候補にする。グループ情報がなければ音符ID単位。 |
| 8 | `convert_rimshot_to_snare` | なし | 奏法が `RIMSHOT` の音を通常の `snare` へ置換する。`snare_rim` という楽器名だけでは対象にしない。 |
| 9 | `convert_cross_stick_to_snare` | なし | 奏法 `XSTICK` または楽器 `snare_rim` の音を通常の `snare` へ置換する。 |
| 10 | `remove_choke` | なし | 奏法 `CHOKE` を `HIT` に変更する。位置・楽器・強弱は維持する。 |
| 11 | `add_ghost_notes` | なし | 時刻順で隣接する `snare` 2音の間隔が2位置以上なら、中点の切り捨て位置にSnareのGhost音を追加する。`snare_rim` は基準音に含まない。 |
| 12 | `add_open_hh` | なし | 各拍の裏拍、すなわち `position % 24 == 12` のClosed HHをOpen HHへ置換する。 |
| 13 | `add_flams` | `targets=None`, `offset=1` | Snare群・Tom群の対象音を `FLAM` にし、offset位置前へ同楽器のGhost副打を追加する。`targets` は音符ID一覧で、省略時は対象楽器の全編集可能音。副打位置が小節外または既に占有されていれば、その主打は変更しない。 |
| 14 | `add_roll` | `start=None`, `end=None`, `instrument='snare'`, `step=3` | 明示区間にstep間隔で通常打撃を追加し、区間内の同楽器の既存音とともにrollグループを付ける。手の楽器を指定する。区間未指定なら適用しない。 |

Flam・Drag除去では、保護された該当奏法の音を含む装飾グループも編集対象から外す。
近接した通常打撃からFlam・Drag・Rollを推定する処理はない。
単一の奏法トークンに副打の位置情報がなければ、副打を復元せず奏法だけを通常打撃へ変更する。

## 2. 手の連打速度・打撃密度：13種類

| No. | ルール名 | 既定引数 | 編集内容・適用条件 |
| --- | --- | --- | --- |
| 15 | `hh_16_to_8` | なし | HH群の6位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 16 | `hh_32_to_16` | なし | HH群の3位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 17 | `snare_16_to_8` | なし | Snare群の6位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 18 | `snare_32_to_16` | なし | Snare群の3位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 19 | `tom_16_to_8` | なし | Tom群全体の6位置間隔・4打点以上の連続区間を1つおきに間引く。Tomの種類が途中で変わっても対象になる。 |
| 20 | `reduce_hand_roll_density` | `r=0.5` | 役割が `roll` でグループIDのある手の音を、グループごとに `ceil(N*r)` 音残すよう間引く。**rは残存割合**。注釈がなければ適用しない。 |
| 21 | `remove_consecutive_hand_hits` | `n=3` | 手の発音位置が6位置以内の間隔で続く区間について、`n+1` 個ごとの打点を削除対象にする。既定では4・8・12……番目の位置にある手の音を対象にする。 |
| 22 | `limit_hand_min_interval` | `t=0.125` | 手の打点を保護・優先順位順に選び、既に選んだ打点との時間差がt秒未満の打点を削除候補にする。同時打ちは1時刻として扱う。 |
| 23 | `hh_8_to_16` | なし | `hh_closed` の12位置間隔の連続区間に、中間の6位置後へ通常音を追加する。Open HHは対象外。 |
| 24 | `snare_8_to_16` | なし | `snare` の12位置間隔の連続区間に、中間の6位置後へ通常音を追加する。 |
| 25 | `add_hand_double_stroke` | `offset=3` | 手の各音のoffset位置後へ、同楽器・同強弱・奏法 `HIT` の音を追加する。保護音も追加位置の基準にできる。 |
| 26 | `add_hand_roll` | `start=None`, `end=None`, `instrument='snare'`, `step=3` | 指定区間にrollグループ付き打撃を追加する。現在は `add_roll` と同じ内部処理。 |
| 27 | `quarter_to_triplet_hand` | `instrument='hh_closed'` | 指定した手の楽器の24位置間隔の連続区間を3等分し、8・16位置後へ通常音を追加する。 |

最小間隔制限では、同じ時刻に保護音がある打点は保持するため、最終的にt秒未満の間隔が残る場合がある。

## 3. 足の連打速度・打撃密度：13種類

| No. | ルール名 | 既定引数 | 編集内容・適用条件 |
| --- | --- | --- | --- |
| 28 | `kick_16_to_8` | なし | Kickの6位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 29 | `kick_32_to_16` | なし | Kickの3位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 30 | `remove_kick_double` | なし | 間隔6位置以内でまとまるKickの連続区間がちょうど2打の場合、低優先度の1打を削除候補にする。 |
| 31 | `remove_kick_triple` | なし | 間隔6位置以内でまとまるKickの連続区間がちょうど3打の場合、中間の1打を削除候補にする。 |
| 32 | `reduce_kick_density` | `r=0.5` | Kick数をNとして、低優先度から `N-ceil(N*r)` 音を削除候補にする。**rは残存割合**。 |
| 33 | `limit_kick_min_interval` | `t=0.125` | Kickの打点を保護・優先順位順に選び、t秒未満の間隔になる打点を削除候補にする。 |
| 34 | `pedal_hh_16_to_8` | なし | ペダルHHの6位置間隔・4打点以上の連続区間を1つおきに間引く。 |
| 35 | `remove_alternating_feet` | なし | 時刻順で隣接する足の音が異なる楽器で、間隔が1〜6位置なら、その組の低優先度音を削除候補にする。 |
| 36 | `kick_8_to_16` | なし | Kickの12位置間隔の連続区間に、中間の6位置後へ通常音を追加する。 |
| 37 | `add_kick_double` | `offset=3` | 各Kickのoffset位置後へ、同強弱の通常Kickを追加する。 |
| 38 | `add_kick_triplet` | `start=None`, `end=None` | 明示区間を3等分して通常Kickを3つ配置する。区間長は3で割り切れる必要がある。 |
| 39 | `add_double_bass_pattern` | `start=None`, `end=None`, `step=6` | 明示区間の空き位置へstep間隔の通常Kickを追加する。左右の足の割当ては行わない。 |
| 40 | `quarter_to_triplet_foot` | `instrument='kick'` | 指定した足の楽器の24位置間隔の連続区間を3等分し、8・16位置後へ通常音を追加する。 |

## 4. 手足のリズム独立性・同期：13種類

| No. | ルール名 | 既定引数 | 編集内容・適用条件 |
| --- | --- | --- | --- |
| 41 | `align_kick_to_hh` | `max_shift=6` | Kickを最寄りのHH群の発音位置へ移動する。距離がmax_shiftを超える音は変更しない。 |
| 42 | `align_kick_to_beat_grid` | `max_shift=12` | Kickを最寄りの拍頭 `0, 24, 48, 72` へ移動する。次小節の先頭96は移動先に含めない。 |
| 43 | `remove_offbeat_kick` | なし | `position % 24 != 0` のKickを削除する。8分裏拍だけでなく、拍頭以外の位置すべてが対象。 |
| 44 | `remove_syncopated_kick` | なし | `align_kick_to_beat_grid` を既定値で呼ぶ。現在は最大12位置の拍頭への移動であり、削除や音楽的シンコペーションの認識は行わない。 |
| 45 | `merge_hand_foot_onsets` | `max_shift=3` | 足の音を、最寄りの手の発音位置へ移動する。手の音は動かさない。 |
| 46 | `remove_interleaved_hand_foot_hits` | なし | 連続する4打点が手のみ／足のみの交互で、各間隔が6位置以内なら、中間2打点の音から低優先度の1音を削除候補にする。4打点の窓を順に走査する。 |
| 47 | `simplify_hand_foot_pattern` | `step=12` | 全楽器の音を共通のstep格子へ、最大 `step//2` 位置の範囲で移動する。同位置・同楽器の重複は、保護音・優先順位で代表音を残して削除する。 |
| 48 | `reduce_limb_polyrhythm` | なし | 手・足それぞれに3打点以上あり、各群が等間隔、かつ周期が異なり整数倍関係でもない場合、共通12位置格子へ簡略化する。条件外なら適用しない。 |
| 49 | `remove_note_from_simultaneous_hit` | `position=None` | 同時打ちから編集可能な低優先度音を1音削除する。position未指定なら、時刻順で最初の編集可能な同時打ちだけを処理する。 |
| 50 | `add_offbeat_kick` | なし | `12, 36, 60, 84` の空き位置へ通常Kickを追加する。 |
| 51 | `shift_kick_off_grid` | `offset=6` | 拍頭のKickをoffset位置後へ移動する。既定の骨格保護では拍頭Kickが全て保護されるため、通常の `apply_rule` では変更されない。 |
| 52 | `add_hand_foot_interleave` | `start=None`, `end=None`, `step=6` | 明示区間に、Closed HHから始めてKickと交互にstep間隔で通常音を追加する。既存音は残すため、完成譜面全体の交互性は保証しない。 |
| 53 | `add_polyrhythmic_relation` | `start=None`, `end=None` | 明示区間の先頭を基準に、Closed HHを8位置間隔、Kickを12位置間隔で追加する。 |

## 5. 打点間の移動距離・楽器配置：12種類

距離に基づく置換は、v2難易度と共通の楽器間距離表を使う。
各対象音について、異なる時刻の直前・直後の手の音をそれぞれ1つ参照し、距離の合計で置換先を選ぶ。
前後の参照音がなければSnareを基準にする。左右の手の運指は推定しない。
同じルール内の各音の置換先は、置換前の小節を参照して決める。

| No. | ルール名 | 既定引数 | 編集内容・適用条件 |
| --- | --- | --- | --- |
| 54 | `tom_to_nearest_tom` | なし | Tom群の中から前後の参照音への距離合計が最小のTomを選び、合計距離が短くなる場合に置換する。 |
| 55 | `floor_tom_to_tom2` | なし | `floor` を同位置の `tom_mid` へ置換する。Tom2はMid Tomを指す。 |
| 56 | `floor_tom_to_snare` | なし | `floor` を同位置の `snare` へ置換する。 |
| 57 | `crash_to_hh` | なし | `crash` を同位置の `hh_closed` へ置換する。 |
| 58 | `ride_to_hh` | なし | `ride_cymbal` と `ride_bell` を同位置の `hh_closed` へ置換する。 |
| 59 | `replace_far_cymbal` | なし | シンバル群の中から前後の参照音への距離合計が最小の楽器を選び、合計距離が短くなる場合に置換する。 |
| 60 | `reduce_tom_span` | なし | Tomの置換先を `tom_high`, `tom_mid` に限定する。範囲外のTomは距離が減らなくても置換し、範囲内のTomは距離が短くなる場合に置換する。 |
| 61 | `remove_large_jump` | `threshold=0.6` | 前後参照音への距離の最大値が閾値を超える手の音について、`snare`, `hh_closed`, `tom_mid` から距離合計を短くできる楽器へ置換する。現在は削除しない。 |
| 62 | `simplify_tom_fill` | なし | 明示された `fill_regions` 内だけで `reduce_tom_span` を適用する。Fill区間がなければ適用しない。 |
| 63 | `expand_tom_span` | なし | 編集可能なSnare群・Tom群を時刻・ID順に並べ、High TomとFloor Tomへ交互に置換する。 |
| 64 | `snare_to_tom_sequence` | なし | 編集可能なSnare群を時刻・ID順に並べ、2・4・6……番目だけをTomに置換する。0始まりの元の列添字kに対し、`(tom_high, tom_mid, floor)[k % 3]` を使うため、置換先はMid→High→Floorの繰り返し。 |
| 65 | `add_large_jump` | なし | Snare群・Tom群を、`tom_high`, `floor`, `crash`, `ride_cymbal` の中で前後参照音への距離合計が最大の楽器へ置換する。現在は音符を追加せず、難易度増加を必須条件にもしていない。 |

この節の楽器置換では、位置・強弱を保持し、奏法を通常の `HIT` に変更する。

## 6. 骨格・安全制約：11項目

以下は候補譜面を編集する65ルールとは別の、`Guards` を設定する関数である。
標準79アクションには含まれない。

| 関数 | 既定の対象・制限 | 動作 |
| --- | --- | --- |
| `preserve_main_kick` | 全拍頭のKick | 対象IDを `locked` に追加する。`ids` によるKick IDの明示指定も可能。 |
| `preserve_snare_backbeat` | 位置24・72のSnare群 | 2・4拍目のSnareを `locked` に追加する。 |
| `preserve_downbeat` | 位置0・24・48・72の各1音 | 各拍頭で優先順位が最も高い音を `locked` に追加する。 |
| `preserve_accent` | 全Accent音 | 対象IDを `accents` に追加する。適格性検査では削除・位置変更・弱化を禁止し、候補生成では編集対象から外す。 |
| `preserve_fill_boundary` | 各明示Fill区間内の最初・最後の実打点 | その打点にある音をすべて `locked` に追加する。区間内に音がなければ追加しない。 |
| `preserve_bpm` | `keep_bpm=True` | 原譜とBPMが異なる候補を拒否する。 |
| `preserve_meter` | `keep_meter=True` | 原譜と拍子が異なる候補を拒否する。内部表現も4/4に限定。 |
| `preserve_bar_length` | `keep_length=True` | 原譜と小節長が異なる候補を拒否する。内部表現も96位置に限定。 |
| `limit_added_notes` | `Guards.max_added=8` | 原譜に対する累積追加音数の上限をnに設定する。 |
| `limit_deleted_notes` | `Guards.max_deleted=16` | 原譜に対する累積削除音数の上限をnに設定する。 |
| `limit_edit_ratio` | `Guards.max_edit_ratio=0.5` | 原譜に対する累積編集率の上限をrに設定する。 |

`default_guards(original)` は `Guards` の既定値に、表の先頭5つの保護設定を加える。
`locked` の音は削除・位置・楽器・強弱・奏法の変更を禁止する。
このほか同位置・同楽器の重複と、同位置で手の音が3つ以上鳴る候補も拒否する。

編集率は音符IDで対応を取り、次のように計算する。

```text
編集率 = (追加音数 + 削除音数 + 変更音数) / max(1, 原譜の音符数)
```

同じIDの位置・楽器・強弱・奏法・強弱明示フラグのいずれかが変わると1変更と数える。
複数の項目を変えても同じ音符なら1変更で、装飾グループ等の注釈だけの変更は数えない。
制限は1ルールごとの編集量ではなく、探索開始時の原譜に対する累積編集量に適用する。

## 7. 標準探索で使用する引数

各ルールを基本的に既定引数で1回ずつ登録する。ただし、次の6ルールは
`start=72, end=96`、すなわち第4拍の区間を明示して登録する。

```text
add_roll
add_hand_roll
add_kick_triplet
add_double_bass_pattern
add_hand_foot_interleave
add_polyrhythmic_relation
```

これは追加処理のために指定した区間であり、Fillの自動認識結果ではない。
関数を直接呼び、`start` または `end` が `None` のままなら、区間が必要なため適用されない。

さらに以下の引数違いを探索候補に加える。

| ルール | 既定値 | 追加する値 | 合計アクション数 |
| --- | --- | --- | --- |
| `reduce_ghost_note_ratio` | `r=0.5` | `r=0.25, 0.75` | 3 |
| `reduce_kick_density` | `r=0.5` | `r=0.25, 0.75` | 3 |
| `limit_hand_min_interval` | `t=0.125` | `t=1/12, 1/6, 0.25` 秒 | 4 |
| `limit_kick_min_interval` | `t=0.125` | `t=1/12, 1/6, 0.25` 秒 | 4 |
| `simplify_hand_foot_pattern` | `step=12` | `step=6, 24` | 3 |
| `merge_hand_foot_onsets` | `max_shift=3` | `max_shift=1, 6` | 3 |

追加分は `2+2+3+3+2+2=14` なので、合計は `65+14=79` アクションになる。
`add_roll` と `add_hand_roll` のように同じ出力を作り得る別名のルールも個別に登録されるが、
探索側では同じ候補状態の重複を除く。

確認時点の [選定済み設定](../config/beamsearch_selected.json) は `beam8_d3_target` で、
この標準79アクションと一致している。探索幅8、最大深さ3、ルール試行上限5,000を使用する。
`SearchConfig` 自体の既定値である探索幅4・最大深さ2とは区別する。

## 8. 適用結果と探索への採用

安全制約付きの入口は `apply_rule`。個別の編集関数を直接呼ぶと、既定の骨格保護や
累積編集制限は自動では適用されない。

| 結果 | 意味 |
| --- | --- |
| `applied` | 変更があり、適用時の安全制約を通過した。後続の採点対象になる。 |
| `unchanged` | 対象音がない、保護で変更できない、必要な区間・奏法注釈がない等により変更しなかった。 |
| `rejected` | 候補が安全制約に違反した。変更を採用せず、適用前の譜面と違反理由を返す。 |

引数の不正などは例外として扱い、`unchanged` とは区別する。
`applied` でも、後続の採点で語彙外トークン・難易度算出不能などが検出されると探索候補に残らない。
とくにFlamの副打や特殊奏法は、内部表現で作れても使用語彙で表現できるかの確認が必要となる。

探索は適格な候補を難易度誤差・骨格保持・ルール数・編集量のスコアで比較する。
原譜自体も最良候補として初期化されるため、編集が採用されず原譜が返る場合がある。
ルールが登録されていること、ある入力で適用できること、最終出力に採用されることはそれぞれ異なる。

## 確認用コマンド・関連資料

リポジトリ直下から実行する。

```bash
python -m scripts.evaluate.review_arrangement_rules --help
python -m pytest -q tests/test_arrangement_rules.py tests/test_arrangement_beam_search.py
```

各ルールの人工例・変更前後の打点図・MIDIを生成する場合は、次を実行する。

```bash
python -m scripts.evaluate.review_arrangement_rules
```

生成先は `output/beamsearch_step1/`。人工例は編集動作を確認するためのもので、実データでの採用率ではない。
既存の詳細は [ステップ1確認資料](BEAMSEARCH_STEP1_REVIEW.md)、
[ステップ5の探索・選定資料](BEAMSEARCH_STEP5_REVIEW.md)、
[研究用ツール一覧](RESEARCH_TOOLS.md)を参照。
