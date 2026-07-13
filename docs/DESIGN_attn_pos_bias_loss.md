# 設計書: Attention Bias Loss への Position モード追加（`pos_over_other` / `other_over_pos`）

作成日: 2026-07-08
対象ブランチ: `data_revise`
関連設計書: `docs/DESIGN_attn_bias_loss.md`（本機能の基盤）

**修正対象ファイル: `scripts/train/train_drum.py`, `src/models/position_bias.py`, `config/*` のみ**
（`src/models/limb_bias.py` / `src/models/musemorphose.py` / `src/models/transformer_encoder.py`
は**無改変**）

---

## 1. 目的・背景

既存の Attention Bias Loss（`DESIGN_attn_bias_loss.md`）は、config の
`training.bias_loss.decoder_layer_modes` / `encoder_layer_modes` に

- `limb_over_other` : `max(0, margin − (Attn_limb − Attn_other))`
- `other_over_limb` : `max(0, margin − (Attn_other − Attn_limb))`
- `none`            : 損失なし

の 3 モードを持ち、**同一四肢ペア（Attn_limb）** と **異四肢ペア（Attn_other）** の注意対比を
hinge 損失で誘導する。

本機能は、`limb_bias` の要素と対になる **position 軸（同一タイミング打撃）** のモードを
**同じ layer_modes リストに追加**し、層ごとに「同四肢 / 同タイミング / 全体(=`none`)」の
役割を割り当て可能にする。これにより、役割分担型スケジュール（例: Decoder =
`pos, pos, limb, limb, none, none`）を **config だけで** 構成できる。

### 追加モード

| mode 文字列 | 損失式 | 意図 |
|---|---|---|
| `pos_over_other` | `max(0, margin − (Attn_pos − Attn_otherpos))` | 同一タイミング間の注意を margin 分だけ優位にする |
| `other_over_pos` | `max(0, margin − (Attn_otherpos − Attn_pos))` | 異タイミング間の注意を margin 分だけ優位にする |

- **Position-Group** は `position_bias.py` の `build_position_ids` が与える「直前の `<POS_X>`
  イベントに属するノート群」。同一グループ = 同一タイミングの打撃。
- **Attn_pos** = 同一 Position-Group の非メタペア平均注意、
  **Attn_otherpos** = 異なる Position-Group の非メタペア平均注意（limb の `other` とは別集合）。
- position 軸は limb 軸と直交（`position_bias.py` 冒頭 docstring の設計方針）。
  **1 層につきモードは 1 つ**（limb 系 / pos 系 / `none` のいずれか）。

### 要件と対応

| # | 要件 | 対応 |
|---|------|------|
| R1 | 修正は `train_drum.py` / `position_bias.py` / `config` のみ | **達成**。マスク生成・損失は position_bias.py、配線は train_drum.py |
| R2 | `limb_bias` と同様に position 要素を layer_modes に追加 | `pos_over_other` / `other_over_pos` を同リストに追加（§3.1, §3.2a） |
| R3 | 既存 limb モード・`none` と混在可能 | 1 層 1 モード。limb/pos/none を任意に配置可（§4） |
| R4 | 後方互換 | pos モード未使用・`bias_loss` キー不在時は現行挙動と完全一致（§3.3, §6） |

---

## 2. 全体アーキテクチャ

```
train_drum.py (_apply_bias_loss)
  │  maps_dec/maps_enc = capture.pop_maps()          # 既存: mode!='none' の全層の注意重み（勾配付き）
  │
  ├─ [limb 分岐] 既存
  │    M_limb, M_other = build_limb_pair_masks(build_limb_ids(...))
  │    dl, _ = compute_attn_bias_loss(maps, M_limb, M_other, modes, margin)   # limb 系層のみ処理
  │
  └─ [pos 分岐] 新規（pos_token_ids が非 None のときのみ）
       pos_ids = build_position_ids(tokens, **pos_token_ids, limb_lookup_table=lookup)   # 既存関数
       M_pos, M_other = build_position_pair_masks(pos_ids, causal=...)                    # 新規
       pl, _ = compute_attn_pos_bias_loss(maps, M_pos, M_other, modes, margin)            # 新規: pos 系層のみ処理

  bias_loss = mean(dec-limb, enc-limb, dec-pos, enc-pos)   # active 項のみ
  total_loss += λ * bias_loss
```

- **新規モデルパラメータなし**。state_dict は完全互換。
- `AttentionWeightCapture`（`limb_bias.py`）は役割非依存で `mode != 'none'` の全層を捕捉するため
  **無改変で共有**。捕捉した maps を limb/pos 両方の compute 関数に渡す。

---

## 3. 変更内容

### 3.1 `src/models/position_bias.py`（追加のみ・既存関数は無変更）

#### (a) モード定数

```python
BIAS_MODE_POS_OVER_OTHER = 'pos_over_other'
BIAS_MODE_OTHER_OVER_POS = 'other_over_pos'
POS_BIAS_MODES = (BIAS_MODE_POS_OVER_OTHER, BIAS_MODE_OTHER_OVER_POS)
```

#### (b) `build_position_pair_masks(pos_group_ids, causal=False) -> (M_pos, M_other)`

`limb_bias.build_limb_pair_masks` と同型。

```python
def build_position_pair_masks(pos_group_ids, causal=False):
    """pos_group_ids [B, N] → (M_pos, M_other) 各 [B, N, N] float。
    M_pos  : 同一 Position-Group の非メタペア（build_position_bias_mask と同一判定）
    M_other: 両端が非メタ かつ 異なる Position-Group のペア
    causal=True で下三角のみ有効（デコーダの因果 Attention 用）。
    """
```

- `ids_i = pos_group_ids.unsqueeze(2)`, `ids_j = pos_group_ids.unsqueeze(1)`。
- `has_no_pos = (ids_i == 0) | (ids_j == 0)`（`build_position_ids` はメタ/未出現を 0 にする）、
  `non_meta = ~has_no_pos`。
- `M_pos   = (ids_i == ids_j) & non_meta`（`build_position_bias_mask` と一致）。
- `M_other = (ids_i != ids_j) & non_meta`。
- `causal=True` のとき `torch.tril` を両マスクに適用（掛けないと上三角ペアが平均の分母を希釈）。
- 返り値は `float()`。

#### (c) `compute_attn_pos_bias_loss(attn_maps, M_pos, M_other, layer_modes, margin) -> (loss, metrics)`

`limb_bias.compute_attn_bias_loss` と同型だが **pos 系モードのみ処理**。

```python
def compute_attn_pos_bias_loss(attn_maps, M_pos, M_other, layer_modes, margin):
    device = M_pos.device
    denom_pos   = M_pos.float().sum().clamp_min(1.0)
    denom_other = M_other.float().sum().clamp_min(1.0)
    hinges, metrics = [], {}
    for idx, A in attn_maps.items():
        mode = layer_modes[idx]
        if mode not in POS_BIAS_MODES:      # limb 系・none はここでスキップ
            continue
        Af = A.float()                       # AMP 対策で fp32 総和
        attn_pos   = (Af * M_pos).sum()   / denom_pos
        attn_other = (Af * M_other).sum() / denom_other
        metrics[f'attn_pos/{idx}']      = attn_pos.item()
        metrics[f'attn_otherpos/{idx}'] = attn_other.item()
        if mode == BIAS_MODE_POS_OVER_OTHER:
            hinge = torch.clamp(margin - (attn_pos - attn_other), min=0.0)
        else:  # BIAS_MODE_OTHER_OVER_POS
            hinge = torch.clamp(margin - (attn_other - attn_pos), min=0.0)
        hinges.append(hinge)
    loss = torch.stack(hinges).mean() if hinges else torch.zeros((), device=device)
    return loss, metrics
```

- metrics キーは limb（`attn_limb/<idx>`, `attn_other/<idx>`）と衝突しないよう
  `attn_pos/<idx>`, `attn_otherpos/<idx>` とする。

### 3.2 `scripts/train/train_drum.py`

#### (a) モード検証の拡張（`_normalize_layer_modes`, `train_drum.py:~290-315`）

- `position_bias` から `POS_BIAS_MODES` を import。
- 許容集合を **train_drum 側で** 和集合として定義（`limb_bias.py` は変更不可）:
  ```python
  ALL_BIAS_MODES = tuple(VALID_BIAS_MODES) + POS_BIAS_MODES
  ```
  検証ループ（`310-314`）の `VALID_BIAS_MODES` を `ALL_BIAS_MODES` に置換。
- 「全層 none」警告（`358-360`）はそのまま（pos モードは `none` ではないので無影響）。

#### (b) pos モード時の前提リソース構築（`train_drum.py:1280-1342` 付近）

- 正規化後の `dec_modes`/`enc_modes` から
  `bias_loss_uses_pos = any(m in POS_BIAS_MODES for m in dec_modes + enc_modes)` を算出。
- `pos_token_ids` 構築のゲート条件（`1307`）に
  `or (bias_loss_enabled and bias_loss_uses_pos)` を追加。
- `pos_token_ids` 辞書に **`'bar': runtime_opts['bar_token_idx']`** を追加
  （`build_position_ids` の `bar_token_idx` 引数に必要。既存の additive position bias 経路は
  このキーを参照しないため無影響のキー追加）。
- **POS/BEAT トークンが見つからない場合**: 警告を出し、`dec_modes`/`enc_modes` 内の pos モードを
  `BIAS_MODE_NONE` に降格（`token2idx` 欠如・POS/BEAT 欠如の両ケース）。全層 none 化時は既存の
  「all none」経路に合流し bias_loss は無効化。

#### (c) `bias_ctx` の拡張（`train_drum.py:1454-1470`）

- `'pos_token_ids'` を格納（bar 追加済み。pos モード未使用/構築失敗時は `None`）。
- `capture_dec`/`capture_enc` は既存どおり `mode != 'none'` の全層に付与（pos 層も自動対象）。

#### (d) `_apply_bias_loss` に pos 分岐を追加（`train_drum.py:459-478`）

limb ブロックの直後、`bias_ctx.get('pos_token_ids')` が非 None のときのみ:

```python
pt = bias_ctx['pos_token_ids']
margin = bias_ctx['margin']
if pt is not None:
    if maps_dec:
        dec_pos_ids = build_position_ids(
            batch_dec_inp,                    # [dec_seqlen, B]（sequence-first）
            pos_token_start=pt['pos_start'], pos_token_end=pt['pos_end'],
            beat_token_start=pt['beat_start'], beat_token_end=pt['beat_end'],
            bar_token_idx=pt['bar'], pad_token_idx=pt['pad'],
            limb_lookup_table=bias_ctx['lookup'], eos_token_idx=pt['eos'])
        Md_pos, Md_other = build_position_pair_masks(dec_pos_ids, causal=True)
        dpl, dpm = compute_attn_pos_bias_loss(
            maps_dec, Md_pos, Md_other, bias_ctx['dec_modes'], margin)
        bias_terms.append(dpl)
        metrics.update({f'dec/{k}': v for k, v in dpm.items()})
    if maps_enc:
        enc_tokens_flat = batch_enc_inp.reshape(batch_enc_inp.size(0), -1)
        enc_pos_ids = build_position_ids(enc_tokens_flat, ... , limb_lookup_table=bias_ctx['lookup'])
        Me_pos, Me_other = build_position_pair_masks(enc_pos_ids, causal=False)
        epl, epm = compute_attn_pos_bias_loss(
            maps_enc, Me_pos, Me_other, bias_ctx['enc_modes'], margin)
        bias_terms.append(epl)
        metrics.update({f'enc/{k}': v for k, v in epm.items()})
```

- **maps は limb 経路と共有**: `pop_maps()`（`453-454`）は関数冒頭で 1 回だけ呼び、
  戻り値 `maps_dec`/`maps_enc` を limb・pos 両方の compute に渡す（`pop_maps` を二度呼ばない）。
- 既存の `bias_terms` 平均・`losses['total_loss'] += lam * bias_loss`（`479-484`）はそのまま。
  最大 4 項（dec-limb / enc-limb / dec-pos / enc-pos）が平均される。

#### (e) import 追加（`train_drum.py` 冒頭）

```python
from src.models.position_bias import (
    build_position_ids, build_position_pair_masks,
    compute_attn_pos_bias_loss, POS_BIAS_MODES,
)
```

### 3.3 config（`config/*`）

- `bias_loss` ブロックの `decoder_layer_modes` / `encoder_layer_modes` コメントに 2 モードを追記:
  - `'pos_over_other'` : `max(0, margin - (Attn_pos - Attn_otherpos))`
  - `'other_over_pos'` : `max(0, margin - (Attn_otherpos - Attn_pos))`
- 役割分担型の例 config を 1 つ追加（例: `config/drum_config_standard-lakhaug1-biasloss-role.yaml`）:

```yaml
training:
  bias_loss:
    enabled: true
    margin: 0.05
    # 'limb_over_other' / 'other_over_limb' / 'pos_over_other' / 'other_over_pos' / 'none'
    decoder_layer_modes: ['pos_over_other', 'pos_over_other', 'limb_over_other',
                          'limb_over_other', 'none', 'none']
    encoder_layer_modes: ['none', 'none', 'limb_over_other',
                          'limb_over_other', 'pos_over_other', 'pos_over_other']
    max_weight: 1.0
    sched: 'linear_warmup'
    no_loss_steps: 1000
    warmup_steps: 10000
    cycle_steps: 10000
```

- 既存 config（`bias_loss` キー不在 / pos モード未使用）は**完全後方互換**。

---

## 4. limb_bias.py 無改変が成立する根拠

- `compute_attn_bias_loss`（`limb_bias.py:339-355`）は `layer_modes[idx]` が limb 系でも `none`
  でもない場合、hinge 分岐の `else: continue`（`353-354`）に落ち、**その層を損失に加えない**。
  よって pos モードを含む modes リストを渡しても、limb 損失は limb 系層のみを対象にする。
  - 副作用: pos 層に対して `attn_limb/<idx>`・`attn_other/<idx>` の metrics が計算・記録される
    （ログのみ・損失非寄与）。実害なしのため許容。
- 新規 `compute_attn_pos_bias_loss` は `mode not in POS_BIAS_MODES → continue` で limb 層・`none`
  層をスキップし、pos 層のみ処理する。
- 両者は同じ maps を受け取り、**モードで自層のみを選別**するため、1 層 1 モードの前提で
  二重計上は起きない。

---

## 5. 既存機能との相互作用

| 既存機能 | 相互作用 |
|---|---|
| bias_loss（limb モード） | 独立。同一 layer_modes 内に混在可。maps を共有し、モードで選別 |
| Limb-Relation Bias（加算, `use_limb_bias_*`） | 独立。併用可（加算バイアス後の分布に対し hinge が評価される） |
| Position-Same-Time Bias（加算, `use_position_bias_*`） | 独立。`pos_token_ids` 構築ロジックを共有するが、bias_loss 経路では `'bar'` キーを追加して `build_position_ids` に渡す |
| AMP (`use_amp`) | 捕捉重みは fp16。損失内部で `.float()` 化（§3.1c） |
| checkpoint 互換 | 新規パラメータなし。strict=True で相互ロード可 |
| Early Stopping / best model 選択 | 判定は従来どおり再構成損失のみ。pos bias_loss も選択基準に混ぜない |
| cp_limb_v1 | 非対応。既存 `_validate_bias_loss_config`（`train_drum.py:353-356`）の ValueError に従う |

---

## 6. テスト・検証計画

### 6.1 単体（`tests/test_attn_bias_loss.py` へ追記）
- `build_position_pair_masks`: 手作り `pos_group_ids`（メタ=0 / 複数グループ / causal 有無）で
  M_pos・M_other の相互排他・メタ除外・下三角を確認。M_pos が `build_position_bias_mask` と一致。
- `compute_attn_pos_bias_loss`: 一様注意（Attn_pos==Attn_otherpos）で loss==margin、
  完全同タイミング集中で `pos_over_other` loss==0。limb 系・`none` を混ぜた modes で pos 層のみ
  処理されること。3 モード（pos_over_other / other_over_pos / 非該当スキップ）を網羅。
- 勾配到達: pos 損失 `backward()` で QKV 重みに勾配が届くこと。

### 6.2 回帰（後方互換）
- `bias_loss.enabled:false` および pos モード未使用の既存 config で、同一 seed 1 epoch の
  loss 系列が現行コードと一致すること。

### 6.3 統合（スモーク）
- §3.3 の role 例 config で数十 step 学習し
  - OOM しない、`loss_metrics.jsonl` に `bias_loss` と `dec/attn_pos/<l>`・
    `dec/attn_otherpos/<l>` が出力される、
  - `pos_over_other` 指定層で `Attn_pos − Attn_otherpos` が単調増加傾向を示す。
- 学習後モデルを `scripts/analysis/visualize_attention_maps.py` / `qk_limb_bias_ratio.py` で
  可視化し、指定層で同タイミング注意が強まっているか定性確認。

---

## 7. 制約・将来拡張

- **margin は limb/pos 共通**（既存 `bias_loss.margin` を共用）。層別 margin は将来拡張。
- **メモリ**: pos モード層がある component でのみ M_pos/M_other を追加生成（[B,N,N]×2）。
  pos モード未使用時は `pos_token_ids is None` で一切構築しない。
- **head 別役割**: 現設計は head 平均に対する損失（既存踏襲）。head 単位割り当ては将来拡張。
- **cp_limb_v1 非対応**（§5）: `build_position_ids` が standard 語彙前提のため。

---

## 8. 実装順序（目安）

1. `position_bias.py`: モード定数 + `build_position_pair_masks`（単体テスト付き）
2. `position_bias.py`: `compute_attn_pos_bias_loss`（単体テスト付き）
3. `train_drum.py`: `_normalize_layer_modes` の許容集合拡張 + import
4. `train_drum.py`: `pos_token_ids` ゲート拡張（`'bar'` 追加・POS/BEAT 欠如時の降格）
5. `train_drum.py`: `bias_ctx` に `pos_token_ids` 追加 → `_apply_bias_loss` に pos 分岐
6. 回帰確認（pos 未使用一致）→ スモーク学習 → 可視化確認
7. config: コメント追記 + role 例 config 追加、必要なら `DESIGN_attn_bias_loss.md` へ相互参照追記
