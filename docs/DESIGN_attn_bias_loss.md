> モデル改良時の参考資料として保持しています。本文の旧実験設定名・配置は当時のものです。現行の実行方法は [研究用ツール](RESEARCH_TOOLS.md) と [README](../README.md) を参照してください。

# 設計書: Attention Bias Loss（Attn_limb / Attn_other 対比増大バイアス）

作成日: 2026-07-06
対象ブランチ: `data_revise`

---

## 1. 目的・背景

四肢運動情報トークン間の Attention（**Attn_limb**）と、それ以外の有効トークンペアへの
Attention（**Attn_other**）の「注意の向け方」の対比を、補助損失 **bias_loss** によって
学習時に増大させる。

既存の Limb-Relation Bias（`limb_bias_alpha` による Attention Logit への加算バイアス）が
「構造をモデルに直接注入する」アプローチであるのに対し、本機能は
「Attention 分布そのものに損失をかけ、勾配を通じて注意の向け方を誘導する」アプローチである。
両者は独立に ON/OFF でき、併用も可能。

### 要件（ユーザ指定）

| # | 要件 | 本設計での対応 |
|---|------|----------------|
| R1 | できれば `src/models/limb_bias.py` と `scripts/train/train_drum.py` のみの修正 | **達成**。Attention 重みは self_attn.forward のモンキーパッチで捕捉（既存 `scripts/analysis/visualize_attention_maps.py` の `AttentionCapture` と同方式、勾配保持版）。`musemorphose.py` / `transformer_encoder.py` は無変更 |
| R2 | Attn_limb と Attn_other の注意向け方の特徴を増大するバイアス | hinge 型 bias_loss（§4） |
| R3 | モデル層ごとに別々の Attention バイアスを設定可能 | 層ごとの `layer_modes` リスト。bias_loss モードに加え、加算バイアス α（`limb_bias_alpha`）の層別固定値も指定可能（§5） |
| R4 | 追加ハイパーパラメータは bias_loss とそのスケジューラーのみ | `training.bias_loss` ブロックに集約（§5） |
| R5 | bias_loss は 3 種類設定可能 | `other_over_limb` / `limb_over_other` / `none` の 3 モード（§4） |

---

## 2. 全体アーキテクチャ

```
train_drum.py (train_model ループ)
  │
  │ ① AttentionWeightCapture を decoder（/encoder）の対象層に常設パッチ
  │    （mode != 'none' の層のみ。need_weights=True で softmax 後の重みを捕捉、detach しない）
  │
  ├─ model.forward(...)                      ← モデルコードは無変更
  │     └─ capture.maps[layer_idx] = attn_weights  [B, N, N]（勾配付き）
  │
  │ ② limb_ids = build_limb_ids(dec_inp, lookup_table)          # 既存関数
  │ ③ M_limb, M_other = build_limb_pair_masks(limb_ids)         # 新規関数
  │ ④ bias_loss, metrics = compute_attn_bias_loss(
  │        capture.pop_maps(), M_limb, M_other, layer_modes, margin)   # 新規関数
  │ ⑤ λ = bias_loss_weight_sched(trained_steps, ...)            # 新規関数
  │
  └─ total_loss = 既存 total_loss + λ * bias_loss → backward
```

- **新規モデルパラメータなし**。state_dict は完全互換（既存 checkpoint の読み込み・
  既存モデルとの相互ロードに影響しない）。
- `bias_loss.enabled: false`（既定）時は一切のパッチ・計算を行わず、現行挙動と完全一致。

---

## 3. Attn_limb / Attn_other の定義

### 3.1 ペア分類

`limb_bias.py` の既存 Limb ID 体系（META=0, HAND=1, RFOOT=2, LFOOT=3, HAND_LFOOT=4）を流用。
query 位置 i、key 位置 j に対して:

- **limb ペア** `M_limb[i,j] = 1` : i, j とも非メタ、かつ同一四肢
  （`HAND_LFOOT` は `HAND`・`LFOOT` の双方と同一扱い — 既存 `build_limb_bias_mask` と同一判定）
- **other ペア** `M_other[i,j] = 1` : i, j とも非メタ、かつ同一四肢でない
- **除外**: どちらかがメタトークン（BAR/BEAT/POS/TEMPO/PAD/EOS 等）のペア。
  PAD は LIMB_META にマップされるため、パディング行・列は自動的に両集合から除外される。

デコーダは因果 Attention のため `j > i` の重みは 0 だが、softmax 後の重み行列上で
該当要素は元々 0 であり、平均の分母はマスク要素数で取るため
**下三角制約も明示的に掛ける**（`torch.tril`）。掛けない場合、上三角の limb/other ペアが
分母に入り平均値が系統的に希釈されるため必須。

### 3.2 スカラー化

層 l の head 平均 Attention 重み `A_l ∈ [B, N, N]`（softmax 後、行方向合計 ≒ 1）に対し:

```
Attn_limb(l)  = Σ_{b,i,j} A_l[b,i,j] * M_limb[b,i,j]  / Σ M_limb   （有効ペアの平均注意確率）
Attn_other(l) = Σ_{b,i,j} A_l[b,i,j] * M_other[b,i,j] / Σ M_other
```

- head 方向は `average_attn_weights=True`（PyTorch 既定）で平均済みの重みを使う。
  損失は平均値の差に対する hinge なので、head 平均を先に取っても定義と等価であり、
  メモリを `n_head` 分の 1 に抑えられる（B=8, N=1280 で層あたり約 26–52 MB）。
- 分母が 0（バッチ内に該当ペアなし）の層はその層の損失を 0 とする（`clamp_min(1)` 等でゼロ除算回避）。
- AMP 使用時、総和・平均は `.float()` で fp32 計算する（1280² 要素の fp16 総和は精度不足）。

---

## 4. bias_loss の 3 モード（R5）

層ごとに以下の 3 種類から選択:

| mode 文字列 | 損失式 | 意図 |
|---|---|---|
| `limb_over_other` | `max(0, margin − (Attn_limb − Attn_other))` | 同一四肢間の注意を margin 分だけ優位にする |
| `other_over_limb` | `max(0, margin − (Attn_other − Attn_limb))` | 四肢間（異四肢）の注意を margin 分だけ優位にする |
| `none` | 損失なし（当該層は Attention 捕捉もしない） | limb-bias 未適用（現行挙動） |

全体の bias_loss は **mode != 'none' の層の平均**:

```
bias_loss = mean_{l ∈ active layers} hinge_l
```

（層数に依存せずスケールが安定するため、総和でなく平均とする）

hinge が 0 に達した層は勾配が消えるため、margin 到達後は過剰な歪みを与えない。

---

## 5. ハイパーパラメータとスケジューラー（R3, R4）

### 5.1 config 追加項目（`training.bias_loss` ブロックに集約）

```yaml
training:
  # ===== Attention Bias Loss =====
  bias_loss:
    enabled: false                    # false なら完全に無効（既定・後方互換）
    margin: 0.05                      # hinge の margin（Attention 確率差。0.0〜1.0 目安）
    # 層ごとのモード指定（R3）
    #   リスト長は dec_n_layer / enc_n_layer と一致必須
    #   文字列 1 個を指定した場合は全層同一モードに展開
    decoder_layer_modes: ['none', 'none', 'limb_over_other',
                          'limb_over_other', 'none', 'none']
    encoder_layer_modes: null         # null = エンコーダには不適用（全層 'none' 相当）
    # ===== スケジューラー（R4） =====
    max_weight: 1.0                   # λ の最大値
    sched: 'linear_warmup'            # 'constant' | 'linear_warmup' | 'cyclical'
    no_loss_steps: 1000               # このステップ数までは λ = 0
    warmup_steps: 4000                # no_loss_steps 経過後、λ を 0→max_weight へ線形増加
    cycle_steps: 10000                # sched='cyclical' 時の周期（KL β と同型）
```

### 5.2 スケジューラー仕様

`bias_loss_weight_sched(step, sched, max_weight, no_loss_steps, warmup_steps, cycle_steps) -> float`

- `constant` : `step < no_loss_steps` なら 0、以降 `max_weight`
- `linear_warmup` : `step < no_loss_steps` なら 0、以降
  `max_weight * min(1, (step − no_loss_steps) / warmup_steps)`
- `cyclical` : 既存 `beta_cyclical_sched` と同型
  （周期前半で 0→max_weight 線形増加、後半は max_weight 維持。`no_loss_steps` 前は 0）

推奨初期設定: 再構成損失の学習が立ち上がる前に Attention を歪めないよう、
`no_loss_steps >= no_kl_steps` かつ `linear_warmup` を既定とする。

### 5.3 バリデーションルール（`_validate_training_config` に追加）

- `margin >= 0` でなければ ValueError
- 各 mode 文字列は `{'limb_over_other', 'other_over_limb', 'none'}` のいずれか
- `decoder_layer_modes` のリスト長 ≠ `model.dec_n_layer` なら ValueError（encoder も同様）
- `enabled: true` かつ全層 `'none'` なら warning を出して実質無効化
- `enabled: true` かつ `sched` が未知文字列なら ValueError
- `tokenization_method == 'cp_limb_v1'` の場合: 標準トークン列と limb ID 対応が異なるため
  **v1 では非対応とし、`enabled: true` なら ValueError**（将来拡張。§9 参照）

---

## 6. ファイル別修正内容

### 6.1 `src/models/limb_bias.py`（追加のみ、既存関数は無変更）

#### (a) `build_limb_pair_masks(limb_ids, causal=False) -> (M_limb, M_other)`

```python
def build_limb_pair_masks(limb_ids, causal=False):
    """limb_ids [B, N] → (M_limb, M_other) 各 [B, N, N] float。
    M_limb : 非メタ同一四肢ペア（build_limb_bias_mask と同一判定）
    M_other: 非メタ異四肢ペア
    causal=True で下三角のみ有効（デコーダ用）
    """
```

- 同一四肢判定・メタ除外ロジックは `build_limb_bias_mask` の実装を関数内で共有
  （same_limb / has_meta の判定部を小さなヘルパー `_same_limb_and_meta(limb_ids)` に
  くくり出し、`build_limb_bias_mask` もそれを使うようリファクタ。外部挙動は不変）。

#### (b) `class AttentionWeightCapture`

分析用 `AttentionCapture`（visualize_attention_maps.py）の**勾配保持・常設パッチ版**:

```python
class AttentionWeightCapture:
    """指定した TransformerEncoderLayer 群の self_attn.forward をパッチし、
    head 平均 Attention 重みを勾配付きで捕捉する。

    - layers: {layer_idx: nn.TransformerEncoderLayer} … mode != 'none' の層のみ渡す
    - enabled=False の間は捕捉しない（validation 中の無駄なメモリ保持を防ぐ）
    - pop_maps(): {layer_idx: Tensor[B, N, N]} を返し内部バッファをクリア
    - attach() / detach(): パッチの適用・解除（学習開始時に一度 attach、終了時 detach）
    """
```

実装要点:
- パッチ内で `kwargs['need_weights'] = self.enabled`、
  `kwargs['average_attn_weights'] = True` を設定（PyTorch < 1.13 は TypeError fallback、
  分析スクリプトと同じ手法）。
- **`.detach()` しない**（勾配を流すため）。`.cpu()` もしない。
- `pop_maps()` を毎バッチ必ず呼びクリアする（`try/finally`）。呼び忘れると
  計算グラフが保持され続け OOM の原因になるため、train ループ側で finally 必須。

#### (c) `compute_attn_bias_loss(attn_maps, M_limb, M_other, layer_modes, margin) -> (loss, metrics)`

```python
def compute_attn_bias_loss(attn_maps, M_limb, M_other, layer_modes, margin):
    """
    attn_maps  : {layer_idx: Tensor[B, N, N]}  (AttentionWeightCapture.pop_maps() の戻り値)
    layer_modes: 層ごとの mode 文字列リスト
    Returns:
      loss    : スカラー Tensor（active 層の hinge 平均。active 層ゼロなら 0 テンソル）
      metrics : {'attn_limb/<l>': float, 'attn_other/<l>': float, ...}  ログ用
    """
```

- 総和は `.float()` で fp32 計算（§3.2）。
- metrics は `.item()` 済み float（ログ専用、グラフから切断）。

#### (d) `bias_loss_weight_sched(...)`

§5.2 の仕様どおりの純関数。`train_drum.py` の `beta_cyclical_sched` と対になるが、
limb 機能に属する関数として `limb_bias.py` 側へ置く
（`train_drum.py` の肥大化を避け、単体テストしやすくするため）。

### 6.2 `scripts/train/train_drum.py`

#### (a) `_validate_training_config`

§5.3 のバリデーションを追加。`config['training']['bias_loss']` を正規化して格納
（未指定なら `{'enabled': False}`、文字列 1 個の layer_modes をリスト展開）。

#### (b) `main()` — セットアップ（モデル構築後）

```python
bias_loss_cfg = config['training']['bias_loss']
attn_capture_dec = attn_capture_enc = None
bias_lookup_table = None
if bias_loss_cfg['enabled']:
    # token_to_limb は limb_bias / position_bias 無効時にも必要 →
    # 既存の構築条件（use_limb_bias_* or use_position_bias_*）に bias_loss_enabled を追加
    bias_lookup_table = make_limb_lookup_table(token_to_limb, dset.vocab_size).to(device)

    dec_active = {i: model.decoder.decoder_layers[i]
                  for i, m in enumerate(dec_modes) if m != 'none'}
    if dec_active:
        attn_capture_dec = AttentionWeightCapture(dec_active)
        attn_capture_dec.attach()
    # encoder 側（encoder_layer_modes 指定時のみ）:
    #   hasattr(model.encoder, 'encoder_layers') なら encoder_layers、
    #   なければ model.encoder.tr_encoder.layers   ← 分析スクリプト _get_layer_lists と同じ分岐
```

- `token_to_limb` 構築部（現行 L1092–1101）: 条件式に `or bias_loss_enabled` を追加。
  `idx2token` が無い場合は bias_loss も warning を出して無効化（既存 limb_bias と同じ扱い）。
- capture オブジェクト・lookup_table・cfg は `train_model` / `validate` に引数で渡す
  （関数シグネチャに `bias_ctx=None` を 1 個追加し、dict にまとめて渡す）。

#### (c) `train_model()` — 損失計算（AMP / 非 AMP 両分岐）

forward 直後・backward 前に挿入（autocast ブロック内。AMP 分岐と非 AMP 分岐で共通化のため
ヘルパー関数 `_apply_bias_loss(losses, bias_ctx, batch_dec_inp, enc_inp_flat, trained_steps)`
にまとめる）:

```python
if bias_ctx is not None:
    try:
        lam = bias_loss_weight_sched(trained_steps, **bias_ctx['sched_kwargs'])
        maps_dec = bias_ctx['capture_dec'].pop_maps() if ... else {}
        if lam > 0 and maps_dec:
            dec_limb_ids = build_limb_ids(batch_dec_inp, bias_ctx['lookup'])  # [B, N]
            M_limb, M_other = build_limb_pair_masks(dec_limb_ids, causal=True)
            bias_loss, bias_metrics = compute_attn_bias_loss(
                maps_dec, M_limb, M_other, bias_ctx['dec_modes'], bias_ctx['margin'])
            # (encoder 分があれば同様に加算。enc 側は causal=False、
            #  入力は enc_inp を [enc_seqlen, B*n_bars] に flatten したもの)
            losses['total_loss'] = losses['total_loss'] + lam * bias_loss
            losses['bias_loss'] = bias_loss.detach()
            losses['bias_weight'] = lam
    finally:
        # λ=0 のステップでも必ず捕捉バッファを破棄（計算グラフ滞留 → OOM 防止）
        ...capture の残バッファをクリア...
```

注意点:
- **λ = 0 の間も `need_weights=True` は掛かり続ける**（パッチ常設のため）。
  no_loss_steps が長い設定では速度が惜しいので、`capture.enabled = (λ_next > 0)` を
  ステップ冒頭で更新し、λ=0 の間は捕捉自体をスキップする。
- 勾配クリッピング（0.5）は既存のまま bias_loss 込みの全勾配に適用される。変更不要。

#### (d) `validate()`

- validation 中は `capture.enabled = False` にして捕捉を止める（no_grad 下で害は小さいが、
  fast path が使える場合の速度低下を避ける）。モニタリング用に Attn_limb / Attn_other を
  検証時にも記録したい場合は `enabled=True` のまま loss へは加算せず metrics のみ収集する
  オプションを将来追加（v1 ではスキップ）。

#### (e) ログ

- コンソール: 既存の per-batch print に
  `BL: {bias_loss:.4f} (λ={bias_weight:.3f})` を追記（enabled 時のみ）。
- `loss_metrics.jsonl`（`_log_mode_loss_metrics`）: enabled 時のみ
  `bias_loss`, `bias_weight`, `attn_limb/<layer>`, `attn_other/<layer>` を payload に追加。
- `log.txt` は**フォーマット変更しない**（既存パーサ・比較スクリプトの互換性維持。
  詳細は jsonl 側で追える）。

### 6.3 config ファイル

実験用に `config/drum_config_standard-lakh1-kl13.yaml` をベースとした
`config/drum_config_standard-lakh1-biasloss.yaml` を追加（§5.1 のブロック付き）。
既存 config は無変更（`bias_loss` キー不在 → disabled として扱う）。

---

## 7. 既存機能との相互作用

| 既存機能 | 相互作用 |
|---|---|
| Limb-Relation Bias (`use_limb_bias_*`, `limb_bias_alpha`) | 独立。併用可。併用時は加算バイアスで持ち上がった Attention 分布に対して hinge が評価される（margin 到達済みなら bias_loss=0 で無干渉） |
| Position-Same-Time Bias | 独立。併用可 |
| `layer_bias_schedule`（加算バイアスの層別適用） | 概念は別物（あちらは logit 加算の層別 ON/OFF、こちらは損失の層別モード）。名称衝突を避けるため config キーは `bias_loss.decoder_layer_modes` とした |
| AMP (`use_amp`) | 捕捉重みは fp16。損失内部で fp32 化（§3.2）。`scaler.scale(total_loss)` は合成後の total に掛かるため変更不要 |
| nn.MultiheadAttention fast path | `need_weights=True` で無効化され若干減速。ただし limb/position バイアス使用時は 3D attn_mask で元々無効。λ=0 期間は `capture.enabled=False` で回避（§6.2c） |
| dropout（attention dropout 0.1） | 返却される重みは dropout 適用後。limb/other 双方が同率で影響を受けるため差分 hinge への期待値バイアスは無し（ノイズは増える）。問題になる場合は margin を小さめに |
| Early Stopping / best model 選択 | 判定は従来どおり **再構成損失のみ**。bias_loss は選択基準に混ぜない（生成品質の代理指標を汚さないため） |
| checkpoint 互換 | 新規パラメータなし。strict=True で相互ロード可 |

---

## 8. テスト・検証計画

1. **単体（新規 `tests/test_attn_bias_loss.py` もしくは scratch 検証スクリプト）**
   - `build_limb_pair_masks`: 手作り limb_ids（メタ/HAND/RFOOT/HAND_LFOOT 混在、causal 有無）で
     M_limb / M_other の相互排他・メタ除外・下三角を確認
   - `compute_attn_bias_loss`: 一様 Attention（Attn_limb == Attn_other）で
     loss == margin、完全 limb 集中で `limb_over_other` loss == 0 を確認。3 モード網羅
   - `bias_loss_weight_sched`: 3 種 sched の境界値（no_loss_steps 前後、warmup 完了点）
   - `AttentionWeightCapture`: 捕捉テンソルに `requires_grad=True` が立ち、
     `loss.backward()` でモデルの QKV 重みに勾配が届くこと
2. **回帰（互換性）**
   - `bias_loss.enabled: false`（および キー不在）で、同一 seed の 1 epoch 学習の
     loss 系列が現行コードと一致すること
3. **統合（スモーク）**
   - enabled 設定で数十 step 学習し、(a) OOM しない、(b) jsonl に bias_loss が出る、
     (c) Attn_limb − Attn_other が `limb_over_other` 指定層で単調増加傾向を示す
   - 学習後モデルを `scripts/analysis/visualize_attention_maps.py` /
     `qk_limb_bias_ratio.py` で可視化し、指定層の注意パターン変化を定性確認

---

## 9. 制約・将来拡張

- **cp_limb_v1 非対応（v1）**: CP モードは複合トークンで limb 情報の持ち方が異なるため、
  本設計の lookup_table（standard 語彙前提）が使えない。enabled 時は ValueError（§5.3）。
  対応する場合は cp_hand1/cp_hand2/... 入力から limb_ids を合成する拡張が必要。
- **head 別バイアス**: 現設計は head 平均に対する損失。head ごとに別モードを掛けたい場合は
  `average_attn_weights=False` 捕捉 + head 次元付き mode 指定への拡張が必要
  （メモリ n_head 倍。B=8, N=1280, H=8 で層あたり ~420 MB fp32 のため v1 では見送り）。
- **margin の層別指定**: v1 は全層共通 margin。必要なら `layer_modes` を
  `[{mode: ..., margin: ...}, ...]` 形式に拡張可能（バリデーションで両形式受理）。
- **生成時への影響なし**: bias_loss は学習時のみ。generate 系スクリプトは無変更で動作。

---

## 10. 実装順序（目安）

1. `limb_bias.py`: `_same_limb_and_meta` くくり出し + `build_limb_pair_masks`（単体テスト付き）
2. `limb_bias.py`: `bias_loss_weight_sched` / `compute_attn_bias_loss`（単体テスト付き）
3. `limb_bias.py`: `AttentionWeightCapture`（勾配到達テスト付き）
4. `train_drum.py`: config バリデーション → セットアップ → 損失合成 → ログ
5. 回帰確認（disabled 一致）→ スモーク学習 → 可視化確認
6. 実験 config 追加・README_drum.md への追記（任意）
