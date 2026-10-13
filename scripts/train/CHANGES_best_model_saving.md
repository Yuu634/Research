# 学習スクリプト修正内容

## 概要
`scripts/train/train_drum.py` に以下の機能を追加しました：

1. **検証損失に基づくベストモデルの自動保存**
2. **学習終了時の最終モデル保存**

## 主な変更点

### 1. `train_model()` 関数の修正

#### 引数の追加
```python
def train_model(epoch, model, dloader, dloader_val, optim, sched, config, 
                trained_steps, scaler=None, best_val_loss=float('inf')):
```
- `best_val_loss` パラメータを追加（デフォルト: `float('inf')`）

#### 戻り値の変更
```python
return trained_steps, best_val_loss
```
- `best_val_loss` も返すように変更（エポック間でベスト値を追跡）

### 2. 検証時のベストモデル保存

検証実行時（`val_interval`ごと）に、以下の処理を追加：

```python
# 検証
if not trained_steps % val_interval:
    vallosses = validate(model, dloader_val, config)
    current_val_loss = np.mean(vallosses[0])  # 再構成損失を使用
    
    with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
        f.write('[step {}] RC: {:.4f} | KL: {:.4f} | [val] | RC: {:.4f} | KL: {:.4f}\n'.format(
            trained_steps,
            recons_loss_ema,
            kl_raw_ema,
            current_val_loss,
            np.mean(vallosses[1])
        ))
    
    # ベストモデルの保存
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(
            model.state_dict(),
            os.path.join(params_dir, 'best_params.pt')
        )
        print(f'[info] New best model saved! (val_loss: {best_val_loss:.4f})')
    
    model.train()
```

**ポイント:**
- 検証損失（再構成損失）が改善された場合のみ保存
- `best_params.pt` として保存
- 保存時にコンソールにメッセージを出力

### 3. メインループの修正

```python
# 学習ループ
print('[info] Starting training...')
best_val_loss = float('inf')

for ep in range(config['training']['max_epochs']):
    trained_steps, best_val_loss = train_model(
        ep + 1,
        model,
        dloader,
        dloader_val,
        optimizer,
        scheduler,
        config,
        trained_steps,
        scaler=scaler,
        best_val_loss=best_val_loss
    )
```

**変更点:**
- `best_val_loss` を初期化（`float('inf')`）
- `train_model()` から返される `best_val_loss` を受け取り、次のエポックに渡す

### 4. 学習終了時の処理

学習完了後に以下の処理を追加：

```python
# 学習終了時に最終モデルを保存
print('[info] Saving final model...')
torch.save(
    model.state_dict(),
    os.path.join(params_dir, 'final_params.pt')
)
print(f'[info] Final model saved to {os.path.join(params_dir, "final_params.pt")}')

# best_params.ptが存在しない場合は最終モデルをベストモデルとして保存
best_params_path = os.path.join(params_dir, 'best_params.pt')
if not os.path.exists(best_params_path):
    torch.save(
        model.state_dict(),
        best_params_path
    )
    print(f'[info] Best model saved to {best_params_path}')
else:
    print(f'[info] Best model already exists at {best_params_path} (val_loss: {best_val_loss:.4f})')

print('[info] Training completed!')
```

**機能:**
1. 最終モデルを `final_params.pt` として保存
2. `best_params.pt` が存在しない場合（検証が実行されなかった場合）、最終モデルをベストモデルとして保存
3. 既に存在する場合は、そのベストモデルの損失値を表示

## 保存されるファイル

学習完了後、以下のファイルが `<ckpt_dir>/params/` に保存されます：

1. **`best_params.pt`** - 検証損失が最も低かったモデル（新規追加）
2. **`final_params.pt`** - 学習終了時点のモデル（新規追加）
3. **`step_<N>-RC_<X>-KL_<Y>-model.pt`** - 定期チェックポイント（既存）

## 使用方法

変更なし。従来通りの方法で実行できます：

```bash
python scripts/train/train_drum.py --config config/train_config.yaml
```

## 動作確認

構文チェック済み：
```bash
python -m py_compile scripts/train/train_drum.py
# → エラーなし
```

## 注意事項

- **検証損失の基準**: 再構成損失（`vallosses[0]`の平均）を使用
- **検証間隔**: `config['training']['val_interval']` で設定された間隔で検証を実行
- **上書き動作**: `best_params.pt` は常に最良のモデルで上書きされます
- **フォールバック**: 検証が一度も実行されない場合、最終モデルがベストモデルとして保存されます

## 生成スクリプトとの連携

修正した生成スクリプト `generate_drum_from_input.py` では、以下のように指定できます：

```bash
# ベストモデルを使用
python scripts/generate/generate_drum_from_input.py \
  --model_path trained_model/checkpoints_xxx/params/best_params.pt \
  --input_midi input.mid \
  --output_midi output.mid

# 最終モデルを使用
python scripts/generate/generate_drum_from_input.py \
  --model_path trained_model/checkpoints_xxx/params/final_params.pt \
  --input_midi input.mid \
  --output_midi output.mid
```

通常は `best_params.pt` の使用を推奨します。
