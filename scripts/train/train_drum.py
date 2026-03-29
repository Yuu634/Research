"""
ドラム譜用MuseMorphoseモデルの学習スクリプト
"""
import sys
import os
import time

from src.models.musemorphose import MuseMorphose
from src.data.drum_dataloader import DrumTransformerDataset
from torch.utils.data import DataLoader
from src.utils.tokenization_mode import (
    ensure_runtime_mode_compatibility,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
)

from src.utils.utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

import yaml
import argparse


def log_epoch(log_file, log_data, is_init=False):
    """エポックごとのログを記録"""
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12} {:12} {:12}\n'.format(
                'ep', 'steps', 'recons_loss', 'kldiv_loss', 'kldiv_raw', 'ep_time'
            ))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12} {:<12} {:<12}\n'.format(
            log_data['ep'],
            log_data['steps'],
            round(log_data['recons_loss'], 5),
            round(log_data['kldiv_loss'], 5),
            round(log_data['kldiv_raw'], 5),
            round(log_data['time'], 2)
        ))


def beta_cyclical_sched(step, no_kl_steps, kl_cycle_steps, kl_max_beta):
    """KLベータの周期的スケジューリング"""
    step_in_cycle = (step - 1) % kl_cycle_steps
    cycle_progress = step_in_cycle / kl_cycle_steps

    if step < no_kl_steps:
        return 0.
    if cycle_progress < 0.5:
        return kl_max_beta * cycle_progress * 2.
    else:
        return kl_max_beta


def compute_loss_ema(ema, batch_loss, decay=0.95):
    """EMAを用いた損失の計算"""
    if ema == 0.:
        return batch_loss
    else:
        return batch_loss * (1 - decay) + ema * decay


def train_model(epoch, model, dloader, dloader_val, optim, sched, config, trained_steps, scaler=None, best_val_loss=float('inf')):
    """モデルの学習（1エポック）"""
    model.train()

    device = config['training']['device']
    lr_decay_steps = config['training']['lr_decay_steps']
    lr_warmup_steps = config['training']['lr_warmup_steps']
    no_kl_steps = config['training']['no_kl_steps']
    kl_cycle_steps = config['training']['kl_cycle_steps']
    kl_max_beta = config['training']['kl_max_beta']
    free_bit_lambda = config['training']['free_bit_lambda']
    max_lr = config['training']['max_lr']
    constant_kl = config['training']['constant_kl']
    ckpt_dir = config['training']['ckpt_dir']
    ckpt_interval = config['training']['ckpt_interval']
    log_interval = config['training']['log_interval']
    val_interval = config['training']['val_interval']
    use_amp = config['training'].get('use_amp', False) and device == 'cuda'
    use_difficulty = config['model'].get('use_difficulty', False)
    tokenization_method = config['data'].get('tokenization_method', 'standard')
    cp_loss_weights = config['model'].get('cp_loss_weights', {})

    params_dir = os.path.join(ckpt_dir, 'params/')
    optim_dir = os.path.join(ckpt_dir, 'optim/')

    recons_loss_ema = 0.
    kl_loss_ema = 0.
    kl_raw_ema = 0.

    print(f'[epoch {epoch:03d}] training ...')
    print(f'[epoch {epoch:03d}] # batches = {len(dloader)}')
    if use_amp:
        print(f'[epoch {epoch:03d}] Mixed Precision Training enabled')
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()

        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_inp_lens = batch_samples['length']
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)

        cp_event_type_inp = batch_samples['cp_event_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_struct_inp = batch_samples['cp_struct_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_pos_inp = batch_samples['cp_pos_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_hand1_inp = batch_samples['cp_hand1_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_hand2_inp = batch_samples['cp_hand2_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_rf_inp = batch_samples['cp_right_foot_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_lf_inp = batch_samples['cp_left_foot_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None

        cp_event_type_tgt = batch_samples['cp_event_type_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_struct_tgt = batch_samples['cp_struct_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_pos_tgt = batch_samples['cp_pos_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_hand1_tgt = batch_samples['cp_hand1_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_hand2_tgt = batch_samples['cp_hand2_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_rf_tgt = batch_samples['cp_right_foot_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_lf_tgt = batch_samples['cp_left_foot_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None

        s_tech_cls_seq = batch_samples['s_tech_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_tech_cls_seq' in batch_samples else None
        s_indep_cls_seq = batch_samples['s_indep_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_indep_cls_seq' in batch_samples else None
        s_hand_cls_seq = batch_samples['s_hand_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_hand_cls_seq' in batch_samples else None
        s_foot_cls_seq = batch_samples['s_foot_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_foot_cls_seq' in batch_samples else None
        s_move_cls_seq = batch_samples['s_move_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_move_cls_seq' in batch_samples else None

        # ドラム譜では属性は使用しない（Noneを渡す）
        trained_steps += 1

        # Mixed Precision Training
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if tokenization_method == 'cp_limb_v1':
                    mu, logvar, cp_logits = model(
                        batch_enc_inp,
                        batch_dec_inp,
                        batch_inp_bar_pos,
                        None,
                        None,
                        s_tech_cls=s_tech_cls_seq,
                        s_indep_cls=s_indep_cls_seq,
                        s_hand_cls=s_hand_cls_seq,
                        s_foot_cls=s_foot_cls_seq,
                        s_move_cls=s_move_cls_seq,
                        cp_event_type_inp=cp_event_type_inp,
                        cp_struct_inp=cp_struct_inp,
                        cp_pos_inp=cp_pos_inp,
                        cp_hand1_inp=cp_hand1_inp,
                        cp_hand2_inp=cp_hand2_inp,
                        cp_right_foot_inp=cp_rf_inp,
                        cp_left_foot_inp=cp_lf_inp,
                        padding_mask=batch_padding_mask
                    )
                else:
                    mu, logvar, dec_logits = model(
                        batch_enc_inp,
                        batch_dec_inp,
                        batch_inp_bar_pos,
                        None,  # rhythm frequency class (not used for drums)
                        None,  # polyphony class (not used for drums)
                        s_tech_cls=s_tech_cls_seq,
                        s_indep_cls=s_indep_cls_seq,
                        s_hand_cls=s_hand_cls_seq,
                        s_foot_cls=s_foot_cls_seq,
                        s_move_cls=s_move_cls_seq,
                        padding_mask=batch_padding_mask
                    )

                # KLベータのスケジューリング
                if not constant_kl:
                    kl_beta = beta_cyclical_sched(trained_steps, no_kl_steps, kl_cycle_steps, kl_max_beta)
                else:
                    kl_beta = kl_max_beta

                if tokenization_method == 'cp_limb_v1':
                    losses = model.compute_cp_loss(
                        mu,
                        logvar,
                        kl_beta,
                        free_bit_lambda,
                        cp_logits,
                        {
                            'event_type': cp_event_type_tgt,
                            'structural': cp_struct_tgt,
                            'cp_pos': cp_pos_tgt,
                            'cp_hand1': cp_hand1_tgt,
                            'cp_hand2': cp_hand2_tgt,
                            'cp_right_foot': cp_rf_tgt,
                            'cp_left_foot': cp_lf_tgt,
                        },
                        loss_weights=cp_loss_weights
                    )
                else:
                    losses = model.compute_loss(mu, logvar, kl_beta, free_bit_lambda, dec_logits, batch_dec_tgt)

            # 学習率のアニーリング
            if trained_steps < lr_warmup_steps:
                curr_lr = max_lr * trained_steps / lr_warmup_steps
                optim.param_groups[0]['lr'] = curr_lr
            else:
                sched.step(trained_steps - lr_warmup_steps)

            # Scaled backward & 勾配クリッピング & モデル更新
            scaler.scale(losses['total_loss']).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optim)
            scaler.update()

        else:
            # 通常の学習（AMP無し）
            if tokenization_method == 'cp_limb_v1':
                mu, logvar, cp_logits = model(
                    batch_enc_inp,
                    batch_dec_inp,
                    batch_inp_bar_pos,
                    None,
                    None,
                    s_tech_cls=s_tech_cls_seq,
                    s_indep_cls=s_indep_cls_seq,
                    s_hand_cls=s_hand_cls_seq,
                    s_foot_cls=s_foot_cls_seq,
                    s_move_cls=s_move_cls_seq,
                    cp_event_type_inp=cp_event_type_inp,
                    cp_struct_inp=cp_struct_inp,
                    cp_pos_inp=cp_pos_inp,
                    cp_hand1_inp=cp_hand1_inp,
                    cp_hand2_inp=cp_hand2_inp,
                    cp_right_foot_inp=cp_rf_inp,
                    cp_left_foot_inp=cp_lf_inp,
                    padding_mask=batch_padding_mask
                )
            else:
                mu, logvar, dec_logits = model(
                    batch_enc_inp,
                    batch_dec_inp,
                    batch_inp_bar_pos,
                    None,  # rhythm frequency class (not used for drums)
                    None,  # polyphony class (not used for drums)
                    s_tech_cls=s_tech_cls_seq,
                    s_indep_cls=s_indep_cls_seq,
                    s_hand_cls=s_hand_cls_seq,
                    s_foot_cls=s_foot_cls_seq,
                    s_move_cls=s_move_cls_seq,
                    padding_mask=batch_padding_mask
                )

            # KLベータのスケジューリング
            if not constant_kl:
                kl_beta = beta_cyclical_sched(trained_steps, no_kl_steps, kl_cycle_steps, kl_max_beta)
            else:
                kl_beta = kl_max_beta

            if tokenization_method == 'cp_limb_v1':
                losses = model.compute_cp_loss(
                    mu,
                    logvar,
                    kl_beta,
                    free_bit_lambda,
                    cp_logits,
                    {
                        'event_type': cp_event_type_tgt,
                        'structural': cp_struct_tgt,
                        'cp_pos': cp_pos_tgt,
                        'cp_hand1': cp_hand1_tgt,
                        'cp_hand2': cp_hand2_tgt,
                        'cp_right_foot': cp_rf_tgt,
                        'cp_left_foot': cp_lf_tgt,
                    },
                    loss_weights=cp_loss_weights
                )
            else:
                losses = model.compute_loss(mu, logvar, kl_beta, free_bit_lambda, dec_logits, batch_dec_tgt)

            # 学習率のアニーリング
            if trained_steps < lr_warmup_steps:
                curr_lr = max_lr * trained_steps / lr_warmup_steps
                optim.param_groups[0]['lr'] = curr_lr
            else:
                sched.step(trained_steps - lr_warmup_steps)

            # 勾配クリッピング & モデル更新
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()

        # EMAの更新
        recons_loss_ema = compute_loss_ema(recons_loss_ema, losses['recons_loss'].item())
        kl_loss_ema = compute_loss_ema(kl_loss_ema, losses['kldiv_loss'].item())
        kl_raw_ema = compute_loss_ema(kl_raw_ema, losses['kldiv_raw'].item())

        print(f' -- epoch {epoch:03d} | batch {batch_idx:03d}: len: {batch_inp_lens}\n'
              f'\t * loss = (RC: {recons_loss_ema:.4f} | KL: {kl_loss_ema:.4f} | '
              f'KL_raw: {kl_raw_ema:.4f}), step = {trained_steps}, beta: {kl_beta:.4f} '
              f'time_elapsed = {time.time() - st:.2f} secs')

        if tokenization_method == 'cp_limb_v1':
            print(
                ' \t * cp_loss = '
                f"(evt: {losses['cp_event_type_loss'].item():.4f} | "
                f"struct: {losses['cp_structural_loss'].item():.4f} | "
                f"pos: {losses['cp_pos_loss'].item():.4f} | "
                f"h1: {losses['cp_hand1_loss'].item():.4f} | "
                f"h2: {losses['cp_hand2_loss'].item():.4f} | "
                f"rf: {losses['cp_right_foot_loss'].item():.4f} | "
                f"lf: {losses['cp_left_foot_loss'].item():.4f})"
            )

        # ログの記録
        if not trained_steps % log_interval:
            log_data = {
                'ep': epoch,
                'steps': trained_steps,
                'recons_loss': recons_loss_ema,
                'kldiv_loss': kl_loss_ema,
                'kldiv_raw': kl_raw_ema,
                'time': time.time() - st
            }
            log_epoch(
                os.path.join(ckpt_dir, 'log.txt'),
                log_data,
                is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
            )

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

        # チェックポイントの保存
        if not trained_steps % ckpt_interval:
            torch.save(
                model.state_dict(),
                os.path.join(params_dir, f'step_{trained_steps:d}-RC_{recons_loss_ema:.3f}-KL_{kl_raw_ema:.3f}-model.pt')
            )
            torch.save(
                optim.state_dict(),
                os.path.join(optim_dir, f'step_{trained_steps:d}-RC_{recons_loss_ema:.3f}-KL_{kl_raw_ema:.3f}-optim.pt')
            )

    print(f'[epoch {epoch:03d}] training completed\n'
          f'  -- loss = (RC: {recons_loss_ema:.4f} | KL: {kl_loss_ema:.4f} | KL_raw: {kl_raw_ema:.4f})\n'
          f'  -- time elapsed = {time.time() - st:.2f} secs.')

    log_data = {
        'ep': epoch,
        'steps': trained_steps,
        'recons_loss': recons_loss_ema,
        'kldiv_loss': kl_loss_ema,
        'kldiv_raw': kl_raw_ema,
        'time': time.time() - st
    }
    log_epoch(
        os.path.join(ckpt_dir, 'log.txt'),
        log_data,
        is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
    )

    return trained_steps, best_val_loss


def validate(model, dloader, config, n_rounds=8):
    """モデルの検証"""
    model.eval()
    device = config['training']['device']
    use_difficulty = config['model'].get('use_difficulty', False)
    tokenization_method = config['data'].get('tokenization_method', 'standard')
    cp_loss_weights = config['model'].get('cp_loss_weights', {})

    loss_rec = []
    kl_loss_rec = []

    print('[info] validating ...')
    with torch.no_grad():
        for i in range(n_rounds):
            print(f'[round {i+1}]')

            for batch_idx, batch_samples in enumerate(dloader):
                model.zero_grad()

                batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
                batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
                batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
                batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
                batch_padding_mask = batch_samples['enc_padding_mask'].to(device)

                cp_event_type_inp = batch_samples['cp_event_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_struct_inp = batch_samples['cp_struct_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_pos_inp = batch_samples['cp_pos_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_hand1_inp = batch_samples['cp_hand1_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_hand2_inp = batch_samples['cp_hand2_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_rf_inp = batch_samples['cp_right_foot_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_lf_inp = batch_samples['cp_left_foot_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None

                cp_event_type_tgt = batch_samples['cp_event_type_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_struct_tgt = batch_samples['cp_struct_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_pos_tgt = batch_samples['cp_pos_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_hand1_tgt = batch_samples['cp_hand1_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_hand2_tgt = batch_samples['cp_hand2_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_rf_tgt = batch_samples['cp_right_foot_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_lf_tgt = batch_samples['cp_left_foot_target'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None

                s_tech_cls_seq = batch_samples['s_tech_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_tech_cls_seq' in batch_samples else None
                s_indep_cls_seq = batch_samples['s_indep_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_indep_cls_seq' in batch_samples else None
                s_hand_cls_seq = batch_samples['s_hand_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_hand_cls_seq' in batch_samples else None
                s_foot_cls_seq = batch_samples['s_foot_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_foot_cls_seq' in batch_samples else None
                s_move_cls_seq = batch_samples['s_move_cls_seq'].permute(1, 0).to(device) if use_difficulty and 's_move_cls_seq' in batch_samples else None

                if tokenization_method == 'cp_limb_v1':
                    mu, logvar, cp_logits = model(
                        batch_enc_inp,
                        batch_dec_inp,
                        batch_inp_bar_pos,
                        None,
                        None,
                        s_tech_cls=s_tech_cls_seq,
                        s_indep_cls=s_indep_cls_seq,
                        s_hand_cls=s_hand_cls_seq,
                        s_foot_cls=s_foot_cls_seq,
                        s_move_cls=s_move_cls_seq,
                        cp_event_type_inp=cp_event_type_inp,
                        cp_struct_inp=cp_struct_inp,
                        cp_pos_inp=cp_pos_inp,
                        cp_hand1_inp=cp_hand1_inp,
                        cp_hand2_inp=cp_hand2_inp,
                        cp_right_foot_inp=cp_rf_inp,
                        cp_left_foot_inp=cp_lf_inp,
                        padding_mask=batch_padding_mask
                    )
                    losses = model.compute_cp_loss(
                        mu,
                        logvar,
                        0.0,
                        0.0,
                        cp_logits,
                        {
                            'event_type': cp_event_type_tgt,
                            'structural': cp_struct_tgt,
                            'cp_pos': cp_pos_tgt,
                            'cp_hand1': cp_hand1_tgt,
                            'cp_hand2': cp_hand2_tgt,
                            'cp_right_foot': cp_rf_tgt,
                            'cp_left_foot': cp_lf_tgt,
                        },
                        loss_weights=cp_loss_weights
                    )
                else:
                    mu, logvar, dec_logits = model(
                        batch_enc_inp,
                        batch_dec_inp,
                        batch_inp_bar_pos,
                        None,
                        None,
                        s_tech_cls=s_tech_cls_seq,
                        s_indep_cls=s_indep_cls_seq,
                        s_hand_cls=s_hand_cls_seq,
                        s_foot_cls=s_foot_cls_seq,
                        s_move_cls=s_move_cls_seq,
                        padding_mask=batch_padding_mask
                    )

                    losses = model.compute_loss(mu, logvar, 0.0, 0.0, dec_logits, batch_dec_tgt)
                if not (batch_idx + 1) % 10:
                    print(f'batch #{batch_idx + 1}:', round(losses['recons_loss'].item(), 3))

                loss_rec.append(losses['recons_loss'].item())
                kl_loss_rec.append(losses['kldiv_raw'].item())

    return loss_rec, kl_loss_rec


def main():
    parser = argparse.ArgumentParser(description='Train MuseMorphose model on drum notation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Overrides config file.')
    args = parser.parse_args()

    # 設定を読み込み
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method'
    )
    config['data']['tokenization_method'] = tokenization_method
    use_difficulty = config['model'].get('use_difficulty', False)

    ensure_runtime_mode_compatibility(
        requested_mode=tokenization_method,
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
    )
    artifact_mode, artifact_mode_source = resolve_artifact_tokenization_method(
        config['data']['vocab_path'],
        config['data']['data_dir'],
    )

    # デバイスの選択（コマンドライン引数が優先）
    if args.device is not None:
        device = args.device
    else:
        device = config['training']['device']

    # CUDAの利用可能性をチェック
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('[warning] CUDA is not available. Falling back to CPU.')
            device = 'cpu'
        else:
            try:
                # CUDAデバイスをテスト
                test_tensor = torch.zeros(1).to(device)
                print(f'[info] Using CUDA device: {torch.cuda.get_device_name(0)}')
                print(f'[info] CUDA capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
            except RuntimeError as e:
                print(f'[warning] CUDA error detected: {e}')
                print('[warning] Falling back to CPU.')
                device = 'cpu'

    print(f'[info] Using device: {device}')
    print(f'[info] Tokenization method: {tokenization_method}')
    if artifact_mode is not None:
        print(f'[info] Artifact tokenization mode: {artifact_mode} ({artifact_mode_source})')
    else:
        print('[info] Artifact tokenization mode: unknown (legacy standard fallback)')
    trained_steps = config['training']['trained_steps']
    lr_decay_steps = config['training']['lr_decay_steps']
    max_lr = config['training']['max_lr']
    min_lr = config['training']['min_lr']
    ckpt_dir = config['training']['ckpt_dir']
    params_dir = os.path.join(ckpt_dir, 'params/')
    optim_dir = os.path.join(ckpt_dir, 'optim/')
    pretrained_params_path = config['model'].get('pretrained_params_path', None)
    pretrained_optim_path = config['model'].get('pretrained_optim_path', None)

    # データセットを読み込み
    print('[info] Loading training dataset...')
    dset = DrumTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['data']['dec_seqlen'],
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['train_split']),
        pad_to_same=True,
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method
    )

    print('[info] Loading validation dataset...')
    dset_val = DrumTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['data']['dec_seqlen'],
        model_max_bars=config['data']['max_bars'],
        pieces=pickle_load(config['data']['val_split']),
        pad_to_same=True,
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method
    )

    print(f'[info] # training samples: {len(dset.pieces)}')
    print(f'[info] # validation samples: {len(dset_val.pieces)}')

    dloader = DataLoader(
        dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 8)
    )
    dloader_val = DataLoader(
        dset_val,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 8)
    )

    # モデルを構築
    print('[info] Building model...')
    mconf = config['model']

    # まずCPU上でモデルを構築
    model = MuseMorphose(
        mconf['enc_n_layer'],
        mconf['enc_n_head'],
        mconf['enc_d_model'],
        mconf['enc_d_ff'],
        mconf['dec_n_layer'],
        mconf['dec_n_head'],
        mconf['dec_d_model'],
        mconf['dec_d_ff'],
        mconf['d_latent'],
        mconf['d_embed'],
        dset.vocab_size,
        d_polyph_emb=mconf.get('d_polyph_emb', 0),
        d_rfreq_emb=mconf.get('d_rfreq_emb', 0),
        d_s_tech_emb=mconf.get('d_s_tech_emb', 32),
        d_s_indep_emb=mconf.get('d_s_indep_emb', 32),
        d_s_hand_emb=mconf.get('d_s_hand_emb', 32),
        d_s_foot_emb=mconf.get('d_s_foot_emb', 32),
        d_s_move_emb=mconf.get('d_s_move_emb', 32),
        n_s_tech_cls=mconf.get('n_s_tech_cls', 8),
        n_s_indep_cls=mconf.get('n_s_indep_cls', 8),
        n_s_hand_cls=mconf.get('n_s_hand_cls', 8),
        n_s_foot_cls=mconf.get('n_s_foot_cls', 8),
        n_s_move_cls=mconf.get('n_s_move_cls', 8),
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method,
        cp_event_type_vocab_size=len(getattr(dset, 'event_type2idx', {'<PAD_EVENT>': 2})) if tokenization_method == 'cp_limb_v1' else 3,
        cp_struct_vocab_size=dset.vocab_size,
        cp_pos_vocab_size=(getattr(dset, 'pos_pad_value', 24) + 1) if tokenization_method == 'cp_limb_v1' else 25,
        cp_limb_vocab_size=len(getattr(dset, 'limb_token2idx', {'<PAD>': 0})) if tokenization_method == 'cp_limb_v1' else 2,
        cp_event_pad_idx=getattr(dset, 'event_type2idx', {'<PAD_EVENT>': 2}).get('<PAD_EVENT>', 2) if tokenization_method == 'cp_limb_v1' else 2,
        cp_struct_pad_idx=getattr(dset, 'struct_token2idx', {'<PAD>': 0}).get('<PAD>', 0) if tokenization_method == 'cp_limb_v1' else 0,
        cp_pos_pad_idx=getattr(dset, 'pos_pad_value', 24),
        cp_limb_pad_idx=getattr(dset, 'limb_token2idx', {'<PAD>': 0}).get('<PAD>', 0) if tokenization_method == 'cp_limb_v1' else 0,
        d_cp_pos_emb=mconf.get('d_cp_pos_emb', 64),
        d_cp_limb_emb=mconf.get('d_cp_limb_emb', 64),
        cond_mode=mconf.get('cond_mode', 'none')
    )

    # パラメータ数を計算
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[info] model # params: {n_params}')
    print(f'[info] vocabulary size: {dset.vocab_size}')

    # デバイスに移動
    try:
        model = model.to(device)
        print(f'[info] Model successfully moved to {device}')
    except RuntimeError as e:
        print(f'[error] Failed to move model to {device}: {e}')
        print('[error] Try running with --device cpu')
        sys.exit(1)

    if pretrained_params_path:
        print(f'[info] Loading pretrained params from {pretrained_params_path}')
        # デバイスを考慮してロード
        if device == 'cpu':
            model.load_state_dict(torch.load(pretrained_params_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(pretrained_params_path))

    model.train()

    # オプティマイザーを構築
    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)

    if pretrained_optim_path:
        print(f'[info] Loading pretrained optimizer from {pretrained_optim_path}')
        # デバイスを考慮してロード
        if device == 'cpu':
            optimizer.load_state_dict(torch.load(pretrained_optim_path, map_location='cpu'))
        else:
            optimizer.load_state_dict(torch.load(pretrained_optim_path))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )

    # Mixed Precision Training用のGradScaler（CUDAのみ）
    use_amp = config['training'].get('use_amp', False) and device == 'cuda'
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print('[info] Mixed Precision Training (AMP) enabled')

    # ディレクトリを作成
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optim_dir):
        os.makedirs(optim_dir)

    # 設定を保存
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

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


if __name__ == "__main__":
    main()
