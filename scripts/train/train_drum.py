"""
ドラム譜用MuseMorphoseモデルの学習スクリプト
"""
import sys
import os
import time
import json

from src.models.musemorphose import MuseMorphose
from src.models.limb_bias import build_token_to_limb_map
from src.data.drum_dataloader import DrumTransformerDataset
from torch.utils.data import DataLoader
from src.utils.tokenization_mode import (
    ensure_batch_mode_consistency,
    ensure_runtime_mode_compatibility,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
)

from src.utils.utils import pickle_load
from torch import nn, optim
import torch
import numpy as np
import random

import yaml
import argparse

class EarlyStopping:
    """
    検証損失を監視し、patience 回改善しなければ学習を停止する。

    Args:
        patience             : 改善なしを何検証回分許容するか
        min_delta            : 改善とみなす最小変化量（この値以上減少しなければ改善なしとみなす）
        restore_best_weights : 停止時に最良モデルの重みを復元するか
    """
    def __init__(self, patience: int, min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0        # 改善なし検証回数
        self.best_loss = None   # EarlyStopping 用ベスト損失（checkpoint 保存とは独立）
        self.triggered = False  # 停止フラグ

    def check(self, current_val_loss: float) -> bool:
        """
        val_loss を受け取り、停止すべきかどうかを返す。
        Returns: True = 停止すべき
        """
        if self.best_loss is None or current_val_loss < self.best_loss - self.min_delta:
            self.best_loss = current_val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'[early_stopping] no improvement {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def _safe_positive_int(value, default_value, field_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        print(f'[warning] Invalid {field_name}={value}. Falling back to {default_value}.')
        return default_value

    if parsed <= 0:
        print(f'[warning] Non-positive {field_name}={parsed}. Falling back to {default_value}.')
        return default_value

    return parsed


def _resolve_runtime_model_options(mconf, dset, tokenization_method):
    cond_mode = mconf.get('cond_mode', 'in-attn')
    if cond_mode not in ('in-attn', 'pre-attn'):
        print(f"[warning] Unsupported cond_mode={cond_mode}. Falling back to 'in-attn'.")
        cond_mode = 'in-attn'

    bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
    if bar_anchor_mode not in ('first_token', 'bar_token'):
        print(f"[warning] Unsupported bar_anchor_mode={bar_anchor_mode}. Falling back to 'first_token'.")
        bar_anchor_mode = 'first_token'

    bar_token_idx = getattr(dset, 'bar_token', None)
    if bar_anchor_mode == 'bar_token' and bar_token_idx is None:
        print("[warning] bar_anchor_mode=bar_token but bar token is unavailable. Falling back to first_token anchor.")
        bar_anchor_mode = 'first_token'

    use_hierarchical_pe = bool(mconf.get('use_hierarchical_pe', False))
    hier_pe_bar_vocab_size = _safe_positive_int(
        mconf.get('hier_pe_bar_vocab_size', 512),
        512,
        'model.hier_pe_bar_vocab_size'
    )
    hier_pe_beat_vocab_size = _safe_positive_int(
        mconf.get('hier_pe_beat_vocab_size', 8),
        8,
        'model.hier_pe_beat_vocab_size'
    )
    hier_pe_pos_vocab_size = _safe_positive_int(
        mconf.get('hier_pe_pos_vocab_size', 32),
        32,
        'model.hier_pe_pos_vocab_size'
    )

    d_cp_type_emb = _safe_positive_int(
        mconf.get('d_cp_type_emb', 64),
        64,
        'model.d_cp_type_emb'
    )

    if tokenization_method == 'cp_limb_v1':
        required_cp_attrs = (
            'type_token2idx',
            'event_type2idx',
            'limb_token2idx',
            'pos_pad_value',
        )
        missing_cp_attrs = [attr for attr in required_cp_attrs if not hasattr(dset, attr)]
        if missing_cp_attrs:
            raise ValueError(
                'cp_limb_v1 mode requires CP-compatible vocab/artifacts. '
                f'Missing keys: {missing_cp_attrs}. '
                'Please regenerate vocab/dataset artifacts for cp_limb_v1.'
            )

    cp_type_vocab_size = len(getattr(dset, 'type_token2idx', getattr(dset, 'struct_token2idx', {})))
    if cp_type_vocab_size <= 0:
        cp_type_vocab_size = dset.vocab_size

    return {
        'cond_mode': cond_mode,
        'bar_anchor_mode': bar_anchor_mode,
        'bar_token_idx': bar_token_idx,
        'use_hierarchical_pe': use_hierarchical_pe,
        'hier_pe_bar_vocab_size': hier_pe_bar_vocab_size,
        'hier_pe_beat_vocab_size': hier_pe_beat_vocab_size,
        'hier_pe_pos_vocab_size': hier_pe_pos_vocab_size,
        'hier_pe_use_struct_bias': bool(mconf.get('hier_pe_use_struct_bias', True)),
        'd_cp_type_emb': d_cp_type_emb,
        'cp_type_vocab_size': cp_type_vocab_size,
    }


def _ensure_difficulty_batch_contract(batch_samples, context):
    required_keys = (
        's_tech_cls_seq',
        's_indep_cls_seq',
        's_hand_cls_seq',
        's_foot_cls_seq',
        's_move_cls_seq',
    )
    missing = [key for key in required_keys if key not in batch_samples]
    if missing:
        raise ValueError(
            'Difficulty conditioning batch mismatch: '
            f'context={context}, missing_keys={missing}'
        )


def _shape_or_none(value):
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def _append_jsonl(log_path, payload):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _log_mode_loss_metrics(ckpt_dir, epoch, trained_steps, tokenization_method, losses):
    payload = {
        'epoch': int(epoch),
        'step': int(trained_steps),
        'mode': tokenization_method,
        'total_loss': float(losses['total_loss'].item()),
        'recons_loss': float(losses['recons_loss'].item()),
        'kldiv_loss': float(losses['kldiv_loss'].item()),
        'kldiv_raw': float(losses['kldiv_raw'].item()),
    }
    if tokenization_method == 'cp_limb_v1':
        payload.update({
            'cp_event_type_loss': float(losses['cp_event_type_loss'].item()),
            'cp_structural_loss': float(losses['cp_structural_loss'].item()),
            'cp_pos_loss': float(losses['cp_pos_loss'].item()),
            'cp_hand1_loss': float(losses['cp_hand1_loss'].item()),
            'cp_hand2_loss': float(losses['cp_hand2_loss'].item()),
            'cp_right_foot_loss': float(losses['cp_right_foot_loss'].item()),
            'cp_left_foot_loss': float(losses['cp_left_foot_loss'].item()),
        })

    _append_jsonl(os.path.join(ckpt_dir, 'loss_metrics.jsonl'), payload)


def _log_loss_failure(ckpt_dir, epoch, trained_steps, batch_idx, tokenization_method, error, shape_context):
    payload = {
        'epoch': int(epoch),
        'step': int(trained_steps),
        'batch_idx': int(batch_idx),
        'mode': tokenization_method,
        'error_type': type(error).__name__,
        'error': str(error),
        'shape_context': shape_context,
    }
    _append_jsonl(os.path.join(ckpt_dir, 'loss_failures.jsonl'), payload)


def _validate_training_config(config):
    required_sections = ('data', 'model', 'training')
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(
            f'Config sections are missing: {missing_sections}. Required: {required_sections}.'
        )

    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method'
    )
    config['data']['tokenization_method'] = tokenization_method

    use_difficulty = bool(config['model'].get('use_difficulty', False))
    config['model']['use_difficulty'] = use_difficulty

    use_attr_cls = bool(config['model'].get('use_attr_cls', True))
    if use_difficulty and not use_attr_cls:
        raise ValueError(
            'Invalid config combination: model.use_difficulty=true requires model.use_attr_cls=true '
            'because difficulty conditions are composed as decoder attributes.'
        )

    cp_loss_weights = config['model'].get('cp_loss_weights', {})
    if cp_loss_weights is None:
        cp_loss_weights = {}
        config['model']['cp_loss_weights'] = cp_loss_weights
    if not isinstance(cp_loss_weights, dict):
        raise ValueError(
            'Invalid config: model.cp_loss_weights must be a mapping (dict) of head_name -> weight.'
        )

    if tokenization_method != 'cp_limb_v1' and cp_loss_weights:
        print('[warning] model.cp_loss_weights is set but mode is not cp_limb_v1. The weights will be ignored.')

    if tokenization_method == 'cp_limb_v1' and 'structural' in cp_loss_weights:
        print('[warning] cp_loss_weights.structural is ignored because CP optimization uses structural-excluded 6-head loss.')

    if not config['data'].get('vocab_path'):
        raise ValueError('Invalid config: data.vocab_path must be provided.')
    if not config['data'].get('data_dir'):
        raise ValueError('Invalid config: data.data_dir must be provided.')

    if not config['training'].get('ckpt_dir'):
        raise ValueError('Invalid config: training.ckpt_dir must be provided.')

    return config


def _load_state_dict_with_compat(model, state_dict, desc='checkpoint'):
    try:
        model.load_state_dict(state_dict)
        print(f'[info] Loaded {desc} with strict=True')
    except RuntimeError as e:
        print(f'[warning] strict=True load failed for {desc}: {e}')
        print(f'[warning] Retrying {desc} load with strict=False for backward compatibility.')
        model.load_state_dict(state_dict, strict=False)


def train_model(epoch, model, dloader, dloader_val, optim, sched, config, trained_steps, scaler=None, best_val_loss=float('inf'), early_stopper=None):
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
        ensure_batch_mode_consistency(
            tokenization_method,
            batch_samples,
            context=f'train epoch={epoch} batch={batch_idx}',
        )
        if use_difficulty:
            _ensure_difficulty_batch_contract(
                batch_samples,
                context=f'train epoch={epoch} batch={batch_idx}',
            )
        model.zero_grad()

        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_inp_lens = batch_samples['length']
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
        enc_bar_idx = batch_samples['enc_bar_idx'].permute(2, 0, 1).to(device) if 'enc_bar_idx' in batch_samples else None
        enc_beat_idx = batch_samples['enc_beat_idx'].permute(2, 0, 1).to(device) if 'enc_beat_idx' in batch_samples else None
        enc_pos_idx = batch_samples['enc_pos_idx'].permute(2, 0, 1).to(device) if 'enc_pos_idx' in batch_samples else None
        enc_is_struct = batch_samples['enc_is_struct'].permute(2, 0, 1).to(device) if 'enc_is_struct' in batch_samples else None
        dec_bar_idx = batch_samples['dec_bar_idx'].permute(1, 0).to(device) if 'dec_bar_idx' in batch_samples else None
        dec_beat_idx = batch_samples['dec_beat_idx'].permute(1, 0).to(device) if 'dec_beat_idx' in batch_samples else None
        dec_pos_idx = batch_samples['dec_pos_idx'].permute(1, 0).to(device) if 'dec_pos_idx' in batch_samples else None
        dec_is_struct = batch_samples['dec_is_struct'].permute(1, 0).to(device) if 'dec_is_struct' in batch_samples else None

        cp_event_type_inp = batch_samples['cp_event_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
        cp_type_inp = batch_samples['cp_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
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
                        cp_type_inp=cp_type_inp,
                        cp_struct_inp=cp_struct_inp,
                        cp_pos_inp=cp_pos_inp,
                        cp_hand1_inp=cp_hand1_inp,
                        cp_hand2_inp=cp_hand2_inp,
                        cp_right_foot_inp=cp_rf_inp,
                        cp_left_foot_inp=cp_lf_inp,
                        enc_bar_idx=enc_bar_idx,
                        enc_beat_idx=enc_beat_idx,
                        enc_pos_idx=enc_pos_idx,
                        enc_is_struct=enc_is_struct,
                        dec_bar_idx=dec_bar_idx,
                        dec_beat_idx=dec_beat_idx,
                        dec_pos_idx=dec_pos_idx,
                        dec_is_struct=dec_is_struct,
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
                        enc_bar_idx=enc_bar_idx,
                        enc_beat_idx=enc_beat_idx,
                        enc_pos_idx=enc_pos_idx,
                        enc_is_struct=enc_is_struct,
                        dec_bar_idx=dec_bar_idx,
                        dec_beat_idx=dec_beat_idx,
                        dec_pos_idx=dec_pos_idx,
                        dec_is_struct=dec_is_struct,
                        padding_mask=batch_padding_mask
                    )

                # KLベータのスケジューリング
                if not constant_kl:
                    kl_beta = beta_cyclical_sched(trained_steps, no_kl_steps, kl_cycle_steps, kl_max_beta)
                else:
                    kl_beta = kl_max_beta

                if tokenization_method == 'cp_limb_v1':
                    try:
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
                    except Exception as e:
                        _log_loss_failure(
                            ckpt_dir,
                            epoch,
                            trained_steps,
                            batch_idx,
                            tokenization_method,
                            e,
                            {
                                'cp_event_type_target': _shape_or_none(cp_event_type_tgt),
                                'cp_struct_target': _shape_or_none(cp_struct_tgt),
                                'cp_pos_target': _shape_or_none(cp_pos_tgt),
                                'cp_hand1_target': _shape_or_none(cp_hand1_tgt),
                                'cp_hand2_target': _shape_or_none(cp_hand2_tgt),
                                'cp_right_foot_target': _shape_or_none(cp_rf_tgt),
                                'cp_left_foot_target': _shape_or_none(cp_lf_tgt),
                            }
                        )
                        raise
                else:
                    try:
                        losses = model.compute_loss(mu, logvar, kl_beta, free_bit_lambda, dec_logits, batch_dec_tgt)
                    except Exception as e:
                        _log_loss_failure(
                            ckpt_dir,
                            epoch,
                            trained_steps,
                            batch_idx,
                            tokenization_method,
                            e,
                            {
                                'dec_logits': _shape_or_none(dec_logits),
                                'dec_target': _shape_or_none(batch_dec_tgt),
                            }
                        )
                        raise

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
                    cp_type_inp=cp_type_inp,
                    cp_struct_inp=cp_struct_inp,
                    cp_pos_inp=cp_pos_inp,
                    cp_hand1_inp=cp_hand1_inp,
                    cp_hand2_inp=cp_hand2_inp,
                    cp_right_foot_inp=cp_rf_inp,
                    cp_left_foot_inp=cp_lf_inp,
                    enc_bar_idx=enc_bar_idx,
                    enc_beat_idx=enc_beat_idx,
                    enc_pos_idx=enc_pos_idx,
                    enc_is_struct=enc_is_struct,
                    dec_bar_idx=dec_bar_idx,
                    dec_beat_idx=dec_beat_idx,
                    dec_pos_idx=dec_pos_idx,
                    dec_is_struct=dec_is_struct,
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
                    enc_bar_idx=enc_bar_idx,
                    enc_beat_idx=enc_beat_idx,
                    enc_pos_idx=enc_pos_idx,
                    enc_is_struct=enc_is_struct,
                    dec_bar_idx=dec_bar_idx,
                    dec_beat_idx=dec_beat_idx,
                    dec_pos_idx=dec_pos_idx,
                    dec_is_struct=dec_is_struct,
                    padding_mask=batch_padding_mask
                )

            # KLベータのスケジューリング
            if not constant_kl:
                kl_beta = beta_cyclical_sched(trained_steps, no_kl_steps, kl_cycle_steps, kl_max_beta)
            else:
                kl_beta = kl_max_beta

            if tokenization_method == 'cp_limb_v1':
                try:
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
                except Exception as e:
                    _log_loss_failure(
                        ckpt_dir,
                        epoch,
                        trained_steps,
                        batch_idx,
                        tokenization_method,
                        e,
                        {
                            'cp_event_type_target': _shape_or_none(cp_event_type_tgt),
                            'cp_struct_target': _shape_or_none(cp_struct_tgt),
                            'cp_pos_target': _shape_or_none(cp_pos_tgt),
                            'cp_hand1_target': _shape_or_none(cp_hand1_tgt),
                            'cp_hand2_target': _shape_or_none(cp_hand2_tgt),
                            'cp_right_foot_target': _shape_or_none(cp_rf_tgt),
                            'cp_left_foot_target': _shape_or_none(cp_lf_tgt),
                        }
                    )
                    raise
            else:
                try:
                    losses = model.compute_loss(mu, logvar, kl_beta, free_bit_lambda, dec_logits, batch_dec_tgt)
                except Exception as e:
                    _log_loss_failure(
                        ckpt_dir,
                        epoch,
                        trained_steps,
                        batch_idx,
                        tokenization_method,
                        e,
                        {
                            'dec_logits': _shape_or_none(dec_logits),
                            'dec_target': _shape_or_none(batch_dec_tgt),
                        }
                    )
                    raise

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
            _log_mode_loss_metrics(
                ckpt_dir,
                epoch,
                trained_steps,
                tokenization_method,
                losses,
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

            # Early Stopping チェック
            if early_stopper is not None:
                if early_stopper.check(current_val_loss):
                    print(f'[early_stopping] triggered after {trained_steps} steps '
                          f'(patience={early_stopper.patience})')
                    return trained_steps, best_val_loss

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
    n_rounds = config['training'].get('val_n_rounds', n_rounds)
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
                ensure_batch_mode_consistency(
                    tokenization_method,
                    batch_samples,
                    context=f'validate round={i + 1} batch={batch_idx}',
                )
                if use_difficulty:
                    _ensure_difficulty_batch_contract(
                        batch_samples,
                        context=f'validate round={i + 1} batch={batch_idx}',
                    )
                model.zero_grad()

                batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
                batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
                batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
                batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
                batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
                enc_bar_idx = batch_samples['enc_bar_idx'].permute(2, 0, 1).to(device) if 'enc_bar_idx' in batch_samples else None
                enc_beat_idx = batch_samples['enc_beat_idx'].permute(2, 0, 1).to(device) if 'enc_beat_idx' in batch_samples else None
                enc_pos_idx = batch_samples['enc_pos_idx'].permute(2, 0, 1).to(device) if 'enc_pos_idx' in batch_samples else None
                enc_is_struct = batch_samples['enc_is_struct'].permute(2, 0, 1).to(device) if 'enc_is_struct' in batch_samples else None
                dec_bar_idx = batch_samples['dec_bar_idx'].permute(1, 0).to(device) if 'dec_bar_idx' in batch_samples else None
                dec_beat_idx = batch_samples['dec_beat_idx'].permute(1, 0).to(device) if 'dec_beat_idx' in batch_samples else None
                dec_pos_idx = batch_samples['dec_pos_idx'].permute(1, 0).to(device) if 'dec_pos_idx' in batch_samples else None
                dec_is_struct = batch_samples['dec_is_struct'].permute(1, 0).to(device) if 'dec_is_struct' in batch_samples else None

                cp_event_type_inp = batch_samples['cp_event_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
                cp_type_inp = batch_samples['cp_type_input'].permute(1, 0).to(device) if tokenization_method == 'cp_limb_v1' else None
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
                        cp_type_inp=cp_type_inp,
                        cp_struct_inp=cp_struct_inp,
                        cp_pos_inp=cp_pos_inp,
                        cp_hand1_inp=cp_hand1_inp,
                        cp_hand2_inp=cp_hand2_inp,
                        cp_right_foot_inp=cp_rf_inp,
                        cp_left_foot_inp=cp_lf_inp,
                        enc_bar_idx=enc_bar_idx,
                        enc_beat_idx=enc_beat_idx,
                        enc_pos_idx=enc_pos_idx,
                        enc_is_struct=enc_is_struct,
                        dec_bar_idx=dec_bar_idx,
                        dec_beat_idx=dec_beat_idx,
                        dec_pos_idx=dec_pos_idx,
                        dec_is_struct=dec_is_struct,
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
                        enc_bar_idx=enc_bar_idx,
                        enc_beat_idx=enc_beat_idx,
                        enc_pos_idx=enc_pos_idx,
                        enc_is_struct=enc_is_struct,
                        dec_bar_idx=dec_bar_idx,
                        dec_beat_idx=dec_beat_idx,
                        dec_pos_idx=dec_pos_idx,
                        dec_is_struct=dec_is_struct,
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
    config = _validate_training_config(config)

    seed = config['training'].get('seed', None)
    if seed is not None:
        set_seed(seed)
        print(f'[info] Random seed fixed to {seed}')

    tokenization_method = config['data']['tokenization_method']
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

    dl_generator = torch.Generator()
    if seed is not None:
        dl_generator.manual_seed(seed)

    dloader = DataLoader(
        dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 8),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=dl_generator,
    )
    dloader_val = DataLoader(
        dset_val,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 8),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=dl_generator,
    )

    # モデルを構築
    print('[info] Building model...')
    mconf = config['model']
    runtime_opts = _resolve_runtime_model_options(mconf, dset, tokenization_method)

    resolved_n_s_tech_cls = mconf.get('n_s_tech_cls', 8)
    resolved_n_s_indep_cls = mconf.get('n_s_indep_cls', 8)
    resolved_n_s_hand_cls = mconf.get('n_s_hand_cls', 8)
    resolved_n_s_foot_cls = mconf.get('n_s_foot_cls', 8)
    resolved_n_s_move_cls = mconf.get('n_s_move_cls', 8)

    if use_difficulty:
        dataset_cls = getattr(dset, 'difficulty_num_classes', {})
        resolved_n_s_tech_cls = max(resolved_n_s_tech_cls, int(dataset_cls.get('s_tech', resolved_n_s_tech_cls)))
        resolved_n_s_indep_cls = max(resolved_n_s_indep_cls, int(dataset_cls.get('s_indep', resolved_n_s_indep_cls)))
        resolved_n_s_hand_cls = max(resolved_n_s_hand_cls, int(dataset_cls.get('s_hand', resolved_n_s_hand_cls)))
        resolved_n_s_foot_cls = max(resolved_n_s_foot_cls, int(dataset_cls.get('s_foot', resolved_n_s_foot_cls)))
        resolved_n_s_move_cls = max(resolved_n_s_move_cls, int(dataset_cls.get('s_move', resolved_n_s_move_cls)))

        pad_label = getattr(dset, 'difficulty_pad_label', {})
        print('[info] Difficulty padding labels: '
              f"tech={pad_label.get('s_tech')}, indep={pad_label.get('s_indep')}, "
              f"hand={pad_label.get('s_hand')}, foot={pad_label.get('s_foot')}, "
              f"move={pad_label.get('s_move')}")
        print('[info] Difficulty class sizes (including PAD): '
              f'tech={resolved_n_s_tech_cls}, indep={resolved_n_s_indep_cls}, '
              f'hand={resolved_n_s_hand_cls}, foot={resolved_n_s_foot_cls}, '
              f'move={resolved_n_s_move_cls}')

    # limb bias 用の token_to_limb マップを構築
    use_limb_bias_encoder = bool(mconf.get('use_limb_bias_encoder', False))
    use_limb_bias_decoder = bool(mconf.get('use_limb_bias_decoder', False))
    alpha_limb_fixed = mconf.get('limb_bias_alpha', None)
    if alpha_limb_fixed is not None:
        alpha_limb_fixed = float(alpha_limb_fixed)
        print(f'[info] limb bias alpha: fixed={alpha_limb_fixed}')
    else:
        print('[info] limb bias alpha: learnable')
    token_to_limb = None
    if use_limb_bias_encoder or use_limb_bias_decoder:
        idx2token = getattr(dset, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(dset.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            print('[warning] idx2token not found in dataset; limb bias disabled.')
            use_limb_bias_encoder = False
            use_limb_bias_decoder = False

    # position bias 用のパラメータと token ID マップを構築
    use_position_bias_encoder = bool(mconf.get('use_position_bias_encoder', False))
    use_position_bias_decoder = bool(mconf.get('use_position_bias_decoder', False))
    position_bias_mode = mconf.get('position_bias_mode', 'uniform')
    alpha_pos_fixed = mconf.get('position_bias_alpha', None)
    if alpha_pos_fixed is not None:
        alpha_pos_fixed = float(alpha_pos_fixed)
        print(f'[info] position bias alpha: fixed={alpha_pos_fixed}')
    else:
        print('[info] position bias alpha: learnable')
    position_bias_per_layer = bool(mconf.get('position_bias_per_layer', True))
    pos_token_ids = None
    if use_position_bias_encoder or use_position_bias_decoder:
        t2i = getattr(dset, 'token2idx', None)
        if t2i is not None:
            pos_keys  = sorted([k for k in t2i if k.startswith('<POS_')],  key=lambda k: t2i[k])
            beat_keys = sorted([k for k in t2i if k.startswith('<BEAT_')], key=lambda k: t2i[k])
            if pos_keys and beat_keys:
                pos_token_ids = {
                    'pos_start':  t2i[pos_keys[0]],
                    'pos_end':    t2i[pos_keys[-1]],
                    'beat_start': t2i[beat_keys[0]],
                    'beat_end':   t2i[beat_keys[-1]],
                    'pad':        t2i.get('<PAD>', 0),
                    'eos':        t2i.get('<EOS>', 1),
                }
                print(f'[info] position bias token IDs: {pos_token_ids}')
            else:
                print('[warning] POS/BEAT tokens not found in token2idx; position bias disabled.')
                use_position_bias_encoder = False
                use_position_bias_decoder = False
        else:
            print('[warning] token2idx not found in dataset; position bias disabled.')
            use_position_bias_encoder = False
            use_position_bias_decoder = False

    # position bias は token_to_limb_table も必要（forward の _pos_ready チェック用）
    # limb_bias が無効でも position_bias が有効なら token_to_limb を構築する
    if (use_position_bias_encoder or use_position_bias_decoder) and token_to_limb is None:
        idx2token = getattr(dset, 'idx2token', None)
        if idx2token is not None:
            vocab_list = [idx2token.get(i, '') for i in range(dset.vocab_size)]
            token_to_limb = build_token_to_limb_map(vocab_list)
        else:
            print('[warning] idx2token not found in dataset; position bias disabled.')
            use_position_bias_encoder = False
            use_position_bias_decoder = False
            pos_token_ids = None

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
        n_s_tech_cls=resolved_n_s_tech_cls,
        n_s_indep_cls=resolved_n_s_indep_cls,
        n_s_hand_cls=resolved_n_s_hand_cls,
        n_s_foot_cls=resolved_n_s_foot_cls,
        n_s_move_cls=resolved_n_s_move_cls,
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method,
        cp_event_type_vocab_size=len(getattr(dset, 'event_type2idx', {'<PAD_EVENT>': 2})) if tokenization_method == 'cp_limb_v1' else 3,
        cp_type_vocab_size=runtime_opts['cp_type_vocab_size'],
        cp_struct_vocab_size=dset.vocab_size,
        cp_pos_vocab_size=(getattr(dset, 'pos_pad_value', 24) + 1) if tokenization_method == 'cp_limb_v1' else 25,
        cp_limb_vocab_size=len(getattr(dset, 'limb_token2idx', {'<PAD>': 0})) if tokenization_method == 'cp_limb_v1' else 2,
        cp_event_pad_idx=getattr(dset, 'event_type2idx', {'<PAD_EVENT>': 2}).get('<PAD_EVENT>', 2) if tokenization_method == 'cp_limb_v1' else 2,
        cp_struct_pad_idx=getattr(dset, 'struct_token2idx', {'<PAD>': 0}).get('<PAD>', 0) if tokenization_method == 'cp_limb_v1' else 0,
        cp_pos_pad_idx=getattr(dset, 'pos_pad_value', 24),
        cp_limb_pad_idx=getattr(dset, 'limb_token2idx', {'<PAD>': 0}).get('<PAD>', 0) if tokenization_method == 'cp_limb_v1' else 0,
        bar_anchor_mode=runtime_opts['bar_anchor_mode'],
        bar_token_idx=runtime_opts['bar_token_idx'],
        d_cp_type_emb=runtime_opts['d_cp_type_emb'],
        d_cp_pos_emb=mconf.get('d_cp_pos_emb', 64),
        d_cp_limb_emb=mconf.get('d_cp_limb_emb', 64),
        cond_mode=runtime_opts['cond_mode'],
        use_hierarchical_pe=runtime_opts['use_hierarchical_pe'],
        hier_pe_bar_vocab_size=runtime_opts['hier_pe_bar_vocab_size'],
        hier_pe_beat_vocab_size=runtime_opts['hier_pe_beat_vocab_size'],
        hier_pe_pos_vocab_size=runtime_opts['hier_pe_pos_vocab_size'],
        hier_pe_use_struct_bias=runtime_opts['hier_pe_use_struct_bias'],
        use_limb_bias_encoder=use_limb_bias_encoder,
        use_limb_bias_decoder=use_limb_bias_decoder,
        token_to_limb=token_to_limb,
        alpha_limb_fixed=alpha_limb_fixed,
        use_position_bias_encoder=use_position_bias_encoder,
        use_position_bias_decoder=use_position_bias_decoder,
        position_bias_mode=position_bias_mode,
        alpha_pos_fixed=alpha_pos_fixed,
        position_bias_per_layer=position_bias_per_layer,
        pos_token_ids=pos_token_ids,
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
            pretrained_state = torch.load(pretrained_params_path, map_location='cpu')
        else:
            pretrained_state = torch.load(pretrained_params_path)
        _load_state_dict_with_compat(model, pretrained_state, desc='pretrained params')

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

    # Early Stopping の初期化
    es_cfg = config['training'].get('early_stopping', {})
    early_stopper = None
    if es_cfg.get('enabled', False):
        early_stopper = EarlyStopping(
            patience=int(es_cfg['patience']),
            min_delta=float(es_cfg.get('min_delta', 0.0)),
            restore_best_weights=bool(es_cfg.get('restore_best_weights', True)),
        )
        print(f'[info] EarlyStopping enabled: patience={early_stopper.patience}, '
              f'min_delta={early_stopper.min_delta}, '
              f'restore_best_weights={early_stopper.restore_best_weights}')

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
            best_val_loss=best_val_loss,
            early_stopper=early_stopper,
        )
        if early_stopper is not None and early_stopper.triggered:
            break

    # Early Stopping: best weights の復元
    if early_stopper is not None and early_stopper.triggered:
        if early_stopper.restore_best_weights:
            best_params_path_es = os.path.join(params_dir, 'best_params.pt')
            if os.path.exists(best_params_path_es):
                _load_state_dict_with_compat(
                    model,
                    torch.load(best_params_path_es, map_location=device),
                    desc='best_params.pt'
                )
                print(f'[early_stopping] Restored best weights from {best_params_path_es}')
            else:
                print('[early_stopping] Warning: best_params.pt not found, weights not restored')
        print(f'[early_stopping] Training stopped early at step {trained_steps}')

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
