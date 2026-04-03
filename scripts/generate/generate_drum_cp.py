"""
CP方式専用ドラム生成スクリプト

tokenization_method=cp_limb_v1 で学習したモデルを使い、
CP多頭出力（event_type/struct/pos/hand1/hand2/rf/lf）から
自己回帰生成してMIDIへ変換する。

Usage:
  python generate_drum_cp.py <config> <ckpt> <output_dir> <n_pieces> <n_samples_per_piece>
"""

import os
import sys
import time
import random
import argparse
from copy import deepcopy

import numpy as np
import torch
import yaml
from scipy.stats import entropy

from drum_dataloader import DrumTransformerDataset
from model.musemorphose import MuseMorphose
from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from drum_to_midi import cp_data_to_midi

try:
    from src.utils.tokenization_mode import (
        ensure_runtime_mode_compatibility,
        resolve_artifact_tokenization_method,
        validate_tokenization_method,
    )
except ModuleNotFoundError:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.utils.tokenization_mode import (
        ensure_runtime_mode_compatibility,
        resolve_artifact_tokenization_method,
        validate_tokenization_method,
    )


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


def _resolve_runtime_model_options(mconf, dset):
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

    return {
        'cond_mode': cond_mode,
        'bar_anchor_mode': bar_anchor_mode,
        'bar_token_idx': bar_token_idx,
        'use_hierarchical_pe': bool(mconf.get('use_hierarchical_pe', False)),
        'hier_pe_bar_vocab_size': _safe_positive_int(mconf.get('hier_pe_bar_vocab_size', 512), 512, 'model.hier_pe_bar_vocab_size'),
        'hier_pe_beat_vocab_size': _safe_positive_int(mconf.get('hier_pe_beat_vocab_size', 8), 8, 'model.hier_pe_beat_vocab_size'),
        'hier_pe_pos_vocab_size': _safe_positive_int(mconf.get('hier_pe_pos_vocab_size', 32), 32, 'model.hier_pe_pos_vocab_size'),
        'hier_pe_use_struct_bias': bool(mconf.get('hier_pe_use_struct_bias', True)),
        'd_cp_type_emb': _safe_positive_int(mconf.get('d_cp_type_emb', 64), 64, 'model.d_cp_type_emb'),
    }


def _load_state_dict_with_compat(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
        print('[info] Loaded checkpoint with strict=True')
    except RuntimeError as e:
        print(f'[warning] strict=True load failed: {e}')
        print('[warning] Retrying checkpoint load with strict=False for backward compatibility.')
        model.load_state_dict(state_dict, strict=False)


def temperatured_softmax(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except Exception:
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs = probs / np.sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p

    if np.sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0]
        candi_index = sorted_index[:max(last_index, 1)]
    else:
        candi_index = sorted_index[:3]

    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= np.sum(candi_probs)
    return np.random.choice(candi_index, size=1, p=candi_probs)[0]


def sample_excluding(logits, temperature, nucleus_p, forbidden_idx=None):
    probs = temperatured_softmax(logits, temperature)
    if forbidden_idx is not None and 0 <= forbidden_idx < len(probs):
        probs[forbidden_idx] = 0.0
        s = probs.sum()
        if s > 0:
            probs /= s
    word = nucleus(probs, nucleus_p)
    return int(word), float(entropy(probs))


def get_latent_embedding_fast(model, piece_data, device='cuda', use_sampling=False, sampling_var=0.0):
    batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)
    enc_bar_idx = piece_data['enc_bar_idx'].permute(1, 0).long().to(device) if 'enc_bar_idx' in piece_data else None
    enc_beat_idx = piece_data['enc_beat_idx'].permute(1, 0).long().to(device) if 'enc_beat_idx' in piece_data else None
    enc_pos_idx = piece_data['enc_pos_idx'].permute(1, 0).long().to(device) if 'enc_pos_idx' in piece_data else None
    enc_is_struct = piece_data['enc_is_struct'].permute(1, 0).long().to(device) if 'enc_is_struct' in piece_data else None

    with torch.no_grad():
        piece_latents = model.get_sampled_latent(
            batch_inp,
            padding_mask=batch_padding_mask,
            use_sampling=use_sampling,
            sampling_var=sampling_var,
            enc_bar_idx=enc_bar_idx,
            enc_beat_idx=enc_beat_idx,
            enc_pos_idx=enc_pos_idx,
            enc_is_struct=enc_is_struct,
        )
    return piece_latents


def _parse_beat_token(token):
    if not isinstance(token, str):
        return None
    if token.startswith('<BEAT_') and token.endswith('>'):
        try:
            beat_idx = int(token[len('<BEAT_'):-1])
            return max(1, min(4, beat_idx))
        except ValueError:
            return None
    return None


def generate_cp_sequence(
    model,
    latents,
    struct_token2idx,
    idx2struct_token,
    idx2limb_token,
    pos_pad_value,
    limb_pad_idx,
    event_type_pad_idx,
    device='cuda',
    max_events=12800,
    max_input_len=1280,
    truncate_len=512,
    nucleus_p=0.9,
    temperature=1.2,
    use_difficulty=False,
    s_tech_cls=None,
    s_indep_cls=None,
    s_hand_cls=None,
    s_foot_cls=None,
    s_move_cls=None,
    use_hierarchical_pe=False,
):
    target_bars = latents.size(0)
    note_type_idx = struct_token2idx.get('<NOTE>', struct_token2idx['<PAD>'])
    pad_type_idx = struct_token2idx['<PAD>']

    latent_placeholder = torch.zeros(max_events, 1, latents.size(-1), device=device)
    event_type_placeholder = torch.full((max_events, 1), event_type_pad_idx, dtype=torch.long, device=device)
    cp_type_placeholder = torch.full((max_events, 1), pad_type_idx, dtype=torch.long, device=device)
    struct_placeholder = torch.full((max_events, 1), struct_token2idx['<PAD>'], dtype=torch.long, device=device)
    pos_placeholder = torch.full((max_events, 1), pos_pad_value, dtype=torch.long, device=device)
    h1_placeholder = torch.full((max_events, 1), limb_pad_idx, dtype=torch.long, device=device)
    h2_placeholder = torch.full((max_events, 1), limb_pad_idx, dtype=torch.long, device=device)
    rf_placeholder = torch.full((max_events, 1), limb_pad_idx, dtype=torch.long, device=device)
    lf_placeholder = torch.full((max_events, 1), limb_pad_idx, dtype=torch.long, device=device)

    if use_hierarchical_pe:
        dec_bar_idx_ph = torch.zeros((max_events, 1), dtype=torch.long, device=device)
        dec_beat_idx_ph = torch.ones((max_events, 1), dtype=torch.long, device=device)
        dec_pos_idx_ph = torch.zeros((max_events, 1), dtype=torch.long, device=device)
        dec_is_struct_ph = torch.zeros((max_events, 1), dtype=torch.long, device=device)
        hier_state_valid = True
    else:
        dec_bar_idx_ph = dec_beat_idx_ph = dec_pos_idx_ph = dec_is_struct_ph = None
        hier_state_valid = False

    if use_difficulty:
        s_tech_ph = torch.zeros(max_events, 1, dtype=torch.long, device=device)
        s_indep_ph = torch.zeros(max_events, 1, dtype=torch.long, device=device)
        s_hand_ph = torch.zeros(max_events, 1, dtype=torch.long, device=device)
        s_foot_ph = torch.zeros(max_events, 1, dtype=torch.long, device=device)
        s_move_ph = torch.zeros(max_events, 1, dtype=torch.long, device=device)
    else:
        s_tech_ph = s_indep_ph = s_hand_ph = s_foot_ph = s_move_ph = None

    # primer: <TEMPO_120>, <BAR>
    generated = []
    primer_struct = '<TEMPO_120>' if '<TEMPO_120>' in struct_token2idx else '<BAR>'
    for tok in [primer_struct, '<BAR>']:
        generated.append((0, struct_token2idx[tok], pos_pad_value, limb_pad_idx, limb_pad_idx, limb_pad_idx, limb_pad_idx))

    for i, ev in enumerate(generated):
        event_type_placeholder[i, 0] = ev[0]
        cp_type_placeholder[i, 0] = ev[1]
        struct_placeholder[i, 0] = ev[1]
        pos_placeholder[i, 0] = ev[2]
        h1_placeholder[i, 0] = ev[3]
        h2_placeholder[i, 0] = ev[4]
        rf_placeholder[i, 0] = ev[5]
        lf_placeholder[i, 0] = ev[6]
        if use_hierarchical_pe:
            dec_bar_idx_ph[i, 0] = 0
            dec_beat_idx_ph[i, 0] = 1
            dec_pos_idx_ph[i, 0] = 0
            dec_is_struct_ph[i, 0] = 1

    generated_bars = 0
    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []
    current_beat = 1
    current_pos = 0

    time_st = time.time()

    while generated_bars < target_bars and len(generated_final) < max_events:
        step_idx = len(generated) - 1
        bar_idx = min(max(generated_bars, 0), target_bars - 1)

        latent_placeholder[step_idx, 0, :] = latents[bar_idx]

        if use_difficulty and s_tech_cls is not None:
            s_tech_ph[step_idx, 0] = s_tech_cls[bar_idx]
            s_indep_ph[step_idx, 0] = s_indep_cls[bar_idx]
            s_hand_ph[step_idx, 0] = s_hand_cls[bar_idx]
            s_foot_ph[step_idx, 0] = s_foot_cls[bar_idx]
            s_move_ph[step_idx, 0] = s_move_cls[bar_idx]

        seq_len = len(generated)
        dec_struct = struct_placeholder[:seq_len, :]

        with torch.no_grad():
            cp_logits = model.generate(
                dec_struct,
                latent_placeholder[:seq_len, :],
                s_tech_cls=s_tech_ph[:seq_len, :] if use_difficulty else None,
                s_indep_cls=s_indep_ph[:seq_len, :] if use_difficulty else None,
                s_hand_cls=s_hand_ph[:seq_len, :] if use_difficulty else None,
                s_foot_cls=s_foot_ph[:seq_len, :] if use_difficulty else None,
                s_move_cls=s_move_ph[:seq_len, :] if use_difficulty else None,
                cp_event_type_inp=event_type_placeholder[:seq_len, :],
                cp_type_inp=cp_type_placeholder[:seq_len, :],
                cp_struct_inp=dec_struct,
                cp_pos_inp=pos_placeholder[:seq_len, :],
                cp_hand1_inp=h1_placeholder[:seq_len, :],
                cp_hand2_inp=h2_placeholder[:seq_len, :],
                cp_right_foot_inp=rf_placeholder[:seq_len, :],
                cp_left_foot_inp=lf_placeholder[:seq_len, :],
                dec_bar_idx=dec_bar_idx_ph[:seq_len, :] if hier_state_valid else None,
                dec_beat_idx=dec_beat_idx_ph[:seq_len, :] if hier_state_valid else None,
                dec_pos_idx=dec_pos_idx_ph[:seq_len, :] if hier_state_valid else None,
                dec_is_struct=dec_is_struct_ph[:seq_len, :] if hier_state_valid else None,
                keep_last_only=True,
            )

        # cp_logits shape: [bsz=1, vocab]
        event_logits = tensor_to_numpy(cp_logits['event_type'][0])
        ev_type, e_entropy = sample_excluding(event_logits, temperature, nucleus_p, forbidden_idx=event_type_pad_idx)

        if ev_type == 0:
            struct_logits = tensor_to_numpy(cp_logits['structural'][0])
            struct_idx, s_entropy = sample_excluding(struct_logits, temperature, nucleus_p, forbidden_idx=struct_token2idx['<PAD>'])

            pos_idx = pos_pad_value
            h1_idx = limb_pad_idx
            h2_idx = limb_pad_idx
            rf_idx = limb_pad_idx
            lf_idx = limb_pad_idx

            st_tok = idx2struct_token.get(struct_idx, '<PAD>')
            if st_tok == '<BAR>':
                generated_bars += 1
            if st_tok == '<EOS>' and generated_bars >= max(target_bars - 1, 0):
                generated_final.append((ev_type, struct_idx, pos_idx, h1_idx, h2_idx, rf_idx, lf_idx))
                entropies.append((e_entropy + s_entropy) / 2.0)
                break

            entropies.append((e_entropy + s_entropy) / 2.0)

            if hier_state_valid:
                beat_token = _parse_beat_token(st_tok)
                if st_tok == '<BAR>':
                    current_beat = 1
                    current_pos = 0
                elif beat_token is not None:
                    current_beat = beat_token
                    current_pos = 0
                elif st_tok.startswith('<TEMPO_') or st_tok in ('<EOS>', '<PAD>'):
                    pass
                else:
                    hier_state_valid = False
        else:
            pos_logits = tensor_to_numpy(cp_logits['cp_pos'][0])
            pos_idx, p_entropy = sample_excluding(pos_logits, temperature, nucleus_p, forbidden_idx=pos_pad_value)

            h1_logits = tensor_to_numpy(cp_logits['cp_hand1'][0])
            h2_logits = tensor_to_numpy(cp_logits['cp_hand2'][0])
            rf_logits = tensor_to_numpy(cp_logits['cp_right_foot'][0])
            lf_logits = tensor_to_numpy(cp_logits['cp_left_foot'][0])

            h1_idx, h1_entropy = sample_excluding(h1_logits, temperature, nucleus_p, forbidden_idx=limb_pad_idx)
            h2_idx, h2_entropy = sample_excluding(h2_logits, temperature, nucleus_p, forbidden_idx=limb_pad_idx)
            rf_idx, rf_entropy = sample_excluding(rf_logits, temperature, nucleus_p, forbidden_idx=limb_pad_idx)
            lf_idx, lf_entropy = sample_excluding(lf_logits, temperature, nucleus_p, forbidden_idx=limb_pad_idx)

            # 全肢が <NONE> の無音CPは生成効率が悪いため、手1だけ強制的に有効化
            if (
                idx2limb_token.get(h1_idx, '<NONE>') == '<NONE>' and
                idx2limb_token.get(h2_idx, '<NONE>') == '<NONE>' and
                idx2limb_token.get(rf_idx, '<NONE>') == '<NONE>' and
                idx2limb_token.get(lf_idx, '<NONE>') == '<NONE>'
            ):
                h1_idx = 1  # <NONE> の次の語彙が演奏トークンである保証は無いが、PAD回避優先

            struct_idx = struct_token2idx['<PAD>']
            entropies.append(np.mean([e_entropy, p_entropy, h1_entropy, h2_entropy, rf_entropy, lf_entropy]))

            if hier_state_valid:
                if 0 <= pos_idx < pos_pad_value:
                    current_pos = int(pos_idx)
                    steps_per_beat = max(1, pos_pad_value // 4)
                    current_beat = max(1, min(4, (current_pos // steps_per_beat) + 1))
                else:
                    hier_state_valid = False

        next_ev = (ev_type, struct_idx, pos_idx, h1_idx, h2_idx, rf_idx, lf_idx)
        generated.append(next_ev)
        generated_final.append(next_ev)

        next_idx = len(generated) - 1
        event_type_placeholder[next_idx, 0] = next_ev[0]
        if int(next_ev[0]) == 1:
            cp_type_placeholder[next_idx, 0] = note_type_idx
        elif int(next_ev[0]) == 0:
            cp_type_placeholder[next_idx, 0] = next_ev[1]
        else:
            cp_type_placeholder[next_idx, 0] = pad_type_idx
        struct_placeholder[next_idx, 0] = next_ev[1]
        pos_placeholder[next_idx, 0] = next_ev[2]
        h1_placeholder[next_idx, 0] = next_ev[3]
        h2_placeholder[next_idx, 0] = next_ev[4]
        rf_placeholder[next_idx, 0] = next_ev[5]
        lf_placeholder[next_idx, 0] = next_ev[6]

        if hier_state_valid:
            is_struct = int(int(next_ev[0]) == 0 and int(next_ev[1]) != struct_token2idx['<PAD>'])
            dec_bar_idx_ph[next_idx, 0] = min(max(generated_bars, 0), max(target_bars - 1, 0))
            dec_beat_idx_ph[next_idx, 0] = current_beat
            dec_pos_idx_ph[next_idx, 0] = current_pos
            dec_is_struct_ph[next_idx, 0] = is_struct

        cur_input_len += 1
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]

            latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]
            event_type_placeholder[:len(generated)-1, 0] = event_type_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            cp_type_placeholder[:len(generated)-1, 0] = cp_type_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            struct_placeholder[:len(generated)-1, 0] = struct_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            pos_placeholder[:len(generated)-1, 0] = pos_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            h1_placeholder[:len(generated)-1, 0] = h1_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            h2_placeholder[:len(generated)-1, 0] = h2_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            rf_placeholder[:len(generated)-1, 0] = rf_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
            lf_placeholder[:len(generated)-1, 0] = lf_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

            if hier_state_valid:
                dec_bar_idx_ph[:len(generated)-1, 0] = dec_bar_idx_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                dec_beat_idx_ph[:len(generated)-1, 0] = dec_beat_idx_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                dec_pos_idx_ph[:len(generated)-1, 0] = dec_pos_idx_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                dec_is_struct_ph[:len(generated)-1, 0] = dec_is_struct_ph[cur_input_len-truncate_len:cur_input_len-1, 0]

            if use_difficulty and s_tech_ph is not None:
                s_tech_ph[:len(generated)-1, 0] = s_tech_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                s_indep_ph[:len(generated)-1, 0] = s_indep_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                s_hand_ph[:len(generated)-1, 0] = s_hand_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                s_foot_ph[:len(generated)-1, 0] = s_foot_ph[cur_input_len-truncate_len:cur_input_len-1, 0]
                s_move_ph[:len(generated)-1, 0] = s_move_ph[cur_input_len-truncate_len:cur_input_len-1, 0]

            cur_input_len = len(generated)

    cp_type_seq = []
    for ev_type, struct_idx, *_ in generated_final:
        if int(ev_type) == 1:
            cp_type_seq.append(note_type_idx)
        elif int(ev_type) == 0:
            cp_type_seq.append(int(struct_idx))
        else:
            cp_type_seq.append(pad_type_idx)

    cp_data = {
        'cp_type': cp_type_seq,
        'cp_pos': [x[2] for x in generated_final],
        'cp_hand1': [x[3] for x in generated_final],
        'cp_hand2': [x[4] for x in generated_final],
        'cp_right_foot': [x[5] for x in generated_final],
        'cp_left_foot': [x[6] for x in generated_final],
    }

    return cp_data, time.time() - time_st, np.array(entropies)


def main():
    parser = argparse.ArgumentParser(
        description='Generate drum patterns with CP tokenization and difficulty control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_drum_cp.py --config configs/train_drum_cp.yaml \\
    --ckpt ckpts/model.pt --output output --n_pieces 5 --n_samples_per_piece 3
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n_pieces', type=int, required=True,
                        help='Number of pieces to generate from')
    parser.add_argument('--n_samples_per_piece', type=int, required=True,
                        help='Number of samples per piece')
    args = parser.parse_args()

    config_path = args.config
    ckpt_path = args.ckpt
    out_dir = args.output
    n_pieces = args.n_pieces
    n_samples_per_piece = args.n_samples_per_piece

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method',
    )
    if tokenization_method != 'cp_limb_v1':
        raise ValueError(
            'This script requires data.tokenization_method=cp_limb_v1. '
            f'Given: {tokenization_method}'
        )

    ensure_runtime_mode_compatibility(
        requested_mode=tokenization_method,
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
    )
    artifact_mode, artifact_mode_source = resolve_artifact_tokenization_method(
        config['data']['vocab_path'],
        config['data']['data_dir'],
    )
    print(f"[info] Tokenization method: {tokenization_method}")
    if artifact_mode is not None:
        print(f"[info] Artifact tokenization mode: {artifact_mode} ({artifact_mode_source})")

    device = config['training']['device']
    use_difficulty = config['model'].get('use_difficulty', False)

    dset = DrumTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=pickle_load(config['data']['test_split']),
        pad_to_same=False,
        use_difficulty=use_difficulty,
        tokenization_method=tokenization_method,
    )

    mconf = config['model']
    runtime_opts = _resolve_runtime_model_options(mconf, dset)
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
        cp_event_type_vocab_size=len(dset.event_type2idx),
        cp_type_vocab_size=len(dset.type_token2idx),
        cp_struct_vocab_size=len(dset.struct_token2idx),
        cp_pos_vocab_size=dset.pos_pad_value + 1,
        cp_limb_vocab_size=len(dset.limb_token2idx),
        cp_event_pad_idx=dset.event_type2idx['<PAD_EVENT>'],
        cp_struct_pad_idx=dset.struct_token2idx['<PAD>'],
        cp_pos_pad_idx=dset.pos_pad_value,
        cp_limb_pad_idx=dset.limb_token2idx['<PAD>'],
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
    ).to(device)

    _load_state_dict_with_compat(model, ckpt_path, device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    pieces = random.sample(range(len(dset)), min(n_pieces, len(dset)))
    print('[sampled pieces]', pieces)

    times = []
    all_entropies = []

    for p in pieces:
        p_data = dset[p]
        p_id = p_data['piece_id']
        p_bar_id = p_data['st_bar_id']

        p_data['enc_input'] = p_data['enc_input'][: p_data['enc_n_bars']]
        p_data['enc_padding_mask'] = p_data['enc_padding_mask'][: p_data['enc_n_bars']]

        for k in list(p_data.keys()):
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        p_latents = get_latent_embedding_fast(
            model,
            p_data,
            device=device,
            use_sampling=config['generate'].get('use_latent_sampling', True),
            sampling_var=config['generate'].get('latent_sampling_var', 0.1),
        )

        if use_difficulty:
            s_tech_bar = p_data['s_tech_cls'].long()
            s_indep_bar = p_data['s_indep_cls'].long()
            s_hand_bar = p_data['s_hand_cls'].long()
            s_foot_bar = p_data['s_foot_cls'].long()
            s_move_bar = p_data['s_move_cls'].long()
        else:
            s_tech_bar = s_indep_bar = s_hand_bar = s_foot_bar = s_move_bar = None

        for samp in range(n_samples_per_piece):
            out_prefix = os.path.join(out_dir, f'id{p}_bar{p_bar_id}_cp_sample{samp+1:02d}')
            if os.path.exists(out_prefix + '.mid'):
                print('[skip] already exists:', out_prefix + '.mid')
                continue

            cp_data, t_sec, entropies = generate_cp_sequence(
                model,
                p_latents,
                dset.struct_token2idx,
                dset.idx2struct_token,
                dset.idx2limb_token,
                dset.pos_pad_value,
                dset.limb_token2idx['<PAD>'],
                dset.event_type2idx['<PAD_EVENT>'],
                device=device,
                max_events=config['generate'].get('dec_seqlen', 12800),
                max_input_len=config['generate'].get('max_input_dec_seqlen', 1280),
                truncate_len=512,
                nucleus_p=config['generate'].get('nucleus_p', 0.9),
                temperature=config['generate'].get('temperature', 1.2),
                use_difficulty=use_difficulty,
                s_tech_cls=s_tech_bar,
                s_indep_cls=s_indep_bar,
                s_hand_cls=s_hand_bar,
                s_foot_cls=s_foot_bar,
                s_move_cls=s_move_bar,
                use_hierarchical_pe=runtime_opts['use_hierarchical_pe'],
            )

            cp_data_to_midi(
                cp_data,
                dset.idx2struct_token,
                dset.idx2limb_token,
                out_prefix + '.mid',
                bpm=120,
            )

            with open(out_prefix + '.txt', 'w') as f:
                for cp_type, pos, h1, h2, rf, lf in zip(
                    cp_data['cp_type'],
                    cp_data['cp_pos'],
                    cp_data['cp_hand1'],
                    cp_data['cp_hand2'],
                    cp_data['cp_right_foot'],
                    cp_data['cp_left_foot'],
                ):
                    cp_type_tok = dset.idx2struct_token.get(cp_type, '<UNK>')
                    if cp_type_tok != '<NOTE>':
                        f.write(f"STRUCT {cp_type_tok}\n")
                    else:
                        f.write(
                            'CP '
                            f"POS={pos} "
                            f"H1={dset.idx2limb_token.get(h1, '<UNK>')} "
                            f"H2={dset.idx2limb_token.get(h2, '<UNK>')} "
                            f"RF={dset.idx2limb_token.get(rf, '<UNK>')} "
                            f"LF={dset.idx2limb_token.get(lf, '<UNK>')}\n"
                        )

            print('[done]', out_prefix + '.mid')
            times.append(t_sec)
            all_entropies.extend(entropies.tolist())

    if times:
        print('[stats] avg generation time:', float(np.mean(times)))
    if all_entropies:
        print('[stats] avg entropy:', float(np.mean(all_entropies)))


if __name__ == '__main__':
    main()
