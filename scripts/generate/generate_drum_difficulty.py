"""
ドラム譜難易度制御生成スクリプト

5つの難易度指標を制御してドラム譜を生成します:
- S_tech: 特殊奏法の頻度
- S_indep: 手足の独立性
- S_hand: 手の連打スピード
- S_foot: 足の連打スピード
- S_move: 打点間移動速度

使用方法:
    python generate_drum_difficulty.py <config> <ckpt> <output_dir> <n_pieces> <n_samples_per_piece>

例:
    python generate_drum_difficulty.py configs/train_drum_difficulty.yaml ckpts/model.pt output 5 3
"""

import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from src.data.drum_dataloader import DrumTransformerDataset
from src.models.musemorphose import MuseMorphose
from src.utils.utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from src.utils.drum_to_midi import tokens_to_midi
from src.utils.tokenization_mode import (
    ensure_runtime_mode_compatibility,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
)

import torch
import yaml
import numpy as np
from scipy.stats import entropy
import argparse


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
    }


###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
    """Temperature付きソフトマックス"""
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('[info] overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    """Nucleus (top-p) sampling"""
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]

    if len(candi_index) == 0:
        candi_index = sorted_index[:3]

    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


########################################
# latent embedding extraction
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0., device='cuda'):
    """潜在表現を抽出"""
    # reshape
    batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)
    enc_bar_idx = piece_data['enc_bar_idx'].permute(1, 0).long().to(device) if 'enc_bar_idx' in piece_data else None
    enc_beat_idx = piece_data['enc_beat_idx'].permute(1, 0).long().to(device) if 'enc_beat_idx' in piece_data else None
    enc_pos_idx = piece_data['enc_pos_idx'].permute(1, 0).long().to(device) if 'enc_pos_idx' in piece_data else None
    enc_is_struct = piece_data['enc_is_struct'].permute(1, 0).long().to(device) if 'enc_is_struct' in piece_data else None

    # get latent conditioning vectors
    with torch.no_grad():
        piece_latents = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask,
            use_sampling=use_sampling, sampling_var=sampling_var,
            enc_bar_idx=enc_bar_idx,
            enc_beat_idx=enc_beat_idx,
            enc_pos_idx=enc_pos_idx,
            enc_is_struct=enc_is_struct,
        )

    return piece_latents


########################################
# generation with difficulty control
########################################
def generate_with_difficulty_control(
        model, latents,
        s_tech_cls, s_indep_cls, s_hand_cls, s_foot_cls, s_move_cls,
        tokenizer, device='cuda',
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512,
        nucleus_p=0.9, temperature=1.2
):
    """
    5つの難易度パラメータを制御して生成

    Args:
        model: MuseMorphoseモデル
        latents: 潜在表現 (n_bars, d_latent)
        s_tech_cls: 特殊奏法クラス (n_bars,) Tensor
        s_indep_cls: 独立性クラス (n_bars,) Tensor
        s_hand_cls: 手スピードクラス (n_bars,) Tensor
        s_foot_cls: 足スピードクラス (n_bars,) Tensor
        s_move_cls: 移動速度クラス (n_bars,) Tensor
        tokenizer: DrumTokenizer
        device: デバイス
        max_events: 最大イベント数
        primer: プライマー（オプション）
        max_input_len: 最大入力長
        truncate_len: トランケート長
        nucleus_p: Nucleus sampling の p
        temperature: サンプリング温度

    Returns:
        generated: 生成されたトークンインデックス列
        time_elapsed: 生成時間（秒）
        entropies: エントロピー列
    """
    # プレースホルダーを準備
    latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
    s_tech_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_indep_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_hand_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_foot_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)
    s_move_placeholder = torch.zeros(max_events, 1, dtype=torch.long).to(device)

    print('[info] Difficulty classes per bar:')
    print(f'  S_tech:  {tensor_to_numpy(s_tech_cls)}')
    print(f'  S_indep: {tensor_to_numpy(s_indep_cls)}')
    print(f'  S_hand:  {tensor_to_numpy(s_hand_cls)}')
    print(f'  S_foot:  {tensor_to_numpy(s_foot_cls)}')
    print(f'  S_move:  {tensor_to_numpy(s_move_cls)}')

    # 初期化
    if primer is None:
        generated = [tokenizer.token2idx['<BAR>']]
    else:
        generated = [tokenizer.token2idx[e] for e in primer]
        latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
        s_tech_placeholder[:len(generated), 0] = s_tech_cls[0]
        s_indep_placeholder[:len(generated), 0] = s_indep_cls[0]
        s_hand_placeholder[:len(generated), 0] = s_hand_cls[0]
        s_foot_placeholder[:len(generated), 0] = s_foot_cls[0]
        s_move_placeholder[:len(generated), 0] = s_move_cls[0]

    target_bars = latents.size(0)
    generated_bars = 0

    time_st = time.time()
    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []

    # 生成ループ
    while generated_bars < target_bars:
        # 入力の準備
        if len(generated) == 1:
            dec_input = numpy_to_tensor([generated], device=device).long()
        else:
            dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

        # 現在の小節に対応する潜在表現と難易度クラスを設定
        latent_placeholder[len(generated)-1, 0, :] = latents[generated_bars]
        s_tech_placeholder[len(generated)-1, 0] = s_tech_cls[generated_bars]
        s_indep_placeholder[len(generated)-1, 0] = s_indep_cls[generated_bars]
        s_hand_placeholder[len(generated)-1, 0] = s_hand_cls[generated_bars]
        s_foot_placeholder[len(generated)-1, 0] = s_foot_cls[generated_bars]
        s_move_placeholder[len(generated)-1, 0] = s_move_cls[generated_bars]

        dec_seg_emb = latent_placeholder[:len(generated), :]
        dec_s_tech = s_tech_placeholder[:len(generated), :]
        dec_s_indep = s_indep_placeholder[:len(generated), :]
        dec_s_hand = s_hand_placeholder[:len(generated), :]
        dec_s_foot = s_foot_placeholder[:len(generated), :]
        dec_s_move = s_move_placeholder[:len(generated), :]

        # サンプリング
        with torch.no_grad():
            logits = model.generate(
                dec_input, dec_seg_emb,
                s_tech_cls=dec_s_tech,
                s_indep_cls=dec_s_indep,
                s_hand_cls=dec_s_hand,
                s_foot_cls=dec_s_foot,
                s_move_cls=dec_s_move
            )

        logits = tensor_to_numpy(logits[0])
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)
        word_token = tokenizer.idx2token[word]

        # <BAR>トークンで小節をカウント
        if word_token == '<BAR>':
            generated_bars += 1
            print(f'[info] Generated {generated_bars}/{target_bars} bars, #events = {len(generated_final)}')

        # <PAD>はスキップ
        if word_token == '<PAD>':
            continue

        # 終了条件
        if len(generated) > max_events or (word_token == '<EOS>' and generated_bars == target_bars - 1):
            generated_bars += 1
            generated.append(tokenizer.token2idx['<BAR>'])
            print('[info] Generation completed (EOS or max_events)')
            break

        generated.append(word)
        generated_final.append(word)
        entropies.append(entropy(probs))

        cur_input_len += 1

        # コンテキストウィンドウのトランケート
        assert cur_input_len == len(generated)
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]

            # プレースホルダーもシフト
            latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0, :
            ]
            s_tech_placeholder[:len(generated)-1, 0] = s_tech_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_indep_placeholder[:len(generated)-1, 0] = s_indep_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_hand_placeholder[:len(generated)-1, 0] = s_hand_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_foot_placeholder[:len(generated)-1, 0] = s_foot_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]
            s_move_placeholder[:len(generated)-1, 0] = s_move_placeholder[
                cur_input_len-truncate_len:cur_input_len-1, 0
            ]

            print(f'[info] Context truncated: accumulated_len={len(generated_final)}')
            cur_input_len = len(generated)

    assert generated_bars == target_bars
    print(f'-- Generated {len(generated_final)} events')
    print(f'-- Time elapsed: {time.time() - time_st:.2f} secs')

    return generated_final[:-1], time.time() - time_st, np.array(entropies)


########################################
# difficulty class shifting
########################################
def random_shift_difficulty_cls(n_samples, n_bars, upper=4, lower=-3):
    """
    難易度クラスをランダムにシフト

    Args:
        n_samples: サンプル数
        n_bars: 小節数
        upper: 上限シフト
        lower: 下限シフト

    Returns:
        shifts: (n_samples, 5, n_bars) の配列
            5次元 = [s_tech, s_indep, s_hand, s_foot, s_move]
    """
    shifts = np.random.randint(lower, upper, (n_samples, 5, n_bars))
    return shifts


########################################
# main
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate drum patterns with difficulty control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_drum_difficulty.py --config configs/train_drum_difficulty.yaml \\
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

    # 設定を読み込み
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method',
    )
    config['data']['tokenization_method'] = tokenization_method
    device = config['training']['device']
    if tokenization_method == 'cp_limb_v1':
        raise NotImplementedError(
            'generate_drum_difficulty.py is standard-tokenization oriented. '
            'Use a dedicated CP generation path for cp_limb_v1.'
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
    print(f'[info] Tokenization method: {tokenization_method}')
    if artifact_mode is not None:
        print(f'[info] Artifact tokenization mode: {artifact_mode} ({artifact_mode_source})')

    # データセット読み込み
    print('[info] Loading dataset...')
    dset = DrumTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=pickle_load(config['data']['test_split']),
        pad_to_same=False,
        use_difficulty=True,
        tokenization_method=tokenization_method
    )

    print(f'[info] Dataset size: {len(dset)} pieces')
    print(f'[info] Vocabulary size: {dset.vocab_size}')

    # モデル読み込み
    print('[info] Building model...')
    mconf = config['model']
    runtime_opts = _resolve_runtime_model_options(mconf, dset)
    model = MuseMorphose(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
        d_s_tech_emb=mconf.get('d_s_tech_emb', 32),
        d_s_indep_emb=mconf.get('d_s_indep_emb', 32),
        d_s_hand_emb=mconf.get('d_s_hand_emb', 32),
        d_s_foot_emb=mconf.get('d_s_foot_emb', 32),
        d_s_move_emb=mconf.get('d_s_move_emb', 32),
        use_difficulty=True,
        use_attr_cls=True,
        cond_mode=runtime_opts['cond_mode'],
        bar_anchor_mode=runtime_opts['bar_anchor_mode'],
        bar_token_idx=runtime_opts['bar_token_idx'],
        use_hierarchical_pe=runtime_opts['use_hierarchical_pe'],
        hier_pe_bar_vocab_size=runtime_opts['hier_pe_bar_vocab_size'],
        hier_pe_beat_vocab_size=runtime_opts['hier_pe_beat_vocab_size'],
        hier_pe_pos_vocab_size=runtime_opts['hier_pe_pos_vocab_size'],
        hier_pe_use_struct_bias=runtime_opts['hier_pe_use_struct_bias'],
    ).to(device)

    print(f'[info] Loading checkpoint from {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ランダムにピースを選択
    pieces = random.sample(range(len(dset)), min(n_pieces, len(dset)))
    print(f'[info] Sampled pieces: {pieces}')

    times = []
    all_entropies = []

    for p_idx, p in enumerate(pieces):
        print(f'\n{"="*60}')
        print(f'Processing piece {p_idx+1}/{len(pieces)} (dataset index: {p})')
        print(f'{"="*60}')

        p_data = dset[p]
        p_id = p_data['piece_id']
        p_bar_id = p_data['st_bar_id']

        # データをデバイスに移動
        for k in p_data.keys():
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        # 元のトークン列を保存（参照用）
        original_tokens = [dset.idx2token[idx] for idx in tensor_to_numpy(p_data['dec_input'][:p_data['length']])]
        original_file = os.path.join(out_dir, f'id{p}_bar{p_bar_id}_original.mid')
        tokens_to_midi(original_tokens, original_file, bpm=120)
        print(f'[info] Saved original MIDI: {original_file}')

        # 潜在表現を抽出
        print('[info] Extracting latent embeddings...')
        p_latents = get_latent_embedding_fast(
            model, p_data,
            use_sampling=config['generate'].get('use_latent_sampling', True),
            sampling_var=config['generate'].get('latent_sampling_var', 0.1),
            device=device
        )

        # 元の難易度クラス
        original_s_tech = p_data['s_tech_cls']
        original_s_indep = p_data['s_indep_cls']
        original_s_hand = p_data['s_hand_cls']
        original_s_foot = p_data['s_foot_cls']
        original_s_move = p_data['s_move_cls']

        n_bars = p_data['enc_n_bars']

        # 難易度クラスのランダムシフトを生成
        diff_shifts = random_shift_difficulty_cls(n_samples_per_piece, n_bars, upper=4, lower=-3)

        for samp in range(n_samples_per_piece):
            print(f'\n[info] Generating sample {samp+1}/{n_samples_per_piece}')

            # シフトした難易度クラス
            s_tech_cls = torch.clamp(original_s_tech + diff_shifts[samp, 0], 0, 7).long()
            s_indep_cls = torch.clamp(original_s_indep + diff_shifts[samp, 1], 0, 7).long()
            s_hand_cls = torch.clamp(original_s_hand + diff_shifts[samp, 2], 0, 7).long()
            s_foot_cls = torch.clamp(original_s_foot + diff_shifts[samp, 3], 0, 7).long()
            s_move_cls = torch.clamp(original_s_move + diff_shifts[samp, 4], 0, 7).long()

            out_file = os.path.join(
                out_dir,
                f'id{p}_bar{p_bar_id}_sample{samp+1:02d}_'
                f'tech{diff_shifts[samp,0,0]:+d}_indep{diff_shifts[samp,1,0]:+d}_'
                f'hand{diff_shifts[samp,2,0]:+d}_foot{diff_shifts[samp,3,0]:+d}_'
                f'move{diff_shifts[samp,4,0]:+d}'
            )

            # 生成
            song_indices, t_sec, ent = generate_with_difficulty_control(
                model, p_latents,
                s_tech_cls, s_indep_cls, s_hand_cls, s_foot_cls, s_move_cls,
                dset.tokenizer,
                device=device,
                max_input_len=config['generate']['max_input_dec_seqlen'],
                truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32),
                nucleus_p=config['generate']['nucleus_p'],
                temperature=config['generate']['temperature']
            )
            times.append(t_sec)
            all_entropies.append(ent)

            # トークン列に変換
            song_tokens = [dset.idx2token[idx] for idx in song_indices]

            # MIDI保存
            tokens_to_midi(song_tokens, out_file + '.mid', bpm=120)
            print(f'[info] Saved MIDI: {out_file}.mid')

            # トークン列をテキスト保存
            with open(out_file + '.txt', 'w') as f:
                for token in song_tokens:
                    f.write(token + '\n')

            # 難易度クラスを保存
            np.save(out_file + '_S_TECH_CLS.npy', tensor_to_numpy(s_tech_cls))
            np.save(out_file + '_S_INDEP_CLS.npy', tensor_to_numpy(s_indep_cls))
            np.save(out_file + '_S_HAND_CLS.npy', tensor_to_numpy(s_hand_cls))
            np.save(out_file + '_S_FOOT_CLS.npy', tensor_to_numpy(s_foot_cls))
            np.save(out_file + '_S_MOVE_CLS.npy', tensor_to_numpy(s_move_cls))

            # エントロピー統計を保存
            np.save(out_file + '_entropies.npy', ent)

    print(f'\n{"="*60}')
    print('Generation completed!')
    print(f'{"="*60}')
    print(f'Total pieces: {len(pieces)}')
    print(f'Samples per piece: {n_samples_per_piece}')
    print(f'Total samples: {len(pieces) * n_samples_per_piece}')
    print(f'Average generation time: {np.mean(times):.2f} ± {np.std(times):.2f} secs')
    print(f'Average entropy: {np.mean([e.mean() for e in all_entropies]):.3f}')
    print(f'Output directory: {out_dir}')
