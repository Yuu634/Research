"""
ドラム譜面生成スクリプト（入力MIDIから）

入力されたドラム譜MIDIファイルから学習済みモデルを用いて、
指定した演奏難易度レベルに沿った新しいドラム譜を生成します。

処理フロー:
1. 学習済みモデルのロード（トークン設計手法はconfig.yamlから取得）
2. 入力譜面データ（ドラム譜MIDI）の読み込み
3. 入力譜面データから各小節の演奏難易度レベル算出
4. 入力譜面データ&各小節の演奏難易度レベルをモデルに入力し、
   指定した演奏難易度レベルに沿ったドラム譜を生成
5. 指定した保存先パスにMIDIファイルを保存

使用方法:
    python generate_drum_from_input.py --model_path <model_path> \\
        --input_midi <input_midi> --output_midi <output_midi>
"""

import sys, os, time
from copy import deepcopy
import pickle
import argparse
from pathlib import Path

import torch
import yaml
import numpy as np
from scipy.stats import entropy
import miditoolkit

# パス設定
sys.path.append('./model')

from src.models.musemorphose import MuseMorphose
from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
from src.data.drum_difficulty_calculator import (
    compute_all_difficulty_scores,
    discretize_difficulty_score
)
from src.utils.utils import numpy_to_tensor, tensor_to_numpy
from src.utils.drum_to_midi import tokens_to_midi
from src.utils.tokenization_mode import (
    ensure_runtime_mode_compatibility,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
)



###########################################
# Checkpoint互換性処理
###########################################
def adapt_checkpoint_to_model(state_dict, model_vocab_size, old_vocab_size):
    """
    古いcheckpointを新しいモデル構造に適応
    
    Args:
        state_dict: 読み込まれたcheckpointの状態辞書
        model_vocab_size: 現在のモデルのトークン数
        old_vocab_size: checkpointの古いトークン数
        
    Returns:
        adjusted_state_dict: 適応されたstate_dict
    """
    adjusted_state_dict = {}
    
    # 削除すべき古い属性埋め込みキーのプレフィックス
    old_attr_prefixes = ['rfreq_attr_emb', 'polyph_attr_emb']
    
    # 新しい難易度属性キー（checkpointに存在しないはず）
    new_difficulty_prefixes = ['s_tech_attr_emb', 's_indep_attr_emb', 
                               's_hand_attr_emb', 's_foot_attr_emb', 's_move_attr_emb']
    
    for key, param in state_dict.items():
        # 古い属性埋め込みは削除
        skip = False
        for prefix in old_attr_prefixes:
            if key.startswith(prefix):
                print(f'[info] Removing outdated key: {key}')
                skip = True
                break
        
        if skip:
            continue
        
        # tokenサイズが異なる場合の処理
        if key in ['token_emb.emb_lookup.weight', 'dec_out_proj.weight', 'dec_out_proj.bias']:
            if key == 'dec_out_proj.bias':
                expected_size = model_vocab_size
            else:
                expected_size = model_vocab_size
            
            if param.size(0) != expected_size:
                old_size = param.size(0)
                print(f'[warning] Size mismatch for {key}: checkpoint={old_size}, model={expected_size}')
                
                if key in ['token_emb.emb_lookup.weight', 'dec_out_proj.weight']:
                    # 重みを切り詰めまたはパディング
                    if old_size > expected_size:
                        # 古いvocabサイズが大きい場合は切り詰め
                        adjusted_state_dict[key] = param[:expected_size, :]
                        print(f'[info] Truncated {key} from {old_size} to {expected_size}')
                    else:
                        # 新しいvocabサイズが大きい場合はパディング
                        padding = torch.zeros(expected_size - old_size, param.size(1), 
                                             dtype=param.dtype, device=param.device)
                        adjusted_state_dict[key] = torch.cat([param, padding], dim=0)
                        print(f'[info] Padded {key} from {old_size} to {expected_size}')
                
                elif key == 'dec_out_proj.bias':
                    # バイアスの処理
                    if old_size > expected_size:
                        adjusted_state_dict[key] = param[:expected_size]
                        print(f'[info] Truncated {key} from {old_size} to {expected_size}')
                    else:
                        padding = torch.zeros(expected_size - old_size, 
                                             dtype=param.dtype, device=param.device)
                        adjusted_state_dict[key] = torch.cat([param, padding], dim=0)
                        print(f'[info] Padded {key} from {old_size} to {expected_size}')
            else:
                adjusted_state_dict[key] = param
        else:
            adjusted_state_dict[key] = param
    
    return adjusted_state_dict


###########################################
# MIDI読み込みとトークン化
###########################################
def load_midi_and_tokenize(midi_path, tokenizer):
    """
    MIDIファイルを読み込み、トークン列に変換
    
    Args:
        midi_path: MIDIファイルパス
        tokenizer: DrumTokenizer または DrumCPTokenizer
        
    Returns:
        tokens: トークン列のリスト
        bar_positions: 各小節の開始位置インデックス
    """
    # midi_to_tokens は既に bar_positions を計算して返す
    tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
    
    return tokens, bar_positions


###########################################
# 難易度レベル計算
###########################################
def compute_difficulty_levels(tokens, bar_positions, difficulty_bounds, bpm=120.0):
    """
    トークン列から各小節の難易度レベルを計算
    
    Args:
        tokens: トークン列
        bar_positions: 小節の開始位置リスト
        difficulty_bounds: 難易度境界値辞書（pkl）
        bpm: テンポ
        
    Returns:
        difficulty_levels: dict of numpy arrays
            's_tech_cls': (n_bars,)
            's_indep_cls': (n_bars,)
            's_hand_cls': (n_bars,)
            's_foot_cls': (n_bars,)
            's_move_cls': (n_bars,)
    """
    n_bars = len(bar_positions) - 1
    
    s_tech_levels = []
    s_indep_levels = []
    s_hand_levels = []
    s_foot_levels = []
    s_move_levels = []
    
    for i in range(n_bars):
        start_pos = bar_positions[i]
        end_pos = bar_positions[i + 1]
        bar_tokens = tokens[start_pos:end_pos]
        
        # 難易度スコア計算
        scores = compute_all_difficulty_scores(bar_tokens, bpm=bpm)
        
        # スコアをクラスに離散化
        s_tech_cls = discretize_difficulty_score(scores['s_tech'], difficulty_bounds['s_tech'])
        s_indep_cls = discretize_difficulty_score(scores['s_indep'], difficulty_bounds['s_indep'])
        s_hand_cls = discretize_difficulty_score(scores['s_hand'], difficulty_bounds['s_hand'])
        s_foot_cls = discretize_difficulty_score(scores['s_foot'], difficulty_bounds['s_foot'])
        s_move_cls = discretize_difficulty_score(scores['s_move'], difficulty_bounds['s_move'])
        
        s_tech_levels.append(s_tech_cls)
        s_indep_levels.append(s_indep_cls)
        s_hand_levels.append(s_hand_cls)
        s_foot_levels.append(s_foot_cls)
        s_move_levels.append(s_move_cls)
    
    return {
        's_tech_cls': np.array(s_tech_levels, dtype=np.int64),
        's_indep_cls': np.array(s_indep_levels, dtype=np.int64),
        's_hand_cls': np.array(s_hand_levels, dtype=np.int64),
        's_foot_cls': np.array(s_foot_levels, dtype=np.int64),
        's_move_cls': np.array(s_move_levels, dtype=np.int64),
    }


###########################################
# トークン列から潜在表現を抽出
###########################################
def tokens_to_model_input(tokens, tokenizer, max_len=128, device='cuda'):
    """
    トークン列をモデル入力形式に変換
    
    Args:
        tokens: トークン列
        tokenizer: トークナイザー
        max_len: 最大系列長
        device: デバイス
        
    Returns:
        model_input: dict
    """
    # トークンをインデックスに変換
    token_indices = [tokenizer.token2idx.get(token, tokenizer.token2idx['<PAD>']) 
                     for token in tokens]
    
    # パディングまたは切り詰め
    if len(token_indices) > max_len:
        token_indices = token_indices[:max_len]
    else:
        padding_len = max_len - len(token_indices)
        token_indices = token_indices + [tokenizer.token2idx['<PAD>']] * padding_len
    
    # Tensorに変換
    enc_input = torch.tensor(token_indices, dtype=torch.long).unsqueeze(1).to(device)
    
    # パディングマスク
    padding_mask = torch.tensor(
        [0 if idx != tokenizer.token2idx['<PAD>'] else 1 for idx in token_indices],
        dtype=torch.bool
    ).unsqueeze(0).to(device)
    
    return {
        'enc_input': enc_input,
        'enc_padding_mask': padding_mask
    }


def get_latent_embedding(model, model_input, use_sampling=False, sampling_var=0., device='cuda'):
    """潜在表現を抽出"""
    batch_inp = model_input['enc_input']
    batch_padding_mask = model_input['enc_padding_mask']
    
    with torch.no_grad():
        piece_latents = model.get_sampled_latent(
            batch_inp, padding_mask=batch_padding_mask,
            use_sampling=use_sampling, sampling_var=sampling_var
        )
    
    return piece_latents


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


###########################################
# 生成
###########################################
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
        s_tech_cls: 特殊奏法クラス (n_bars,) array
        s_indep_cls: 独立性クラス (n_bars,) array
        s_hand_cls: 手スピードクラス (n_bars,) array
        s_foot_cls: 足スピードクラス (n_bars,) array
        s_move_cls: 移動速度クラス (n_bars,) array
        tokenizer: トークナイザー
        device: デバイス
        max_events: 最大イベント数
        primer: プライマー
        max_input_len: 最大入力長
        truncate_len: トランケート長
        nucleus_p: Nucleus sampling の p
        temperature: サンプリング温度
        
    Returns:
        generated: 生成されたトークンインデックス列
        time_elapsed: 生成時間（秒）
        entropies: エントロピー列
    """
    # Tensorに変換
    s_tech_cls = numpy_to_tensor(s_tech_cls, device=device).long()
    s_indep_cls = numpy_to_tensor(s_indep_cls, device=device).long()
    s_hand_cls = numpy_to_tensor(s_hand_cls, device=device).long()
    s_foot_cls = numpy_to_tensor(s_foot_cls, device=device).long()
    s_move_cls = numpy_to_tensor(s_move_cls, device=device).long()
    
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


###########################################
# main
###########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate drum patterns from input MIDI with difficulty control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_drum_from_input.py \\
    --model_path trained_model/checkpoints_drum_standard_with_difficulty/params/best_params.pt \\
    --input_midi input_drum.mid \\
    --output_midi output_drum.mid \\
    --difficulty_shift 0 0 2 1 0
        """
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--input_midi', type=str, required=True,
                        help='Path to input drum MIDI file')
    parser.add_argument('--output_midi', type=str, required=True,
                        help='Path to output MIDI file')
    parser.add_argument('--difficulty_shift', type=int, nargs=5, default=[0, 0, 0, 0, 0],
                        help='Difficulty level shifts for [s_tech, s_indep, s_hand, s_foot, s_move] (e.g., 0 0 2 1 0)')
    parser.add_argument('--nucleus_p', type=float, default=0.9,
                        help='Nucleus sampling p value (default: 0.9)')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature (default: 1.2)')
    parser.add_argument('--use_latent_sampling', action='store_true',
                        help='Use sampling for latent extraction')
    parser.add_argument('--latent_sampling_var', type=float, default=0.1,
                        help='Latent sampling variance (default: 0.1)')
    parser.add_argument('--bpm', type=int, default=120,
                        help='Tempo in BPM (default: 120)')
    args = parser.parse_args()
    
    # パスの検証
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f'[ERROR] Model checkpoint not found: {model_path}')
        sys.exit(1)
    
    input_midi_path = Path(args.input_midi)
    if not input_midi_path.exists():
        print(f'[ERROR] Input MIDI file not found: {input_midi_path}')
        sys.exit(1)
    
    # モデルのディレクトリからconfig.yamlを探す
    checkpoint_dir = model_path.parent.parent  # params/ -> checkpoints_xxx/
    config_path = checkpoint_dir / 'config.yaml'
    
    if not config_path.exists():
        print(f'[ERROR] config.yaml not found in {checkpoint_dir}')
        sys.exit(1)
    
    print(f'[info] Loading config from {config_path}')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    
    # トークン設計手法を取得
    tokenization_method = validate_tokenization_method(
        config['data'].get('tokenization_method', 'standard'),
        context='data.tokenization_method',
    )
    print(f'[info] Tokenization method: {tokenization_method}')
    
    # CP形式は未対応
    if tokenization_method == 'cp_limb_v1':
        print('[ERROR] This script does not support cp_limb_v1 tokenization method.')
        print('        Please use standard tokenization model.')
        sys.exit(1)
    
    device = config['training']['device']
    
    # 語彙パスを取得
    vocab_path = Path(config['data']['vocab_path'])
    if not vocab_path.exists():
        print(f'[ERROR] Vocabulary file not found: {vocab_path}')
        sys.exit(1)

    data_dir = config['data'].get('data_dir')
    try:
        ensure_runtime_mode_compatibility(
            requested_mode=tokenization_method,
            vocab_path=str(vocab_path),
            data_dir=data_dir,
        )
        artifact_mode, artifact_mode_source = resolve_artifact_tokenization_method(
            str(vocab_path),
            data_dir,
        )
        if artifact_mode is not None:
            print(f'[info] Artifact tokenization mode: {artifact_mode} ({artifact_mode_source})')
    except ValueError as e:
        print(f'[ERROR] {e}')
        sys.exit(1)
    
    print(f'[info] Loading vocabulary from {vocab_path}')
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # トークナイザーを初期化
    if isinstance(vocab_data, dict) and vocab_data.get('tokenization_method') == 'cp_limb_v1':
        print('[ERROR] CP tokenization detected but not supported.')
        sys.exit(1)
    else:
        # 標準トークナイザー
        tokenizer = DrumTokenizer()
        tokenizer.token2idx, tokenizer.idx2token = vocab_data
        tokenizer.vocab_size = len(tokenizer.token2idx)
    
    print(f'[info] Vocabulary size: {tokenizer.vocab_size}')
    
    # 難易度境界値を読み込み
    difficulty_bounds_path = Path('data/metadata/difficulty_bounds_standard.pkl')
    if not difficulty_bounds_path.exists():
        print(f'[ERROR] Difficulty bounds file not found: {difficulty_bounds_path}')
        sys.exit(1)
    
    print(f'[info] Loading difficulty bounds from {difficulty_bounds_path}')
    with open(difficulty_bounds_path, 'rb') as f:
        difficulty_bounds = pickle.load(f)
    
    # モデル構築
    print('[info] Building model...')
    mconf = config['model']
    cond_mode = mconf.get('cond_mode', 'in-attn')
    if cond_mode not in ('in-attn', 'pre-attn'):
        print(f"[warning] Unsupported cond_mode={cond_mode}. Falling back to 'in-attn'.")
        cond_mode = 'in-attn'

    bar_anchor_mode = mconf.get('bar_anchor_mode', 'first_token')
    if bar_anchor_mode not in ('first_token', 'bar_token'):
        print(f"[warning] Unsupported bar_anchor_mode={bar_anchor_mode}. Falling back to 'first_token'.")
        bar_anchor_mode = 'first_token'

    model = MuseMorphose(
        mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
        mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
        mconf['d_latent'], mconf['d_embed'], tokenizer.vocab_size,
        d_s_tech_emb=mconf.get('d_s_tech_emb', 32),
        d_s_indep_emb=mconf.get('d_s_indep_emb', 32),
        d_s_hand_emb=mconf.get('d_s_hand_emb', 32),
        d_s_foot_emb=mconf.get('d_s_foot_emb', 32),
        d_s_move_emb=mconf.get('d_s_move_emb', 32),
        use_difficulty=True,
        use_attr_cls=True,
        cond_mode=cond_mode,
        bar_anchor_mode=bar_anchor_mode,
        bar_token_idx=tokenizer.token2idx.get('<BAR>', None),
        use_hierarchical_pe=bool(mconf.get('use_hierarchical_pe', False)),
        hier_pe_bar_vocab_size=max(1, int(mconf.get('hier_pe_bar_vocab_size', 512))),
        hier_pe_beat_vocab_size=max(1, int(mconf.get('hier_pe_beat_vocab_size', 8))),
        hier_pe_pos_vocab_size=max(1, int(mconf.get('hier_pe_pos_vocab_size', 32))),
        hier_pe_use_struct_bias=bool(mconf.get('hier_pe_use_struct_bias', True)),
    ).to(device)
    
    print(f'[info] Loading model checkpoint from {model_path}')
    #model.load_state_dict(torch.load(model_path, map_location=device))
    #model.eval()
    print(f'[info] Loading model checkpoint from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    
    # checkpointが state_dict（辞書）直接の場合と、
    # Checkpoint型（'state_dict'キーを持つ）の場合に対応
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['state_dict']
    else:
        checkpoint_state_dict = checkpoint
    
    # 互換性処理を適用
    adapted_state_dict = adapt_checkpoint_to_model(
        checkpoint_state_dict, 
        tokenizer.vocab_size,
        old_vocab_size=checkpoint_state_dict['token_emb.emb_lookup.weight'].size(0)
    )
    
    # 適応されたstate_dictをロード（strict=Falseで新しい難易度層は初期化）
    model.load_state_dict(adapted_state_dict, strict=False)
    model.eval()
    
    # 入力MIDIを読み込み、トークン化
    print(f'[info] Loading and tokenizing input MIDI: {input_midi_path}')
    tokens, bar_positions = load_midi_and_tokenize(str(input_midi_path), tokenizer)
    
    n_bars = len(bar_positions) - 1
    print(f'[info] Input MIDI contains {n_bars} bars')
    
    if n_bars == 0:
        print('[ERROR] No bars found in input MIDI')
        sys.exit(1)
    
    # 難易度レベル計算
    print('[info] Computing difficulty levels for each bar...')
    difficulty_levels = compute_difficulty_levels(
        tokens, bar_positions, difficulty_bounds, bpm=args.bpm
    )
    
    print('[info] Original difficulty levels:')
    print(f'  S_tech:  {difficulty_levels["s_tech_cls"]}')
    print(f'  S_indep: {difficulty_levels["s_indep_cls"]}')
    print(f'  S_hand:  {difficulty_levels["s_hand_cls"]}')
    print(f'  S_foot:  {difficulty_levels["s_foot_cls"]}')
    print(f'  S_move:  {difficulty_levels["s_move_cls"]}')
    
    # 難易度シフトを適用
    shift_tech, shift_indep, shift_hand, shift_foot, shift_move = args.difficulty_shift
    
    shifted_s_tech = np.clip(difficulty_levels['s_tech_cls'] + shift_tech, 0, 7)
    shifted_s_indep = np.clip(difficulty_levels['s_indep_cls'] + shift_indep, 0, 7)
    shifted_s_hand = np.clip(difficulty_levels['s_hand_cls'] + shift_hand, 0, 7)
    shifted_s_foot = np.clip(difficulty_levels['s_foot_cls'] + shift_foot, 0, 7)
    shifted_s_move = np.clip(difficulty_levels['s_move_cls'] + shift_move, 0, 7)
    
    print(f'\n[info] Applied difficulty shifts: {args.difficulty_shift}')
    print('[info] Target difficulty levels:')
    print(f'  S_tech:  {shifted_s_tech}')
    print(f'  S_indep: {shifted_s_indep}')
    print(f'  S_hand:  {shifted_s_hand}')
    print(f'  S_foot:  {shifted_s_foot}')
    print(f'  S_move:  {shifted_s_move}')
    
    # モデル入力形式に変換
    print('\n[info] Converting tokens to model input format...')
    model_input = tokens_to_model_input(
        tokens, tokenizer,
        max_len=config['data']['enc_seqlen'],
        device=device
    )
    
    # 潜在表現を抽出
    print('[info] Extracting latent embeddings...')
    latents = get_latent_embedding(
        model, model_input,
        use_sampling=args.use_latent_sampling,
        sampling_var=args.latent_sampling_var,
        device=device
    )
    
    print(f'[info] Latent shape: {latents.shape}')
    
    # 生成
    print('\n[info] Generating drum pattern with difficulty control...')
    
    # generate設定のデフォルト値
    generate_config = config.get('generate', {})
    max_input_dec_seqlen = generate_config.get('max_input_dec_seqlen', 1280)
    
    generated_indices, time_elapsed, entropies = generate_with_difficulty_control(
        model, latents,
        shifted_s_tech, shifted_s_indep, shifted_s_hand, shifted_s_foot, shifted_s_move,
        tokenizer,
        device=device,
        max_input_len=max_input_dec_seqlen,
        truncate_len=min(512, max_input_dec_seqlen - 32),
        nucleus_p=args.nucleus_p,
        temperature=args.temperature
    )
    
    # トークン列に変換
    generated_tokens = [tokenizer.idx2token[idx] for idx in generated_indices]
    
    # MIDIに変換して保存
    print(f'\n[info] Saving generated MIDI to {args.output_midi}')
    tokens_to_midi(generated_tokens, args.output_midi, bpm=args.bpm)
    
    print(f'\n{"="*60}')
    print('Generation completed successfully!')
    print(f'{"="*60}')
    print(f'Model: {model_path}')
    print(f'Input MIDI: {input_midi_path}')
    print(f'Output MIDI: {args.output_midi}')
    print(f'Number of bars: {n_bars}')
    print(f'Generation time: {time_elapsed:.2f} seconds')
    print(f'Average entropy: {entropies.mean():.3f}')
    print(f'Difficulty shifts: tech={shift_tech:+d}, indep={shift_indep:+d}, hand={shift_hand:+d}, foot={shift_foot:+d}, move={shift_move:+d}')
