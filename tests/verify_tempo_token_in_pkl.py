#!/usr/bin/env python3
"""
テンポトークンがpklファイルに正しく含まれているか検証

使用方法:
    python verify_tempo_token_in_pkl.py <midi_file> <output_pkl>

例:
    python verify_tempo_token_in_pkl.py test_token/sample_midis/simple_beat.mid output_test.pkl
"""
import sys
import os
import pickle
import argparse
from drum_tokenizer import DrumTokenizer


def verify_tempo_tokens_in_pkl(midi_path: str, pkl_output_path: str, vocab_path: str = None):
    """
    MIDIをトークン化 → pkl化 → 検証
    
    Args:
        midi_path: MIDIファイルパス
        pkl_output_path: 出力pkl ファイルパス
        vocab_path: 語彙ファイルパス（オプション）
    """
    # トークナイザーを初期化
    tokenizer = DrumTokenizer()
    
    # 語彙を保存（オプション）
    if vocab_path:
        tokenizer.save_vocab(vocab_path)
        print(f"✓ Vocabulary saved to: {vocab_path}")
    
    print("\n" + "="*70)
    print("TEMPO TOKEN VERIFICATION")
    print("="*70)
    
    # Step 1: MIDIをトークン化
    print(f"\n[Step 1] Converting MIDI → Tokens")
    print(f"  Input MIDI: {midi_path}")
    try:
        tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
        print(f"  ✓ Tokenization complete")
        print(f"    - Total tokens: {len(tokens)}")
        print(f"    - Total bars: {len(bar_positions)}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Step 2: トークンをインデックス化
    print(f"\n[Step 2] Converting Tokens → Indices")
    try:
        token_indices = tokenizer.tokens_to_indices(tokens)
        print(f"  ✓ Conversion complete")
        print(f"    - Total indices: {len(token_indices)}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Step 3: pkl化
    print(f"\n[Step 3] Saving to PKL")
    print(f"  Output PKL: {pkl_output_path}")
    try:
        os.makedirs(os.path.dirname(pkl_output_path) or '.', exist_ok=True)
        with open(pkl_output_path, 'wb') as f:
            pickle.dump((bar_positions, token_indices.tolist()), f)
        print(f"  ✓ PKL saved successfully")
        print(f"    - File size: {os.path.getsize(pkl_output_path)} bytes")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Step 4: pkl から読み込み
    print(f"\n[Step 4] Loading from PKL")
    try:
        with open(pkl_output_path, 'rb') as f:
            loaded_bar_pos, loaded_indices = pickle.load(f)
        print(f"  ✓ PKL loaded successfully")
        print(f"    - Bar positions: {loaded_bar_pos}")
        print(f"    - Total indices: {len(loaded_indices)}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Step 5: テンポトークン検証
    print(f"\n[Step 5] Verifying Tempo Tokens")
    
    # テンポトークンをフィルタリング
    tempo_tokens = [t for t in tokens if t.startswith('<TEMPO_')]
    tempo_token_indices = [
        tokenizer.token2idx.get(t, -1) for t in tokens if t.startswith('<TEMPO_')
    ]
    
    print(f"  ✓ Tempo tokens found: {len(tempo_tokens)}")
    print(f"    - Unique tempo values: {sorted(set(tempo_tokens))}")
    
    # 詳細表示
    print(f"\n  Detailed token sequence (with tempo tokens):")
    in_bar = False
    bar_count = 0
    for i, token in enumerate(tokens):
        if token == '<BAR>':
            in_bar = True
            bar_count += 1
        elif token.startswith('<TEMPO_'):
            if bar_count > 0 or i > 0:
                print(f"  ...")
            print(f"\n    Bar {bar_count}: {token}")
    
    # 検証チェック
    print(f"\n[Verification Checks]")
    checks = [
        (len(tempo_tokens) == len(bar_positions), 
         f"Tempo token count matches bar count: {len(tempo_tokens)} == {len(bar_positions)}"),
        (len(tokens) == len(token_indices), 
         f"Token count matches index count: {len(tokens)} == {len(token_indices)}"),
        (all(t in tokenizer.vocab for t in tokens), 
         f"All tokens in vocabulary"),
        (all(idx >= 0 and idx < tokenizer.vocab_size for idx in token_indices), 
         f"All indices are valid: 0-{tokenizer.vocab_size-1}"),
    ]
    
    all_passed = True
    for passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {message}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Tempo tokens are correctly integrated!")
    else:
        print("✗ SOME CHECKS FAILED - Please review the log above")
    print("="*70 + "\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Verify tempo tokens are correctly included in pkl files'
    )
    parser.add_argument(
        'midi_file',
        help='Input MIDI file path'
    )
    parser.add_argument(
        'output_pkl',
        nargs='?',
        default='test_output.pkl',
        help='Output PKL file path (default: test_output.pkl)'
    )
    parser.add_argument(
        '--vocab',
        help='Optional: save vocabulary to this path'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.midi_file):
        print(f"Error: MIDI file not found: {args.midi_file}")
        return 1
    
    success = verify_tempo_tokens_in_pkl(
        args.midi_file,
        args.output_pkl,
        args.vocab
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
