"""
テンポトークン機能のテストスクリプト

MIDI → トークン → MIDI の変換でテンポが正しく保持されるかテストします。
"""

import sys
sys.path.append('.')

from drum_tokenizer import DrumTokenizer
from drum_to_midi import tokens_to_midi
import miditoolkit

def test_tempo_token(input_midi_path: str, output_midi_path: str):
    """テンポトークンのテスト"""

    print("="*60)
    print("Tempo Token Test")
    print("="*60)

    # 元のMIDIファイルを読み込み
    print(f"\n1. Loading original MIDI: {input_midi_path}")
    original_midi = miditoolkit.MidiFile(input_midi_path)

    # テンポ情報を表示
    print(f"   Original tempo changes:")
    if original_midi.tempo_changes:
        for tc in original_midi.tempo_changes:
            print(f"     - Time: {tc.time} ticks, Tempo: {tc.tempo} BPM")
    else:
        print(f"     - No tempo changes (default: 120 BPM)")

    # MIDI → トークン変換
    print(f"\n2. Converting MIDI to tokens...")
    tokenizer = DrumTokenizer()
    tokens, bar_positions = tokenizer.midi_to_tokens(input_midi_path)

    print(f"   - num_tokens: {len(tokens)}")
    print(f"   - num_bars: {len(bar_positions)}")

    # テンポトークンを抽出して表示
    print(f"\n3. Tempo tokens in the sequence:")
    tempo_tokens = [t for t in tokens if t.startswith('<TEMPO_')]
    if tempo_tokens:
        print(f"   Found {len(tempo_tokens)} tempo tokens:")
        for i, tt in enumerate(tempo_tokens[:10]):  # 最初の10個を表示
            tempo_value = int(tt.replace('<TEMPO_', '').replace('>', ''))
            bar_idx = tokens[:tokens.index(tt)].count('<BAR>')
            print(f"     - Bar {bar_idx}: {tt} ({tempo_value} BPM)")
        if len(tempo_tokens) > 10:
            print(f"     ... and {len(tempo_tokens) - 10} more")
    else:
        print(f"   No tempo tokens found!")

    # 最初の50トークンを表示
    print(f"\n4. First 50 tokens:")
    print(f"   {tokens[:50]}")

    # トークン → MIDI変換
    print(f"\n5. Converting tokens back to MIDI...")
    reconstructed_midi = tokens_to_midi(tokens, output_midi_path)

    print(f"   Reconstructed MIDI saved to: {output_midi_path}")

    # 再構築されたテンポ情報を表示
    print(f"\n6. Reconstructed tempo changes:")
    if reconstructed_midi.tempo_changes:
        for tc in reconstructed_midi.tempo_changes:
            print(f"     - Time: {tc.time} ticks, Tempo: {tc.tempo} BPM")
    else:
        print(f"     - No tempo changes")

    # 比較
    print(f"\n7. Comparison:")
    original_tempo = original_midi.tempo_changes[0].tempo if original_midi.tempo_changes else 120.0
    reconstructed_tempo = reconstructed_midi.tempo_changes[0].tempo if reconstructed_midi.tempo_changes else 120.0

    # 量子化されたテンポを計算
    quantized_original = int(original_tempo / 5) * 5

    print(f"   Original tempo: {original_tempo} BPM")
    print(f"   Quantized tempo (5 BPM steps): {quantized_original} BPM")
    print(f"   Reconstructed tempo: {reconstructed_tempo} BPM")

    if reconstructed_tempo == quantized_original:
        print(f"   ✓ Tempo correctly preserved (with 5 BPM quantization)")
    else:
        print(f"   ✗ Tempo mismatch: {abs(reconstructed_tempo - quantized_original)} BPM difference")

    print("\n" + "="*60)

    return tokens


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_tempo_token.py <input_midi> <output_midi>")
        print("\nExample:")
        print("  python test_tempo_token.py input.mid output.mid")
        sys.exit(1)

    input_midi = sys.argv[1]
    output_midi = sys.argv[2]

    tokens = test_tempo_token(input_midi, output_midi)

    print("\n✓ Test completed!")
    print(f"  Generated {len(tokens)} tokens")
    print(f"  Output: {output_midi}")
