"""
MIDI往復変換テストスクリプト
MIDIファイル → トークン → MIDIファイルの変換が正しく動作するかテスト
"""

import sys
sys.path.append('.')

from drum_tokenizer import DrumTokenizer
from drum_to_midi import tokens_to_midi
import miditoolkit

def test_roundtrip(input_midi_path: str, output_midi_path: str):
    """MIDIファイルの往復変換テスト"""

    print("="*60)
    print("MIDI Roundtrip Conversion Test")
    print("="*60)

    # 元のMIDIファイルを読み込み
    print(f"\n1. Loading original MIDI: {input_midi_path}")
    original_midi = miditoolkit.MidiFile(input_midi_path)
    print(f"   - ticks_per_beat: {original_midi.ticks_per_beat}")
    print(f"   - num_tracks: {len(original_midi.instruments)}")

    drum_track = None
    for inst in original_midi.instruments:
        if inst.is_drum:
            drum_track = inst
            break

    if drum_track:
        print(f"   - drum_notes: {len(drum_track.notes)}")
        if drum_track.notes:
            print(f"   - note range: {min(n.start for n in drum_track.notes)} ~ {max(n.end for n in drum_track.notes)} ticks")

    # MIDI → トークン変換
    print(f"\n2. Converting MIDI to tokens...")
    tokenizer = DrumTokenizer()
    tokens, bar_positions = tokenizer.midi_to_tokens(input_midi_path)

    print(f"   - num_tokens: {len(tokens)}")
    print(f"   - num_bars: {len(bar_positions)}")
    print(f"   - first 50 tokens: {tokens[:50]}")

    # トークン → MIDI変換
    print(f"\n3. Converting tokens back to MIDI...")
    reconstructed_midi = tokens_to_midi(tokens, output_midi_path, bpm=120)

    print(f"   - ticks_per_beat: {reconstructed_midi.ticks_per_beat}")
    print(f"   - num_tracks: {len(reconstructed_midi.instruments)}")

    recon_drum_track = None
    for inst in reconstructed_midi.instruments:
        if inst.is_drum:
            recon_drum_track = inst
            break

    if recon_drum_track:
        print(f"   - drum_notes: {len(recon_drum_track.notes)}")
        if recon_drum_track.notes:
            print(f"   - note range: {min(n.start for n in recon_drum_track.notes)} ~ {max(n.end for n in recon_drum_track.notes)} ticks")

    # 結果の比較
    print(f"\n4. Comparison:")
    if drum_track and recon_drum_track:
        orig_notes = len(drum_track.notes)
        recon_notes = len(recon_drum_track.notes)

        print(f"   - Original notes: {orig_notes}")
        print(f"   - Reconstructed notes: {recon_notes}")
        print(f"   - Difference: {abs(orig_notes - recon_notes)} ({abs(orig_notes - recon_notes) / orig_notes * 100:.1f}%)")

        # 音符の位置を比較（最初の10個）
        print(f"\n   First 10 notes comparison:")
        print(f"   {'Original':>15} | {'Reconstructed':>15} | {'Diff':>10}")
        print(f"   {'-'*15}-|-{'-'*15}-|-{'-'*10}")

        for i in range(min(10, orig_notes, recon_notes)):
            orig_start = drum_track.notes[i].start
            recon_start = recon_drum_track.notes[i].start

            # 正規化後の比較（元のticks_per_beatで割る）
            orig_normalized = orig_start / original_midi.ticks_per_beat
            recon_normalized = recon_start / reconstructed_midi.ticks_per_beat

            diff = abs(orig_normalized - recon_normalized)

            print(f"   {orig_start:>15} | {recon_start:>15} | {diff:>10.3f}")

    print(f"\n5. Output saved to: {output_midi_path}")
    print("="*60)

    return tokens


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_midi_roundtrip.py <input_midi> <output_midi>")
        print("\nExample:")
        print("  python test_midi_roundtrip.py input.mid output.mid")
        sys.exit(1)

    input_midi = sys.argv[1]
    output_midi = sys.argv[2]

    tokens = test_roundtrip(input_midi, output_midi)

    print("\n✓ Test completed!")
    print(f"  Generated {len(tokens)} tokens")
    print(f"  Output: {output_midi}")
