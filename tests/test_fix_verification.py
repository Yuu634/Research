#!/usr/bin/env python
"""
修正の検証スクリプト
ticks_per_beatの正規化とテンポトークンが正しく動作することを確認
"""
import sys
import os
from drum_tokenizer import DrumTokenizer
from drum_to_midi import tokens_to_midi
import miditoolkit

def test_ticks_normalization(midi_path: str):
    """ticks_per_beatの正規化をテスト"""
    print("=" * 80)
    print("ticks_per_beat正規化テスト")
    print("=" * 80)

    # 1. 元のMIDIファイルを読み込み
    print(f"\n1. 元のMIDIファイル: {midi_path}")
    original_midi = miditoolkit.MidiFile(midi_path)
    print(f"   - ticks_per_beat: {original_midi.ticks_per_beat}")

    drum_track = None
    for inst in original_midi.instruments:
        if inst.is_drum:
            drum_track = inst
            break

    if not drum_track or not drum_track.notes:
        print("   - ドラムトラックまたはノートが見つかりません")
        return False

    # 最初の数個のノート位置を記録
    original_notes = sorted(drum_track.notes, key=lambda n: n.start)[:10]
    print(f"   - ノート数: {len(drum_track.notes)}")
    print(f"   - 最初の3ノート位置 (ticks): {[n.start for n in original_notes[:3]]}")

    # 元のticks_per_beatで正規化した位置（beat単位）
    original_normalized = [n.start / original_midi.ticks_per_beat for n in original_notes[:3]]
    print(f"   - 最初の3ノート位置 (beats): {[f'{x:.4f}' for x in original_normalized]}")

    # 2. トークン化
    print(f"\n2. トークン化")
    tokenizer = DrumTokenizer()
    tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
    print(f"   - トークン数: {len(tokens)}")

    # テンポトークンを確認
    tempo_tokens = [t for t in tokens if t.startswith('<TEMPO_')]
    print(f"   - 検出されたテンポトークン: {tempo_tokens[:3]}")

    # トークン列の最初の部分を表示
    print(f"   - 最初の30トークン:")
    print(f"     {tokens[:30]}")

    # 3. MIDI再構築
    print(f"\n3. MIDI再構築")
    output_path = '/tmp/test_reconstructed.mid'
    reconstructed_midi = tokens_to_midi(tokens, output_path)
    print(f"   - ticks_per_beat: {reconstructed_midi.ticks_per_beat}")
    print(f"   - 出力: {output_path}")

    recon_drum_track = reconstructed_midi.instruments[0]
    reconstructed_notes = sorted(recon_drum_track.notes, key=lambda n: n.start)[:10]
    print(f"   - ノート数: {len(recon_drum_track.notes)}")
    print(f"   - 最初の3ノート位置 (ticks): {[n.start for n in reconstructed_notes[:3]]}")

    # 再構築後のticks_per_beatで正規化した位置（beat単位）
    reconstructed_normalized = [n.start / reconstructed_midi.ticks_per_beat for n in reconstructed_notes[:3]]
    print(f"   - 最初の3ノート位置 (beats): {[f'{x:.4f}' for x in reconstructed_normalized]}")

    # 4. 比較
    print(f"\n4. 比較結果")
    print(f"   {'元 (beats)':>12} | {'再構築 (beats)':>15} | {'差':>10}")
    print(f"   {'-'*12}-+-{'-'*15}-+-{'-'*10}")

    all_match = True
    for i in range(min(3, len(original_normalized), len(reconstructed_normalized))):
        orig = original_normalized[i]
        recon = reconstructed_normalized[i]
        diff = abs(orig - recon)
        match = "✓" if diff < 0.01 else "✗"
        print(f"   {orig:>12.4f} | {recon:>15.4f} | {diff:>9.4f} {match}")
        if diff >= 0.01:
            all_match = False

    if all_match:
        print(f"\n✓ 成功: すべてのノート位置が正しく保持されています!")
        return True
    else:
        print(f"\n✗ 失敗: ノート位置に不一致があります")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_fix_verification.py <input_midi>")
        print("\nExample:")
        print("  python test_fix_verification.py input.mid")
        sys.exit(1)

    midi_path = sys.argv[1]

    if not os.path.exists(midi_path):
        print(f"エラー: ファイルが見つかりません: {midi_path}")
        sys.exit(1)

    success = test_ticks_normalization(midi_path)
    sys.exit(0 if success else 1)
