#!/usr/bin/env python
"""
テスト用のドラムMIDIファイルを生成するユーティリティ
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import miditoolkit

# 定数
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_BPM = 120


def create_simple_beat(output_path: str):
    """
    シンプルな8ビートのドラムパターンを生成

    パターン:
    - Bar 1-2: 基本的な8ビート
    - Bar 3-4: フィルイン付き
    """
    midi_obj = miditoolkit.MidiFile()
    midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL

    # ドラムトラック作成
    drum_track = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')

    # テンポ設定
    midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(DEFAULT_BPM, 0)
    )

    # カスタムドラムマップ（新しい設計に対応）
    KICK = 36
    SNARE = 38      # D2
    HH_CLOSED = 42  # F#2
    HH_OPEN = 46    # A#2
    CRASH = 49
    TOM1 = 48       # C3 - 修正
    TOM2 = 47       # B2 - 修正
    FLOOR = 41      # F2

    # 小節1-2: 基本8ビート
    for bar in range(2):
        bar_start = bar * DEFAULT_BAR_RESOL

        for beat in range(4):
            tick = bar_start + beat * DEFAULT_BEAT_RESOL
            eighth_note = DEFAULT_BEAT_RESOL // 2

            # キック: 1拍目と3拍目
            if beat in [0, 2]:
                drum_track.notes.append(miditoolkit.Note(
                    velocity=100, pitch=KICK,
                    start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
                ))

            # スネア: 2拍目と4拍目
            if beat in [1, 3]:
                drum_track.notes.append(miditoolkit.Note(
                    velocity=90, pitch=SNARE,
                    start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
                ))

            # ハイハット: 8分音符刻み
            for i in range(2):
                hh_tick = tick + i * eighth_note
                velocity = 80 if i == 0 else 60
                drum_track.notes.append(miditoolkit.Note(
                    velocity=velocity, pitch=HH_CLOSED,
                    start=hh_tick, end=hh_tick + DEFAULT_BEAT_RESOL // 16
                ))

    # 小節3: フィルイン
    bar_start = 2 * DEFAULT_BAR_RESOL

    # 1-2拍目: 通常パターン
    for beat in range(2):
        tick = bar_start + beat * DEFAULT_BEAT_RESOL
        eighth_note = DEFAULT_BEAT_RESOL // 2

        if beat == 0:
            drum_track.notes.append(miditoolkit.Note(
                velocity=100, pitch=KICK,
                start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
            ))
        else:
            drum_track.notes.append(miditoolkit.Note(
                velocity=90, pitch=SNARE,
                start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
            ))

        for i in range(2):
            hh_tick = tick + i * eighth_note
            velocity = 80 if i == 0 else 60
            drum_track.notes.append(miditoolkit.Note(
                velocity=velocity, pitch=HH_CLOSED,
                start=hh_tick, end=hh_tick + DEFAULT_BEAT_RESOL // 16
            ))

    # 3-4拍目: タムフィルイン
    sixteenth = DEFAULT_BEAT_RESOL // 4
    fill_start = bar_start + 2 * DEFAULT_BEAT_RESOL

    # 16分音符でタムを叩く
    toms = [TOM1, TOM1, TOM2, TOM2, TOM2, TOM1, TOM2, SNARE]
    for i, tom in enumerate(toms):
        tick = fill_start + i * sixteenth
        velocity = 100 if i % 2 == 0 else 85
        drum_track.notes.append(miditoolkit.Note(
            velocity=velocity, pitch=tom,
            start=tick, end=tick + sixteenth // 2
        ))

    # 小節4: クラッシュで終わる
    bar_start = 3 * DEFAULT_BAR_RESOL

    # 1拍目: キック + クラッシュ
    drum_track.notes.append(miditoolkit.Note(
        velocity=110, pitch=KICK,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL // 8
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=120, pitch=CRASH,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL * 2  # 長めに
    ))

    # 残りは8ビート
    for beat in range(1, 4):
        tick = bar_start + beat * DEFAULT_BEAT_RESOL
        eighth_note = DEFAULT_BEAT_RESOL // 2

        if beat in [2]:
            drum_track.notes.append(miditoolkit.Note(
                velocity=100, pitch=KICK,
                start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
            ))

        if beat in [1, 3]:
            drum_track.notes.append(miditoolkit.Note(
                velocity=90, pitch=SNARE,
                start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
            ))

        for i in range(2):
            hh_tick = tick + i * eighth_note
            velocity = 80 if i == 0 else 60
            drum_track.notes.append(miditoolkit.Note(
                velocity=velocity, pitch=HH_CLOSED,
                start=hh_tick, end=hh_tick + DEFAULT_BEAT_RESOL // 16
            ))

    midi_obj.instruments.append(drum_track)
    midi_obj.dump(output_path)

    print(f"✓ シンプルなビートを生成: {output_path}")
    print(f"  小節数: 4")
    print(f"  ノート数: {len(drum_track.notes)}")
    return output_path


def create_complex_beat(output_path: str):
    """
    複雑なドラムパターンを生成（様々な技法を含む）

    含まれる要素:
    - ゴーストノート
    - アクセント
    - クラッシュチョーク
    - 複雑なリズム (16分音符、3連符風)
    """
    midi_obj = miditoolkit.MidiFile()
    midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL

    drum_track = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')

    midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(DEFAULT_BPM, 0)
    )

    # カスタムドラムマップ（新しい設計に対応）
    KICK = 36
    SNARE = 38        # D2
    SIDE_STICK = 37
    HH_CLOSED = 42    # F#2
    HH_PEDAL = 44
    HH_OPEN = 46      # A#2
    CRASH = 49
    RIDE = 51         # D#3
    TOM1 = 48         # C3 - 修正
    TOM2 = 47         # B2
    FLOOR = 41        # F2

    # 小節1: ゴーストノートとアクセントを含む複雑なパターン
    bar_start = 0

    # 1拍目: キック + ハイハット (アクセント ベロシティ120)
    drum_track.notes.append(miditoolkit.Note(
        velocity=110, pitch=KICK,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL // 8
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=120, pitch=HH_CLOSED,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL // 16
    ))

    # 1拍目裏: ゴーストノート (ベロシティ30)
    sixteenth = DEFAULT_BEAT_RESOL // 4
    drum_track.notes.append(miditoolkit.Note(
        velocity=30, pitch=SNARE,
        start=bar_start + 2 * sixteenth, end=bar_start + 2 * sixteenth + sixteenth // 2
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=35, pitch=HH_CLOSED,
        start=bar_start + 2 * sixteenth, end=bar_start + 2 * sixteenth + sixteenth // 2
    ))

    # 2拍目: スネア (ノーマル ベロシティ85)
    tick = bar_start + DEFAULT_BEAT_RESOL
    drum_track.notes.append(miditoolkit.Note(
        velocity=85, pitch=SNARE,
        start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=80, pitch=HH_CLOSED,
        start=tick, end=tick + DEFAULT_BEAT_RESOL // 16
    ))

    # 2拍目後半: 16分音符連打
    for i in range(2, 4):
        t = tick + i * sixteenth
        drum_track.notes.append(miditoolkit.Note(
            velocity=60, pitch=HH_CLOSED,
            start=t, end=t + sixteenth // 2
        ))

    # 3拍目: キック + サイドスティック
    tick = bar_start + 2 * DEFAULT_BEAT_RESOL
    drum_track.notes.append(miditoolkit.Note(
        velocity=100, pitch=KICK,
        start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=80, pitch=SIDE_STICK,
        start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
    ))

    # 4拍目: スネア (アクセント ベロシティ115)
    tick = bar_start + 3 * DEFAULT_BEAT_RESOL
    drum_track.notes.append(miditoolkit.Note(
        velocity=115, pitch=SNARE,
        start=tick, end=tick + DEFAULT_BEAT_RESOL // 8
    ))

    # 小節2: クラッシュチョークとハイハットペダル
    bar_start = DEFAULT_BAR_RESOL

    # 1拍目: クラッシュ（短い音価でチョーク）
    drum_track.notes.append(miditoolkit.Note(
        velocity=110, pitch=CRASH,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL // 6  # 極端に短い
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=100, pitch=KICK,
        start=bar_start, end=bar_start + DEFAULT_BEAT_RESOL // 8
    ))

    # ハイハットペダル → クローズドハイハットに変更(MIDI 44をMIDI 42に)
    for beat in range(4):
        tick = bar_start + beat * DEFAULT_BEAT_RESOL
        drum_track.notes.append(miditoolkit.Note(
            velocity=70, pitch=HH_CLOSED,  # HH_PEDAL → HH_CLOSEDに変更
            start=tick, end=tick + DEFAULT_BEAT_RESOL // 16
        ))

        # 裏拍にオープンハイハット
        if beat in [1, 3]:
            t = tick + DEFAULT_BEAT_RESOL // 2
            drum_track.notes.append(miditoolkit.Note(
                velocity=90, pitch=HH_OPEN,
                start=t, end=t + DEFAULT_BEAT_RESOL // 4
            ))

    # 小節3: フラム（短い間隔の同楽器連打）
    bar_start = 2 * DEFAULT_BAR_RESOL

    # 1拍目: スネアフラム（前打音 + 主音符）
    # 前打音（ゴーストノート）- bar内に収める
    drum_track.notes.append(miditoolkit.Note(
        velocity=35, pitch=SNARE,
        start=max(0, bar_start), end=max(0, bar_start + 20)  # bar内に配置
    ))
    # 主音符
    drum_track.notes.append(miditoolkit.Note(
        velocity=95, pitch=SNARE,
        start=bar_start + 30, end=bar_start + 30 + DEFAULT_BEAT_RESOL // 8  # 30 ticks後に配置
    ))

    # 2拍目: タムフラム
    tick = bar_start + DEFAULT_BEAT_RESOL
    drum_track.notes.append(miditoolkit.Note(
        velocity=40, pitch=TOM1,
        start=tick, end=tick + 20  # bar内に配置
    ))
    drum_track.notes.append(miditoolkit.Note(
        velocity=100, pitch=TOM1,
        start=tick + 35, end=tick + 35 + DEFAULT_BEAT_RESOL // 8  # 35 ticks後に配置
    ))

    # 3-4拍目: ライドシンバル
    for beat in [2, 3]:
        tick = bar_start + beat * DEFAULT_BEAT_RESOL
        for i in range(3):
            t = tick + i * (DEFAULT_BEAT_RESOL // 3)
            vel = 95 if i == 0 else 70
            drum_track.notes.append(miditoolkit.Note(
                velocity=vel, pitch=RIDE,
                start=t, end=t + DEFAULT_BEAT_RESOL // 16
            ))

    midi_obj.instruments.append(drum_track)
    midi_obj.dump(output_path)

    print(f"✓ 複雑なビートを生成: {output_path}")
    print(f"  小節数: 3")
    print(f"  ノート数: {len(drum_track.notes)}")
    print(f"  含まれる要素: ゴーストノート, アクセント, フラム, チョーク")
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='テスト用ドラムMIDIファイルを生成')
    parser.add_argument('--output_dir', type=str, default='./test_token/sample_midis',
                        help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)

    print("テスト用MIDIファイルを生成中...")
    print("=" * 60)

    # サンプルファイルを生成
    simple_path = os.path.join(args.output_dir, 'simple_beat.mid')
    complex_path = os.path.join(args.output_dir, 'complex_beat.mid')

    create_simple_beat(simple_path)
    create_complex_beat(complex_path)

    print("\n" + "=" * 60)
    print("✓ すべてのサンプルファイルを生成しました")
    print(f"\n生成されたファイル:")
    print(f"  1. {simple_path}")
    print(f"  2. {complex_path}")
