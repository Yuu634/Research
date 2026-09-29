#!/usr/bin/env python
"""
ドラム専用トークン検証テスト - 統合テストランナー

このスクリプトは以下のテストを順番に実行します:
1. サンプルMIDIファイルの生成
2. 往復変換テスト (MIDI → Token → MIDI)
3. トークン内容の検証
"""
import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from pathlib import Path


def print_section(title: str):
    """セクションヘッダーを表示"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def run_command(cmd: list, description: str) -> bool:
    """
    コマンドを実行

    Args:
        cmd: 実行するコマンド（リスト形式）
        description: コマンドの説明

    Returns:
        成功したかどうか
    """
    print(f"実行中: {description}")
    print(f"コマンド: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} - 成功\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - 失敗")
        print(f"エラーコード: {e.returncode}\n")
        return False

    except Exception as e:
        print(f"\n✗ {description} - エラー: {e}\n")
        return False


def main():
    """メインテスト実行関数"""

    print_section("ドラム専用トークン検証テスト - 統合テストスイート")

    # ディレクトリの設定
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    sample_midis_dir = script_dir / "sample_midis"
    output_dir = script_dir / "output"

    # Python実行ファイルのパス
    python = sys.executable

    # テスト結果を記録
    test_results = []

    # ステップ1: サンプルMIDIファイルの生成
    print_section("ステップ1: サンプルMIDIファイルの生成")

    sample_generator = script_dir / "sample_midi_generator.py"

    if not sample_midis_dir.exists() or len(list(sample_midis_dir.glob("*.[mM][iI][dD]*"))) == 0:
        success = run_command(
            [python, str(sample_generator), "--output_dir", str(sample_midis_dir)],
            "サンプルMIDIファイルの生成"
        )
        test_results.append(("サンプルMIDI生成", success))

        if not success:
            print("✗ サンプルMIDIファイルの生成に失敗しました")
            print("後続のテストを中止します")
            return 1
    else:
        print("✓ サンプルMIDIファイルは既に存在します")
        test_results.append(("サンプルMIDI生成", True))

    # 生成されたMIDIファイルを確認
    midi_files = list(sample_midis_dir.glob("*.[mM][iI][dD]*"))
    print(f"\n利用可能なMIDIファイル: {len(midi_files)}個")
    for midi_file in midi_files:
        print(f"  - {midi_file.name}")

    # ステップ2: 往復変換テスト
    print_section("ステップ2: 往復変換テスト (MIDI → pkl → MIDI, 3方式)")

    round_trip_script = script_dir / "test_round_trip.py"

    success = run_command(
        [
            python,
            str(round_trip_script),
            "--midi_dir", str(sample_midis_dir),
            "--output_dir", str(output_dir),
            "--all_methods",
        ],
        "往復変換テスト (standard/remi/cp_limb_v1)"
    )
    test_results.append(("往復変換テスト", success))

    if not success:
        print("⚠ 往復変換テストに失敗しましたが、続行します")

    # ステップ3: トークン内容検証
    print_section("ステップ3: トークン内容の検証")

    verification_script = script_dir / "test_token_verification.py"

    # 生成されたpklファイルを探す（方式別サブディレクトリも含む）
    pkl_files = list(output_dir.rglob("*_tokens.pkl"))

    if len(pkl_files) == 0:
        print("✗ pklファイルが見つかりません")
        print("往復変換テストが成功していない可能性があります")
        test_results.append(("トークン内容検証", False))
    else:
        print(f"検証対象のpklファイル: {len(pkl_files)}個\n")

        verification_success = True

        for pkl_file in pkl_files:
            # 対応するMIDIファイルを探す
            base_name = pkl_file.stem.replace("_tokens", "")
            midi_file = sample_midis_dir / f"{base_name}.mid"

            # .midがなければ.midiを試す
            if not midi_file.exists():
                midi_file = sample_midis_dir / f"{base_name}.midi"

            if not midi_file.exists():
                print(f"⚠ 対応するMIDIファイルが見つかりません: {base_name}(.mid or .midi)")
                continue

            print(f"\n検証中: {pkl_file.name} ↔ {midi_file.name}")

            success = run_command(
                [
                    python,
                    str(verification_script),
                    "--pkl_path", str(pkl_file),
                    "--midi_path", str(midi_file)
                ],
                f"{base_name}のトークン内容検証"
            )

            if not success:
                verification_success = False

        test_results.append(("トークン内容検証", verification_success))

    # 最終結果のサマリー
    print_section("テスト結果サマリー")

    all_success = True

    for test_name, success in test_results:
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{status:10s} - {test_name}")

        if not success:
            all_success = False

    print("\n" + "=" * 80)

    if all_success:
        print("✓ すべてのテストが成功しました!")
        print("\n生成されたファイル:")
        print(f"  サンプルMIDI: {sample_midis_dir}")
        print(f"  テスト結果: {output_dir}")
        print(f"    - *_tokens.pkl: トークン列データ")
        print(f"    - *_reconstructed.mid: 復元されたMIDIファイル")
        print("\nトークン設計は正しく機能しています。")
        return 0
    else:
        print("✗ 一部のテストが失敗しました")
        print("\n詳細は上記のログを確認してください。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
