#!/usr/bin/env python
"""
CUDA修正の検証スクリプト
train_drum.pyのデバイス選択機能が正しく動作するか確認します
"""
import sys
import torch

def test_device_detection():
    """デバイス検出のテスト"""
    print("=" * 60)
    print("デバイス検出テスト")
    print("=" * 60)

    print(f"\nPyTorch バージョン: {torch.__version__}")
    print(f"CUDA 利用可能: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA バージョン: {torch.version.cuda}")
        print(f"GPU デバイス: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA capability: sm_{cap[0]}{cap[1]}")

        # CUDAテスト
        try:
            test_tensor = torch.zeros(10, 10).cuda()
            print("\n✓ CUDA動作テスト成功")
            print("  GPUでの学習が可能です")
            return 'cuda'
        except RuntimeError as e:
            print(f"\n✗ CUDA動作テスト失敗: {e}")
            print("  CPUでの学習を推奨します")
            return 'cpu'
    else:
        print("\nCUDAが利用できません")
        print("  CPUでの学習のみ可能です")
        return 'cpu'


def test_config_loading():
    """設定ファイルの読み込みテスト"""
    print("\n" + "=" * 60)
    print("設定ファイル読み込みテスト")
    print("=" * 60)

    try:
        import yaml
        config_path = 'config/drum_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        device = config['training']['device']
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']

        print(f"\n設定ファイル: {config_path}")
        print(f"  デバイス: {device}")
        print(f"  バッチサイズ: {batch_size}")
        print(f"  ワーカー数: {num_workers}")

        if device == 'cpu':
            print("\n✓ CPUモードに設定されています（推奨）")
        else:
            print("\n⚠ CUDAモードに設定されています")
            print("  CUDAエラーが発生する場合は device: 'cpu' に変更してください")

        return True

    except Exception as e:
        print(f"\n✗ 設定ファイルの読み込み失敗: {e}")
        return False


def test_imports():
    """必要なモジュールのインポートテスト"""
    print("\n" + "=" * 60)
    print("モジュールインポートテスト")
    print("=" * 60)

    modules = [
        ('torch', 'PyTorch'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy'),
    ]

    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未インストール")
            all_ok = False

    # ローカルモジュール
    try:
        import drum_tokenizer
        print(f"✓ drum_tokenizer")
    except ImportError as e:
        print(f"✗ drum_tokenizer - {e}")
        all_ok = False

    try:
        import drum_dataloader
        print(f"✓ drum_dataloader")
    except ImportError as e:
        print(f"✗ drum_dataloader - {e}")
        all_ok = False

    return all_ok


def main():
    """メイン関数"""
    print("\nドラム譜MuseMorphose - CUDA修正検証\n")

    # テスト実行
    recommended_device = test_device_detection()
    config_ok = test_config_loading()
    imports_ok = test_imports()

    # 結果サマリー
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)

    print(f"\n推奨デバイス: {recommended_device.upper()}")

    if config_ok:
        print("設定ファイル: ✓ OK")
    else:
        print("設定ファイル: ✗ エラー")

    if imports_ok:
        print("依存関係: ✓ OK")
    else:
        print("依存関係: ✗ 不足")

    # 実行コマンドの提案
    print("\n" + "=" * 60)
    print("次のステップ")
    print("=" * 60)

    if not imports_ok:
        print("\n依存関係をインストールしてください:")
        print("  pip install -r requirements_drum.txt")
        return 1

    if recommended_device == 'cpu':
        print("\nCPUで学習を開始できます:")
        print("  bash quickstart_cpu.sh")
        print("\nまたは:")
        print("  python train_drum.py --config config/drum_config.yaml --device cpu")
    else:
        print("\nGPUで学習を開始できます:")
        print("  python train_drum.py --config config/drum_config.yaml --device cuda")
        print("\nエラーが発生する場合はCPUで実行してください:")
        print("  python train_drum.py --config config/drum_config.yaml --device cpu")

    print("\n詳細なトラブルシューティング:")
    print("  CUDA_TROUBLESHOOTING.md を参照")

    return 0


if __name__ == '__main__':
    sys.exit(main())
