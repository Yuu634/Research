#!/usr/bin/env python
"""
ドラム譜トークンとMIDI変換のテストスクリプト
依存関係がインストールされているか確認し、基本的な動作をテストします
"""
import sys
import os

def test_imports():
    """必要なモジュールのインポートをテスト"""
    print("=" * 60)
    print("依存関係のチェック")
    print("=" * 60)

    modules = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'yaml': 'PyYAML',
        'miditoolkit': 'miditoolkit',
        'tqdm': 'tqdm'
    }

    missing = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未インストール")
            missing.append(name)

    if missing:
        print(f"\n警告: {', '.join(missing)} がインストールされていません")
        print("pip install -r requirements_drum.txt を実行してください")
        return False

    print("\n✓ すべての依存関係がインストールされています")
    return True


def test_tokenizer():
    """トークナイザーのテスト"""
    print("\n" + "=" * 60)
    print("トークナイザーのテスト")
    print("=" * 60)

    try:
        from drum_tokenizer import DrumTokenizer

        tokenizer = DrumTokenizer()
        print(f"✓ トークナイザーの初期化成功")
        print(f"  語彙サイズ: {tokenizer.vocab_size}")

        # トークンの例をいくつか表示
        print(f"\n  構造トークンの例:")
        structure_tokens = [t for t in tokenizer.vocab if t.startswith('<')][:10]
        for token in structure_tokens:
            print(f"    - {token}")

        print(f"\n  ドラムトークンの例:")
        drum_tokens = [t for t in tokenizer.vocab if not t.startswith('<')][:10]
        for token in drum_tokens:
            print(f"    - {token}")

        # 語彙を保存
        vocab_path = '/tmp/test_drum_vocab.pkl'
        tokenizer.save_vocab(vocab_path)
        print(f"\n✓ テスト語彙ファイルを保存: {vocab_path}")

        return True

    except Exception as e:
        print(f"✗ トークナイザーのテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cp_tokenizer():
    """CPトークナイザーのテスト"""
    print("\n" + "=" * 60)
    print("CPトークナイザーのテスト")
    print("=" * 60)

    try:
        from drum_cp_tokenizer import DrumCPTokenizer

        tokenizer = DrumCPTokenizer()
        print("✓ CPトークナイザーの初期化成功")
        print(f"  EventType語彙サイズ: {len(tokenizer.event_type2idx)}")
        print(f"  Structural語彙サイズ: {len(tokenizer.struct_token2idx)}")
        print(f"  Limb語彙サイズ: {len(tokenizer.limb_token2idx)}")

        vocab_path = '/tmp/test_drum_cp_vocab.pkl'
        tokenizer.save_vocab(vocab_path)
        print(f"✓ テストCP語彙ファイルを保存: {vocab_path}")

        return True

    except Exception as e:
        print(f"✗ CPトークナイザーのテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_token_to_midi():
    """トークンからMIDIへの変換をテスト"""
    print("\n" + "=" * 60)
    print("Token to MIDI 変換のテスト")
    print("=" * 60)

    try:
        from drum_to_midi import tokens_to_midi

        # テスト用のトークン列
        test_tokens = [
            '<BAR>',
            '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'HH_CLOSED_HIT_Normal',
            '<POS_6>', 'HH_CLOSED_HIT_Ghost',
            '<POS_12>', 'SNARE_HIT_Normal', 'HH_CLOSED_HIT_Normal',
            '<POS_18>', 'HH_CLOSED_HIT_Ghost',
            '<BEAT_2>', '<POS_0>', 'KICK_HIT_Accent', 'HH_CLOSED_HIT_Normal',
            '<POS_6>', 'HH_CLOSED_HIT_Ghost',
            '<POS_12>', 'SNARE_HIT_Accent', 'HH_CLOSED_HIT_Normal',
            '<POS_18>', 'HH_CLOSED_HIT_Ghost',
            '<BAR>',
            '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'CRASH_HIT_Accent',
            '<POS_12>', 'SNARE_FLAM_Normal',
            '<BEAT_2>', '<POS_0>', 'KICK_HIT_Normal',
            '<POS_12>', 'SNARE_HIT_Normal',
            '<EOS>'
        ]

        output_path = '/tmp/test_drum_output.mid'
        midi_obj = tokens_to_midi(test_tokens, output_path, bpm=120)

        print(f"✓ MIDIファイルの生成成功")
        print(f"  出力: {output_path}")
        print(f"  ノート数: {len(midi_obj.instruments[0].notes)}")
        print(f"  BPM: 120")

        return True

    except Exception as e:
        print(f"✗ Token to MIDI 変換のテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_imports():
    """データローダーのインポートをテスト"""
    print("\n" + "=" * 60)
    print("データローダーのインポートテスト")
    print("=" * 60)

    try:
        from drum_dataloader import DrumTransformerDataset
        print(f"✓ データローダーのインポート成功")
        return True

    except Exception as e:
        print(f"✗ データローダーのインポート失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cp_model_smoke():
    """CPモデルの前向き/損失計算スモークテスト"""
    print("\n" + "=" * 60)
    print("CPモデルスモークテスト")
    print("=" * 60)

    try:
        import torch
        from model.musemorphose import MuseMorphose

        bsz = 2
        n_bars = 4
        enc_len = 16
        dec_len = 32

        model = MuseMorphose(
            enc_n_layer=2,
            enc_n_head=2,
            enc_d_model=64,
            enc_d_ff=128,
            dec_n_layer=2,
            dec_n_head=2,
            dec_d_model=64,
            dec_d_ff=128,
            d_vae_latent=64,
            d_embed=64,
            n_token=32,
            tokenization_method='cp_limb_v1',
            cp_event_type_vocab_size=3,
            cp_struct_vocab_size=32,
            cp_pos_vocab_size=25,
            cp_limb_vocab_size=40,
            cp_event_pad_idx=2,
            cp_struct_pad_idx=0,
            cp_pos_pad_idx=24,
            cp_limb_pad_idx=0,
            use_attr_cls=False,
            cond_mode='in-attn'
        )

        enc_inp = torch.randint(0, 31, (enc_len, bsz, n_bars))
        dec_inp = torch.randint(0, 31, (dec_len, bsz))
        bar_pos = torch.tensor([[0, 8, 16, 24, 32], [0, 8, 16, 24, 32]], dtype=torch.long)
        padding_mask = torch.zeros((bsz, n_bars, enc_len), dtype=torch.bool)

        cp_event_type_inp = torch.randint(0, 2, (dec_len, bsz))
        cp_struct_inp = torch.randint(0, 31, (dec_len, bsz))
        cp_pos_inp = torch.randint(0, 24, (dec_len, bsz))
        cp_h1_inp = torch.randint(0, 39, (dec_len, bsz))
        cp_h2_inp = torch.randint(0, 39, (dec_len, bsz))
        cp_rf_inp = torch.randint(0, 39, (dec_len, bsz))
        cp_lf_inp = torch.randint(0, 39, (dec_len, bsz))

        mu, logvar, cp_logits = model(
            enc_inp,
            dec_inp,
            bar_pos,
            cp_event_type_inp=cp_event_type_inp,
            cp_struct_inp=cp_struct_inp,
            cp_pos_inp=cp_pos_inp,
            cp_hand1_inp=cp_h1_inp,
            cp_hand2_inp=cp_h2_inp,
            cp_right_foot_inp=cp_rf_inp,
            cp_left_foot_inp=cp_lf_inp,
            padding_mask=padding_mask,
        )

        losses = model.compute_cp_loss(
            mu,
            logvar,
            0.1,
            0.0,
            cp_logits,
            {
                'event_type': cp_event_type_inp,
                'structural': cp_struct_inp,
                'cp_pos': cp_pos_inp,
                'cp_hand1': cp_h1_inp,
                'cp_hand2': cp_h2_inp,
                'cp_right_foot': cp_rf_inp,
                'cp_left_foot': cp_lf_inp,
            },
        )

        print(f"✓ CPモデル前向き成功: logits keys = {list(cp_logits.keys())}")
        print(f"✓ CP損失計算成功: total_loss = {losses['total_loss'].item():.4f}")
        return True

    except Exception as e:
        print(f"✗ CPモデルスモークテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト関数"""
    print("\nドラム譜MuseMorphose - システムテスト\n")

    results = []

    # 依存関係のチェック
    results.append(("依存関係", test_imports()))

    # トークナイザーのテスト
    results.append(("トークナイザー", test_tokenizer()))

    # Token to MIDI 変換のテスト
    results.append(("Token→MIDI変換", test_token_to_midi()))

    # CPトークナイザーのテスト
    results.append(("CPトークナイザー", test_cp_tokenizer()))

    # CPモデルのスモークテスト
    results.append(("CPモデルスモーク", test_cp_model_smoke()))

    # データローダーのインポートテスト
    results.append(("データローダー", test_dataloader_imports()))

    # 結果のサマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    for name, result in results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{name:20s}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ すべてのテストが成功しました!")
        print("\n次は以下のコマンドでデータセットを準備してください:")
        print("  python prepare_drum_dataset.py --midi_dir /path/to/midis --output_dir ./drum_dataset --vocab_path ./drum_vocab.pkl")
        return 0
    else:
        print("\n✗ いくつかのテストが失敗しました")
        print("\n依存関係がすべてインストールされているか確認してください:")
        print("  pip install -r requirements_drum.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
