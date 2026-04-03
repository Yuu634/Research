import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if torch is not None:
    from src.models.musemorphose import MuseMorphose
else:
    MuseMorphose = None


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class MuseMorphoseHierarchicalCompatTests(unittest.TestCase):
    def _build_standard_or_remi_model(self, tokenization_method="standard", use_hierarchical_pe=False):
        return MuseMorphose(
            enc_n_layer=1,
            enc_n_head=2,
            enc_d_model=8,
            enc_d_ff=16,
            dec_n_layer=1,
            dec_n_head=2,
            dec_d_model=8,
            dec_d_ff=16,
            d_vae_latent=4,
            d_embed=8,
            n_token=32,
            enc_dropout=0.0,
            dec_dropout=0.0,
            tokenization_method=tokenization_method,
            use_attr_cls=False,
            cond_mode="in-attn",
            use_hierarchical_pe=use_hierarchical_pe,
            hier_pe_bar_vocab_size=64,
            hier_pe_beat_vocab_size=8,
            hier_pe_pos_vocab_size=32,
        )

    def _build_cp_model(self, use_hierarchical_pe=False):
        return MuseMorphose(
            enc_n_layer=1,
            enc_n_head=2,
            enc_d_model=8,
            enc_d_ff=16,
            dec_n_layer=1,
            dec_n_head=2,
            dec_d_model=8,
            dec_d_ff=16,
            d_vae_latent=4,
            d_embed=8,
            n_token=16,
            enc_dropout=0.0,
            dec_dropout=0.0,
            tokenization_method="cp_limb_v1",
            use_attr_cls=False,
            cond_mode="in-attn",
            cp_event_type_vocab_size=3,
            cp_type_vocab_size=16,
            cp_struct_vocab_size=16,
            cp_pos_vocab_size=25,
            cp_limb_vocab_size=8,
            cp_event_pad_idx=2,
            cp_struct_pad_idx=0,
            cp_pos_pad_idx=24,
            cp_limb_pad_idx=0,
            d_cp_type_emb=4,
            d_cp_pos_emb=2,
            d_cp_limb_emb=1,
            use_hierarchical_pe=use_hierarchical_pe,
            hier_pe_bar_vocab_size=64,
            hier_pe_beat_vocab_size=8,
            hier_pe_pos_vocab_size=32,
        )

    def _make_bar_pos(self, batch_size=2, dec_len=8, n_bars=2):
        if n_bars != 2:
            raise ValueError("This helper currently assumes n_bars=2")
        return torch.tensor([[0, dec_len // 2, dec_len]] * batch_size, dtype=torch.long)

    def _make_hier_indices(self, enc_inp, dec_len, batch_size):
        enc_bar_idx = torch.zeros_like(enc_inp)
        enc_beat_idx = torch.ones_like(enc_inp)
        enc_pos_idx = torch.zeros_like(enc_inp)
        enc_is_struct = torch.zeros_like(enc_inp)

        dec_bar_idx = torch.zeros((dec_len, batch_size), dtype=torch.long)
        dec_beat_idx = torch.ones((dec_len, batch_size), dtype=torch.long)
        dec_pos_idx = torch.zeros((dec_len, batch_size), dtype=torch.long)
        dec_is_struct = torch.zeros((dec_len, batch_size), dtype=torch.long)

        return {
            "enc_bar_idx": enc_bar_idx,
            "enc_beat_idx": enc_beat_idx,
            "enc_pos_idx": enc_pos_idx,
            "enc_is_struct": enc_is_struct,
            "dec_bar_idx": dec_bar_idx,
            "dec_beat_idx": dec_beat_idx,
            "dec_pos_idx": dec_pos_idx,
            "dec_is_struct": dec_is_struct,
        }

    # 7.1: Barアンカー分岐
    def test_bar_anchor_selection_prefers_unique_bar_token_with_fallback(self):
        model = self._build_standard_or_remi_model(tokenization_method="standard")
        model.bar_anchor_mode = "bar_token"
        model.bar_token_idx = 3

        # col0: unique bar at row=2, col1: duplicated bar -> fallback(0)
        enc_tokens_flat = torch.tensor(
            [
                [1, 1],
                [2, 3],
                [3, 4],
                [5, 6],
                [7, 3],
            ],
            dtype=torch.long,
        )

        anchor_positions = model._resolve_encoder_anchor_positions(enc_tokens_flat)
        self.assertEqual(anchor_positions.tolist(), [2, 0])

    # 7.1: PE selectorのフォールバック
    def test_position_encoding_falls_back_to_absolute_when_hier_indices_missing(self):
        model = self._build_standard_or_remi_model(tokenization_method="standard", use_hierarchical_pe=True)
        seq_len = 6

        pe_abs = model.pe(seq_len)
        pe_selected = model._build_position_encoding(seq_len, bar_idx=None, beat_idx=None, pos_idx=None)
        self.assertTrue(torch.allclose(pe_abs, pe_selected))

    # 7.1: 階層PE選択（インデックス完備時）
    def test_position_encoding_uses_hierarchical_when_indices_available(self):
        model = self._build_standard_or_remi_model(tokenization_method="standard", use_hierarchical_pe=True)
        seq_len = 5
        bsz = 2

        with torch.no_grad():
            model.hier_pe.bar_emb.weight.zero_()
            model.hier_pe.beat_emb.weight.zero_()
            model.hier_pe.pos_emb.weight.zero_()
            model.hier_pe.bar_emb.weight[0, :] = 1.0

        bar_idx = torch.zeros((seq_len, bsz), dtype=torch.long)
        beat_idx = torch.zeros((seq_len, bsz), dtype=torch.long)
        pos_idx = torch.zeros((seq_len, bsz), dtype=torch.long)

        pe_hier = model._build_position_encoding(seq_len, bar_idx=bar_idx, beat_idx=beat_idx, pos_idx=pos_idx)
        pe_abs = model.pe(seq_len)

        self.assertEqual(pe_hier.shape, pe_abs.shape)
        self.assertFalse(torch.allclose(pe_hier, pe_abs))

    # 7.2: mode別1-step学習スモーク
    def test_one_step_train_smoke_all_modes(self):
        torch.manual_seed(42)
        modes = ["standard", "remi", "cp_limb_v1"]

        for mode in modes:
            if mode == "cp_limb_v1":
                model = self._build_cp_model(use_hierarchical_pe=True)
                vocab_size = 16
                enc_len, dec_len, bsz, n_bars = 4, 8, 2, 2
                enc_inp = torch.randint(0, vocab_size, (enc_len, bsz, n_bars), dtype=torch.long)
                dec_inp = torch.randint(0, vocab_size, (dec_len, bsz), dtype=torch.long)
                bar_pos = self._make_bar_pos(batch_size=bsz, dec_len=dec_len, n_bars=n_bars)
                padding_mask = torch.zeros((bsz, n_bars, enc_len), dtype=torch.bool)
                hier = self._make_hier_indices(enc_inp, dec_len, bsz)

                cp_event_type_inp = torch.randint(0, 2, (dec_len, bsz), dtype=torch.long)
                cp_type_inp = torch.randint(0, vocab_size, (dec_len, bsz), dtype=torch.long)
                cp_struct_inp = torch.randint(0, vocab_size, (dec_len, bsz), dtype=torch.long)
                cp_pos_inp = torch.randint(0, 25, (dec_len, bsz), dtype=torch.long)
                cp_h1_inp = torch.randint(0, 8, (dec_len, bsz), dtype=torch.long)
                cp_h2_inp = torch.randint(0, 8, (dec_len, bsz), dtype=torch.long)
                cp_rf_inp = torch.randint(0, 8, (dec_len, bsz), dtype=torch.long)
                cp_lf_inp = torch.randint(0, 8, (dec_len, bsz), dtype=torch.long)

                mu, logvar, cp_logits = model(
                    enc_inp,
                    dec_inp,
                    bar_pos,
                    cp_event_type_inp=cp_event_type_inp,
                    cp_type_inp=cp_type_inp,
                    cp_struct_inp=cp_struct_inp,
                    cp_pos_inp=cp_pos_inp,
                    cp_hand1_inp=cp_h1_inp,
                    cp_hand2_inp=cp_h2_inp,
                    cp_right_foot_inp=cp_rf_inp,
                    cp_left_foot_inp=cp_lf_inp,
                    padding_mask=padding_mask,
                    **hier,
                )

                losses = model.compute_cp_loss(
                    mu,
                    logvar,
                    beta=0.1,
                    fb_lambda=0.0,
                    cp_logits=cp_logits,
                    cp_targets={
                        "event_type": cp_event_type_inp,
                        "structural": cp_struct_inp,
                        "cp_pos": cp_pos_inp,
                        "cp_hand1": cp_h1_inp,
                        "cp_hand2": cp_h2_inp,
                        "cp_right_foot": cp_rf_inp,
                        "cp_left_foot": cp_lf_inp,
                    },
                )
                losses["total_loss"].backward()
                self.assertTrue(torch.isfinite(losses["total_loss"]))
            else:
                model = self._build_standard_or_remi_model(tokenization_method=mode, use_hierarchical_pe=True)
                vocab_size = 32
                enc_len, dec_len, bsz, n_bars = 4, 8, 2, 2
                enc_inp = torch.randint(0, vocab_size, (enc_len, bsz, n_bars), dtype=torch.long)
                dec_inp = torch.randint(0, vocab_size, (dec_len, bsz), dtype=torch.long)
                dec_tgt = torch.randint(0, vocab_size, (dec_len, bsz), dtype=torch.long)
                bar_pos = self._make_bar_pos(batch_size=bsz, dec_len=dec_len, n_bars=n_bars)
                padding_mask = torch.zeros((bsz, n_bars, enc_len), dtype=torch.bool)
                hier = self._make_hier_indices(enc_inp, dec_len, bsz)

                mu, logvar, dec_logits = model(
                    enc_inp,
                    dec_inp,
                    bar_pos,
                    padding_mask=padding_mask,
                    **hier,
                )
                losses = model.compute_loss(mu, logvar, beta=0.1, fb_lambda=0.0, dec_logits=dec_logits, dec_tgt=dec_tgt)
                losses["total_loss"].backward()
                self.assertTrue(torch.isfinite(losses["total_loss"]))

    # 7.3: use_hierarchical_pe=false で旧挙動回帰
    def test_absolute_pe_path_unchanged_when_hierarchical_disabled(self):
        torch.manual_seed(7)
        model = self._build_standard_or_remi_model(tokenization_method="standard", use_hierarchical_pe=False)
        model.eval()

        enc_len, dec_len, bsz, n_bars = 4, 8, 2, 2
        enc_inp = torch.randint(0, 32, (enc_len, bsz, n_bars), dtype=torch.long)
        dec_inp = torch.randint(0, 32, (dec_len, bsz), dtype=torch.long)
        bar_pos = self._make_bar_pos(batch_size=bsz, dec_len=dec_len, n_bars=n_bars)
        padding_mask = torch.zeros((bsz, n_bars, enc_len), dtype=torch.bool)
        hier = self._make_hier_indices(enc_inp, dec_len, bsz)

        torch.manual_seed(123)
        mu_a, logvar_a, dec_logits_a = model(
            enc_inp,
            dec_inp,
            bar_pos,
            padding_mask=padding_mask,
        )

        torch.manual_seed(123)
        mu_b, logvar_b, dec_logits_b = model(
            enc_inp,
            dec_inp,
            bar_pos,
            padding_mask=padding_mask,
            **hier,
        )

        self.assertTrue(torch.allclose(mu_a, mu_b, atol=1e-6))
        self.assertTrue(torch.allclose(logvar_a, logvar_b, atol=1e-6))
        self.assertTrue(torch.allclose(dec_logits_a, dec_logits_b, atol=1e-6))

    # 7.3: cp_type 未指定時の後方互換
    def test_cp_type_backward_compatibility_fallback_matches_struct_input(self):
        torch.manual_seed(11)
        model = self._build_cp_model(use_hierarchical_pe=False)
        model.eval()

        seq_len, bsz = 8, 2
        dec_inp = torch.randint(0, 16, (seq_len, bsz), dtype=torch.long)
        cp_struct_inp = torch.randint(0, 16, (seq_len, bsz), dtype=torch.long)
        cp_pos_inp = torch.randint(0, 25, (seq_len, bsz), dtype=torch.long)
        cp_h1_inp = torch.randint(0, 8, (seq_len, bsz), dtype=torch.long)
        cp_h2_inp = torch.randint(0, 8, (seq_len, bsz), dtype=torch.long)
        cp_rf_inp = torch.randint(0, 8, (seq_len, bsz), dtype=torch.long)
        cp_lf_inp = torch.randint(0, 8, (seq_len, bsz), dtype=torch.long)

        emb_legacy = model._build_decoder_input_emb(
            dec_inp,
            cp_type_inp=None,
            cp_struct_inp=cp_struct_inp,
            cp_pos_inp=cp_pos_inp,
            cp_hand1_inp=cp_h1_inp,
            cp_hand2_inp=cp_h2_inp,
            cp_right_foot_inp=cp_rf_inp,
            cp_left_foot_inp=cp_lf_inp,
        )
        emb_explicit = model._build_decoder_input_emb(
            dec_inp,
            cp_type_inp=cp_struct_inp,
            cp_struct_inp=cp_struct_inp,
            cp_pos_inp=cp_pos_inp,
            cp_hand1_inp=cp_h1_inp,
            cp_hand2_inp=cp_h2_inp,
            cp_right_foot_inp=cp_rf_inp,
            cp_left_foot_inp=cp_lf_inp,
        )

        self.assertTrue(torch.allclose(emb_legacy, emb_explicit, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
