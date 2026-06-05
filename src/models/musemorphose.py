import torch
from torch import nn
import torch.nn.functional as F
from src.models.transformer_encoder import VAETransformerEncoder
from src.models.transformer_helpers import (
  weights_init, PositionalEncoding, HierarchicalDrumPositionalEncoding,
  TokenEmbedding, CpCompositeEmbedding, generate_causal_mask
)
from src.models.limb_bias import (
  build_limb_ids, build_limb_bias_mask, expand_for_mha, make_limb_lookup_table
)

class VAETransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu',
               cond_mode='in-attn', use_limb_bias: bool = False, alpha_limb_fixed=None):
    super(VAETransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dropout = dropout
    self.activation = activation
    self.cond_mode = cond_mode
    self.use_limb_bias = use_limb_bias

    if cond_mode == 'in-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    elif cond_mode == 'pre-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

    if use_limb_bias:
      if alpha_limb_fixed is None:
        # 可学習スカラー α（初期値 0.0: 学習開始時はバイアス無効）
        self.alpha_limb = nn.ParameterList([
          nn.Parameter(torch.zeros(1)) for _ in range(n_layer)
        ])
        self.register_buffer('alpha_limb_fixed', None)
      else:
        # 固定スカラー α（勾配なし、全レイヤー共通）
        self.alpha_limb = None
        self.register_buffer('alpha_limb_fixed', torch.tensor(float(alpha_limb_fixed)))

  def forward(self, x, seg_emb, limb_bias=None):
    if not hasattr(self, 'cond_mode'):
      self.cond_mode = 'in-attn'
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)   # [N, N]

    if self.cond_mode == 'in-attn':
      seg_emb = self.seg_emb_proj(seg_emb)
    elif self.cond_mode == 'pre-attn':
      x = torch.cat([x, seg_emb], dim=-1)
      x = self.seg_emb_proj(x)

    out = x
    for i in range(self.n_layer):
      if self.cond_mode == 'in-attn':
        out += seg_emb

      if self.use_limb_bias and limb_bias is not None:
        alpha = self.alpha_limb_fixed if self.alpha_limb_fixed is not None else self.alpha_limb[i]
        # causal_mask [N,N] + α * M_limb [B*n_heads, N, N] → [B*n_heads, N, N]
        effective_mask = attn_mask.unsqueeze(0) + alpha * limb_bias
      else:
        effective_mask = attn_mask

      out = self.decoder_layers[i](out, src_mask=effective_mask)

    return out

class MuseMorphose(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    # ===== 新規: 5つの難易度埋め込みパラメータ =====
    d_s_tech_emb=32, d_s_indep_emb=32, d_s_hand_emb=32, d_s_foot_emb=32, d_s_move_emb=32,
    n_s_tech_cls=8, n_s_indep_cls=8, n_s_hand_cls=8, n_s_foot_cls=8, n_s_move_cls=8,
    use_difficulty=False,  # 難易度属性を使用するかどうか
    tokenization_method='standard',
    cp_event_type_vocab_size=3,
    cp_type_vocab_size=None,
    cp_struct_vocab_size=None,
    cp_pos_vocab_size=25,
    cp_limb_vocab_size=2,
    cp_event_pad_idx=2,
    cp_struct_pad_idx=0,
    cp_pos_pad_idx=24,
    cp_limb_pad_idx=0,
    bar_anchor_mode='first_token',
    bar_token_idx=None,
    d_cp_type_emb=64,
    d_cp_pos_emb=64,
    d_cp_limb_emb=64,
    # ==============================================
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn',
    use_hierarchical_pe=False,
    hier_pe_bar_vocab_size=512,
    hier_pe_beat_vocab_size=8,
    hier_pe_pos_vocab_size=32,
    hier_pe_use_struct_bias=True,
    # ===== Limb-Relation Bias =====
    use_limb_bias_encoder: bool = False,
    use_limb_bias_decoder: bool = False,
    token_to_limb=None,
    alpha_limb_fixed=None,
  ):
    super(MuseMorphose, self).__init__()
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation  

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training
    self.use_difficulty = use_difficulty  # ===== 追加 =====
    self.tokenization_method = tokenization_method

    self.cond_mode = cond_mode
    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.use_hierarchical_pe = use_hierarchical_pe
    if use_hierarchical_pe:
      self.hier_pe = HierarchicalDrumPositionalEncoding(
        d_embed=d_embed,
        bar_vocab_size=hier_pe_bar_vocab_size,
        beat_vocab_size=hier_pe_beat_vocab_size,
        pos_vocab_size=hier_pe_pos_vocab_size,
        use_struct_bias=hier_pe_use_struct_bias,
      )
    else:
      self.hier_pe = None
    self.dec_out_proj = nn.Linear(dec_d_model, n_token)

    # ===== Limb-Relation Bias セットアップ =====
    self.use_limb_bias_encoder = use_limb_bias_encoder
    self.use_limb_bias_decoder = use_limb_bias_decoder
    if (use_limb_bias_encoder or use_limb_bias_decoder) and token_to_limb is not None:
      table = make_limb_lookup_table(token_to_limb, n_token)
      self.register_buffer('token_to_limb_table', table)  # [vocab_size], long

    self.encoder = VAETransformerEncoder(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation,
      use_limb_bias=use_limb_bias_encoder,
      alpha_limb_fixed=alpha_limb_fixed,
    )

    self.use_attr_cls = use_attr_cls
    if self.use_difficulty and not self.use_attr_cls:
      raise ValueError('use_difficulty=True requires use_attr_cls=True for conditional embedding composition.')

    self.cp_event_type_vocab_size = cp_event_type_vocab_size
    self.cp_struct_vocab_size = cp_struct_vocab_size if cp_struct_vocab_size is not None else n_token
    self.cp_type_vocab_size = cp_type_vocab_size if cp_type_vocab_size is not None else self.cp_struct_vocab_size
    self.cp_pos_vocab_size = cp_pos_vocab_size
    self.cp_limb_vocab_size = cp_limb_vocab_size
    self.cp_event_pad_idx = cp_event_pad_idx
    self.cp_struct_pad_idx = cp_struct_pad_idx
    self.cp_pos_pad_idx = cp_pos_pad_idx
    self.cp_limb_pad_idx = cp_limb_pad_idx

    if bar_anchor_mode not in ('first_token', 'bar_token'):
      raise ValueError(
        f'Invalid bar_anchor_mode: {bar_anchor_mode}. Supported: first_token, bar_token'
      )
    self.bar_anchor_mode = bar_anchor_mode
    self.bar_token_idx = bar_token_idx

    # ===== 難易度属性を使用する場合 =====
    if use_attr_cls and use_difficulty:
      # 5つの難易度埋め込みの合計サイズ
      total_attr_emb_size = d_s_tech_emb + d_s_indep_emb + d_s_hand_emb + d_s_foot_emb + d_s_move_emb

      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + total_attr_emb_size,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode, use_limb_bias=use_limb_bias_decoder,
        alpha_limb_fixed=alpha_limb_fixed,
      )

      # 5つの属性埋め込み層
      self.d_s_tech_emb = d_s_tech_emb
      self.d_s_indep_emb = d_s_indep_emb
      self.d_s_hand_emb = d_s_hand_emb
      self.d_s_foot_emb = d_s_foot_emb
      self.d_s_move_emb = d_s_move_emb

      self.s_tech_attr_emb = TokenEmbedding(n_s_tech_cls, d_s_tech_emb, d_s_tech_emb)
      self.s_indep_attr_emb = TokenEmbedding(n_s_indep_cls, d_s_indep_emb, d_s_indep_emb)
      self.s_hand_attr_emb = TokenEmbedding(n_s_hand_cls, d_s_hand_emb, d_s_hand_emb)
      self.s_foot_attr_emb = TokenEmbedding(n_s_foot_cls, d_s_foot_emb, d_s_foot_emb)
      self.s_move_attr_emb = TokenEmbedding(n_s_move_cls, d_s_move_emb, d_s_move_emb)

      # 従来の属性は使用しない
      self.rfreq_attr_emb = None
      self.polyph_attr_emb = None

    # ===== 従来の属性を使用する場合 =====
    elif use_attr_cls:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_polyph_emb + d_rfreq_emb,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode, use_limb_bias=use_limb_bias_decoder,
        alpha_limb_fixed=alpha_limb_fixed,
      )

      self.d_rfreq_emb = d_rfreq_emb
      self.d_polyph_emb = d_polyph_emb
      self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
      self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)

      # 難易度属性は使用しない
      self.s_tech_attr_emb = None
      self.s_indep_attr_emb = None
      self.s_hand_attr_emb = None
      self.s_foot_attr_emb = None
      self.s_move_attr_emb = None

    # ===== 属性を使用しない場合 =====
    else:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode, use_limb_bias=use_limb_bias_decoder,
        alpha_limb_fixed=alpha_limb_fixed,
      )

      self.rfreq_attr_emb = None
      self.polyph_attr_emb = None
      self.s_tech_attr_emb = None
      self.s_indep_attr_emb = None
      self.s_hand_attr_emb = None
      self.s_foot_attr_emb = None
      self.s_move_attr_emb = None

    self.emb_dropout = nn.Dropout(self.enc_dropout)

    # ===== CP方式の埋め込み/出力ヘッド =====
    if self.tokenization_method == 'cp_limb_v1':
      self.cp_composite_emb = CpCompositeEmbedding(
        n_type=self.cp_type_vocab_size,
        n_pos=self.cp_pos_vocab_size,
        n_limb_token=self.cp_limb_vocab_size,
        d_type=d_cp_type_emb,
        d_pos=d_cp_pos_emb,
        d_limb=d_cp_limb_emb,
        d_proj=enc_d_model
      )
      self.cp_event_type_head = nn.Linear(dec_d_model, self.cp_event_type_vocab_size)
      self.cp_struct_head = nn.Linear(dec_d_model, self.cp_struct_vocab_size)
      self.cp_pos_head = nn.Linear(dec_d_model, self.cp_pos_vocab_size)
      self.cp_hand1_head = nn.Linear(dec_d_model, self.cp_limb_vocab_size)
      self.cp_hand2_head = nn.Linear(dec_d_model, self.cp_limb_vocab_size)
      self.cp_right_foot_head = nn.Linear(dec_d_model, self.cp_limb_vocab_size)
      self.cp_left_foot_head = nn.Linear(dec_d_model, self.cp_limb_vocab_size)
    else:
      self.cp_composite_emb = None
      self.cp_event_type_head = None
      self.cp_struct_head = None
      self.cp_pos_head = None
      self.cp_hand1_head = None
      self.cp_hand2_head = None
      self.cp_right_foot_head = None
      self.cp_left_foot_head = None

    self.apply(weights_init)
    

  def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
    std = torch.exp(0.5 * logvar).to(mu.device)
    if use_sampling:
      eps = torch.randn_like(std).to(mu.device) * sampling_var
    else:
      eps = torch.zeros_like(std).to(mu.device)

    return eps * std + mu

  def _resolve_encoder_anchor_positions(self, enc_tokens_flat):
    if self.bar_anchor_mode != 'bar_token' or self.bar_token_idx is None:
      return None

    bar_mask = enc_tokens_flat.eq(self.bar_token_idx)
    match_counts = bar_mask.sum(dim=0)
    has_unique_bar = match_counts.eq(1)

    first_hit = torch.argmax(bar_mask.long(), dim=0)
    anchor_positions = torch.zeros(enc_tokens_flat.size(1), dtype=torch.long, device=enc_tokens_flat.device)
    anchor_positions[has_unique_bar] = first_hit[has_unique_bar]

    return anchor_positions

  def _flatten_hier_index_tensor(self, index_tensor):
    if index_tensor is None:
      return None
    if index_tensor.dim() == 3:
      return index_tensor.reshape(index_tensor.size(0), -1)
    return index_tensor

  def _build_position_encoding(self, seq_len, bar_idx=None, beat_idx=None, pos_idx=None, is_struct=None):
    if (
      self.use_hierarchical_pe and self.hier_pe is not None and
      bar_idx is not None and beat_idx is not None and pos_idx is not None
    ):
      return self.hier_pe(bar_idx, beat_idx, pos_idx, is_struct=is_struct)

    return self.pe(seq_len)

  def get_sampled_latent(
    self,
    inp,
    padding_mask=None,
    use_sampling=False,
    sampling_var=0.,
    enc_bar_idx=None,
    enc_beat_idx=None,
    enc_pos_idx=None,
    enc_is_struct=None,
  ):
    if inp.dim() == 3:
      enc_tokens_flat = inp.reshape(inp.size(0), -1)
      token_emb = self.token_emb(inp).reshape(inp.size(0), -1, self.enc_d_model)
      if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
    else:
      enc_tokens_flat = inp
      token_emb = self.token_emb(inp)

    enc_bar_idx = self._flatten_hier_index_tensor(enc_bar_idx)
    enc_beat_idx = self._flatten_hier_index_tensor(enc_beat_idx)
    enc_pos_idx = self._flatten_hier_index_tensor(enc_pos_idx)
    enc_is_struct = self._flatten_hier_index_tensor(enc_is_struct)

    enc_pe = self._build_position_encoding(
      inp.size(0),
      bar_idx=enc_bar_idx,
      beat_idx=enc_beat_idx,
      pos_idx=enc_pos_idx,
      is_struct=enc_is_struct,
    )
    enc_inp = self.emb_dropout(token_emb) + enc_pe
    enc_anchor_positions = self._resolve_encoder_anchor_positions(enc_tokens_flat)

    _, mu, logvar = self.encoder(
      enc_inp,
      padding_mask=padding_mask,
      anchor_positions=enc_anchor_positions,
    )
    mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
    vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

    return vae_latent

  def _build_decoder_input_emb(
    self,
    dec_inp,
    cp_type_inp=None,
    cp_struct_inp=None,
    cp_pos_inp=None,
    cp_hand1_inp=None,
    cp_hand2_inp=None,
    cp_right_foot_inp=None,
    cp_left_foot_inp=None,
  ):
    if self.tokenization_method != 'cp_limb_v1':
      return self.token_emb(dec_inp)

    # Backward compatibility fallback: if cp_type is absent, use legacy structural input.
    if cp_type_inp is None:
      if cp_struct_inp is not None:
        cp_type_inp = cp_struct_inp
      else:
        cp_type_inp = dec_inp

    if cp_pos_inp is None:
      cp_pos_inp = torch.full_like(cp_type_inp, self.cp_pos_pad_idx)
    if cp_hand1_inp is None:
      cp_hand1_inp = torch.full_like(cp_type_inp, self.cp_limb_pad_idx)
    if cp_hand2_inp is None:
      cp_hand2_inp = torch.full_like(cp_type_inp, self.cp_limb_pad_idx)
    if cp_right_foot_inp is None:
      cp_right_foot_inp = torch.full_like(cp_type_inp, self.cp_limb_pad_idx)
    if cp_left_foot_inp is None:
      cp_left_foot_inp = torch.full_like(cp_type_inp, self.cp_limb_pad_idx)

    return self.cp_composite_emb(
      cp_type_inp,
      cp_pos_inp,
      cp_hand1_inp,
      cp_hand2_inp,
      cp_right_foot_inp,
      cp_left_foot_inp,
    )

  def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None,
               s_tech_cls=None, s_indep_cls=None, s_hand_cls=None, s_foot_cls=None, s_move_cls=None,
               cp_event_type_inp=None, cp_type_inp=None, cp_struct_inp=None,
               cp_pos_inp=None, cp_hand1_inp=None, cp_hand2_inp=None,
               cp_right_foot_inp=None, cp_left_foot_inp=None,
               dec_bar_idx=None, dec_beat_idx=None, dec_pos_idx=None, dec_is_struct=None,
               keep_last_only=True):
    dec_seg_emb_cat = self._build_conditional_seg_emb(
      dec_seg_emb,
      rfreq_cls=rfreq_cls,
      polyph_cls=polyph_cls,
      s_tech_cls=s_tech_cls,
      s_indep_cls=s_indep_cls,
      s_hand_cls=s_hand_cls,
      s_foot_cls=s_foot_cls,
      s_move_cls=s_move_cls
    )

    if self.tokenization_method == 'cp_limb_v1':
      dec_token_emb = self._build_decoder_input_emb(
        inp,
        cp_type_inp=cp_type_inp,
        cp_struct_inp=cp_struct_inp,
        cp_pos_inp=cp_pos_inp,
        cp_hand1_inp=cp_hand1_inp,
        cp_hand2_inp=cp_hand2_inp,
        cp_right_foot_inp=cp_right_foot_inp,
        cp_left_foot_inp=cp_left_foot_inp,
      )
      dec_pe = self._build_position_encoding(
        inp.size(0),
        bar_idx=dec_bar_idx,
        beat_idx=dec_beat_idx,
        pos_idx=dec_pos_idx,
        is_struct=dec_is_struct,
      )
      dec_inp = self.emb_dropout(dec_token_emb) + dec_pe

      dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
      cp_logits = {
        'event_type': self.cp_event_type_head(dec_out),
        'structural': self.cp_struct_head(dec_out),
        'cp_pos': self.cp_pos_head(dec_out),
        'cp_hand1': self.cp_hand1_head(dec_out),
        'cp_hand2': self.cp_hand2_head(dec_out),
        'cp_right_foot': self.cp_right_foot_head(dec_out),
        'cp_left_foot': self.cp_left_foot_head(dec_out),
      }

      if keep_last_only:
        cp_logits = {k: v[-1, ...] for k, v in cp_logits.items()}
      return cp_logits

    token_emb = self.token_emb(inp)
    dec_pe = self._build_position_encoding(
      inp.size(0),
      bar_idx=dec_bar_idx,
      beat_idx=dec_beat_idx,
      pos_idx=dec_pos_idx,
      is_struct=dec_is_struct,
    )
    dec_inp = self.emb_dropout(token_emb) + dec_pe
    out = self.decoder(dec_inp, dec_seg_emb_cat)
    out = self.dec_out_proj(out)

    if keep_last_only:
      out = out[-1, ...]

    return out


  def _build_dec_seg_emb(self, dec_inp, dec_inp_bar_pos, vae_latent):
    enc_bt_size = dec_inp_bar_pos.size(0)
    enc_n_bars = dec_inp_bar_pos.size(1) - 1
    vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
    for n in range(dec_inp.size(1)):
      for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]
    return dec_seg_emb

  def _build_conditional_seg_emb(
    self,
    dec_seg_emb,
    rfreq_cls=None,
    polyph_cls=None,
    s_tech_cls=None,
    s_indep_cls=None,
    s_hand_cls=None,
    s_foot_cls=None,
    s_move_cls=None
  ):
    if self.use_difficulty:
      missing_inputs = []
      if s_tech_cls is None:
        missing_inputs.append('s_tech_cls')
      if s_indep_cls is None:
        missing_inputs.append('s_indep_cls')
      if s_hand_cls is None:
        missing_inputs.append('s_hand_cls')
      if s_foot_cls is None:
        missing_inputs.append('s_foot_cls')
      if s_move_cls is None:
        missing_inputs.append('s_move_cls')

      if missing_inputs:
        raise ValueError(
          'Difficulty conditioning is enabled but required class sequences are missing: '
          f'{missing_inputs}'
        )

      s_tech_emb = self.s_tech_attr_emb(s_tech_cls)
      s_indep_emb = self.s_indep_attr_emb(s_indep_cls)
      s_hand_emb = self.s_hand_attr_emb(s_hand_cls)
      s_foot_emb = self.s_foot_attr_emb(s_foot_cls)
      s_move_emb = self.s_move_attr_emb(s_move_cls)

      return torch.cat([
        dec_seg_emb, s_tech_emb, s_indep_emb, s_hand_emb, s_foot_emb, s_move_emb
      ], dim=-1)

    if rfreq_cls is not None and polyph_cls is not None and self.use_attr_cls:
      dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
      dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
      return torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)

    return dec_seg_emb

  def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None,
              s_tech_cls=None, s_indep_cls=None, s_hand_cls=None, s_foot_cls=None, s_move_cls=None,
              cp_event_type_inp=None, cp_type_inp=None, cp_struct_inp=None,
              cp_pos_inp=None, cp_hand1_inp=None, cp_hand2_inp=None,
              cp_right_foot_inp=None, cp_left_foot_inp=None,
              enc_bar_idx=None, enc_beat_idx=None, enc_pos_idx=None, enc_is_struct=None,
              dec_bar_idx=None, dec_beat_idx=None, dec_pos_idx=None, dec_is_struct=None,
              padding_mask=None):
    # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
    enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
    enc_tokens_flat = enc_inp.reshape(enc_inp.size(0), -1)
    enc_token_emb = self.token_emb(enc_inp)

    enc_token_emb = enc_token_emb.reshape(
      enc_inp.size(0), -1, enc_token_emb.size(-1)
    )
    enc_bar_idx = self._flatten_hier_index_tensor(enc_bar_idx)
    enc_beat_idx = self._flatten_hier_index_tensor(enc_beat_idx)
    enc_pos_idx = self._flatten_hier_index_tensor(enc_pos_idx)
    enc_is_struct = self._flatten_hier_index_tensor(enc_is_struct)

    enc_pe = self._build_position_encoding(
      enc_inp.size(0),
      bar_idx=enc_bar_idx,
      beat_idx=enc_beat_idx,
      pos_idx=enc_pos_idx,
      is_struct=enc_is_struct,
    )
    enc_inp = self.emb_dropout(enc_token_emb) + enc_pe

    dec_token_emb = self._build_decoder_input_emb(
      dec_inp,
      cp_type_inp=cp_type_inp,
      cp_struct_inp=cp_struct_inp,
      cp_pos_inp=cp_pos_inp,
      cp_hand1_inp=cp_hand1_inp,
      cp_hand2_inp=cp_hand2_inp,
      cp_right_foot_inp=cp_right_foot_inp,
      cp_left_foot_inp=cp_left_foot_inp,
    )
    dec_pe = self._build_position_encoding(
      dec_inp.size(0),
      bar_idx=dec_bar_idx,
      beat_idx=dec_beat_idx,
      pos_idx=dec_pos_idx,
      is_struct=dec_is_struct,
    )
    dec_inp_emb = self.emb_dropout(dec_token_emb) + dec_pe

    # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
    # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

    # ─── Limb Bias マスクの1回生成（各ループ前にキャッシュ）─────────────────
    enc_limb_bias_mha = None
    dec_limb_bias_mha = None
    if self.use_limb_bias_encoder and hasattr(self, 'token_to_limb_table'):
      # enc_tokens_flat: [enc_seqlen, B*n_bars]
      enc_limb_ids = build_limb_ids(enc_tokens_flat, self.token_to_limb_table)
      enc_limb_bias = build_limb_bias_mask(enc_limb_ids)
      enc_limb_bias_mha = expand_for_mha(enc_limb_bias, self.enc_n_head)
      # shape: [B*n_bars*enc_n_head, enc_seqlen, enc_seqlen]
    if self.use_limb_bias_decoder and hasattr(self, 'token_to_limb_table'):
      # dec_inp: [dec_seqlen, B]
      dec_limb_ids = build_limb_ids(dec_inp, self.token_to_limb_table)
      dec_limb_bias = build_limb_bias_mask(dec_limb_ids)
      dec_limb_bias_mha = expand_for_mha(dec_limb_bias, self.dec_n_head)
      # shape: [B*dec_n_head, dec_seqlen, dec_seqlen]
    # ─────────────────────────────────────────────────────────────────────────

    enc_anchor_positions = self._resolve_encoder_anchor_positions(enc_tokens_flat)
    _, mu, logvar = self.encoder(
      enc_inp,
      padding_mask=padding_mask,
      anchor_positions=enc_anchor_positions,
      limb_bias=enc_limb_bias_mha,
    )
    vae_latent = self.reparameterize(mu, logvar)
    dec_seg_emb = self._build_dec_seg_emb(dec_inp_emb, dec_inp_bar_pos, vae_latent)
    dec_seg_emb_cat = self._build_conditional_seg_emb(
      dec_seg_emb,
      rfreq_cls=rfreq_cls,
      polyph_cls=polyph_cls,
      s_tech_cls=s_tech_cls,
      s_indep_cls=s_indep_cls,
      s_hand_cls=s_hand_cls,
      s_foot_cls=s_foot_cls,
      s_move_cls=s_move_cls
    )

    if self.tokenization_method == 'cp_limb_v1':
      dec_out = self.decoder(dec_inp_emb, dec_seg_emb_cat, limb_bias=dec_limb_bias_mha)
      cp_logits = {
        'event_type': self.cp_event_type_head(dec_out),
        'structural': self.cp_struct_head(dec_out),
        'cp_pos': self.cp_pos_head(dec_out),
        'cp_hand1': self.cp_hand1_head(dec_out),
        'cp_hand2': self.cp_hand2_head(dec_out),
        'cp_right_foot': self.cp_right_foot_head(dec_out),
        'cp_left_foot': self.cp_left_foot_head(dec_out),
      }
      return mu, logvar, cp_logits

    dec_out = self.decoder(dec_inp_emb, dec_seg_emb_cat, limb_bias=dec_limb_bias_mha)
    dec_logits = self.dec_out_proj(dec_out)

    return mu, logvar, dec_logits

  def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
    if dec_logits.dim() != 3:
      raise ValueError(
        'Loss shape mismatch: dec_logits must be 3D [seq_len, batch, vocab]. '
        f'Got shape={tuple(dec_logits.shape)}'
      )
    if dec_tgt.dim() != 2:
      raise ValueError(
        'Loss shape mismatch: dec_tgt must be 2D [seq_len, batch]. '
        f'Got shape={tuple(dec_tgt.shape)}'
      )
    if dec_logits.size(0) != dec_tgt.size(0) or dec_logits.size(1) != dec_tgt.size(1):
      raise ValueError(
        'Loss shape mismatch between logits and target: '
        f'logits={tuple(dec_logits.shape)}, target={tuple(dec_tgt.shape)}'
      )

    recons_loss = F.cross_entropy(
      dec_logits.reshape(-1, dec_logits.size(-1)), dec_tgt.contiguous().reshape(-1), 
      ignore_index=self.n_token - 1, reduction='mean'
    ).float()

    kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
      'beta': beta,
      'total_loss': recons_loss + beta * kldiv_loss,
      'kldiv_loss': kldiv_loss,
      'kldiv_raw': kl_before_free_bits,
      'recons_loss': recons_loss
    }

  def _masked_ce(self, logits, targets, mask):
    if mask is None:
      return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.contiguous().reshape(-1), reduction='mean')

    if mask.dtype != torch.bool:
      mask = mask.bool()

    if mask.dim() > 1:
      mask_flat = mask.reshape(-1)
    else:
      mask_flat = mask

    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.contiguous().reshape(-1)

    if mask_flat.sum() == 0:
      return logits_flat.new_tensor(0.0)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]
    return F.cross_entropy(logits_sel, targets_sel, reduction='mean')

  def compute_cp_loss(self, mu, logvar, beta, fb_lambda, cp_logits, cp_targets, loss_weights=None):
    if loss_weights is None:
      loss_weights = {}

    required_keys = (
      'event_type',
      'structural',
      'cp_pos',
      'cp_hand1',
      'cp_hand2',
      'cp_right_foot',
      'cp_left_foot',
    )
    missing_logits = [key for key in required_keys if key not in cp_logits]
    if missing_logits:
      raise ValueError(
        'CP loss input mismatch: missing cp_logits keys '
        f'{missing_logits}. Required={required_keys}'
      )
    missing_targets = [key for key in required_keys if key not in cp_targets]
    if missing_targets:
      raise ValueError(
        'CP loss input mismatch: missing cp_targets keys '
        f'{missing_targets}. Required={required_keys}'
      )

    for key in required_keys:
      logits = cp_logits[key]
      targets = cp_targets[key]
      if logits.dim() != 3:
        raise ValueError(
          f'CP loss shape mismatch: cp_logits[{key}] must be 3D [seq_len, batch, vocab], '
          f'got shape={tuple(logits.shape)}'
        )
      if targets.dim() != 2:
        raise ValueError(
          f'CP loss shape mismatch: cp_targets[{key}] must be 2D [seq_len, batch], '
          f'got shape={tuple(targets.shape)}'
        )
      if logits.size(0) != targets.size(0) or logits.size(1) != targets.size(1):
        raise ValueError(
          f'CP loss shape mismatch for head={key}: '
          f'logits={tuple(logits.shape)}, target={tuple(targets.shape)}'
        )

    event_tgt = cp_targets['event_type']
    struct_tgt = cp_targets['structural']
    pos_tgt = cp_targets['cp_pos']
    h1_tgt = cp_targets['cp_hand1']
    h2_tgt = cp_targets['cp_hand2']
    rf_tgt = cp_targets['cp_right_foot']
    lf_tgt = cp_targets['cp_left_foot']

    event_valid = event_tgt != self.cp_event_pad_idx
    struct_mask = (event_tgt == 0) & event_valid
    cp_mask = (event_tgt == 1) & event_valid

    loss_event = self._masked_ce(cp_logits['event_type'], event_tgt, event_valid)
    loss_struct = self._masked_ce(cp_logits['structural'], struct_tgt, struct_mask)
    loss_pos = self._masked_ce(cp_logits['cp_pos'], pos_tgt, cp_mask)
    loss_h1 = self._masked_ce(cp_logits['cp_hand1'], h1_tgt, cp_mask)
    loss_h2 = self._masked_ce(cp_logits['cp_hand2'], h2_tgt, cp_mask)
    loss_rf = self._masked_ce(cp_logits['cp_right_foot'], rf_tgt, cp_mask)
    loss_lf = self._masked_ce(cp_logits['cp_left_foot'], lf_tgt, cp_mask)

    # Structural head is kept for observability/backward-compat metrics,
    # but excluded from optimization to keep 6-head CP loss semantics.
    recons_loss = (
      loss_weights.get('event_type', 1.0) * loss_event +
      loss_weights.get('cp_pos', 1.0) * loss_pos +
      loss_weights.get('cp_hand1', 1.0) * loss_h1 +
      loss_weights.get('cp_hand2', 1.0) * loss_h2 +
      loss_weights.get('cp_right_foot', 1.0) * loss_rf +
      loss_weights.get('cp_left_foot', 1.0) * loss_lf
    )

    kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
      'beta': beta,
      'total_loss': recons_loss + beta * kldiv_loss,
      'kldiv_loss': kldiv_loss,
      'kldiv_raw': kl_before_free_bits,
      'recons_loss': recons_loss,
      'cp_event_type_loss': loss_event,
      'cp_structural_loss': loss_struct,
      'cp_pos_loss': loss_pos,
      'cp_hand1_loss': loss_h1,
      'cp_hand2_loss': loss_h2,
      'cp_right_foot_loss': loss_rf,
      'cp_left_foot_loss': loss_lf,
    }