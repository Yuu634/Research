import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu',
               use_limb_bias: bool = False, alpha_limb_fixed=None,
               # ===== Position-Same-Time Bias =====
               use_position_bias: bool = False,
               position_bias_mode: str = 'uniform',
               alpha_pos_fixed: Optional[float] = None,
               position_bias_per_layer: bool = True,
               # ===== Per-layer Bias Schedule =====
               layer_bias_schedule: Optional[List[str]] = None):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation
    self.use_limb_bias = use_limb_bias
    self.use_position_bias = use_position_bias
    self.layer_bias_schedule = layer_bias_schedule

    # いずれかのバイアスが有効な場合はレイヤーごとのループを使う
    self._use_per_layer_loop = use_limb_bias or use_position_bias

    if self._use_per_layer_loop:
      self.encoder_layers = nn.ModuleList([
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        for _ in range(n_layer)
      ])
    else:
      self.tr_encoder_layer = nn.TransformerEncoderLayer(
        d_model, n_head, d_ff, dropout, activation
      )
      self.tr_encoder = nn.TransformerEncoder(
        self.tr_encoder_layer, n_layer
      )

    # Limb Bias パラメータ
    if use_limb_bias:
      if alpha_limb_fixed is None:
        self.alpha_limb = nn.ParameterList([
          nn.Parameter(torch.zeros(1)) for _ in range(n_layer)
        ])
        self.register_buffer('alpha_limb_fixed', None)
      else:
        self.alpha_limb = None
        self.register_buffer('alpha_limb_fixed', torch.tensor(float(alpha_limb_fixed)))

    # Position Bias パラメータ
    if use_position_bias:
      from src.models.position_bias import PositionBiasModule
      self.position_bias_module = PositionBiasModule(
        mode=position_bias_mode,
        n_layers=n_layer,
        alpha_fixed=alpha_pos_fixed,
        per_layer=position_bias_per_layer,
      )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None, anchor_positions=None,
              limb_bias=None, position_bias=None, limb_ids_for_pos_bias=None):
    if self._use_per_layer_loop:
      from src.models.limb_bias import expand_for_mha
      out = x
      for i in range(self.n_layer):
        effective_mask = None

        # スケジュールが指定されていれば層ごとにバイアス適用可否を決定
        if self.layer_bias_schedule is not None:
          entry = self.layer_bias_schedule[i]
          apply_limb = (entry in ('limb', 'both')) and self.use_limb_bias
          apply_pos  = (entry in ('position', 'both')) and self.use_position_bias
        else:
          apply_limb = self.use_limb_bias
          apply_pos  = self.use_position_bias

        if apply_limb and limb_bias is not None:
          alpha = (self.alpha_limb_fixed
                   if self.alpha_limb_fixed is not None
                   else self.alpha_limb[i])
          effective_mask = alpha * limb_bias   # [B*n_heads, N, N]

        if apply_pos and position_bias is not None:
          pos_contrib = self.position_bias_module.get_bias_for_layer(
            i, position_bias, limb_ids_for_pos_bias
          )                                    # [B, N, N]
          pos_contrib_mha = expand_for_mha(pos_contrib, self.n_head)  # [B*n_heads, N, N]
          effective_mask = (pos_contrib_mha if effective_mask is None
                            else effective_mask + pos_contrib_mha)

        out = self.encoder_layers[i](
          out,
          src_mask=effective_mask,
          src_key_padding_mask=padding_mask,
        )
    else:
      out = self.tr_encoder(x, src_key_padding_mask=padding_mask)

    if anchor_positions is None:
      hidden_out = out[0, :, :]
    else:
      anchor_positions = anchor_positions.to(out.device)
      if anchor_positions.dim() != 1 or anchor_positions.size(0) != out.size(1):
        raise ValueError(
          f'anchor_positions must have shape ({out.size(1)},), got {tuple(anchor_positions.size())}'
        )

      anchor_positions = anchor_positions.clamp(min=0, max=out.size(0) - 1)
      batch_idx = torch.arange(out.size(1), device=out.device)
      hidden_out = out[anchor_positions, batch_idx, :]

    mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)

    return hidden_out, mu, logvar
