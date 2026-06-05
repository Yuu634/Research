import torch
from torch import nn
import torch.nn.functional as F

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu',
               use_limb_bias: bool = False, alpha_limb_fixed=None):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation
    self.use_limb_bias = use_limb_bias

    if use_limb_bias:
      self.encoder_layers = nn.ModuleList([
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        for _ in range(n_layer)
      ])
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
    else:
      self.tr_encoder_layer = nn.TransformerEncoderLayer(
        d_model, n_head, d_ff, dropout, activation
      )
      self.tr_encoder = nn.TransformerEncoder(
        self.tr_encoder_layer, n_layer
      )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None, anchor_positions=None, limb_bias=None):
    if self.use_limb_bias and limb_bias is not None:
      out = x
      for i in range(self.n_layer):
        alpha = self.alpha_limb_fixed if self.alpha_limb_fixed is not None else self.alpha_limb[i]
        effective_mask = alpha * limb_bias   # scalar * [B*n_heads, N, N]
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
