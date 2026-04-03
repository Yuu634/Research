import torch
from torch import nn
import torch.nn.functional as F

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None, anchor_positions=None):
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
