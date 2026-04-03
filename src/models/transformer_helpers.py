import math
import torch

from torch import nn
import torch.nn.functional as F

def generate_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  
def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('GRU') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:  # weights
                weight_init_orthogonal(param, 0.01)
            else:                      # biases
                bias_init(param)
    # else:
    #   print ('[{}] not initialized !!'.format(classname))

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_pos=20480):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len, bsz=None):
        pos_encoding = self.pe[:seq_len, :]

        if bsz is not None:
          pos_encoding = pos_encoding.expand(seq_len, bsz, -1)

        return pos_encoding

class HierarchicalDrumPositionalEncoding(nn.Module):
    def __init__(
      self,
      d_embed,
      bar_vocab_size=512,
      beat_vocab_size=8,
      pos_vocab_size=32,
      use_struct_bias=True,
    ):
      super(HierarchicalDrumPositionalEncoding, self).__init__()
      self.d_embed = d_embed
      self.bar_vocab_size = bar_vocab_size
      self.beat_vocab_size = beat_vocab_size
      self.pos_vocab_size = pos_vocab_size
      self.use_struct_bias = use_struct_bias

      self.bar_emb = nn.Embedding(bar_vocab_size, d_embed)
      self.beat_emb = nn.Embedding(beat_vocab_size, d_embed)
      self.pos_emb = nn.Embedding(pos_vocab_size, d_embed)

      if use_struct_bias:
        # 0: non-structural token, 1: structural token
        self.struct_emb = nn.Embedding(2, d_embed)
      else:
        self.struct_emb = None

    def _sanitize_index(self, idx, vocab_size):
      return idx.long().clamp(min=0, max=vocab_size - 1)

    def forward(self, bar_idx, beat_idx, pos_idx, is_struct=None):
      if not (bar_idx.shape == beat_idx.shape == pos_idx.shape):
        raise ValueError(
          'bar_idx, beat_idx, and pos_idx must share the same shape. '
          f'got {tuple(bar_idx.shape)}, {tuple(beat_idx.shape)}, {tuple(pos_idx.shape)}'
        )

      bar_ids = self._sanitize_index(bar_idx, self.bar_vocab_size)
      beat_ids = self._sanitize_index(beat_idx, self.beat_vocab_size)
      pos_ids = self._sanitize_index(pos_idx, self.pos_vocab_size)

      pe = self.bar_emb(bar_ids) + self.beat_emb(beat_ids) + self.pos_emb(pos_ids)

      if self.struct_emb is not None and is_struct is not None:
        struct_ids = is_struct.long().clamp(min=0, max=1)
        pe = pe + self.struct_emb(struct_ids)

      return pe

class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)


class CpCompositeEmbedding(nn.Module):
  def __init__(
    self,
    n_type,
    n_pos,
    n_limb_token,
    d_type,
    d_pos,
    d_limb,
    d_proj
  ):
    super(CpCompositeEmbedding, self).__init__()

    self.type_emb = nn.Embedding(n_type, d_type)
    self.pos_emb = nn.Embedding(n_pos, d_pos)
    self.hand1_emb = nn.Embedding(n_limb_token, d_limb)
    self.hand2_emb = nn.Embedding(n_limb_token, d_limb)
    self.rf_emb = nn.Embedding(n_limb_token, d_limb)
    self.lf_emb = nn.Embedding(n_limb_token, d_limb)

    d_cat = d_type + d_pos + (d_limb * 4)
    self.proj = nn.Linear(d_cat, d_proj, bias=False)
    self.emb_scale = d_proj ** 0.5

  def forward(self, type_token, pos, hand1, hand2, right_foot, left_foot):
    type_e = self.type_emb(type_token)
    pos_e = self.pos_emb(pos)
    h1_e = self.hand1_emb(hand1)
    h2_e = self.hand2_emb(hand2)
    rf_e = self.rf_emb(right_foot)
    lf_e = self.lf_emb(left_foot)

    cp_cat = torch.cat([type_e, pos_e, h1_e, h2_e, rf_e, lf_e], dim=-1)
    cp_proj = self.proj(cp_cat)
    return cp_proj.mul_(self.emb_scale)