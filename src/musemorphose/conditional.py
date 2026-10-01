"""比較用の連続難易度条件付きVAE。既存MuseMorphoseのEncoder/Decoderを共用する。"""
from dataclasses import dataclass
import math
import torch
from torch import nn

from src.musemorphose.model import VAETransformerDecoder
from src.musemorphose.transformer_encoder import VAETransformerEncoder
from src.musemorphose.transformer_helpers import PositionalEncoding
from src.common.skeleton import FEATURE_SIZE
from src.common.conditioning import difficulty_condition


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    d_latent: int = 512
    n_head: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    d_ff: int = 2048
    d_condition: int = 128
    dropout: float = .1
    cond_mode: str = 'in-attn'
    difficulty_mode: str = 'relative'
    use_skeleton: bool = True
    max_tokens: int = 512

    def validate(self):
        for name in ('d_model', 'd_latent', 'n_head', 'enc_layers', 'dec_layers', 'd_ff', 'd_condition', 'max_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + 'は正整数')
        if self.d_model % self.n_head or self.d_model % 2 or self.max_tokens < 4:
            raise ValueError('d_modelは偶数かつhead数の倍数、max_tokens>=4が必要')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropoutは0以上1未満')
        if self.cond_mode not in ('in-attn', 'pre-attn') or self.difficulty_mode not in ('relative', 'absolute'):
            raise ValueError('条件入力モードが不正')
        if type(self.use_skeleton) is not bool:
            raise ValueError('use_skeletonはbool')
        return self


class ConditionalMuseMorphose(nn.Module):
    """前・対象・後の小節を個別に符号化し、対象小節のトークン列を因果的に復号する。

    単体比較では原譜を対象入力にし、use_skeleton=Falseを用いる。
    旧クラスのcheckpoint形式とは別形式。旧クラスは__init__.pyから利用できる。
    """
    def __init__(self, n_token, pad_id, bar_id, config=ModelConfig()):
        super().__init__()
        self.config = c = config.validate()
        self.pad_id, self.bar_id, self.n_token = pad_id, bar_id, n_token
        self.embedding = nn.Embedding(n_token, c.d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(c.d_model, c.max_tokens)
        self.dropout = nn.Dropout(c.dropout)
        self.encoder = VAETransformerEncoder(c.enc_layers, c.n_head, c.d_model, c.d_ff, c.d_latent, c.dropout)
        self.difficulty_embedding = nn.Sequential(nn.Linear(10, c.d_condition), nn.GELU())
        self.skeleton_embedding = (nn.Sequential(nn.Linear(FEATURE_SIZE, c.d_condition), nn.GELU())
                                   if c.use_skeleton else None)
        condition_size = c.d_condition * (2 if c.use_skeleton else 1)
        self.encoder_condition = nn.Linear(condition_size, c.d_model, bias=False)
        self.context_embedding = nn.Linear(2 * c.d_latent + 2, c.d_condition)
        self.decoder = VAETransformerDecoder(c.dec_layers, c.n_head, c.d_model, c.d_ff,
                                             c.d_latent + condition_size + c.d_condition,
                                             c.dropout, cond_mode=c.cond_mode)
        self.output = nn.Linear(c.d_model, n_token)

    def encode(self, batch, sample=None):
        tokens, present = batch['encoder_tokens'], batch['context_present']
        if tokens.ndim != 3 or tokens.shape[1] != 3 or present.shape != (tokens.shape[0], 2):
            raise ValueError('encoder_tokensは[B,3,L]、context_presentは[B,2]')
        if tokens.shape[2] > self.config.max_tokens:
            raise ValueError('Encoderの上限超過。切り詰めず入力を除外する')
        condition = self.difficulty_embedding(difficulty_condition(batch['target'], batch['current'], self.config.difficulty_mode))
        if self.skeleton_embedding is not None:
            condition = torch.cat((condition, self.skeleton_embedding(batch['skeleton'])), dim=-1)
        b, _, length = tokens.shape
        flat = tokens.reshape(b * 3, length).T
        mask = flat.T.eq(self.pad_id)
        if mask.all(dim=1).any():
            raise ValueError('欠損文脈にもBAR/EOSを指定し、全padding小節は作らない')
        anchors = flat.eq(self.bar_id)
        if not anchors.sum(dim=0).eq(1).all():
            raise ValueError('Encoderの各小節にBARが1つ必要')
        emb = self.dropout(self.embedding(flat) * self.config.d_model ** .5) + self.pe(length)
        roles = torch.zeros((b, 3, self.config.d_model), device=emb.device)
        roles[:, 1] = self.encoder_condition(condition)
        emb = emb + roles.reshape(b * 3, -1).unsqueeze(0)
        _, mu, logvar = self.encoder(emb, padding_mask=mask, anchor_positions=anchors.long().argmax(dim=0))
        mu, logvar = mu.reshape(b, 3, -1), logvar.reshape(b, 3, -1)
        center_mu, center_logvar = mu[:, 1], logvar[:, 1].clamp(-20, 20)
        sampling = self.training if sample is None else sample
        z = center_mu + torch.exp(.5 * center_logvar) * torch.randn_like(center_mu) if sampling else center_mu
        neighbors = torch.cat((mu[:, 0] * present[:, :1], mu[:, 2] * present[:, 1:], present), dim=-1)
        segment = torch.cat((z, condition, self.context_embedding(neighbors)), dim=-1)
        return dict(mu=center_mu, logvar=center_logvar, segment=segment)

    def decode(self, tokens, encoded):
        if tokens.ndim != 2 or tokens.shape[0] > self.config.max_tokens:
            raise ValueError('Decoder入力は上限内の[T,B]')
        embedded = self.dropout(self.embedding(tokens) * self.config.d_model ** .5) + self.pe(tokens.shape[0])
        segment = encoded['segment'].unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return self.output(self.decoder(embedded, segment))

    def forward(self, batch, sample=None):
        encoded = self.encode(batch, sample=sample)
        return dict(**encoded, logits=self.decode(batch['decoder_input'], encoded))
