import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass, field
from model_config import ModelConfig


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [Batch, Time, Variables]
        # Permute to [Batch, Variables, Time] for inverted embedding
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model, d_state, d_ff=None, expand=2, dropout=0.1, activation="gelu"
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # Bidirectional Mamba with expansion
        self.attention = Mamba(
            d_model=d_model, d_state=d_state, d_conv=4, expand=expand
        )
        self.attention_r = Mamba(
            d_model=d_model, d_state=d_state, d_conv=4, expand=expand
        )

        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Bidirectional Mamba
        forward_out = self.attention(x)
        backward_out = self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        x = self.norm1(x + forward_out + backward_out)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), None


class Decoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class QuantFlow(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_quantiles = len(config.quantiles)
        self.enc_embedding = DataEmbedding_inverted(
            config.seq_len, config.d_model, config.dropout
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_ff=config.d_ff,
                    expand=config.expand,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )
        self.projector = nn.Linear(
            config.d_model, config.pred_len * self.n_quantiles, bias=True
        )

    def forward(self, x_enc):
        # Normalization
        if self.config.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.decoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out)

        # Reshape: [Batch, N_vars, Pred_len, Quantiles]
        dec_out = dec_out.reshape(
            dec_out.shape[0], self.config.n_vars, self.config.pred_len, self.n_quantiles
        )
        # Permute to: [Batch, Pred_len, N_vars, Quantiles]
        dec_out = dec_out.permute(0, 2, 1, 3)

        # De-normalization
        if self.config.use_norm:
            stdev_b = stdev.unsqueeze(-1)
            means_b = means.unsqueeze(-1)
            dec_out = dec_out * stdev_b + means_b

        return dec_out


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert preds.shape[-1] == len(self.quantiles)
        target = target.unsqueeze(-1)
        loss = 0
        for i, q in enumerate(self.quantiles):
            error = target - preds[..., i : i + 1]
            loss += torch.max(q * error, (q - 1) * error)
        return loss.mean()
