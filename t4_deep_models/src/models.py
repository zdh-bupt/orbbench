from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as torch_f


TRAINED_MODELS = ("lstm", "gru", "dlinear", "tcn", "informer", "fedformer")


class RNNDirect(nn.Module):
    def __init__(
        self, kind: str, input_dim: int, pred_len: int, hidden: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.encoder = recurrent(
            input_dim,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, pred_len * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1]).view(x.shape[0], self.pred_len, self.input_dim)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = torch_f.pad(x.transpose(1, 2), (self.padding, self.padding), mode="replicate")
        return torch_f.avg_pool1d(values, self.kernel_size, stride=1).transpose(1, 2)


class DLinear(nn.Module):
    def __init__(self, input_dim: int, input_len: int, pred_len: int, moving_avg: int) -> None:
        super().__init__()
        self.decomp = MovingAverage(moving_avg)
        self.seasonal = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(input_dim)])
        self.trend = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(input_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return torch.stack(
            [
                self.seasonal[index](seasonal[:, :, index])
                + self.trend[index](trend[:, :, index])
                for index in range(x.shape[2])
            ],
            dim=2,
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.conv(x)
        if self.padding:
            value = value[:, :, : -self.padding]
        return x + self.dropout(torch_f.gelu(self.norm(value)))


class TCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        channels: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                TemporalBlock(channels, 2**index, kernel_size, dropout)
                for index in range(layers)
            )
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, pred_len * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.blocks(self.input_proj(x.transpose(1, 2)))
        pooled = torch.cat((value[:, :, -1], value.mean(dim=2)), dim=1)
        return self.head(pooled).view(x.shape[0], self.pred_len, self.input_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]]


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, factor: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        sample_count = min(length, max(1, self.factor * math.ceil(math.log(length + 1))))
        top_count = min(length, max(1, self.factor * math.ceil(math.log(length + 1))))
        query_indices = torch.arange(length, device=x.device).unsqueeze(1)
        offsets = torch.arange(sample_count, device=x.device).unsqueeze(0)
        stride = max(1, length // sample_count)
        sample_indices = (query_indices + offsets * stride) % length
        sampled_keys = key[:, :, sample_indices, :]
        sampled_scores = torch.einsum("bhld,bhlsd->bhls", query, sampled_keys)
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        top_indices = sparsity.topk(top_count, dim=-1, sorted=False).indices
        gather = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        top_queries = query.gather(2, gather)
        scores = torch.matmul(top_queries, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        updates = torch.matmul(weights, value)
        context = value.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()
        context.scatter_(2, gather, updates)
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context)


class InformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, heads: int, factor: int, feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = ProbSparseAttention(d_model, heads, factor, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.feedforward(self.norm2(x)))


class QueryDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(query)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        query = query + self.dropout(attended)
        attended, _ = self.cross_attention(
            self.norm2(query), memory, memory, need_weights=False
        )
        query = query + self.dropout(attended)
        return query + self.dropout(self.feedforward(self.norm3(query)))


class HorizonQueries(nn.Module):
    def __init__(self, target_steps: list[int], d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(len(target_steps), d_model))
        nn.init.normal_(self.embedding, std=0.02)
        values = torch.log1p(torch.as_tensor(target_steps, dtype=torch.float32)).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(len(target_steps), d_model)
        encoding[:, 0::2] = torch.sin(values * frequencies)
        encoding[:, 1::2] = torch.cos(values * frequencies)
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(self, batch: int) -> torch.Tensor:
        return (self.embedding + self.encoding).unsqueeze(0).expand(batch, -1, -1)


class Informer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_len: int,
        pred_len: int,
        target_steps: list[int],
        d_model: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        factor: int,
        feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, input_len)
        self.encoder = nn.ModuleList(
            [
                InformerEncoderLayer(d_model, heads, factor, feedforward, dropout)
                for _ in range(encoder_layers)
            ]
        )
        self.queries = HorizonQueries(target_steps, d_model)
        self.decoder = nn.ModuleList(
            [QueryDecoderLayer(d_model, heads, feedforward, dropout) for _ in range(decoder_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, input_dim))
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.position(self.input(x) * math.sqrt(self.input.out_features))
        for layer in self.encoder:
            memory = layer(memory)
        query = self.queries(len(x))
        for layer in self.decoder:
            query = layer(query, memory)
        return self.output(query)


class FourierEncoderLayer(nn.Module):
    def __init__(self, d_model: int, modes: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(modes, d_model, 2) * 0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = torch.fft.rfft(self.norm1(x), dim=1, norm="ortho")
        mixed = torch.zeros_like(frequency)
        keep = min(self.modes, frequency.shape[1])
        weights = torch.view_as_complex(self.weight[:keep].contiguous())
        mixed[:, :keep] = frequency[:, :keep] * weights.unsqueeze(0)
        value = torch.fft.irfft(mixed, n=x.shape[1], dim=1, norm="ortho")
        x = x + self.dropout(value)
        return x + self.dropout(self.feedforward(self.norm2(x)))


class FEDformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_len: int,
        pred_len: int,
        target_steps: list[int],
        d_model: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        feedforward: int,
        dropout: float,
        moving_avg: int,
        modes: int,
    ) -> None:
        super().__init__()
        self.decomp = MovingAverage(moving_avg)
        self.input = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, input_len)
        self.encoder = nn.ModuleList(
            [FourierEncoderLayer(d_model, modes, feedforward, dropout) for _ in range(encoder_layers)]
        )
        self.queries = HorizonQueries(target_steps, d_model)
        self.decoder = nn.ModuleList(
            [QueryDecoderLayer(d_model, heads, feedforward, dropout) for _ in range(decoder_layers)]
        )
        self.seasonal_output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, input_dim))
        self.trend_output = nn.ModuleList(
            [nn.Linear(input_len, pred_len) for _ in range(input_dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        memory = self.position(self.input(seasonal) * math.sqrt(self.input.out_features))
        for layer in self.encoder:
            memory = layer(memory)
        query = self.queries(len(x))
        for layer in self.decoder:
            query = layer(query, memory)
        seasonal_output = self.seasonal_output(query)
        trend_output = torch.stack(
            [self.trend_output[index](trend[:, :, index]) for index in range(x.shape[2])],
            dim=2,
        )
        return seasonal_output + trend_output


def model_config(model_name: str, input_steps: int, target_steps: list[int]) -> dict[str, Any]:
    common: dict[str, Any] = {"input_dim": 6, "pred_len": len(target_steps)}
    if model_name in {"lstm", "gru"}:
        return {**common, "kind": model_name, "hidden": 128, "layers": 2, "dropout": 0.1}
    if model_name == "dlinear":
        return {**common, "input_len": input_steps, "moving_avg": 25}
    if model_name == "tcn":
        return {
            **common,
            "channels": 64,
            "layers": 8,
            "kernel_size": 7,
            "dropout": 0.1,
        }
    transformer = {
        **common,
        "input_len": input_steps,
        "target_steps": target_steps,
        "d_model": 128,
        "heads": 4,
        "feedforward": 512,
        "dropout": 0.1,
    }
    if model_name == "informer":
        return {**transformer, "encoder_layers": 3, "decoder_layers": 1, "factor": 5}
    if model_name == "fedformer":
        return {
            **transformer,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "moving_avg": 25,
            "modes": 64,
        }
    raise ValueError(f"unsupported trained model: {model_name}")


def build_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    if model_name in {"lstm", "gru"}:
        return RNNDirect(**config)
    if model_name == "dlinear":
        return DLinear(**config)
    if model_name == "tcn":
        return TCN(**config)
    if model_name == "informer":
        return Informer(**config)
    if model_name == "fedformer":
        return FEDformer(**config)
    raise ValueError(model_name)
