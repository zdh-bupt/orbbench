from __future__ import annotations

import torch
from torch import nn


class ObservableARTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 16, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.embedding_projection = nn.Linear(d_model, embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, input_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        length = contexts.shape[1]
        if length > self.position_embedding.shape[1]:
            raise ValueError(f"Sequence length {length} exceeds supported length")
        causal_mask = torch.triu(
            torch.ones((length, length), device=contexts.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.input_projection(contexts) + self.position_embedding[:, :length]
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.embedding_projection(hidden[:, -1])

    def forward(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(contexts)
        return embedding, self.prediction_head(embedding)

    def config(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
        }

