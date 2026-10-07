"""Portable cross-attention regression with optional PyTorch fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic


@dataclass(frozen=True, slots=True)
class PortableCrossAttentionRegressor:
    """A fitted attention model whose inference path requires only NumPy."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    bank_embedding: np.ndarray
    bank_od: np.ndarray
    embed_weight: np.ndarray
    embed_bias: np.ndarray
    attention_weight: np.ndarray
    attention_bias: np.ndarray
    attention_output_weight: np.ndarray
    attention_output_bias: np.ndarray
    head0_weight: np.ndarray
    head0_bias: np.ndarray
    head1_weight: np.ndarray
    head1_bias: np.ndarray
    heads: int
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        width = len(np.asarray(self.embed_bias))
        feature_count = len(np.asarray(self.feature_mean))
        if np.asarray(self.feature_scale).shape != (feature_count,):
            raise ValueError("attention feature normalization is malformed")
        if np.asarray(self.embed_weight).shape != (width, feature_count):
            raise ValueError("attention embedding weight is malformed")
        if (
            np.asarray(self.bank_embedding).ndim != 2
            or self.bank_embedding.shape[1] != width
        ):
            raise ValueError("attention bank embedding is malformed")
        if np.asarray(self.bank_od).shape != (len(self.bank_embedding),):
            raise ValueError("attention bank outcomes are malformed")
        if width % self.heads or self.heads < 1:
            raise ValueError("attention head count is incompatible with width")
        if np.asarray(self.attention_weight).shape != (3 * width, width):
            raise ValueError("attention projection weight is malformed")
        if np.asarray(self.attention_bias).shape != (3 * width,):
            raise ValueError("attention projection bias is malformed")
        expected = _fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("portable attention fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def predict(
        self, features: np.ndarray, *, batch_size: int = 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict nonnegative reference OD and neighbor-dispersion uncertainty."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("attention features have the wrong shape")
        output = np.empty(len(values), dtype=np.float32)
        uncertainty = np.empty(len(values), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            embedded = _gelu(
                ((values[start:stop] - self.feature_mean) / self.feature_scale)
                @ self.embed_weight.T
                + self.embed_bias
            )
            neighbors = _nearest_bank(embedded, self.bank_embedding, count=16)
            context = _attention_context(self, embedded, neighbors)
            hidden = _gelu(
                np.concatenate((embedded, context), axis=1) @ self.head0_weight.T
                + self.head0_bias
            )
            output[start:stop] = np.maximum(
                (hidden @ self.head1_weight.T + self.head1_bias)[:, 0], 0
            )
            uncertainty[start:stop] = np.std(self.bank_od[neighbors], axis=1)
        return output, uncertainty

    def predict_accelerated(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 2048,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use optional PyTorch acceleration with the exact portable weights."""

        try:
            import torch
        except ImportError:
            return self.predict(features, batch_size=batch_size)
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("attention features have the wrong shape")
        selected_device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        tensor = lambda value: torch.from_numpy(  # noqa: E731
            np.asarray(value, dtype=np.float32)
        ).to(selected_device)
        mean, scale = tensor(self.feature_mean), tensor(self.feature_scale)
        embed_weight, embed_bias = tensor(self.embed_weight), tensor(self.embed_bias)
        bank = tensor(self.bank_embedding)
        bank_od = tensor(self.bank_od)
        attention_weight = tensor(self.attention_weight)
        attention_bias = tensor(self.attention_bias)
        output_weight = tensor(self.attention_output_weight)
        output_bias = tensor(self.attention_output_bias)
        head0_weight, head0_bias = tensor(self.head0_weight), tensor(self.head0_bias)
        head1_weight, head1_bias = tensor(self.head1_weight), tensor(self.head1_bias)
        output = np.empty(len(values), dtype=np.float32)
        uncertainty = np.empty(len(values), dtype=np.float32)
        width = bank.shape[1]
        head_width = width // self.heads
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                raw = tensor(values[start:stop])
                query = torch.nn.functional.gelu(
                    torch.nn.functional.linear(
                        (raw - mean) / scale, embed_weight, embed_bias
                    ),
                    approximate="tanh",
                )
                nearest = (query @ bank.T).topk(min(16, len(bank)), dim=1).indices
                neighbors = bank[nearest]
                q = torch.nn.functional.linear(
                    query,
                    attention_weight[:width],
                    attention_bias[:width],
                ).reshape(len(query), self.heads, head_width)
                key = torch.nn.functional.linear(
                    neighbors,
                    attention_weight[width : 2 * width],
                    attention_bias[width : 2 * width],
                ).reshape(len(query), neighbors.shape[1], self.heads, head_width)
                value = torch.nn.functional.linear(
                    neighbors,
                    attention_weight[2 * width :],
                    attention_bias[2 * width :],
                ).reshape(len(query), neighbors.shape[1], self.heads, head_width)
                scores = torch.einsum("bhd,bnhd->bhn", q, key) / np.sqrt(head_width)
                weights = torch.softmax(scores, dim=2)
                context = torch.einsum("bhn,bnhd->bhd", weights, value).reshape(
                    len(query), width
                )
                context = torch.nn.functional.linear(
                    context, output_weight, output_bias
                )
                hidden = torch.nn.functional.gelu(
                    torch.nn.functional.linear(
                        torch.cat((query, context), dim=1),
                        head0_weight,
                        head0_bias,
                    ),
                    approximate="tanh",
                )
                prediction = torch.nn.functional.linear(
                    hidden, head1_weight, head1_bias
                )[:, 0]
                output[start:stop] = torch.clamp_min(prediction, 0).cpu().numpy()
                uncertainty[start:stop] = (
                    torch.std(bank_od[nearest], dim=1, correction=0).cpu().numpy()
                )
        return output, uncertainty

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        metadata = {
            "schema_version": 1,
            "architecture": "cross_attention",
            "heads": self.heads,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                **_arrays(self),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> PortableCrossAttentionRegressor:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported portable attention schema")
            return cls(
                heads=int(metadata["heads"]),
                provenance=dict(metadata["provenance"]),
                fingerprint=str(metadata["fingerprint"]),
                **{name: data[name] for name in _ARRAY_NAMES},
            )


def fit_portable_cross_attention(
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 2048,
    bank_limit: int = 12_000,
) -> PortableCrossAttentionRegressor:
    """Fit cross-attention with PyTorch and export a NumPy-only artifact."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "cross-attention fitting requires the 'cells' extra"
        ) from error
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(measured_od, dtype=np.float32)
    indices = np.asarray(train, dtype=np.int64)
    valid = indices[np.isfinite(y[indices])]
    if len(valid) < 4:
        raise ValueError("cross-attention requires four measured training cells")
    center, scale = x[valid].mean(0), x[valid].std(0)
    scale[scale < 1e-6] = 1
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy((x[valid] - center) / scale).to(device)
    train_y = torch.from_numpy(y[valid]).to(device)
    if batch_size < 1 or bank_limit < 16:
        raise ValueError("attention batch and bank sizes must be positive")
    generator = np.random.default_rng(seed)
    bank_rows = (
        np.sort(generator.choice(len(valid), bank_limit, replace=False))
        if len(valid) > bank_limit
        else np.arange(len(valid), dtype=np.int64)
    )
    bank_x = train_x[torch.from_numpy(bank_rows).to(device)]
    width = min(128, max(16, x.shape[1]))
    heads = 4 if width % 4 == 0 else 1

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(x.shape[1], width)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.head0 = nn.Linear(width * 2, width)
            self.head1 = nn.Linear(width, 1)

        def forward(self, query, bank):  # type: ignore[no-untyped-def]
            embedded = torch.nn.functional.gelu(self.embed(query), approximate="tanh")
            bank_embedding = torch.nn.functional.gelu(
                self.embed(bank), approximate="tanh"
            )
            similarity = embedded @ bank_embedding.T
            nearest = similarity.topk(min(16, len(bank_embedding)), dim=1).indices
            neighbor = bank_embedding[nearest]
            context, _weights = self.attention(embedded[:, None, :], neighbor, neighbor)
            joined = torch.cat((embedded, context[:, 0, :]), dim=1)
            hidden = torch.nn.functional.gelu(self.head0(joined), approximate="tanh")
            return self.head1(hidden)[:, 0]

    model = Network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), batch_size):
            selected = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(
                model(train_x[selected], bank_x), train_y[selected]
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        bank = (
            torch.nn.functional.gelu(model.embed(bank_x), approximate="tanh")
            .cpu()
            .numpy()
        )

    def numpy(value) -> np.ndarray:  # type: ignore[no-untyped-def]
        return value.detach().cpu().numpy().astype(np.float32)

    return PortableCrossAttentionRegressor(
        feature_mean=center.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        bank_embedding=bank.astype(np.float32),
        bank_od=y[valid[bank_rows]].astype(np.float32),
        embed_weight=numpy(model.embed.weight),
        embed_bias=numpy(model.embed.bias),
        attention_weight=numpy(model.attention.in_proj_weight),
        attention_bias=numpy(model.attention.in_proj_bias),
        attention_output_weight=numpy(model.attention.out_proj.weight),
        attention_output_bias=numpy(model.attention.out_proj.bias),
        head0_weight=numpy(model.head0.weight),
        head0_bias=numpy(model.head0.bias),
        head1_weight=numpy(model.head1.weight),
        head1_bias=numpy(model.head1.bias),
        heads=heads,
        provenance={
            "seed": seed,
            "epochs": epochs,
            "training_cells": len(valid),
            "attention_bank_cells": len(bank_rows),
            "batch_size": batch_size,
        },
    )


def _nearest_bank(query: np.ndarray, bank: np.ndarray, *, count: int) -> np.ndarray:
    similarity = query @ bank.T
    number = min(count, len(bank))
    selected = np.argpartition(similarity, -number, axis=1)[:, -number:]
    scores = np.take_along_axis(similarity, selected, axis=1)
    return np.take_along_axis(selected, np.argsort(-scores, axis=1), axis=1)


def _attention_context(
    model: PortableCrossAttentionRegressor,
    query: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    width = query.shape[1]
    weight = np.asarray(model.attention_weight)
    bias = np.asarray(model.attention_bias)
    q = query @ weight[:width].T + bias[:width]
    bank = np.asarray(model.bank_embedding)[neighbors]
    key = bank @ weight[width : 2 * width].T + bias[width : 2 * width]
    value = bank @ weight[2 * width :].T + bias[2 * width :]
    head_width = width // model.heads
    q = q.reshape(len(q), model.heads, head_width)
    key = key.reshape(len(q), key.shape[1], model.heads, head_width)
    value = value.reshape(len(q), value.shape[1], model.heads, head_width)
    scores = np.einsum("bhd,bnhd->bhn", q, key) / np.sqrt(head_width)
    scores -= scores.max(axis=2, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=2, keepdims=True)
    context = np.einsum("bhn,bnhd->bhd", weights, value).reshape(len(q), width)
    return context @ model.attention_output_weight.T + model.attention_output_bias


def _gelu(values: np.ndarray) -> np.ndarray:
    return (
        0.5
        * values
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))
    )


_ARRAY_NAMES = (
    "feature_mean",
    "feature_scale",
    "bank_embedding",
    "bank_od",
    "embed_weight",
    "embed_bias",
    "attention_weight",
    "attention_bias",
    "attention_output_weight",
    "attention_output_bias",
    "head0_weight",
    "head0_bias",
    "head1_weight",
    "head1_bias",
)


def _arrays(model: PortableCrossAttentionRegressor) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(model, name), dtype=np.float32)
        for name in _ARRAY_NAMES
    }


def _fingerprint(model: PortableCrossAttentionRegressor) -> str:
    digest = hashlib.sha256(b"histopia-portable-cross-attention-v1\0")
    digest.update(
        json.dumps(
            {"heads": model.heads, "provenance": model.provenance},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name, value in sorted(_arrays(model).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
