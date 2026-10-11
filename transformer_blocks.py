
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Single Causal Self-Attention Head
# ============================================================

class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        block_size,
        head_size,
        dropout=0.1
    ):
        super().__init__()

        self.key = nn.Linear(
            embedding_dim, head_size, bias=False
        )
        self.query = nn.Linear(
            embedding_dim, head_size, bias=False
        )
        self.value = nn.Linear(
            embedding_dim, head_size, bias=False
        )

        # Lower-triangular mask prevents looking at future tokens.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
            persistent=False
        )

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Scaled dot-product attention scores.
        scores = q @ k.transpose(-2, -1)
        scores = scores / (k.shape[-1] ** 0.5)

        # Causal mask: each position sees only current and past tokens.
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.attention_dropout(
            attention_weights
        )

        output = attention_weights @ v

        return output


# ============================================================
# 2. Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        block_size,
        num_heads,
        dropout=0.1
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                embedding_dim,
                block_size,
                head_size,
                dropout
            )
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            embedding_dim, embedding_dim
        )

        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the output of all attention heads.
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.projection(output)
        output = self.residual_dropout(output)

        return output


# ============================================================
# 3. Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Transformer Block
# ============================================================

class Block(nn.Module):
    def __init__(
        self,
        embedding_dim,
        block_size,
        n_heads,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        # Pre-normalization improves training stability.
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.sa = MultiHeadAttention(
            embedding_dim,
            block_size,
            n_heads,
            dropout
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffwrd = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):
        # Attention residual connection.
        x = x + self.sa(self.ln1(x))

        # Feed-forward residual connection.
        x = x + self.ffwrd(self.ln2(x))

        return x
