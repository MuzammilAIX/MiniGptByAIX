
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from transformer_blocks import Block


# ============================================================
# 1. Paths and device
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

CHECKPOINT_PATH = MODEL_DIR / "sheikh_peer_gpt_bpe.pth"
DEFAULT_TOKENIZER_PATH = MODEL_DIR / "sheikh_peer_bpe.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. GPT model
# Must match the architecture used during training.
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.positional_embedding = nn.Embedding(
            block_size, embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            Block(
                embedding_dim,
                block_size,
                num_heads,
                dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.language_head = nn.Linear(
            embedding_dim, vocab_size, bias=False
        )

        # Same weight sharing as the training model.
        self.language_head.weight = self.token_embedding.weight

    def forward(self, idx):
        _, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError("Input exceeds the model context length.")

        positions = torch.arange(
            seq_len, device=idx.device
        )

        x = (
            self.token_embedding(idx)
            + self.positional_embedding(positions)
        )

        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.language_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            # Only use the most recent context tokens.
            context = idx[:, -self.block_size:]

            logits = self(context)
            logits = logits[:, -1, :]

            logits = logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k)
                cutoff = top_values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================
# 3. Load trained model and tokenizer
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run demo.py to train and save your model first."
        )

    # Only load checkpoints you trust, such as your own model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    vocab_size = checkpoint["vocab_size"]

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=checkpoint["block_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved_path = Path(checkpoint["tokenizer_path"])

    # Prefer the saved tokenizer, with a local fallback.
    if not saved_path.exists():
        saved_path = DEFAULT_TOKENIZER_PATH

    if not saved_path.exists():
        raise FileNotFoundError(
            "BPE tokenizer not found. Expected:\n"
            f"{saved_path}"
        )

    tokenizer = Tokenizer.from_file(str(saved_path))

    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(
            "Tokenizer vocabulary does not match the checkpoint."
        )

    return model, tokenizer, checkpoint


# ============================================================
# 4. Generate text
# ============================================================

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
):
    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError("Please enter a non-empty prompt.")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return tokenizer.decode(output_ids[0].tolist())


# ============================================================
# 5. Interactive terminal interface
# ============================================================

def main():
    print("=" * 60)
    print("MiniGPT - Interactive Text Generation")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model, tokenizer, checkpoint = load_model()

    print("Checkpoint:", CHECKPOINT_PATH.name)
    print("Vocabulary size:", checkpoint["vocab_size"])
    print("Context length:", checkpoint["block_size"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("\nModel loaded successfully. No retraining required.")
    print("Enter 'exit' or 'quit' to stop.")

    while True:
        print("\n" + "-" * 60)
        prompt = input("Enter your prompt: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter some text.")
            continue

        try:
            token_limit = input(
                "New tokens (default 100): "
            ).strip()

            max_new_tokens = int(token_limit) if token_limit else 100

            if not 1 <= max_new_tokens <= 500:
                print("Choose a value between 1 and 500.")
                continue

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=40,
            )

            print("\nGenerated text:\n")
            print(result)

        except (ValueError, RuntimeError) as error:
            print("Generation error:", error)


if __name__ == "__main__":
    main()
