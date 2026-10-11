
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from transformer_blocks import Block


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sheikhs_peer.txt"
MODEL_DIR = PROJECT_DIR / "models"

TOKENIZER_PATH = MODEL_DIR / "sheikh_peer_bpe.json"
CHECKPOINT_PATH = MODEL_DIR / "sheikh_peer_gpt_bpe.pth"

SEED = 42

BLOCK_SIZE = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

BATCH_SIZE = 4
MAX_STEPS = 8000
EVAL_INTERVAL = 250
EVAL_BATCHES = 10

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

GENERATE_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
PROMPT = "First Citizen:"

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Device information
# ============================================================

print("=" * 60)
print("MiniGPT - BPE Text Training")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Training device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2,
        ),
        "GB",
    )


# ============================================================
# 3. Validate files and load BPE tokenizer
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"BPE tokenizer not found: {TOKENIZER_PATH}\n"
        "Run: python tokenizer_train.py"
    )

text = DATA_PATH.read_text(encoding="utf-8-sig")

if not text.strip():
    raise ValueError("The dataset is empty.")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab_size = tokenizer.get_vocab_size()

# Encode the entire corpus using the trained BPE tokenizer.
encoded = tokenizer.encode(text)
all_data = torch.tensor(encoded.ids, dtype=torch.long)

if len(all_data) <= BLOCK_SIZE * 2:
    raise ValueError(
        "Not enough encoded tokens for training and validation."
    )

# Sequential split: first 90% for training, last 10% for validation.
split_index = int(len(all_data) * 0.90)

train_data = all_data[:split_index]
val_data = all_data[split_index:]

if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        "Training or validation data is too short for BLOCK_SIZE."
    )

print("\nDataset:", DATA_PATH.name)
print("Characters:", f"{len(text):,}")
print("BPE tokens:", f"{len(all_data):,}")
print("Training tokens:", f"{len(train_data):,}")
print("Validation tokens:", f"{len(val_data):,}")
print("BPE vocabulary size:", f"{vocab_size:,}")
print("Block size:", BLOCK_SIZE)


# ============================================================
# 4. Create training batches
# ============================================================

def get_batch(split):
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        low=0,
        high=len(source) - BLOCK_SIZE,
        size=(BATCH_SIZE,),
    )

    x = torch.stack([
        source[i:i + BLOCK_SIZE]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


# ============================================================
# 5. MiniGPT architecture
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.block_size = BLOCK_SIZE

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
        )

        self.positional_embedding = nn.Embedding(
            BLOCK_SIZE,
            EMBEDDING_DIM,
        )

        self.embedding_dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.Sequential(*[
            Block(
                EMBEDDING_DIM,
                BLOCK_SIZE,
                NUM_HEADS,
                DROPOUT,
            )
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = nn.LayerNorm(EMBEDDING_DIM)

        self.language_head = nn.Linear(
            EMBEDDING_DIM,
            vocab_size,
            bias=False,
        )

        # Share input token embeddings and output weights.
        self.language_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError("Input sequence exceeds BLOCK_SIZE.")

        positions = torch.arange(seq_len, device=idx.device)

        x = (
            self.token_embedding(idx)
            + self.positional_embedding(positions)
        )

        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.language_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]

            logits, _ = self(context)
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
                probabilities,
                num_samples=1,
            )

            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================
# 6. Evaluate training and validation loss
# ============================================================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}

    for split in ("train", "val"):
        batch_losses = torch.zeros(EVAL_BATCHES)

        for i in range(EVAL_BATCHES):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses[i] = loss.item()

        losses[split] = batch_losses.mean().item()

    model.train()
    return losses


# ============================================================
# 7. Initialize model and optimizer
# ============================================================

model = TinyGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=MIN_LEARNING_RATE,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:", f"{parameter_count:,}")
print("Model device:", next(model.parameters()).device)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
best_step = 0

print("\nStarting BPE training...")


# ============================================================
# 8. Training loop
# ============================================================

try:
    for step in range(1, MAX_STEPS + 1):
        model.train()

        x_batch, y_batch = get_batch("train")
        _, loss = model(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        ):
            metrics = estimate_loss(model)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Step {step:5d}/{MAX_STEPS} | "
                f"Train Loss: {metrics['train']:.4f} | "
                f"Val Loss: {metrics['val']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                best_step = step

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "block_size": BLOCK_SIZE,
                        "embedding_dim": EMBEDDING_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "vocab_size": vocab_size,
                        "tokenizer_path": str(TOKENIZER_PATH),
                        "best_val_loss": best_val_loss,
                        "best_step": best_step,
                    },
                    CHECKPOINT_PATH,
                )

                print(f"  Saved best checkpoint at step {step}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

print("\nTraining finished.")
print("Best validation loss:", round(best_val_loss, 4))
print("Best step:", best_step)
print("Checkpoint:", CHECKPOINT_PATH)


# ============================================================
# 9. Load best checkpoint and generate text
# ============================================================

if not CHECKPOINT_PATH.exists() or best_step == 0:
    raise RuntimeError(
        "No checkpoint was saved during this run. "
        "Allow at least one evaluation step to complete."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use the same BPE tokenizer that encoded the training corpus.
saved_tokenizer_path = Path(checkpoint["tokenizer_path"])

if not saved_tokenizer_path.exists():
    raise FileNotFoundError(
        f"Checkpoint tokenizer not found: {saved_tokenizer_path}"
    )

tokenizer = Tokenizer.from_file(str(saved_tokenizer_path))

print("\n" + "=" * 60)
print("BPE Text Generation")
print("=" * 60)

prompt_ids = tokenizer.encode(PROMPT).ids

if not prompt_ids:
    raise ValueError(
        "The prompt produced no token IDs. Choose another prompt."
    )

start_context = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    generated_ids = model.generate(
        start_context,
        max_new_tokens=GENERATE_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )[0].tolist()

generated_text = tokenizer.decode(generated_ids)

print("Prompt:", PROMPT)
print("\nGenerated text:\n")
print(generated_text)

if torch.cuda.is_available():
    print(
        "\nPeak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )

print("\nTraining and generation completed.")
