
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sheikhs_peer.txt"
MODEL_DIR = PROJECT_DIR / "models"
TOKENIZER_PATH = MODEL_DIR / "sheikh_peer_bpe.json"

# BPE settings
VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    text = DATA_PATH.read_text(encoding="utf-8-sig")

    if not text.strip():
        raise ValueError("Dataset is empty.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[UNK]", "[BOS]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(
        iter(text.splitlines()),
        trainer=trainer
    )

    tokenizer.save(str(TOKENIZER_PATH))

    encoded = tokenizer.encode(text)

    print("=" * 50)
    print("BPE TOKENIZER TRAINING COMPLETE")
    print("=" * 50)
    print(f"Dataset characters: {len(text):,}")
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size():,}")
    print(f"Encoded tokens: {len(encoded.ids):,}")
    print(f"Tokenizer saved to: {TOKENIZER_PATH}")

    sample = text[:200]
    print("\nOriginal sample:")
    print(sample)

    print("\nDecoded sample:")
    print(tokenizer.decode(tokenizer.encode(sample).ids))

if __name__ == "__main__":
    main()
