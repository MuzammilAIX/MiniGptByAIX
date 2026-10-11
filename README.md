# MiniGPT — Small GPT-Style Language Model

## Overview

MiniGPT is a self-learning project that trains a small GPT-style language model on a Shakespeare-style text corpus using PyTorch, Byte Pair Encoding (BPE), and CUDA.

The purpose of this project is to explore the fundamentals of language-model training, next-token prediction, validation, checkpointing, and text generation.

## Technical Details

- **Framework:** PyTorch
- **Tokenizer:** Byte Pair Encoding (BPE)
- **Vocabulary size:** 8,000 tokens
-
- **Context block size:** 128 tokens
- **Model parameters:** 1,832,192
- **Training steps:** 8,000
- **Hardware:** NVIDIA GeForce RTX 2050, 4 GB VRAM

## Training Results

| Metric                |  Result |
| --------------------- | ------: |
| Training tokens       | 286,251 |
| Validation tokens     |  31,806 |
| Best validation loss  |  4.7458 |
| Best checkpoint step  |   6,500 |
| Final training loss   |  4.1705 |
| Final validation loss |  4.9357 |

The best checkpoint was selected using validation loss. The model's generated text demonstrates some source-style patterns, but its output is not consistently coherent.

## Key Learning Outcomes

- BPE tokenization and vocabulary construction
- GPT-style next-token prediction
- PyTorch model training with CUDA
- Training and validation loss monitoring
- Model checkpointing
- Text generation and qualitative evaluation

## Limitations

This is a small experimental language model trained on a limited corpus. It is not a general-purpose chatbot or a production-ready LLM. Its text generation may contain grammatical errors and disconnected sentences.

## Running the Project

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Place the permitted source corpus in the expected data location.
5. Review the training script and run it using the command appropriate to your project.

Exact commands and file paths should be added after confirming the repository's actual script names.

## Future Improvements

Potential extensions include early stopping, larger permitted training corpora, longer context windows, and controlled experiments with model size and generation settings.

## Project Status

Completed as a self-learning project. The implementation and results are documented for educational and portfolio purposes.
