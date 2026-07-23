"""Train the educational MiniGPT on a tiny text corpus.

Run with: ``python examples/train_minigpt.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow ``python examples/train_minigpt.py`` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minigpt import CharTokenizer, MiniGPT, make_language_model_batches


def main() -> None:
    text = ("我爱看猫。猫很可爱。" * 80) + ("hello transformer! " * 80)
    tokenizer = CharTokenizer.from_text(text)
    tokens = tokenizer.encode(text)
    model = MiniGPT(tokenizer.vocab_size, d_model=32, num_heads=4, d_ff=64, block_size=16)
    rng = np.random.default_rng(7)
    for step in range(300):
        inputs, targets = make_language_model_batches(tokens, 16, 8, rng)
        loss = model.train_step(inputs, targets, learning_rate=0.08)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1:>3}: loss={loss:.3f}")
    print(tokenizer.decode(model.generate(tokenizer.encode("我"), 30, seed=1)))


if __name__ == "__main__":
    main()
