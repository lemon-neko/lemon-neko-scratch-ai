import numpy as np

from src.minigpt import CharTokenizer, MiniGPT, make_language_model_batches


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("猫喜欢鱼")
    assert tokenizer.decode(tokenizer.encode("猫喜欢鱼")) == "猫喜欢鱼"


def test_minigpt_shapes_and_causal_attention() -> None:
    model = MiniGPT(vocab_size=7, d_model=12, num_heads=3, d_ff=20, block_size=6)
    logits, cache = model.forward(np.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 7)
    assert np.allclose(cache["attentions"][0][0, :, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]], 0)


def test_a_training_step_reduces_loss_on_repeated_pattern() -> None:
    tokenizer = CharTokenizer.from_text("abababab")
    tokens = tokenizer.encode("abab" * 80)
    model = MiniGPT(tokenizer.vocab_size, d_model=12, num_heads=3, d_ff=24, block_size=4, seed=3)
    rng = np.random.default_rng(4)
    inputs, targets = make_language_model_batches(tokens, 4, 12, rng)
    before, _ = model.loss_and_gradients(inputs, targets)
    for _ in range(80):
        inputs, targets = make_language_model_batches(tokens, 4, 12, rng)
        model.train_step(inputs, targets, learning_rate=0.1)
    after, _ = model.loss_and_gradients(inputs, targets)
    assert after < before
