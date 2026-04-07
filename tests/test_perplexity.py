import math

import pytest

torch = pytest.importorskip("torch")

from quantarena.eval.perplexity import sliding_window_nll  # noqa: E402


class Out:
    def __init__(self, logits):
        self.logits = logits


class UniformLM(torch.nn.Module):
    """Predicts a uniform distribution, so perplexity == vocab size."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        return Out(torch.zeros(ids.size(0), ids.size(1), self.vocab))


class CountingLM(UniformLM):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.window_lens = []

    def forward(self, ids):
        self.window_lens.append(ids.size(1))
        return super().forward(ids)


@pytest.mark.parametrize("n, seq_len, stride", [(1000, 128, 128), (1000, 128, 32), (100, 128, 64), (257, 64, 50)])
def test_every_token_scored_once(n, seq_len, stride):
    ids = torch.randint(0, 50, (1, n))
    nll, tokens, _ = sliding_window_nll(UniformLM(50), ids, seq_len, stride)
    assert tokens == n - 1
    assert math.isclose(math.exp(nll / tokens), 50, rel_tol=1e-5)


def test_windows_never_exceed_seq_len():
    lm = CountingLM(10)
    sliding_window_nll(lm, torch.randint(0, 10, (1, 500)), 64, 16)
    assert max(lm.window_lens) <= 64


def test_max_windows():
    _, tokens, windows = sliding_window_nll(UniformLM(10), torch.randint(0, 10, (1, 1000)), 100, 50, max_windows=3)
    assert windows == 3
    assert tokens == 99 + 50 + 50
