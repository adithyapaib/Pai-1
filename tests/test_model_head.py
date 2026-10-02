"""Unit tests for the decision head math (no backbone download)."""

import pytest
import torch

from app.model import MiniPaiHead, Pai1


def test_mini_pai_head_output_shape_and_masking():
    torch.manual_seed(0)
    head = MiniPaiHead(hidden_size=16, decision_size=8, num_heads=2).eval()
    state = torch.randn(2, 16)
    question = torch.randn(2, 16)
    options = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    with torch.no_grad():
        logits = head(state, question, options, mask)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_mini_pai_head_parameter_count_matches_export():
    head = MiniPaiHead(hidden_size=896, decision_size=256, num_heads=4)
    assert sum(p.numel() for p in head.parameters()) == 1414657


def test_legacy_aliases_still_importable():
    from app.model import MiniClefHead, PaiClef

    assert MiniClefHead is MiniPaiHead
    assert PaiClef is Pai1


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert Pai1._resolve_device("auto") == torch.device("cpu")
    assert Pai1._resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        Pai1._resolve_device("cuda")


def test_pool_span_averages_tokens_in_marker_line():
    # text layout: "STATE: " is 7 chars, "ab" occupies chars 7..9,
    # newline at 9. Two tokens split "a"/"b" so the mean is [2, 0].
    hidden = torch.tensor([[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]])
    text = "STATE: ab\nQUESTION: cd"
    offsets = torch.tensor([[7, 8], [8, 9], [9, 20]])
    pooled = Pai1._pool_span(hidden, offsets, text, "STATE: ")
    assert pooled.tolist() == pytest.approx([2.0, 0.0])


def test_pool_span_missing_marker_raises_value_error():
    # Marker absent from text: str.index raises ValueError (not RuntimeError).
    hidden = torch.zeros(2, 4)
    offsets = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        Pai1._pool_span(hidden, offsets, "STATE: x", "OPTION_1: ")


def test_pool_span_no_overlapping_tokens_raises_runtime_error():
    # Marker present, but no token offsets overlap its line span.
    hidden = torch.zeros(2, 4)
    text = "STATE: x\nNEXT: y"
    offsets = torch.tensor([[10, 11], [12, 13]])
    with pytest.raises(RuntimeError, match="could not locate token span"):
        Pai1._pool_span(hidden, offsets, text, "STATE: ")
