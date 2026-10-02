"""Pai-1 model loading and inference."""

from collections import OrderedDict
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from .config import Settings

LOGGER = logging.getLogger(__name__)


class MiniPaiHead(nn.Module):
    """Decision head matching the exported Pai-1 checkpoint."""

    def __init__(self, hidden_size: int, decision_size: int, num_heads: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(hidden_size, decision_size)
        self.question_proj = nn.Linear(hidden_size, decision_size)
        self.option_proj = nn.Linear(hidden_size, decision_size)
        self.input_norm = nn.LayerNorm(decision_size)
        self.attention = nn.MultiheadAttention(decision_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(decision_size),
            nn.Linear(decision_size, decision_size * 2), nn.GELU(),
            nn.Linear(decision_size * 2, decision_size),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(decision_size * 3),
            nn.Linear(decision_size * 3, decision_size), nn.GELU(),
            nn.Linear(decision_size, 1),
        )

    def forward(self, state: Tensor, question: Tensor, options: Tensor, option_mask: Tensor) -> Tensor:
        state = self.state_proj(state)
        question = self.question_proj(question)
        options = self.option_proj(options)
        sequence = torch.cat((state.unsqueeze(1), question.unsqueeze(1), options), dim=1)
        attended_input = self.input_norm(sequence)
        padding = torch.cat((torch.zeros((sequence.size(0), 2), dtype=torch.bool, device=sequence.device), ~option_mask), dim=1)
        attended, _ = self.attention(attended_input, attended_input, attended_input, key_padding_mask=padding)
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.input_norm(sequence))
        state_context = sequence[:, 0:1].expand(-1, options.size(1), -1)
        question_context = sequence[:, 1:2].expand(-1, options.size(1), -1)
        combined = torch.cat((sequence[:, 2:], state_context, question_context), dim=-1)
        return self.scorer(combined).squeeze(-1)


class Pai1:
    """Load the frozen Qwen backbone and exported decision head once."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_environment()
        self.device = self._resolve_device(self.settings.device)
        config = json.loads((self.settings.model_path / "config.json").read_text(encoding="utf-8"))
        self.model_name = self.settings.model_path.name
        self.hidden_size = int(config["hidden_size"])
        tokenizer_path = self.settings.model_path / "tokenizer"
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.backbone = AutoModel.from_pretrained(self.settings.backbone, torch_dtype=dtype)
        self.backbone.to(self.device).eval()
        self.head = MiniPaiHead(self.hidden_size, int(config["decision_size"]), int(config["num_heads"]))
        checkpoint = torch.load(self.settings.model_path / "decision_head.pt", map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, (dict, OrderedDict)):
            raise RuntimeError("decision head checkpoint does not contain a state dict")
        self.head.load_state_dict(state_dict, strict=True)
        self.head.to(self.device, dtype=torch.float32).eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.head.parameters())
        LOGGER.info("Loaded %s on %s with %d head parameters", self.model_name, self.device, self.parameter_count)

    @staticmethod
    def _load_tokenizer(tokenizer_path: Path):
        """Load exported tokenizers across legacy/new Transformers formats."""
        config_path = tokenizer_path / "tokenizer_config.json"
        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
            return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)

        # Older exports store this as a list; newer Transformers expects a mapping.
        with tempfile.TemporaryDirectory(prefix="pai-1-tokenizer-") as temporary_directory:
            compatible_path = Path(temporary_directory) / "tokenizer"
            shutil.copytree(tokenizer_path, compatible_path)
            tokenizer_config["extra_special_tokens"] = {}
            (compatible_path / "tokenizer_config.json").write_text(
                json.dumps(tokenizer_config), encoding="utf-8"
            )
            return AutoTokenizer.from_pretrained(compatible_path, local_files_only=True, use_fast=True)
    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)

    @staticmethod
    def _pool_span(hidden: Tensor, offsets: Tensor, text: str, marker: str) -> Tensor:
        start = text.index(marker) + len(marker)
        end = text.find("\n", start)
        end = len(text) if end == -1 else end
        token_mask = (offsets[:, 0] < end) & (offsets[:, 1] > start)
        if not token_mask.any():
            raise RuntimeError(f"could not locate token span for {marker}")
        return hidden[token_mask].mean(dim=0)

    def predict(self, state: str, question: str, options: dict[str, str]) -> dict[str, Any]:
        option_items = list(options.items())
        text = "\n".join([f"STATE: {state}", f"QUESTION: {question}"] + [
            f"OPTION_{index}: {description}" for index, (_, description) in enumerate(option_items, 1)
        ])
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.settings.max_length, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].to(self.device)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.backbone(**encoded).last_hidden_state[0].float()
            pooled_state = self._pool_span(hidden, offsets, text, "STATE: ")
            pooled_question = self._pool_span(hidden, offsets, text, "QUESTION: ")
            pooled_options = torch.stack([
                self._pool_span(hidden, offsets, text, f"OPTION_{index}: ")
                for index in range(1, len(option_items) + 1)
            ])
            mask = torch.ones((1, len(option_items)), dtype=torch.bool, device=self.device)
            logits = self.head(pooled_state.unsqueeze(0), pooled_question.unsqueeze(0), pooled_options.unsqueeze(0), mask)[0]
            probabilities = torch.softmax(logits, dim=0)
        ranked = sorted(zip((label for label, _ in option_items), probabilities.tolist()), key=lambda item: item[1], reverse=True)
        return {"choice": ranked[0][0], "confidence": ranked[0][1], "probabilities": dict(ranked)}


# Backward-compatible aliases for the PaiClef name.
MiniClefHead = MiniPaiHead
PaiClef = Pai1