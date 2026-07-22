from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


class HybridIndoBERTBiLSTM(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        bert_config = AutoConfig.from_pretrained(
            model_name
        )
        self.bert = AutoModel.from_config(
            bert_config
        )

        bert_hidden_size = int(
            bert_config.hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.attention = nn.Linear(
            hidden_dim * 2,
            1,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        sequence_lengths = (
            attention_mask.sum(dim=1)
            .clamp_min(1)
            .cpu()
        )

        packed_input = pack_padded_sequence(
            sequence_output,
            sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(
            packed_input
        )

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_output.size(1),
        )

        attention_scores = self.attention(
            lstm_output
        )

        attention_scores = attention_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0,
            -1e9,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        context_vector = torch.sum(
            attention_weights * lstm_output,
            dim=1,
        )

        return self.classifier(
            self.dropout(context_vector)
        )


@dataclass
class Runtime:
    model: HybridIndoBERTBiLSTM
    tokenizer: Any
    max_length: int
    positive_class_id: int
    threshold: float
    model_name: str
    variant: str
    training_seed: int


def _load_checkpoint(
    checkpoint_path: str,
) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint tidak berisi state_dict PyTorch."
        )

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue

        clean_key = str(key)

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        cleaned[clean_key] = value

    if not cleaned:
        raise ValueError(
            "Checkpoint tidak memiliki parameter tensor."
        )

    return cleaned


def _read_json(path: str) -> dict[str, Any]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "run_config harus berupa objek JSON."
        )

    return config


def load_runtime(
    repo_id: str,
    checkpoint_filename: str,
    config_filename: str,
) -> Runtime:
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
    )

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
    )

    config = _read_json(config_path)
    state_dict = _load_checkpoint(
        checkpoint_path
    )

    training_seed = int(
        config.get("training_seed", -1)
    )
    variant = str(
        config.get("variant", "")
    )

    if training_seed != 2027:
        raise ValueError(
            "Checkpoint deployment harus memakai training seed 2027."
        )

    if variant != "target_q0150":
        raise ValueError(
            "Checkpoint deployment harus memakai varian target_q0150."
        )

    model_name = str(
        config.get(
            "model_name",
            "indolem/indobert-base-uncased",
        )
    )

    max_length = int(
        config.get(
            "max_len",
            config.get("max_length", 64),
        )
    )

    positive_class_id = int(
        config.get(
            "positive_class_id",
            1,
        )
    )

    threshold = float(
        config.get(
            "decision_threshold",
            config.get("threshold", 0.5),
        )
    )

    lstm_weight = state_dict.get(
        "lstm.weight_ih_l0"
    )
    classifier_weight = state_dict.get(
        "classifier.weight"
    )

    if (
        lstm_weight is None
        or classifier_weight is None
    ):
        raise ValueError(
            "Checkpoint tidak sesuai dengan arsitektur final."
        )

    hidden_dim = int(
        lstm_weight.shape[0] // 4
    )
    num_classes = int(
        classifier_weight.shape[0]
    )

    if num_classes != 2:
        raise ValueError(
            "Aplikasi hanya mendukung klasifikasi biner."
        )

    if positive_class_id not in (0, 1):
        raise ValueError(
            "positive_class_id harus 0 atau 1."
        )

    model = HybridIndoBERTBiLSTM(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.3,
        lstm_dropout=0.2,
    )

    try:
        incompatible = model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint tidak cocok secara persis dengan "
            f"arsitektur final: {error}"
        ) from error

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "Checkpoint memiliki parameter yang tidak cocok."
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    torch.set_num_threads(
        max(
            1,
            min(
                8,
                os.cpu_count() or 4,
            ),
        )
    )

    del state_dict
    gc.collect()

    return Runtime(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        positive_class_id=positive_class_id,
        threshold=threshold,
        model_name=model_name,
        variant=variant,
        training_seed=training_seed,
    )


def predict_proba_batch(
    runtime: Runtime,
    texts: Sequence[str],
    batch_size: int = 16,
) -> np.ndarray:
    normalized_texts = [
        str(text).strip() or " "
        for text in texts
    ]

    if not normalized_texts:
        return np.empty(
            (
                0,
                runtime.model.classifier.out_features,
            ),
            dtype=np.float32,
        )

    probability_parts: list[np.ndarray] = []

    for start in range(
        0,
        len(normalized_texts),
        batch_size,
    ):
        batch_texts = normalized_texts[
            start:start + batch_size
        ]

        encoded = runtime.tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=runtime.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = runtime.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded[
                    "attention_mask"
                ],
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        probability_parts.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(
        probability_parts,
        axis=0,
    )


def predict(
    runtime: Runtime,
    text: str,
) -> dict[str, float | int]:
    probabilities = predict_proba_batch(
        runtime=runtime,
        texts=[text],
        batch_size=1,
    )[0]

    clickbait_score = float(
        probabilities[
            runtime.positive_class_id
        ]
    )

    negative_class_id = (
        1 - runtime.positive_class_id
    )

    label_id = (
        runtime.positive_class_id
        if clickbait_score >= runtime.threshold
        else negative_class_id
    )

    return {
        "label_id": int(label_id),
        "confidence": float(
            probabilities[label_id]
        ),
        "clickbait_score": clickbait_score,
        "non_clickbait_score": float(
            probabilities[
                negative_class_id
            ]
        ),
    }
