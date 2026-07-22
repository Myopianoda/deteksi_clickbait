from __future__ import annotations

import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_text import LimeTextExplainer

from model import Runtime, predict_proba_batch

TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)

CLASS_NAMES = [
    "NON-CLICKBAIT",
    "CLICKBAIT",
]


def tokenize_with_punctuation(
    text: str,
) -> list[str]:
    return TOKEN_PATTERN.findall(
        str(text)
    )


def explain_prediction(
    runtime: Runtime,
    text: str,
    predicted_label: int,
    num_samples: int = 500,
    num_features: int = 10,
    batch_size: int = 16,
    random_state: int = 42,
) -> dict[str, Any]:
    clean_text = str(text).strip()

    if not clean_text:
        raise ValueError(
            "Teks tidak boleh kosong."
        )

    tokens = tokenize_with_punctuation(
        clean_text
    )

    if not tokens:
        raise ValueError(
            "Teks tidak memiliki fitur yang dapat dijelaskan."
        )

    selected_feature_count = max(
        1,
        min(
            int(num_features),
            len(dict.fromkeys(tokens)),
        ),
    )

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        split_expression=tokenize_with_punctuation,
        bow=True,
        random_state=int(random_state),
    )

    def classifier_fn(texts):
        return predict_proba_batch(
            runtime=runtime,
            texts=[
                str(item)
                for item in texts
            ],
            batch_size=int(batch_size),
        )

    explanation = explainer.explain_instance(
        text_instance=clean_text,
        classifier_fn=classifier_fn,
        labels=(int(predicted_label),),
        num_features=selected_feature_count,
        num_samples=int(num_samples),
    )

    predicted_class_name = CLASS_NAMES[
        int(predicted_label)
    ]

    features = []

    for feature, weight in explanation.as_list(
        label=int(predicted_label)
    ):
        numeric_weight = float(weight)

        features.append(
            {
                "feature": str(feature),
                "weight": numeric_weight,
                "direction": (
                    f"Mendukung {predicted_class_name}"
                    if numeric_weight >= 0
                    else f"Menahan {predicted_class_name}"
                ),
            }
        )

    score = explanation.score
    local_fidelity = (
        float(
            score.get(
                int(predicted_label),
                0.0,
            )
        )
        if isinstance(score, dict)
        else float(score)
    )

    return {
        "predicted_label": int(
            predicted_label
        ),
        "predicted_class_name": (
            predicted_class_name
        ),
        "features": features,
        "num_samples": int(num_samples),
        "num_features": len(features),
        "random_state": int(random_state),
        "local_fidelity": local_fidelity,
    }


def build_lime_figure(
    features: list[dict[str, Any]],
    predicted_class_name: str,
):
    if not features:
        raise ValueError(
            "Bobot fitur LIME kosong."
        )

    feature_df = (
        pd.DataFrame(features)
        .sort_values(
            "weight",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    colors = [
        "#c62828"
        if weight >= 0
        else "#2e7d32"
        for weight in feature_df["weight"]
    ]

    figure_height = max(
        4.5,
        0.52 * len(feature_df) + 1.8,
    )

    figure, axis = plt.subplots(
        figsize=(9, figure_height)
    )

    bars = axis.barh(
        feature_df["feature"],
        feature_df["weight"],
        color=colors,
        height=0.55,
        alpha=0.9,
    )

    axis.axvline(
        0.0,
        color="#424242",
        linewidth=1.0,
    )

    axis.set_xlabel("Bobot LIME")
    axis.set_ylabel("Fitur")
    axis.set_title(
        "Kontribusi fitur terhadap prediksi "
        f"{predicted_class_name}"
    )
    axis.grid(
        axis="x",
        alpha=0.2,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)

    maximum = max(
        float(
            feature_df["weight"].abs().max()
        ),
        1e-8,
    )
    offset = maximum * 0.025

    for bar, value in zip(
        bars,
        feature_df["weight"],
    ):
        numeric_value = float(value)

        axis.text(
            (
                numeric_value + offset
                if numeric_value >= 0
                else numeric_value - offset
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{numeric_value:+.6f}",
            va="center",
            ha=(
                "left"
                if numeric_value >= 0
                else "right"
            ),
            fontsize=9,
        )

    axis.set_xlim(
        min(
            float(
                feature_df["weight"].min()
            ),
            0.0,
        )
        - maximum * 0.25,
        max(
            float(
                feature_df["weight"].max()
            ),
            0.0,
        )
        + maximum * 0.25,
    )

    figure.tight_layout()
    return figure
