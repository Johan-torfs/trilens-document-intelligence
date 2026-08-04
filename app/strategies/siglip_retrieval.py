from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.strategies.retrieval import RetrievalStrategy


SIGLIP_MODEL_NAME = "google/siglip-base-patch16-384"


class SiglipRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name

        self._device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        dtype = (
            torch.float16
            if self._device.type == "cuda"
            else torch.float32
        )

        self._processor = AutoProcessor.from_pretrained(
            model_name
        )

        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model.get_image_features(**inputs)

        features = self._extract_features(outputs)

        return self._normalize(features[0])

    def embed_text(self, text: str) -> np.ndarray:
        cleaned_text = text.strip().lower()

        if not cleaned_text:
            raise ValueError("Text may not be empty.")

        inputs = self._processor(
            text=[cleaned_text],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model.get_text_features(**inputs)

        features = self._extract_features(outputs)

        return self._normalize(features[0])

    @staticmethod
    def _extract_features(outputs: object) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        if isinstance(outputs, torch.Tensor):
            return outputs

        raise TypeError(
            "Model returned an unsupported feature type: "
            f"{type(outputs).__name__}"
        )

    @staticmethod
    def _normalize(features: torch.Tensor) -> np.ndarray:
        normalized = torch.nn.functional.normalize(
            features.float(),
            p=2,
            dim=-1,
        )

        return (
            normalized
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    @staticmethod
    def calibrate_score(raw: float) -> float:
        """Map a raw SigLIP cosine similarity to a 0-1 calibrated range.

        Thresholds are placeholders - replace with benchmark-derived values
        after running the labelled evaluation set.
        """
        _noise_floor = 0.04
        _ceiling = 0.28
        calibrated = (raw - _noise_floor) / (_ceiling - _noise_floor)
        return max(0.0, min(1.0, calibrated))
