import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from app.strategies.retrieval import RetrievalStrategy


class ClipRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self._model_name = model_name
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model.get_image_features(**inputs)

        features = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            else outputs
        )

        return self._normalize(features[0])

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model.get_text_features(**inputs)

        features = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            else outputs
        )

        return self._normalize(features[0])

    @staticmethod
    def _normalize(features: torch.Tensor) -> np.ndarray:
        norm = features.norm(p=2).clamp_min(1e-12)
        normalized = features / norm

        return normalized.cpu().numpy().astype(np.float32)

    @staticmethod
    def calibrate_score(raw: float) -> float:
        """Map a raw CLIP cosine similarity to a 0-1 calibrated range.

        Thresholds are placeholders - replace with benchmark-derived values
        after running the labelled evaluation set.
        """
        _noise_floor = 0.15
        _ceiling = 0.35
        calibrated = (raw - _noise_floor) / (_ceiling - _noise_floor)
        return max(0.0, min(1.0, calibrated))