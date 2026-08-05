from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from app.strategies.embedding import EmbeddingStrategy


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipEmbeddingStrategy(EmbeddingStrategy):
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        model_version: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = self._resolve_device(device)

        dtype = (
            torch.float16
            if self._device.type == "cuda"
            else torch.float32
        )

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            revision=model_version,
        )

        self._model = CLIPModel.from_pretrained(
            model_name,
            revision=model_version,
            torch_dtype=dtype,
        )

        self._model.to(self._device)
        self._model.eval()

        self._model_version = (
            model_version
            or getattr(
                self._model.config,
                "_commit_hash",
                None,
            )
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def embed_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        if not images:
            raise ValueError(
                "At least one image is required."
            )

        inputs = self._processor(
            images=[
                image.convert("RGB")
                for image in images
            ],
            return_tensors="pt",
        )

        inputs = self._move_to_device(inputs)

        with torch.inference_mode():
            outputs = self._model.get_image_features(
                **inputs
            )

        features = self._extract_features(outputs)

        return self._normalize(features)

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Text may not be empty.")

        inputs = self._processor(
            text=[cleaned_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        inputs = self._move_to_device(inputs)

        with torch.inference_mode():
            outputs = self._model.get_text_features(
                **inputs
            )

        features = self._extract_features(outputs)
        embeddings = self._normalize(features)

        return embeddings[0]

    def _move_to_device(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

    @staticmethod
    def _extract_features(
        outputs: object,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        if isinstance(outputs, torch.Tensor):
            return outputs

        raise TypeError(
            "Model returned an unsupported feature type: "
            f"{type(outputs).__name__}"
        )

    @staticmethod
    def _normalize(
        features: torch.Tensor,
    ) -> np.ndarray:
        if features.ndim == 1:
            features = features.unsqueeze(0)

        if features.ndim != 2:
            raise ValueError(
                "Model features must be a two-dimensional batch."
            )

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
    def _resolve_device(
        requested_device: str | None,
    ) -> torch.device:
        if requested_device is not None:
            return torch.device(requested_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")