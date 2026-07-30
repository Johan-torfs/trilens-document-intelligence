import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

from app.strategies.caption import CaptionOptions, CaptionStrategy


class BlipCaptionStrategy(CaptionStrategy):
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        model_version: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._device = self._detect_device()

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            revision=model_version,
        )
        self._model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            revision=model_version,
        )

        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def generate_caption(
        self,
        image: Image.Image,
        options: CaptionOptions,
    ) -> str:
        processor_arguments: dict[str, object] = {
            "images": image,
            "return_tensors": "pt",
        }

        if options.prompt_prefix:
            processor_arguments["text"] = options.prompt_prefix

        inputs = self._processor(
            **processor_arguments,
        ).to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=options.max_new_tokens,
                num_beams=options.num_beams,
                do_sample=options.do_sample,
            )

        caption = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return caption.strip()

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")