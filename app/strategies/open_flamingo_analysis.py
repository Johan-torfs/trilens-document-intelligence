from typing import Any

import torch
from PIL import Image
import gc

from app.strategies.analysis import (
    AnalysisOptions,
    AnalysisStrategy,
)


class OpenFlamingoAnalysisError(RuntimeError):
    pass


class OpenFlamingoAnalysisStrategy(AnalysisStrategy):
    def __init__(
        self,
        checkpoint_repository: str = (
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b"
        ),
        checkpoint_filename: str = "checkpoint.pt",
        model_version: str | None = None,
        device: str | None = None,
    ) -> None:
        self._checkpoint_repository = checkpoint_repository
        self._checkpoint_filename = checkpoint_filename
        self._model_version = model_version

        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Deze resources blijven leeg tot analyze() wordt aangeroepen.
        self._model: Any | None = None
        self._image_processor: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None

    @property
    def model_name(self) -> str:
        return self._checkpoint_repository

    @property
    def model_version(self) -> str | None:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def analyze(
        self,
        image: Image.Image,
        question: str,
        options: AnalysisOptions,
    ) -> str:
        if self._load_error is not None:
            raise OpenFlamingoAnalysisError(
                self._load_error
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("De analysevraag mag niet leeg zijn.")

        try:
            self._ensure_loaded()
        except torch.cuda.OutOfMemoryError as error:
            message = (
                "Onvoldoende GPU-geheugen om OpenFlamingo te laden. "
                "De captionfallback wordt gebruikt."
            )

            self._mark_unavailable(message)

            raise OpenFlamingoAnalysisError(
                message
            ) from error

        assert self._model is not None
        assert self._image_processor is not None
        assert self._tokenizer is not None

        vision_x = self._image_processor(image)
        vision_x = vision_x.unsqueeze(0)
        vision_x = vision_x.unsqueeze(0)
        vision_x = vision_x.unsqueeze(0)
        vision_x = vision_x.to(self._device)

        prompt = (
            "<image>"
            "Carefully inspect this document image. "
            f"Question: {cleaned_question} "
            "Answer in one complete paragraph of two to four sentences. "
            "Describe only visually observable information. "
            "Do not use a numbered list. "
            "If the answer is uncertain, say so. "
            "Answer:"
        )

        self._tokenizer.padding_side = "left"

        language_inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
        )

        input_ids = language_inputs["input_ids"].to(
            self._device
        )
        attention_mask = language_inputs[
            "attention_mask"
        ].to(self._device)

        try:
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    vision_x=vision_x,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=options.max_new_tokens,
                    min_new_tokens=options.min_new_tokens,
                    num_beams=options.num_beams,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
        except torch.cuda.OutOfMemoryError as error:
            message = (
                "Onvoldoende GPU-geheugen om de "
                "OpenFlamingo-analyse uit te voeren. "
                "De captionfallback wordt gebruikt."
            )

            self._mark_unavailable(message)

            raise OpenFlamingoAnalysisError(
                message
            ) from error

        prompt_length = input_ids.shape[1]
        output_ids = generated_ids[0]

        if output_ids.shape[0] > prompt_length:
            output_ids = output_ids[prompt_length:]

        answer = self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise OpenFlamingoAnalysisError(
                "OpenFlamingo retourneerde geen bruikbare analyse."
            )

        return answer

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms

        model, image_processor, tokenizer = (
            create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=(
                    "anas-awadalla/"
                    "mpt-1b-redpajama-200b"
                ),
                tokenizer_path=(
                    "anas-awadalla/"
                    "mpt-1b-redpajama-200b"
                ),
                cross_attn_every_n_layers=1,
            )
        )

        checkpoint_path = hf_hub_download(
            repo_id=self._checkpoint_repository,
            filename=self._checkpoint_filename,
            revision=self._model_version,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint,
            strict=False,
        )

        model.to(self._device)
        model.eval()

        self._model = model
        self._image_processor = image_processor
        self._tokenizer = tokenizer

    def _mark_unavailable(
        self,
        message: str,
    ) -> None:
        self._model = None
        self._image_processor = None
        self._tokenizer = None
        self._load_error = message

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()