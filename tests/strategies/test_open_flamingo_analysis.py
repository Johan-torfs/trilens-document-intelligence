from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.strategies.analysis import AnalysisOptions
from app.strategies.open_flamingo_analysis import (
    OpenFlamingoAnalysisStrategy,
)


def test_model_is_not_loaded_during_initialization() -> None:
    strategy = OpenFlamingoAnalysisStrategy()

    assert strategy.is_loaded is False


def test_empty_question_does_not_load_model() -> None:
    strategy = OpenFlamingoAnalysisStrategy()

    with patch.object(
        strategy,
        "_ensure_loaded",
    ) as ensure_loaded:
        with pytest.raises(
            ValueError,
            match="analysevraag mag niet leeg zijn",
        ):
            strategy.analyze(
                image=Image.new("RGB", (100, 100), "white"),
                question="   ",
                options=AnalysisOptions(),
            )

    ensure_loaded.assert_not_called()


def test_analyze_loads_model_and_decodes_answer() -> None:
    strategy = OpenFlamingoAnalysisStrategy()

    model = MagicMock()
    image_processor = MagicMock()
    tokenizer = MagicMock()

    image_processor.return_value = torch.zeros(
        (3, 224, 224),
        dtype=torch.float32,
    )

    tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 20, 30]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    model.generate.return_value = torch.tensor(
        [[10, 20, 30, 40, 50]]
    )

    tokenizer.decode.return_value = (
        "  a signature is visible  "
    )

    def load_fake_resources() -> None:
        strategy._model = model
        strategy._image_processor = image_processor
        strategy._tokenizer = tokenizer

    with patch.object(
        strategy,
        "_ensure_loaded",
        side_effect=load_fake_resources,
    ) as ensure_loaded:
        answer = strategy.analyze(
            image=Image.new("RGB", (100, 100), "white"),
            question="Is there a signature?",
            options=AnalysisOptions(
                max_new_tokens=32,
                num_beams=2,
            ),
        )

    assert answer == "a signature is visible"
    ensure_loaded.assert_called_once()

    vision_x = model.generate.call_args.kwargs["vision_x"]

    assert vision_x.shape == (
        1,
        1,
        1,
        3,
        224,
        224,
    )

    model.generate.assert_called_once()

    generate_arguments = model.generate.call_args.kwargs

    assert generate_arguments["vision_x"].shape == (
        1,
        1,
        1,
        3,
        224,
        224,
    )

    torch.testing.assert_close(
        generate_arguments["lang_x"].cpu(),
        torch.tensor([[10, 20, 30]]),
    )

    torch.testing.assert_close(
        generate_arguments["attention_mask"].cpu(),
        torch.tensor([[1, 1, 1]]),
    )

    assert generate_arguments["max_new_tokens"] == 32
    assert generate_arguments["num_beams"] == 2
    assert generate_arguments["repetition_penalty"] == 1.15
    assert generate_arguments["no_repeat_ngram_size"] == 3

    tokenizer.decode.assert_called_once()

    decode_arguments = tokenizer.decode.call_args

    decoded_ids = decode_arguments.args[0]
    decode_options = decode_arguments.kwargs

    torch.testing.assert_close(
        decoded_ids.cpu(),
        torch.tensor([40, 50]),
    )

    assert decode_options["skip_special_tokens"] is True

def test_analyze_reports_understandable_memory_error() -> None:
    strategy = OpenFlamingoAnalysisStrategy()

    with patch.object(
        strategy,
        "_ensure_loaded",
        side_effect=torch.cuda.OutOfMemoryError(),
    ):
        with pytest.raises(
            RuntimeError,
            match="Onvoldoende GPU-geheugen",
        ):
            strategy.analyze(
                image=Image.new("RGB", (100, 100), "white"),
                question="Describe this document.",
                options=AnalysisOptions(),
            )


def test_memory_failure_prevents_repeated_model_loading() -> None:
    strategy = OpenFlamingoAnalysisStrategy()
    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    with patch.object(
        strategy,
        "_ensure_loaded",
        side_effect=torch.cuda.OutOfMemoryError(),
    ) as ensure_loaded:
        for _ in range(2):
            with pytest.raises(
                RuntimeError,
                match="Onvoldoende GPU-geheugen",
            ):
                strategy.analyze(
                    image=image,
                    question="Describe this document.",
                    options=AnalysisOptions(),
                )

    ensure_loaded.assert_called_once()