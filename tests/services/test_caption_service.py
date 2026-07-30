import pytest
from PIL import Image

from app.services.caption_service import (
    CaptionGenerationError,
    CaptionService,
)
from app.strategies.caption import CaptionOptions, CaptionStrategy


class FakeCaptionStrategy(CaptionStrategy):
    def __init__(self, caption: str) -> None:
        self._caption = caption
        self.received_options: CaptionOptions | None = None

    @property
    def model_name(self) -> str:
        return "fake-caption-model"

    @property
    def model_version(self) -> str | None:
        return "test-version"

    def generate_caption(
        self,
        image: Image.Image,
        options: CaptionOptions,
    ) -> str:
        self.received_options = options
        return self._caption


def test_generate_returns_caption_metadata() -> None:
    strategy = FakeCaptionStrategy(
        "  an invoice with several product rows  "
    )
    options = CaptionOptions(max_new_tokens=32)

    service = CaptionService(
        strategy=strategy,
        options=options,
    )

    result = service.generate(
        Image.new("RGB", (100, 100), "white")
    )

    assert result.caption == "an invoice with several product rows"
    assert result.model_name == "fake-caption-model"
    assert result.model_version == "test-version"
    assert result.duration_ms >= 0
    assert strategy.received_options is options


def test_generate_rejects_empty_caption() -> None:
    service = CaptionService(
        strategy=FakeCaptionStrategy("   ")
    )

    with pytest.raises(
        CaptionGenerationError,
        match="geen bruikbare tekst",
    ):
        service.generate(
            Image.new("RGB", (100, 100), "white")
        )