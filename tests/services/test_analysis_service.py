import pytest
from PIL import Image

from app.services.analysis_service import (
    AnalysisDisabledError,
    AnalysisService,
)
from app.services.caption_service import CaptionService
from app.strategies.analysis import (
    AnalysisOptions,
    AnalysisStrategy,
)
from app.strategies.caption import CaptionOptions, CaptionStrategy


class FakeAnalysisStrategy(AnalysisStrategy):
    def __init__(
        self,
        answer: str = "a signature is visible",
        should_fail: bool = False,
    ) -> None:
        self.answer = answer
        self.should_fail = should_fail
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-open-flamingo"

    @property
    def model_version(self) -> str | None:
        return "version-1"

    def analyze(
        self,
        image: Image.Image,
        question: str,
        options: AnalysisOptions,
    ) -> str:
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Model kon niet worden uitgevoerd.")

        return self.answer


class FakeCaptionStrategy(CaptionStrategy):
    @property
    def model_name(self) -> str:
        return "fake-blip"

    @property
    def model_version(self) -> str | None:
        return "version-1"

    def generate_caption(
        self,
        image: Image.Image,
        options: CaptionOptions,
    ) -> str:
        return "a document with several rows"


def test_analysis_is_disabled_by_default() -> None:
    strategy = FakeAnalysisStrategy()
    service = AnalysisService(strategy=strategy)

    with pytest.raises(
        AnalysisDisabledError,
        match="uitgeschakeld",
    ):
        service.analyze(
            image=Image.new("RGB", (100, 100), "white"),
            question="Is there a signature?",
        )

    assert strategy.call_count == 0


def test_analysis_returns_model_generated_answer() -> None:
    service = AnalysisService(
        strategy=FakeAnalysisStrategy(),
        enabled=True,
    )

    result = service.analyze(
        image=Image.new("RGB", (100, 100), "white"),
        question="Is there a signature?",
    )

    assert result.text == "a signature is visible"
    assert result.source == "open_flamingo"
    assert result.model_name == "fake-open-flamingo"
    assert result.duration_ms >= 0


def test_analysis_uses_caption_fallback_on_failure() -> None:
    caption_service = CaptionService(
        strategy=FakeCaptionStrategy()
    )

    service = AnalysisService(
        strategy=FakeAnalysisStrategy(should_fail=True),
        enabled=True,
        fallback_caption_service=caption_service,
    )

    result = service.analyze(
        image=Image.new("RGB", (100, 100), "white"),
        question="Is there a signature?",
    )

    assert result.text == "a document with several rows"
    assert result.source == "caption_fallback"
    assert result.model_name == "fake-blip"