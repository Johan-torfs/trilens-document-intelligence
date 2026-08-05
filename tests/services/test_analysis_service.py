import pytest
from PIL import Image

from app.services.analysis_service import (
    AnalysisDisabledError,
    AnalysisService,
)
from app.strategies.analysis import (
    AnalysisOptions,
    AnalysisStrategy,
)


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
    assert result.model_name == "fake-open-flamingo"
    assert result.duration_ms >= 0


def test_analysis_raises_runtime_error_on_failure() -> None:
    service = AnalysisService(
        strategy=FakeAnalysisStrategy(should_fail=True),
        enabled=True,
    )

    with pytest.raises(
        RuntimeError,
        match="Model kon niet",
    ):
        service.analyze(
            image=Image.new("RGB", (100, 100), "white"),
            question="Is there a signature?",
        )