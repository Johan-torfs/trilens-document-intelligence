from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from app.strategies.analysis import (
    AnalysisOptions,
    AnalysisStrategy,
)


class AnalysisDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    model_name: str
    model_version: str | None
    duration_ms: float


class AnalysisService:
    def __init__(
        self,
        strategy: AnalysisStrategy,
        enabled: bool = False,
        options: AnalysisOptions | None = None,
    ) -> None:
        self._strategy = strategy
        self._enabled = enabled
        self._options = options or AnalysisOptions()

    def analyze(
        self,
        image: Image.Image,
        question: str,
    ) -> AnalysisResult:
        if not self._enabled:
            raise AnalysisDisabledError(
                "OpenFlamingo-analyse is uitgeschakeld."
            )

        started_at = perf_counter()

        text = self._strategy.analyze(
            image=image,
            question=question,
            options=self._options,
        )

        return AnalysisResult(
            text=text,
            model_name=self._strategy.model_name,
            model_version=self._strategy.model_version,
            duration_ms=(
                perf_counter() - started_at
            ) * 1000,
        )