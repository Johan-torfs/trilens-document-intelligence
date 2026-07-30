from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from app.services.caption_service import CaptionService
from app.strategies.analysis import AnalysisOptions, AnalysisStrategy


class AnalysisDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    source: str
    model_name: str
    model_version: str | None
    duration_ms: float


class AnalysisService:
    def __init__(
        self,
        strategy: AnalysisStrategy,
        enabled: bool = False,
        options: AnalysisOptions | None = None,
        fallback_caption_service: CaptionService | None = None,
    ) -> None:
        self._strategy = strategy
        self._enabled = enabled
        self._options = options or AnalysisOptions()
        self._fallback_caption_service = fallback_caption_service

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

        try:
            text = self._strategy.analyze(
                image=image,
                question=question,
                options=self._options,
            )

            return AnalysisResult(
                text=text,
                source="open_flamingo",
                model_name=self._strategy.model_name,
                model_version=self._strategy.model_version,
                duration_ms=(perf_counter() - started_at) * 1000,
            )

        except RuntimeError:
            if self._fallback_caption_service is None:
                raise

            caption_result = (
                self._fallback_caption_service.generate(image)
            )

            return AnalysisResult(
                text=caption_result.caption,
                source="caption_fallback",
                model_name=caption_result.model_name,
                model_version=caption_result.model_version,
                duration_ms=(perf_counter() - started_at) * 1000,
            )