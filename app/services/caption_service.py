from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from app.strategies.caption import CaptionOptions, CaptionStrategy


class CaptionGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    model_name: str
    model_version: str | None
    duration_ms: float


class CaptionService:
    def __init__(
        self,
        strategy: CaptionStrategy,
        options: CaptionOptions | None = None,
    ) -> None:
        self._strategy = strategy
        self._options = options or CaptionOptions()

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    @property
    def model_version(self) -> str | None:
        return self._strategy.model_version

    def generate(self, image: Image.Image) -> CaptionResult:
        started_at = perf_counter()

        caption = self._strategy.generate_caption(
            image=image,
            options=self._options,
        ).strip()

        duration_ms = (perf_counter() - started_at) * 1000

        if not caption:
            raise CaptionGenerationError(
                "Het captionmodel retourneerde geen bruikbare tekst."
            )

        return CaptionResult(
            caption=caption,
            model_name=self._strategy.model_name,
            model_version=self._strategy.model_version,
            duration_ms=duration_ms,
        )