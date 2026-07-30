from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class AnalysisOptions:
    max_new_tokens: int = 64
    min_new_tokens: int = 16
    num_beams: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens moet groter zijn dan nul."
            )

        if self.min_new_tokens < 0:
            raise ValueError(
                "min_new_tokens mag niet negatief zijn."
            )

        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens mag niet groter zijn "
                "dan max_new_tokens."
            )

        if self.num_beams <= 0:
            raise ValueError(
                "num_beams moet groter zijn dan nul."
            )


class AnalysisStrategy(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str | None:
        ...

    @abstractmethod
    def analyze(
        self,
        image: Image.Image,
        question: str,
        options: AnalysisOptions,
    ) -> str:
        ...