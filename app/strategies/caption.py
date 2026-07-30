from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class CaptionOptions:
    max_new_tokens: int = 64
    num_beams: int = 3
    do_sample: bool = False
    prompt_prefix: str = ""

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens moet groter zijn dan nul."
            )

        if self.num_beams <= 0:
            raise ValueError(
                "num_beams moet groter zijn dan nul."
            )


class CaptionStrategy(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str | None:
        ...

    @abstractmethod
    def generate_caption(
        self,
        image: Image.Image,
        options: CaptionOptions,
    ) -> str:
        ...