from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


PayloadValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector_name: str
    values: np.ndarray
    payload: dict[str, PayloadValue]


@dataclass(frozen=True)
class VectorSearchResult:
    point_id: str
    score: float
    payload: dict[str, PayloadValue]