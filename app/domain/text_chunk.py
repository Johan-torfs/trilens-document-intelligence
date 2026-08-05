from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_number: int
    text: str
