from app.domain.ocr import OCRResult
from app.domain.text_chunk import TextChunk


MIN_CHUNK_WORDS = 3
MAX_CHUNK_WORDS = 80


class TextChunker:
    """Splits OCR page text into paragraph-level chunks.

    Splits first on docTR block boundaries (double newline), then if a
    block exceeds max_words, splits further on single newlines and groups
    lines greedily to stay within the word limit.
    """

    def __init__(
        self,
        min_words: int = MIN_CHUNK_WORDS,
        max_words: int = MAX_CHUNK_WORDS,
    ) -> None:
        self._min_words = min_words
        self._max_words = max_words

    def chunk(self, ocr_result: OCRResult) -> list[TextChunk]:
        chunks: list[TextChunk] = []

        for page in ocr_result.pages:
            chunk_number = 0

            for block_text in page.text.split("\n\n"):
                cleaned = block_text.strip()

                if not cleaned:
                    continue

                words = cleaned.split()

                if len(words) <= self._max_words:
                    if len(words) >= self._min_words:
                        chunks.append(
                            TextChunk(
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                text=cleaned,
                            )
                        )
                        chunk_number += 1
                else:
                    lines = [
                        line.strip()
                        for line in cleaned.split("\n")
                        if line.strip()
                    ]

                    accumulator: list[str] = []
                    word_count = 0

                    for line in lines:
                        line_words = len(line.split())

                        if (
                            word_count + line_words > self._max_words
                            and accumulator
                        ):
                            sub_text = "\n".join(accumulator)
                            if len(sub_text.split()) >= self._min_words:
                                chunks.append(
                                    TextChunk(
                                        page_number=page.page_number,
                                        chunk_number=chunk_number,
                                        text=sub_text,
                                    )
                                )
                                chunk_number += 1
                            accumulator = [line]
                            word_count = line_words
                        else:
                            accumulator.append(line)
                            word_count += line_words

                    if accumulator:
                        sub_text = "\n".join(accumulator)
                        if len(sub_text.split()) >= self._min_words:
                            chunks.append(
                                TextChunk(
                                    page_number=page.page_number,
                                    chunk_number=chunk_number,
                                    text=sub_text,
                                )
                            )
                            chunk_number += 1

        return chunks
