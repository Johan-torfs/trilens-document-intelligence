from app.domain.ocr import OCRPageResult, OCRResult
from app.services.text_chunker import TextChunker


def make_ocr_result(page_texts: list[str]) -> OCRResult:
    pages = [
        OCRPageResult(
            page_number=index + 1,
            text=text,
            words=[],
            mean_confidence=0.9,
        )
        for index, text in enumerate(page_texts)
    ]

    return OCRResult(
        text="\n\n".join(p.text for p in pages),
        pages=pages,
        mean_confidence=0.9,
        model_name="doctr",
        model_version="1",
    )


def test_chunks_single_page_by_paragraphs() -> None:
    ocr = make_ocr_result(
        ["Invoice number 1234\n\nTotal amount 99.95\n\nThank you for your payment"]
    )

    chunks = TextChunker().chunk(ocr)

    assert len(chunks) == 3
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_number == 0
    assert chunks[0].text == "Invoice number 1234"
    assert chunks[1].chunk_number == 1
    assert chunks[2].chunk_number == 2


def test_chunks_multiple_pages_independently() -> None:
    ocr = make_ocr_result(
        [
            "Page one text block",
            "Page two text block\n\nSecond block on page two",
        ]
    )

    chunks = TextChunker().chunk(ocr)

    assert len(chunks) == 3

    page_one = [c for c in chunks if c.page_number == 1]
    page_two = [c for c in chunks if c.page_number == 2]

    assert len(page_one) == 1
    assert len(page_two) == 2

    # chunk_number resets per page
    assert page_two[0].chunk_number == 0
    assert page_two[1].chunk_number == 1


def test_discards_chunks_below_min_words() -> None:
    ocr = make_ocr_result(["OK\n\nThis is a valid paragraph"])

    chunks = TextChunker(min_words=3).chunk(ocr)

    assert len(chunks) == 1
    assert chunks[0].text == "This is a valid paragraph"


def test_empty_ocr_produces_no_chunks() -> None:
    ocr = make_ocr_result([""])

    assert TextChunker().chunk(ocr) == []


def test_whitespace_only_paragraphs_are_discarded() -> None:
    ocr = make_ocr_result(["   \n\n  \n\nActual content here"])

    chunks = TextChunker().chunk(ocr)

    assert len(chunks) == 1
    assert chunks[0].text == "Actual content here"
