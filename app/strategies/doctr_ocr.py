from importlib.metadata import version

import numpy as np
from doctr.models import ocr_predictor
from PIL import Image

from app.domain.ocr import (
    OCRBoundingBox,
    OCRPageResult,
    OCRResult,
    OCRWord,
)
from app.domain.prepared_document import DocumentPage
from app.strategies.ocr import OCRStrategy


class DocTROCRStrategy(OCRStrategy):
    DETECTION_MODEL = "fast_base"
    RECOGNITION_MODEL = "crnn_vgg16_bn"

    def __init__(self) -> None:
        self._predictor = ocr_predictor(
            det_arch=self.DETECTION_MODEL,
            reco_arch=self.RECOGNITION_MODEL,
            pretrained=True,
            assume_straight_pages=True,
            export_as_straight_boxes=True,
        )

    @property
    def model_name(self) -> str:
        return (
            f"doctr:"
            f"{self.DETECTION_MODEL}+"
            f"{self.RECOGNITION_MODEL}"
        )

    @property
    def model_version(self) -> str:
        return version("python-doctr")

    def extract(self, pages: list[DocumentPage]) -> OCRResult:
        page_arrays = [
            np.asarray(page.image.convert("RGB"))
            for page in pages
        ]

        prediction = self._predictor(page_arrays)
        exported = prediction.export()

        page_results: list[OCRPageResult] = []

        for doc_page, page_data in zip(pages, exported["pages"]):
            words: list[OCRWord] = []
            text_blocks: list[str] = []

            for block in page_data["blocks"]:
                block_lines: list[str] = []

                for line in block["lines"]:
                    line_words: list[str] = []

                    for word in line["words"]:
                        value = word["value"].strip()

                        if not value:
                            continue

                        (left, top), (right, bottom) = (
                            word["geometry"]
                        )

                        words.append(
                            OCRWord(
                                text=value,
                                confidence=float(
                                    word["confidence"]
                                ),
                                bounding_box=OCRBoundingBox(
                                    left=float(left),
                                    top=float(top),
                                    right=float(right),
                                    bottom=float(bottom),
                                ),
                            )
                        )

                        line_words.append(value)

                    if line_words:
                        block_lines.append(
                            " ".join(line_words)
                        )

                if block_lines:
                    text_blocks.append(
                        "\n".join(block_lines)
                    )

            page_text = "\n\n".join(text_blocks)
            page_confidence = (
                sum(w.confidence for w in words) / len(words)
                if words
                else 0.0
            )

            page_results.append(
                OCRPageResult(
                    page_number=doc_page.page_number,
                    text=page_text,
                    words=words,
                    mean_confidence=page_confidence,
                )
            )

        all_words = [w for p in page_results for w in p.words]
        document_confidence = (
            sum(w.confidence for w in all_words) / len(all_words)
            if all_words
            else 0.0
        )

        return OCRResult(
            text="\n\n".join(
                p.text for p in page_results if p.text
            ),
            pages=page_results,
            mean_confidence=document_confidence,
        )
        page = np.asarray(image.convert("RGB"))

        prediction = self._predictor([page])
        exported = prediction.export()

        words: list[OCRWord] = []
        text_blocks: list[str] = []

        for page_result in exported["pages"]:
            for block in page_result["blocks"]:
                block_lines: list[str] = []

                for line in block["lines"]:
                    line_words: list[str] = []

                    for word in line["words"]:
                        value = word["value"].strip()

                        if not value:
                            continue

                        (left, top), (right, bottom) = (
                            word["geometry"]
                        )

                        words.append(
                            OCRWord(
                                text=value,
                                confidence=float(
                                    word["confidence"]
                                ),
                                bounding_box=OCRBoundingBox(
                                    left=float(left),
                                    top=float(top),
                                    right=float(right),
                                    bottom=float(bottom),
                                ),
                            )
                        )

                        line_words.append(value)

                    if line_words:
                        block_lines.append(
                            " ".join(line_words)
                        )

                if block_lines:
                    text_blocks.append(
                        "\n".join(block_lines)
                    )

        mean_confidence = (
            sum(word.confidence for word in words)
            / len(words)
            if words
            else 0.0
        )

        return OCRResult(
            text="\n\n".join(text_blocks),
            words=words,
            mean_confidence=mean_confidence,
        )