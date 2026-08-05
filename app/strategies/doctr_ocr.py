from importlib.metadata import version
from typing import Sequence

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

    def extract(
        self,
        pages: Sequence[DocumentPage],
    ) -> OCRResult:
        if not pages:
            raise ValueError(
                "At least one page is required for OCR."
            )

        page_arrays = [
            np.asarray(page.image.convert("RGB"))
            for page in pages
        ]

        prediction = self._predictor(page_arrays)
        exported = prediction.export()

        exported_pages = exported.get("pages", [])

        if len(exported_pages) != len(pages):
            raise RuntimeError(
                "docTR returned an unexpected number of pages."
            )

        page_results: list[OCRPageResult] = []

        for document_page, page_data in zip(
            pages,
            exported_pages,
            strict=True,
        ):
            words: list[OCRWord] = []
            text_blocks: list[str] = []

            for block in page_data.get("blocks", []):
                block_lines: list[str] = []

                for line in block.get("lines", []):
                    line_words: list[str] = []

                    for word in line.get("words", []):
                        value = str(
                            word.get("value", "")
                        ).strip()

                        if not value:
                            continue

                        geometry = word["geometry"]
                        (left, top), (right, bottom) = geometry

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

            mean_confidence = (
                sum(word.confidence for word in words)
                / len(words)
                if words
                else 0.0
            )

            page_results.append(
                OCRPageResult(
                    page_number=document_page.page_number,
                    text=page_text,
                    words=words,
                    mean_confidence=mean_confidence,
                )
            )

        all_words = [
            word
            for page in page_results
            for word in page.words
        ]

        document_confidence = (
            sum(word.confidence for word in all_words)
            / len(all_words)
            if all_words
            else 0.0
        )

        return OCRResult(
            text="\n\n".join(
                page.text
                for page in page_results
                if page.text
            ),
            pages=page_results,
            mean_confidence=document_confidence,
            model_name=self.model_name,
            model_version=self.model_version,
        )