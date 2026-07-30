from pathlib import Path

import pytest

from app.preprocessing.pipeline import preprocess_image
from app.strategies.blip_caption import BlipCaptionStrategy
from app.strategies.caption import CaptionOptions

pytestmark = pytest.mark.model_integration


def test_blip_generates_caption_for_real_document() -> None:
    image_path = Path(
        "data/generated/invoices/invoice_001.png"
    )

    if not image_path.exists():
        pytest.skip(
            "Generated demo dataset is not included "
            "in a clean repository checkout."
        )

    preprocessing_result = preprocess_image(image_path)

    strategy = BlipCaptionStrategy()

    caption = strategy.generate_caption(
        image=preprocessing_result.image,
        options=CaptionOptions(
            max_new_tokens=32,
            num_beams=3,
            do_sample=False,
            prompt_prefix="a document showing",
        ),
    )

    assert isinstance(caption, str)
    assert caption.strip()