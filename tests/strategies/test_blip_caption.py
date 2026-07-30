from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from app.strategies.blip_caption import BlipCaptionStrategy
from app.strategies.caption import CaptionOptions


@patch(
    "app.strategies.blip_caption."
    "BlipForConditionalGeneration.from_pretrained"
)
@patch(
    "app.strategies.blip_caption.AutoProcessor.from_pretrained"
)
def test_generate_caption_decodes_generated_tokens(
    processor_loader: MagicMock,
    model_loader: MagicMock,
) -> None:
    processor = MagicMock()
    model = MagicMock()

    processor_loader.return_value = processor
    model_loader.return_value = model

    inputs = MagicMock()
    inputs.to.return_value = inputs
    processor.return_value = inputs

    generated_ids = torch.tensor([[101, 200, 102]])
    model.generate.return_value = generated_ids

    processor.batch_decode.return_value = [
        "  a document with a table  "
    ]

    strategy = BlipCaptionStrategy()
    options = CaptionOptions(
        max_new_tokens=32,
        num_beams=2,
        do_sample=False,
    )

    caption = strategy.generate_caption(
        image=Image.new("RGB", (100, 100), "white"),
        options=options,
    )

    assert caption == "a document with a table"

    model.generate.assert_called_once_with(
        **inputs,
        max_new_tokens=32,
        num_beams=2,
        do_sample=False,
    )

    processor.batch_decode.assert_called_once_with(
        generated_ids,
        skip_special_tokens=True,
    )