import pytest

from app.strategies.caption import CaptionOptions


def test_caption_options_have_safe_defaults() -> None:
    options = CaptionOptions()

    assert options.max_new_tokens == 64
    assert options.num_beams == 3
    assert options.do_sample is False
    assert options.prompt_prefix == ""


def test_caption_options_reject_invalid_max_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_new_tokens moet groter zijn dan nul",
    ):
        CaptionOptions(max_new_tokens=0)


def test_caption_options_reject_invalid_beam_count() -> None:
    with pytest.raises(
        ValueError,
        match="num_beams moet groter zijn dan nul",
    ):
        CaptionOptions(num_beams=0)


def test_caption_options_accept_sampling() -> None:
    options = CaptionOptions(
        max_new_tokens=32,
        num_beams=1,
        do_sample=True,
        prompt_prefix="a document showing",
    )

    assert options.max_new_tokens == 32
    assert options.num_beams == 1
    assert options.do_sample is True
    assert options.prompt_prefix == "a document showing"