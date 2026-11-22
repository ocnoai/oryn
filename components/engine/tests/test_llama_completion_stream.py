import pytest

from oryn_engine.runtime_packs.llama_cpp_worker import CompletionText


@pytest.mark.parametrize("chunk_size", range(1, 25))
def test_native_completion_footer_never_leaks_into_text_deltas(chunk_size):
    text = "A literal [end of text] can occur inside generated text."
    raw = text + CompletionText.footer
    stream = CompletionText()
    pieces = [stream.feed(raw[i : i + chunk_size]) for i in range(0, len(raw), chunk_size)]
    pieces.append(stream.finish())
    assert "".join(pieces) == text


def test_token_limit_output_is_not_mistaken_for_native_eog():
    text = "literal [end of text]"
    stream = CompletionText()
    assert stream.feed(text) + stream.finish() == text
