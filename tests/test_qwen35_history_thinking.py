"""Qwen3.5 bridge over a reply that stopped inside its think block.

A text-game rollout (hundreds of one-reply turns, every observation a user message)
must keep each prompt an exact continuation of what the engine's streaming session
holds. A reply that stopped inside its think block made the bridge refuse; the
caller's full re-render then re-tokenized history (and, under the template's rule,
dropped every historical think block), so the engine saw a forked prompt and
reopened the session.
"""

from functools import lru_cache

from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import config_from_name

MODEL = "Qwen/Qwen3.5-4B"


@lru_cache(None)
def _renderer(retention="all"):
    tok = load_tokenizer(MODEL)
    config = config_from_name("qwen3.5").model_copy(
        update={"enable_thinking": True, "thinking_retention": retention}
    )
    return tok, create_renderer(tok, config)


def _encode(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


def _sample(tok, renderer, text):
    """A sampled completion: the model's text after the ``<think>\\n`` generation prompt, then its stop."""
    return _encode(tok, text) + [renderer._im_end]


def test_bridge_keeps_a_reply_that_stopped_inside_its_think_block():
    tok, renderer = _renderer()
    messages = [{"role": "system", "content": "Play."}, {"role": "user", "content": "obs 1"}]
    prompt = renderer.render_ids(messages, add_generation_prompt=True)
    completion = _sample(tok, renderer, "I should chop. <action>DO</action>")  # no </think>
    parsed = renderer.parse_response(completion, prompt_ids=prompt)
    assert parsed.content == "" and parsed.reasoning_complete is False
    bridged = renderer.bridge_to_next_turn(prompt, completion, [{"role": "user", "content": "obs 2"}])
    assert bridged is not None
    assert bridged.token_ids[: len(prompt) + len(completion)] == prompt + completion
    assert not any(bridged.sampled_mask[len(prompt) + len(completion) :])
    assert tok.decode(bridged.token_ids[len(prompt) + len(completion) :]).startswith("\n<|im_start|>user\nobs 2")


def test_bridge_still_closes_a_reply_cut_off_inside_its_think_block():
    tok, renderer = _renderer()
    prompt = renderer.render_ids([{"role": "user", "content": "obs 1"}], add_generation_prompt=True)
    truncated = _encode(tok, "still thinking")  # max_tokens hit: no stop
    bridged = renderer.bridge_to_next_turn(prompt, truncated, [{"role": "user", "content": "obs 2"}])
    assert bridged is not None
    tail = bridged.token_ids[len(prompt) + len(truncated) :]
    assert tail[:2] == [renderer._think_end, renderer._im_end]
