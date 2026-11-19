#!/usr/bin/env python3
"""Closed VLMs as analyst and answerer of the harness: agent.py --model gpt-6-luna | claude-haiku-5-5.

The local model's calls go to the API instead, with the same messages. __call__ is Chat's interface (the object list
and the analyst; text only in --blind runs). answer() sends each final answer exactly as that model's direct-input runner
sends it (answer_gpt.py: Responses API, reasoning effort none, 16 output tokens; answer_claude.py: Messages API, its
THINKING settings, a letter-only instruction instead of a prefill), so the harness rows compare with the model's
image-only and plain rows of Table 1. A request the API rejects (400, e.g. an image over its size limit) makes only
that answer run wrong, with an "error" field, as answer_claude.py records it; any other API error fails the question
(the run's redo pass retries it). Only image latent formats work over an API (no padding tokens), and no --think.

Keys: OPENAI_API_KEY or ~/.config/openai_key; ANTHROPIC_API_KEY or ~/.config/anthropic_key (never printed).
Another API (Gemini): a class with __call__ and answer() like the two below, and its prefix in PROVIDERS.
"""
import os
from pathlib import Path

IMAGE_FORMATS = ("image", "imagez", "tilez")
EMPTY_REPLY = "(no reply)"  # an analyst reply that was empty, sent back in the history (APIs reject empty text)


def _key(env, path):
    p = Path(path).expanduser()
    key = os.environ.get(env) or (p.read_text().strip() if p.exists() else None)
    if not key:
        raise SystemExit(f"no API key: set {env} or write it to {path}")
    return key


def _text(content):
    return content if isinstance(content, str) else "".join(c["text"] for c in content if c["type"] == "text")


def _failed(e):
    return "", 0, {"output_tokens": 0, "error": str(e)[:300]}


class OpenAIChat:
    def __init__(self, model, client=None, effort="none", bad_request=None):
        if client is None:
            import openai

            client = openai.OpenAI(api_key=_key("OPENAI_API_KEY", "~/.config/openai_key"), max_retries=6)
            bad_request = openai.BadRequestError
        self.client, self.model, self.effort = client, model, effort
        self.bad_request = bad_request or type("NoError", (Exception,), {})

    def __call__(self, messages, max_new_tokens, think=False, prefix=""):
        from answer_gpt import to_data_url

        assert not (prefix or think), "no prefill or thinking over an API"
        items = []
        for m in messages:
            if isinstance(m["content"], str) or m["role"] == "assistant":
                items.append({"role": m["role"], "content": _text(m["content"]) or EMPTY_REPLY})
            else:
                items.append({"role": m["role"], "content": [
                    {"type": "input_image", "image_url": to_data_url(c["image"])} if c["type"] == "image"
                    else {"type": "input_text", "text": c["text"]} for c in m["content"]]})
        resp = self.client.responses.create(model=self.model, input=items, reasoning={"effort": self.effort},
                                            max_output_tokens=max_new_tokens)
        return (resp.output_text or "").strip(), resp.usage.input_tokens

    def answer(self, content):
        """(reply, prompt tokens, extra fields of the result row)."""
        from answer_gpt import to_input

        try:
            resp = self.client.responses.create(model=self.model, input=to_input(content),
                                                reasoning={"effort": self.effort}, max_output_tokens=16)
        except self.bad_request as e:
            return _failed(e)
        return (resp.output_text or "").strip(), resp.usage.input_tokens, {"output_tokens": resp.usage.output_tokens}


class ClaudeChat:
    def __init__(self, model, client=None, bad_request=None):
        from answer_claude import THINKING

        if model not in THINKING:
            raise SystemExit(f"{model}: answer_claude.py has no settings for it (one of {sorted(THINKING)})")
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "~/.config/anthropic_key"), max_retries=6)
            bad_request = anthropic.BadRequestError
        self.client, self.model, self.params = client, model, THINKING[model]
        self.bad_request = bad_request or type("NoError", (Exception,), {})

    def __call__(self, messages, max_new_tokens, think=False, prefix=""):
        from answer_claude import image_block

        assert not (prefix or think), "no prefill or thinking over an API"
        system = "\n\n".join(_text(m["content"]) for m in messages if m["role"] == "system")
        msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            if isinstance(m["content"], str) or m["role"] == "assistant":
                parts = [{"type": "text", "text": _text(m["content"]) or EMPTY_REPLY}]
            else:
                parts = [image_block(c["image"]) if c["type"] == "image" else {"type": "text", "text": c["text"]}
                         for c in m["content"]]
            msgs.append({"role": m["role"], "content": parts})
        # Haiku (thinking disabled) and Sonnet (no thinking before the answer) spend max_tokens on the reply only;
        # Opus thinks first, so its thinking allowance comes on top.
        extra = 0 if "thinking" in self.params else self.params["max_tokens"]
        resp = self.client.messages.create(model=self.model, messages=msgs, **({"system": system} if system else {}),
                                           **{**self.params, "max_tokens": max_new_tokens + extra})
        return "".join(b.text for b in resp.content if b.type == "text").strip(), resp.usage.input_tokens

    def answer(self, content):
        """(reply, prompt tokens, extra fields of the result row)."""
        from answer_claude import to_messages

        try:
            resp = self.client.messages.create(model=self.model, messages=to_messages(content), **self.params)
        except self.bad_request as e:
            return _failed(e)
        return ("".join(b.text for b in resp.content if b.type == "text").strip(), resp.usage.input_tokens,
                {"output_tokens": resp.usage.output_tokens, "stop_reason": resp.stop_reason})


PROVIDERS = {"gpt-": OpenAIChat, "claude-": ClaudeChat}


def api_class(model):
    """The chat class for a model name served over an API; None for a local model (a directory or an HF repo id)."""
    if Path(model).exists() or "/" in model:
        return None
    return next((cls for p, cls in PROVIDERS.items() if model.startswith(p)), None)
