"""Anthropic Claude client wrapper that mimics the OpenAI ChatCompletion interface.

The rest of the pipeline (LLMDecompileRecord) expects:
  - response.choices          : list of choice objects
  - choice.message.content    : str
  - response.choices[0].message.content : str

Two backends are provided:

``ClaudeClient`` — uses the Anthropic SDK directly (requires ANTHROPIC_API_KEY).

``ClaudeCliClient`` — shells out to the ``claude -p`` CLI that is already
authenticated by the current Claude Code session.  No API key env var needed.
Since Claude's API does not support ``n`` (parallel completions), both clients
make ``n`` sequential calls and stitch them into a single fake ChatCompletion.
All objects are plain dataclasses so they pickle cleanly.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_TOKENS = 8192
CLAUDE_CLI = os.path.expanduser("~/.local/bin/claude")


# ---------------------------------------------------------------------------
# Shared data-transfer objects (OpenAI-compatible)
# ---------------------------------------------------------------------------


@dataclass
class _Message:
    content: Optional[str]
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int
    finish_reason: str = "stop"


@dataclass
class _ClaudeCompletion:
    """Minimal OpenAI-ChatCompletion-compatible container."""

    choices: List[_Choice]
    model: str = ""
    id: str = ""


# ---------------------------------------------------------------------------
# CLI-based backend (uses ``claude -p``, no API key required)
# ---------------------------------------------------------------------------


class _CliCompletions:
    """Mimics ``openai_client.chat.completions`` using the ``claude`` CLI."""

    def __init__(self, model: str, timeout_secs: int = 7200):
        self._model = model
        self._timeout = timeout_secs

    def create(
        self,
        model: str,
        messages: list,
        n: int = 1,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> _ClaudeCompletion:
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")

        effective_timeout = timeout or self._timeout
        choices: List[_Choice] = []

        for i in range(n):
            content = self._call_cli(user_content, effective_timeout)
            choices.append(_Choice(message=_Message(content=content), index=i))
            if i < n - 1:
                time.sleep(0.5)

        return _ClaudeCompletion(choices=choices, model=model)

    def _call_cli(self, prompt_text: str, timeout: float) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt_text)
            tmp_path = f.name

        try:
            cmd = [
                CLAUDE_CLI,
                "-p",
                "--dangerously-skip-permissions",
            ]
            with open(tmp_path, "r", encoding="utf-8") as stdin_f:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_f,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            if result.returncode != 0:
                logger.warning("claude CLI returned %d: %s", result.returncode, result.stderr[:200])
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("claude CLI timed out after %s seconds", timeout)
            return ""
        except Exception as exc:
            logger.warning("claude CLI error: %s", exc)
            return ""
        finally:
            os.unlink(tmp_path)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class ClaudeCliClient:
    """Drop-in replacement for ``openai.OpenAI`` using the ``claude -p`` CLI.

    Uses the Claude Code session authentication — no ANTHROPIC_API_KEY needed.

    Usage::

        client = ClaudeCliClient(model="claude-sonnet-4-6")
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "..."}],
            n=2,
        )
    """

    def __init__(self, model: str = "claude-sonnet-4-6", timeout_secs: int = 7200):
        completions = _CliCompletions(model=model, timeout_secs=timeout_secs)
        self.chat = _Chat(completions)


# ---------------------------------------------------------------------------
# SDK-based backend (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


class _SdkCompletions:
    """Mimics ``openai_client.chat.completions`` using the Anthropic SDK."""

    def __init__(self, anthropic_client, default_max_tokens: int):
        self._client = anthropic_client
        self._default_max_tokens = default_max_tokens

    def create(
        self,
        model: str,
        messages: list,
        n: int = 1,
        stream: bool = False,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> _ClaudeCompletion:
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        max_tok = max_tokens or self._default_max_tokens
        choices: List[_Choice] = []

        for i in range(n):
            try:
                call_kwargs: dict = dict(
                    model=model,
                    max_tokens=max_tok,
                    messages=[{"role": "user", "content": user_content}],
                )
                if system_content:
                    call_kwargs["system"] = system_content
                if timeout is not None:
                    call_kwargs["timeout"] = timeout

                resp = self._client.messages.create(**call_kwargs)
                content = resp.content[0].text if resp.content else ""
            except Exception as exc:
                logger.warning("Claude SDK call %d/%d failed: %s", i + 1, n, exc)
                content = ""

            choices.append(_Choice(message=_Message(content=content), index=i))
            if i < n - 1:
                time.sleep(0.2)

        return _ClaudeCompletion(choices=choices, model=model)


class ClaudeClient:
    """Drop-in replacement for ``openai.OpenAI`` using the Anthropic Python SDK.

    Requires ``ANTHROPIC_API_KEY`` to be set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        import anthropic
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._anthropic = anthropic.Anthropic(api_key=api_key)
        completions = _SdkCompletions(self._anthropic, default_max_tokens)
        self.chat = _Chat(completions)
