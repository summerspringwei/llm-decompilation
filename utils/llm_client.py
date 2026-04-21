"""LLM client factory.

Consolidates the duplicated ``SERVICE_CONFIG`` dicts from
``gemini_decompilation.py`` and ``iterative_decompile.py`` into a single
registry.

Usage::

    from config import DecompilationConfig
    from utils.llm_client import create_llm_client

    config = DecompilationConfig(model_name="gpt-oss-20b", host="localhost", port=9001)
    client, model_id = create_llm_client(config)
"""

from __future__ import annotations

import os
from typing import Tuple

from openai import OpenAI

from config import DecompilationConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Models that use a local vLLM / compatible endpoint.
_LOCAL_MODELS = {
    "Qwen3-32B",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Qwen3-30B-A3B",
}


def create_llm_client(
    config: DecompilationConfig,
) -> Tuple[OpenAI, str]:
    """Create an OpenAI-compatible client for the requested model.

    Returns:
        ``(client, model_name)`` tuple ready for ``client.chat.completions.create(model=model_name, …)``.
    """
    model = config.model_name

    if model in _LOCAL_MODELS:
        client = OpenAI(
            api_key=config.api_key,
            base_url=f"http://{config.host}:{config.port}/v1",
        )
        return client, model

    if model == "Huoshan-DeepSeek-R1":
        client = OpenAI(
            api_key=os.environ.get("ARK_STREAM_API_KEY", ""),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=1800,
        )
        return client, "ep-20250317013717-m9ksl"

    if model == "OpenAI-GPT-4.1":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        return client, "gpt-4.1"

    raise ValueError(
        f"Unknown model '{model}'. Known models: "
        f"{sorted(_LOCAL_MODELS | {'Huoshan-DeepSeek-R1', 'OpenAI-GPT-4.1'})}"
    )
