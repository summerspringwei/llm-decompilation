"""LLM response parsing utilities.

Responsible for extracting LLVM IR code blocks from LLM-generated text.
Prompt-formatting functions have been moved to :mod:`utils.prompt_builder`;
the :class:`PromptType` enum has been moved to :mod:`utils.prompt_type`.
"""

import re
from typing import List

from utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_llvm_code(markdown_content: str) -> list[str]:
    """Extract LLVM IR code blocks from markdown-formatted text.

    Tries ````` ```llvm ````` fences first, falling back to ``<code>`` tags.

    Returns:
        A list of extracted code strings (may be empty).
    """
    # Try ```llvm fences first.
    pattern = r"```llvm\n(.*?)\n```"
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    if matches:
        return matches

    # Fallback: <code> tags.
    pattern = r"<code>(.*?)</code>"
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    return matches


def extract_llvm_code_from_response_text(result: str) -> str:
    """Extract a single LLVM IR string from raw LLM output text.

    Strips ``<think>`` blocks used by reasoning models, then delegates to
    :func:`extract_llvm_code`.
    """
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()
    llvm_code = extract_llvm_code(result)
    if isinstance(llvm_code, list) and len(llvm_code) > 0:
        llvm_code = llvm_code[0]
    if len(llvm_code) == 0:
        logger.warning("No LLVM code found in the response: %s", result[:200])
    return llvm_code


def extract_llvm_code_from_response(response) -> List[str]:
    """Extract LLVM IR from each choice in an OpenAI ``ChatCompletion``.

    Returns:
        A list with one LLVM IR string per choice.
    """
    predict_llvm_code_list: List[str] = []
    if response.choices and len(response.choices) > 0:
        for choice in response.choices:
            content = choice.message.content
            if content is None:
                logger.info("No result found in response choice")
                predict_llvm_code_list.append("")
                continue
            predict_llvm_code_list.append(
                extract_llvm_code_from_response_text(content)
            )
    else:
        logger.warning("No choices found in the response.")
    return predict_llvm_code_list


def strip_think_block(text: str) -> str:
    """Remove the ``<think>…</think>`` block from reasoning-model output."""
    if "<think>" in text and "</think>" in text:
        start = text.find("</think>") + len("</think>")
        return text[start:].strip()
    return text
