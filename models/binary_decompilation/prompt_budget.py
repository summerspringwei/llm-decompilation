"""Count local-model chat tokens without truncating target binary or IR evidence."""

from __future__ import annotations

import copy
from functools import lru_cache
import json


@lru_cache(maxsize=2)
def tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def with_output_budget(config: dict, budget: dict) -> dict:
    requested = config.get("max_tokens", 8192)
    return {**config, "requested_max_tokens": requested,
            "max_tokens": budget.get("max_output_tokens", requested)}


def fit_prompt(config: dict, instruction: str, evidence: dict, feedback: str = "") -> tuple[str, dict]:
    packed = copy.deepcopy(evidence)
    # Carry semantic contracts, not unrelated validation paths and test reports.
    if "callee_contracts" in packed:
        related = {packed.get("target"), *(call["name"] for call in packed.get("calls", []))}
        packed["callee_contracts"] = {
            name: {"signature": contract.get("signature"), "contract": contract.get("contract"),
                   "accepted": contract.get("accepted", False)}
            for name, contract in packed["callee_contracts"].items() if name in related}
    path = config.get("tokenizer_path")
    context = config.get("context_window", 65536)
    requested = config.get("max_tokens", 8192)
    policy = config.get("max_tokens_policy", "fixed")
    if policy not in ("fixed", "remaining_context"):
        raise ValueError("Unknown max_tokens_policy: " + str(policy))
    if policy == "remaining_context" and not path:
        raise ValueError("remaining_context output budgeting requires a local tokenizer")
    if type(requested) is not int or requested < 1:
        raise ValueError("max_tokens must be a positive integer")
    budget = context - (1 if policy == "remaining_context" else requested) - 64
    removed = []
    while True:
        prompt = instruction + "\nEvidence:\n" + json.dumps(packed)
        if feedback:
            prompt += "\nPrevious compiler/runtime feedback:\n" + feedback
        count = None if not path else len(tokenizer(path).apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            enable_thinking=config.get("enable_thinking", False)))
        effective = min(requested, context - count - 64) if policy == "remaining_context" else requested
        metadata = {"prompt_tokens": count, "input_token_budget": budget,
                    "max_output_tokens": max(0, effective), "requested_max_output_tokens": requested,
                    "max_tokens_policy": policy,
                    "context_window": context, "enable_thinking": config.get("enable_thinking", False),
                    "removed_retrieval_examples": removed, "target_evidence_truncated": False}
        if count is None or count <= budget:
            return prompt, metadata
        examples = packed.get("retrieved_examples", [])
        if examples:
            example = examples.pop()
            removed.append({"collection": example.get("collection"), "index": example.get("index"),
                            "score": example.get("score")})
            continue
        if len(feedback) > 4000:
            feedback = feedback[:4000]
            continue
        raise ValueError(f"Complete target evidence needs {count} tokens; input budget is {budget}. "
                         "Increase the served context or partition the function; evidence was not truncated.")
