"""Evaluate independent completions and feed back the candidate with most progress."""

import hashlib
import json
from pathlib import Path
import time

import httpx
from openai import APIConnectionError, APIStatusError

from models.binary_decompilation.prompt_budget import with_output_budget


def validation_rank(validation):
    tests = validation.get("tests", {})
    passed, failed = tests.get("passed", 0), tests.get("failed", 0)
    if validation.get("accepted"):
        return (4, -len(validation.get("probe_errors", [])), passed)
    if validation.get("execution_observed"):
        coverage = validation.get("coverage", {})
        if "target_passed_cases" in coverage:
            passed, failed = coverage["target_passed_cases"], coverage["target_failed_cases"]
        return (3, passed / max(1, passed + failed), passed)
    if validation.get("manifest") or validation.get("needs_probe"):
        return (2, 0, 0)
    if validation.get("failure_kind") == "response_format":
        return (0, 0, 0)
    return (1, 0, 0)


def search_candidates(client, config, adapter, function, stage, directory,
                      signature, instruction, evidence):
    from models.binary_decompilation.project_pipeline import (
        code_block, fit_prompt, generate, generation_metadata, write_json,
    )

    final_path = directory / "result.json"
    feedback, previous_hashes, stagnant = "", None, 0
    best = None
    for attempt in range(config["max_attempts"]):
        root = directory / f"attempt_{attempt}"
        root.mkdir(exist_ok=True)
        try:
            prompt, budget = fit_prompt(config, instruction, evidence, feedback)
        except ValueError as error:
            result = {"accepted": False, "attempts": attempt, "failure_kind": "context_window",
                      "feedback": str(error)}
            write_json(final_path, result)
            return result
        if not (root / "context_budget.json").exists():
            write_json(root / "context_budget.json", budget)
        response_path = root / "response.json"
        if response_path.exists():
            response = json.loads(response_path.read_text())
        elif (root / "request.json").exists():
            feedback = "Interrupted batch consumes one generation round; it must not be resent."
            continue
        else:
            (root / "prompt.txt").write_text(prompt)
            request_config = with_output_budget(config, budget)
            write_json(root / "generation_config.json", generation_metadata(request_config))
            write_json(root / "request.json", {"state": "started", "n": config["num_generate"],
                       "started_unix": time.time()})
            try:
                response = generate(client, request_config, prompt, root)
            except (APIConnectionError, APIStatusError, httpx.RequestError, TimeoutError) as error:
                response = {"choices": [], "request_error": {"type": type(error).__name__, "message": str(error)}}
                write_json(root / "request_error.json", response["request_error"])
            write_json(response_path, response)
        choices = response.get("choices", [])
        if not choices:
            feedback = "Inference transport failed; previous candidate is unchanged." if response.get("request_error") else "Server returned no candidates."
            write_json(root / "validation.json", {"accepted": False, "feedback": feedback,
                       "failure_kind": "inference_transport" if response.get("request_error") else "response_format"})
            continue
        outcomes, hashes, duplicates = [], set(), {}
        for position, choice in enumerate(choices):
            index = choice.get("index", position)
            # Position owns the path, so duplicate server indices cannot overwrite siblings.
            choice_dir = root / f"choice_{position}"
            choice_dir.mkdir(exist_ok=True)
            text = choice.get("message", {}).get("content") or ""
            candidate, contract, candidate_signature = None, None, signature
            finish = choice.get("finish_reason")
            digest = hashlib.sha256(text.encode()).hexdigest()
            hashes.add(digest)
            try:
                if finish in ("length", "interrupted", "content_filter") or str(finish).startswith("degenerate_"):
                    raise ValueError(f"Incomplete response ({finish}); return a compact complete candidate.")
                if stage in ("signature", "probes"):
                    contract = json.loads(code_block(text, ("json",)))
                    if not isinstance(contract, dict):
                        raise ValueError("Contract response must be a JSON object")
                    if stage == "signature":
                        candidate_signature = contract["c_declaration"]
                    write_json(choice_dir / "contract.json", contract)
                    candidate = choice_dir / "unused.c"
                else:
                    candidate = choice_dir / ("candidate.ll" if stage == "ir" else "candidate.c")
                    candidate.write_text(code_block(text, ("llvm", "llvm-ir", "ll") if stage == "ir" else ("c",)))
                if digest in duplicates:
                    validation = {**duplicates[digest]["validation"], "duplicate_of": duplicates[digest]["candidate"]}
                else:
                    validation = adapter.validate(function["name"], function["source_file"], candidate_signature,
                                                  candidate, choice_dir / "validation", stage)
            except (ValueError, KeyError, TypeError) as error:
                validation = {"accepted": False, "feedback": str(error), "failure_kind": "response_format"}
                if response.get("request_error") and finish in (None, "interrupted"):
                    validation["failure_kind"] = "inference_transport"
            validation.setdefault("failure_kind", "accepted" if validation.get("accepted") else
                                  "runtime" if validation.get("manifest") else "compile_or_contract")
            outcome = {"choice_index": index, "choice_position": position, "signature": candidate_signature,
                       "candidate": str(candidate) if candidate else None, "validation": validation,
                       "text": text, "finish_reason": finish, "contract": contract}
            outcomes.append(outcome)
            duplicates.setdefault(digest, outcome)
            write_json(choice_dir / "validation.json", validation)
        selected = max(outcomes, key=lambda item: validation_rank(item["validation"]))
        write_json(root / "selection.json", {
            "requested_candidates": config["num_generate"], "returned_candidates": len(choices),
            "unique_candidates": len(hashes), "selected_choice_position": selected["choice_position"],
            "candidates": [{k: v for k, v in item.items() if k not in ("text", "contract")} for item in outcomes],
        })
        validation = selected["validation"]
        if validation.get("accepted") and stage in ("signature", "probes"):
            # Signature/probe validation changes adapter state; restore the selected choice.
            validation = adapter.validate(function["name"], function["source_file"], selected["signature"],
                                          Path(selected["candidate"]), root / "selected_validation", stage)
        result = {"accepted": bool(validation.get("accepted")), "attempts": attempt + 1,
                  "candidate": selected["candidate"], "signature": selected["signature"],
                  "validation": validation, "feedback": validation.get("feedback", ""),
                  "selected_choice_position": selected["choice_position"], "candidates_in_round": len(choices)}
        if stage in ("signature", "probes"):
            result["contract"] = selected["contract"]
        for flag in ("needs_probe", "repair_contract"):
            if validation.get(flag):
                result[flag] = True
        write_json(root / "validation.json", validation)
        if validation.get('failure_kind') == 'infrastructure':
            result.update(failure_kind='infrastructure', reason='Validation infrastructure failed; do not spend more generation rounds')
            write_json(final_path, result)
            return result
        if result["accepted"] or result.get("needs_probe") or result.get("repair_contract"):
            write_json(final_path, result)
            return result
        if best is None or validation_rank(validation) >= validation_rank(best["validation"]):
            best = {**selected, "validation": validation}
        summaries = [{"choice": item["choice_position"], "kind": item["validation"]["failure_kind"],
                      "feedback": item["validation"].get("feedback", "")[-1200:]} for item in outcomes]
        previous = best["text"]
        try:
            previous = code_block(previous, ("json",) if stage in ("signature", "probes") else
                                  ("llvm", "llvm-ir", "ll") if stage == "ir" else ("c",))
        except ValueError:
            previous = "No complete artifact available. Reconstruct from the original evidence."
        if best["finish_reason"] in ("length", "interrupted") or str(best["finish_reason"]).startswith("degenerate_"):
            previous = previous[:4000]
        feedback = ("Repair the candidate with greatest validation progress. Preserve working behavior.\n"
                    + best["validation"].get("feedback", "") + "\nPrevious candidate:\n" + previous
                    + "\nOther candidates in the last batch:\n" + json.dumps(summaries))
        if stage == "ir" and config.get("semantic_checks"):
            facts = getattr(adapter, 'semantic_facts', {}).get(function['name'])
            if facts:
                from models.binary_decompilation.repair_evidence import compiler_slice
                localized = compiler_slice(facts, previous, best['validation'].get('feedback', ''))
                write_json(root / 'repair_slice.json', localized)
                feedback += '\nAnalysis repair slice: ' + json.dumps(localized)
        stagnant = stagnant + 1 if hashes == previous_hashes else 1
        previous_hashes = hashes
        if config.get("stagnation_limit") and stagnant >= config["stagnation_limit"]:
            result["reason"] = "identical failed candidate set repeated"
            write_json(final_path, result)
            return result
    result = {"accepted": False, "attempts": config["max_attempts"], "feedback": feedback,
              "reason": "stage generation budget exhausted"}
    if best:
        result.update(candidate=best["candidate"], signature=best["signature"], validation=best["validation"])
    write_json(final_path, result)
    return result
