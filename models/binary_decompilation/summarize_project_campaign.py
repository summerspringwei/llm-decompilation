"""Summarize recorded project outcomes without counting baselines as recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.binary_decompilation.project_pipeline import write_json


def optional_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(output: Path) -> dict:
    config = json.loads((output / "config.json").read_text())
    report = {"model": config["model"], "base_url": config["base_url"],
              "service_preflight": optional_json(output / "service_preflight.json", None),
              "programs": {}, "recorded_model_responses": 0,
              "complete_goal": False,
              "limitations": ["Original-source scaffolds are used for isolated and hybrid validation.",
                              "Baseline and fixture passes are not decompilation outcomes.",
                              "Source-inlined functions are not separate out-of-line targets.",
                              "Skipped tests and invalid runtime traces do not establish coverage.",
                              "Standalone builds require a separate whole-suite and scope audit."]}
    for program in config["programs"]:
        directory = output / program
        plan = optional_json(directory / "plan.json", {})
        names = {f["name"] for f in plan.get("targets", [])}
        results = optional_json(directory / "results.json", [])
        if not results:
            results = [json.loads(path.read_text()) for path in (directory / "functions").glob("*/result.json")]
        by_name = {r["name"]: r for r in results if r["name"] in names}
        baseline = optional_json(directory / "baseline.json", {})
        runtime = optional_json(directory / "runtime.json", {})
        responses = 0
        for path in directory.glob("**/response.json"):
            response = json.loads(path.read_text())
            if response.get("choices"):
                responses += 1
        report["recorded_model_responses"] += responses
        stages = {}
        for stage in ("signature", "ir", "c"):
            recorded = [r[stage] for r in by_name.values() if stage in r]
            passed = sum(r.get("accepted", False) for r in recorded)
            stages[stage] = {"recorded_outcomes": len(recorded), "accepted": passed,
                             "inventory_acceptance_rate": passed / len(names) if responses and names else None}
        report["programs"][program] = {
            "targets": len(names), "processed_functions": len(by_name), "pending": sorted(names - by_name.keys()),
            "recorded_model_responses": responses, "stages": stages,
            "runtime_coverage": sum(bool(runtime.get("hits", {}).get(n, 0)) for n in names) / len(names)
                                if runtime.get("trace_valid") and names else None,
            "baseline": {k: baseline.get(k) for k in ("passed", "failed", "skipped", "unit_vector_cases")},
            "hybrid_integration": optional_json(directory / "integration.json", {}).get("accepted"),
            "standalone_application": optional_json(directory / "standalone" / "result.json", {}).get("accepted"),
            "validation_scope": "configured test suites and CLI probes; not all possible inputs",
        }
    report["inference_status"] = "recorded_outputs_present" if report["recorded_model_responses"] else "not_started"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    report = summarize(root)
    write_json(root / "campaign_report.json", report)
    for program, record in report["programs"].items():
        print(program, "targets", record["targets"], "processed", record["processed_functions"],
              "C inventory acceptance", record["stages"]["c"]["inventory_acceptance_rate"],
              "baseline", record["baseline"])
    print("Inference:", report["inference_status"], "recorded responses:", report["recorded_model_responses"])


if __name__ == "__main__":
    main()
