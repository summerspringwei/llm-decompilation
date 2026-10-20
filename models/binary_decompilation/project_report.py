"""Report actual recovered artifacts separately from baselines and fixtures."""

import argparse
import json
from pathlib import Path

from models.binary_decompilation.project_pipeline import write_json


def report(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    rows = []
    for track in ("retrieval", "no_retrieval", "assisted"):
        for program in config["programs"]:
            directory = root / track / program
            plan_path = directory / "plan.json"
            if not plan_path.exists():
                continue
            plan = json.loads(plan_path.read_text())
            baseline_path = directory / "baseline.json"
            baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            failure_path = directory / "failure.json"
            results_path = directory / "results.json"
            current_results = {r["name"]: r for r in json.loads(results_path.read_text())} if results_path.exists() else {}
            functions = []
            for target in plan["targets"]:
                path = directory / "functions" / target["name"]
                latest = {}
                for stage in ("signature", "probes", "ir", "c"):
                    checkpoint = path / stage / "latest.json"
                    latest[stage] = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
                if target["name"] in current_results:
                    # A completed current attempt supersedes historical successful revisions.
                    current = current_results[target["name"]]
                    latest = {stage: current.get(stage, {}) for stage in latest}
                functions.append({"name": target["name"], "function_id": target["function_id"],
                                  "runtime_observed": bool(plan["runtime"]["hits"].get(target["name"])),
                                  "signature_syntax_valid": latest["signature"].get("accepted", False),
                                  "ir_validated": latest["ir"].get("accepted", False),
                                  "c_validated": bool(latest["ir"].get("accepted") and latest["c"].get("accepted")),
                                  "stage_outcomes": {s: {"accepted": v.get("accepted", False),
                                      "reason": v.get("reason"), "failure_kind": v.get("failure_kind") or
                                      v.get("validation", {}).get("failure_kind"), "error": v.get("error"),
                                      "needs_probe": v.get("needs_probe", False),
                                      "coverage": v.get("validation", {}).get("coverage")}
                                      for s, v in latest.items()},
                                  "error": current_results.get(target["name"], {}).get("error"),
                                  "requests": {s: len(list((path / s).glob("*/attempt_*/request.json"))) for s in latest},
                                  "requested_candidates": {s: sum(json.loads(p.read_text()).get("n", 1)
                                      for p in (path / s).glob("*/attempt_*/request.json")) for s in latest},
                                  "returned_candidates": {s: sum(len(json.loads(p.read_text()).get("choices", []))
                                      for p in (path / s).glob("*/attempt_*/response.json")) for s in latest}})
            row = {"track": track, "program": program, "targets": len(functions), "functions": functions,
                   "status": "partial" if summary.get("partial_run") else "completed" if summary_path.exists() else "failed" if failure_path.exists() else "incomplete",
                   "selected_functions": summary.get("selected_functions"),
                   "validated_ir": sum(f["ir_validated"] for f in functions),
                   "validated_c": sum(f["c_validated"] for f in functions),
                   "runtime_observed": sum(f["runtime_observed"] for f in functions),
                   "baseline": {k: baseline.get(k) for k in ("passed", "failed", "skipped", "unit_vector_cases")},
                   "complete_project_recovery": summary.get("complete_project_recovery", False),
                   "instrumentation_exclusions": plan["instrumented_baseline"].get("instrumentation_exclusions", {})}
            row["c_success_rate"] = row["validated_c"] / row["targets"] if row["targets"] else None
            rows.append(row)
    acceptance_path = root / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {"goal_complete": False}
    result = {"rows": rows, "acceptance": acceptance,
              "metrics": {"signature": "Syntax-valid inferred declarations, not ABI proof",
                          "ir": "LLVM verified, compiled, and accepted by the track's runtime validator",
                          "c": "Compiled and accepted by the track's runtime validator",
                          "application": "Standalone IR and C builds pass configured application suites"}}
    write_json(root / "report.json", result)
    lines = ["# Project-Level Recovery Results", "", "Baseline and fixture passes are not model decompilation successes.", "",
             "| Track | Tool | Status | Targets | Observed | IR Validated | C Validated | Standalone Complete |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['track']} | {row['program']} | {row['status']} | {row['targets']} | {row['runtime_observed']} | "
                     f"{row['validated_ir']} | {row['validated_c']} | {row['complete_project_recovery']} |")
    lines += ["", "## Interpretation", "", "- Function denominators include all retained source-defined targets and private helper targets.",
              "- Runtime hit counts are bounded samples, not profiling frequencies; stacks use fuzzy unwinding.",
              "- Compiler/runtime scaffolding and approved libraries are outside the recovered inventory.",
              "- A function not entered during replacement tests is not counted as binary-first validated.",
              "- Source-scaffold-assisted results are not binary-only recovery results.",
              "- Assisted regression acceptance does not require observed replacement entry.",
              "- Initial pilot safeguards evolved; this is exploratory, not a controlled causal RAG comparison.",
              "- Skipped upstream tests are reported separately and do not count as passes.",
              "- The /proc-masking df test is excluded only from instrumentation, not standalone tests.",
              "- Passing available tests does not establish general semantic equivalence.",
              "", "Goal complete: **" + str(acceptance.get("goal_complete", False)) + "**."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    for row in result["rows"]:
        print(row["track"], row["program"], f"C {row['validated_c']}/{row['targets']}",
              "standalone", row["complete_project_recovery"])


if __name__ == "__main__":
    main()
