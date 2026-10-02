"""Check inference, retrieval, and tracing prerequisites before generation."""

import requests

from models.binary_decompilation.project_pipeline import run


def check_services(client, config: dict) -> dict:
    checks = {}
    try:
        models = client.models.list().data
        matching = next((model for model in models if model.id == config["model"]), None)
        if matching is None:
            raise ValueError("Requested model is not listed by the inference endpoint")
        maximum = getattr(matching, "max_model_len", None)
        if isinstance(maximum, int) and maximum < config.get("context_window", 65536):
            raise ValueError(f"Server context {maximum} is smaller than the configured input/output window")
        checks["inference"] = {"ok": True, "endpoint": config["base_url"],
                               "model": matching.id, "server_max_model_len": maximum}
    except Exception as error:
        checks["inference"] = {"ok": False, "endpoint": config["base_url"],
                               "error_type": type(error).__name__, "error": str(error)}
    if config["retrieval"].get("enabled", True):
        endpoints = {"qdrant": config["retrieval"]["qdrant_url"].rstrip("/") + "/readyz",
                     "hermessim": config["retrieval"]["embedding_url"].rsplit("/embed/", 1)[0] + "/health"}
        for name, endpoint in endpoints.items():
            try:
                response = requests.get(endpoint, timeout=10)
                response.raise_for_status()
                if name == "hermessim" and response.json().get("model_loaded") is not True:
                    raise ValueError("HermesSim reports that its embedding model is not loaded")
                checks[name] = {"ok": True, "endpoint": endpoint}
            except Exception as error:
                checks[name] = {"ok": False, "endpoint": endpoint,
                                "error_type": type(error).__name__, "error": str(error)}
    try:
        probe = run(["gdb", "--batch", "-ex", "run", "--args", "/bin/true"], timeout=30)
        checks["runtime_tracing"] = {"ok": probe["returncode"] == 0, "probe": probe}
    except OSError as error:
        checks["runtime_tracing"] = {"ok": False, "error": str(error)}
    return {"ready": all(check["ok"] for check in checks.values()), "checks": checks,
            "inference_generation_requests": 0}
