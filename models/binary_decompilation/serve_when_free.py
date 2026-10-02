"""Wait for GPUs 6,7, serve the local checkpoint, then run a campaign.

Run on the host, outside a GPU-isolating sandbox. Never stops other workloads.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def gpu_memory() -> dict[int, dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    )
    return {int(i): {"free_mib": int(free), "total_mib": int(total)}
            for i, free, total in csv.reader(result.stdout.splitlines())}


def active_compute_gpus() -> set[int]:
    mapping = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
    indices = {uuid.strip(): int(index) for index, uuid in csv.reader(mapping.stdout.splitlines())}
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True)
    return {indices[line.strip()] for line in processes.stdout.splitlines() if line.strip() in indices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="/data1/xiachunwei/Datasets/Models/Qwen3.8-27B-FP8")
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--vllm-python", default="/usr/bin/python3")
    parser.add_argument("--campaign-config")
    parser.add_argument("--campaign-python", default="/data1/xiachunwei/Software/anaconda3/bin/python")
    parser.add_argument("--check-only", action="store_true", help="Record one GPU probe without serving")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "watcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A watcher already owns this output directory")

    def status(stage: str, **details) -> None:
        temp = output / "status.tmp"
        temp.write_text(json.dumps({"stage": stage, "timestamp": time.time(),
                                    "watcher_pid": os.getpid(), **details}, indent=2) + "\n")
        temp.replace(output / "status.json")

    command = [args.vllm_python, "-m", "vllm.entrypoints.cli.main", "serve",
               args.model_path, "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "2", "--served-model-name", "Qwen3.8-27B-FP8",
               "--language-model-only", "--max-model-len", "131072",
               "--max-num-batched-tokens", "8192",
               "--max-num-seqs", "4", "--gpu-memory-utilization", "0.80"]
    (output / "server_config.json").write_text(json.dumps(
        {"command": command, "CUDA_VISIBLE_DEVICES": "6,7", "poll_seconds": 30,
         "minimum_free_fraction": 0.85, "require_no_active_compute_jobs": True}, indent=2) + "\n")
    try:
        while True:
            try:
                cards = gpu_memory()
                active = active_compute_gpus()
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                status("waiting_for_gpu_access", error=str(error))
                if args.check_only:
                    return
                time.sleep(30)
                continue
            if not all(i in cards for i in (6, 7)):
                status("waiting_for_gpu_access", error="GPUs 6 and 7 are not visible")
                if args.check_only:
                    return
                time.sleep(30)
                continue
            selected = {i: cards[i] for i in (6, 7)}
            available = not active.intersection((6, 7)) and all(
                c["free_mib"] >= 0.85 * c["total_mib"] for c in selected.values())
            if args.check_only:
                status("gpus_available" if available else "waiting_for_gpus",
                       gpus=selected, active_compute_gpus=sorted(active))
                return
            if available:
                break
            status("waiting_for_gpus", gpus=selected, active_compute_gpus=sorted(active))
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="6,7",
                   VLLM_CACHE_ROOT=str(output / "vllm-cache"),
                   TRITON_CACHE_DIR=str(output / "triton-cache"))
        with (output / "server.log").open("a") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        status("starting_server", server_pid=server.pid)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited {server.returncode}; inspect server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(5)
        else:
            server.terminate()
            server.wait(timeout=60)
            raise TimeoutError("vLLM startup exceeded 30 minutes")
        payload = json.dumps({"model": "Qwen3.8-27B-FP8", "n": 1, "max_tokens": 32,
                              "messages": [{"role": "user", "content": "Return the number 1."}],
                              "chat_template_kwargs": {"enable_thinking": False}}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                         data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            smoke = json.load(response)
        if not smoke.get("choices") or not smoke["choices"][0]["message"].get("content"):
            raise RuntimeError("Model inference smoke test returned no content")
        (output / "inference_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
        status("serving", server_pid=server.pid, endpoint=f"http://127.0.0.1:{args.port}/v1")
        if args.campaign_config:
            with (output / "campaign.log").open("a") as log:
                campaign = subprocess.Popen(
                    [args.campaign_python, "-m", "models.binary_decompilation.project_pipeline",
                     "--config", str(Path(args.campaign_config).resolve())],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            status("evaluating", server_pid=server.pid, campaign_pid=campaign.pid)
            code = campaign.wait()
            status("evaluation_complete" if code == 0 else "evaluation_failed",
                   server_pid=server.pid, campaign_returncode=code)
        server.wait()
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
