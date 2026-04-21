"""Centralized configuration for the llm-decompilation pipeline.

All runtime parameters, paths, and service endpoints are defined here as
dataclasses.  Prefer constructing a config from CLI args via
``DecompilationConfig.from_args(args)`` or from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

HOME_DIR = os.path.expanduser("~")


@dataclass
class GhidraConfig:
    """Configuration for the Ghidra decompiler integration."""

    home_dir: str = field(
        default_factory=lambda: os.environ.get(
            "GHIDRA_HOME",
            "/data1/xiachunwei/Software/ghidra_11.4.2_PUBLIC",
        )
    )
    command_timeout: int = 90
    """Default timeout in seconds for Ghidra subprocess calls."""

    @property
    def headless_analyzer_path(self) -> str:
        return os.path.join(self.home_dir, "support", "analyzeHeadless")


@dataclass
class RagConfig:
    """Configuration for retrieval-augmented generation (Qdrant + embeddings)."""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_url: str = "http://localhost:8123/embed/batch"
    collection_name_template: str = (
        "train_synth_rich_io_filtered_{idx}_preprocessed_hermessim"
    )
    dataset_dir: str = field(
        default_factory=lambda: os.path.join(
            HOME_DIR,
            "Datasets",
            "filtered_exebench",
            "train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff",
        )
    )


@dataclass
class EvaluationConfig:
    """Configuration for compilation and execution evaluation."""

    compilation_timeout: int = 30
    """Timeout in seconds for ``llc`` invocations."""
    max_eval_workers: int = 8
    """Maximum number of threads for parallel prediction evaluation."""


@dataclass
class DecompilationConfig:
    """Top-level configuration aggregating all sub-configs.

    Use ``from_args`` to construct from an ``argparse.Namespace`` produced by
    the CLI entry-points.
    """

    # Model / inference settings
    model_name: str = "gpt-oss-20b"
    host: str = "localhost"
    port: int = 9001
    api_key: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_API_KEY", "token-llm4decompilation-abc123"
        )
    )
    num_generate: int = 8
    num_retry: int = 10
    num_processes: int = 1
    llm_timeout: int = 7200
    """Timeout in seconds for LLM API calls."""

    # Prompt behaviour
    prompt_type: str = "in-context-learning"
    fix_prompt_type: str = "compile-fix"
    remove_comments: bool = True
    use_pcode: bool = False

    # Dataset
    dataset_name: str = "sampled_dataset_with_loops_and_only_one_bb_164"

    # Sub-configs
    ghidra: GhidraConfig = field(default_factory=GhidraConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Output
    output_dir: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "DecompilationConfig":
        """Build a config from an ``argparse.Namespace``.

        Unrecognised attributes in *args* are silently ignored so the same
        namespace can feed multiple sub-configs.
        """
        rag = RagConfig(
            qdrant_host=getattr(args, "qdrant_host", "localhost"),
            qdrant_port=int(getattr(args, "qdrant_port", 6333)),
            embedding_url=getattr(
                args, "embedding_url", "http://localhost:8123/embed/batch"
            ),
            collection_name_template=getattr(
                args,
                "collection_name_with_idx",
                RagConfig.collection_name_template,
            ),
        )
        return cls(
            model_name=getattr(args, "model", cls.model_name),
            host=getattr(args, "host", cls.host),
            port=int(getattr(args, "port", cls.port)),
            num_generate=getattr(args, "num_generate", cls.num_generate),
            num_retry=getattr(args, "num_retry", cls.num_retry),
            num_processes=getattr(args, "num_processes", cls.num_processes),
            prompt_type=getattr(args, "prompt_type", cls.prompt_type),
            remove_comments=getattr(args, "remove_comments", cls.remove_comments),
            use_pcode=getattr(args, "use_pcode", cls.use_pcode),
            dataset_name=getattr(args, "dataset_name", cls.dataset_name),
            rag=rag,
        )
