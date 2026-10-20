"""Structural adapter contract for projects other than Coreutils."""

from importlib import import_module
from pathlib import Path
from typing import Protocol


class ProjectAdapter(Protocol):
    config: dict
    program: str
    binary: Path
    clang: str
    header: str
    binary_only: bool
    feedback_config: dict

    def test(self, executable: Path, directory: Path) -> dict: ...
    def feedback_test(self, executable: Path, directory: Path) -> dict: ...
    def validate(self, name: str, source_file: str, signature: str, candidate: Path,
                 directory: Path, stage: str) -> dict: ...


def resolve_factory(value: str):
    module, separator, name = value.partition(":")
    if not separator or not name:
        raise ValueError("Adapter factories must use module:callable syntax")
    return getattr(import_module(module), name)
