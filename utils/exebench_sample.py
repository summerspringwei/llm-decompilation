

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AsmBlock:
    code: List[str]
    target: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AsmBlock":
        return AsmBlock(code=d.get("code", []), target=d.get("target", []))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "target": self.target}


@dataclass
class LLVMIR:
    bb_count: Dict[str, Any]
    code: List[str]
    target: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLVMIR":
        return LLVMIR(
            bb_count=d.get("bb_count", {}),
            code=d.get("code", []),
            target=d.get("target", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"bb_count": self.bb_count, "code": self.code, "target": self.target}


@dataclass
class FuncInfo:
    functions: List[Dict[str, Any]]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FuncInfo":
        return FuncInfo(functions=d.get("functions", []))

    def to_dict(self) -> Dict[str, Any]:
        return {"functions": self.functions}


@dataclass
class SynthIOPairs:
    dummy_funcs: List[str]
    dummy_funcs_seed: List[int]
    input: List[Dict[str, Any]]
    output: List[Dict[str, Any]]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SynthIOPairs":
        return SynthIOPairs(
            dummy_funcs=d.get("dummy_funcs", []),
            dummy_funcs_seed=d.get("dummy_funcs_seed", []),
            input=d.get("input", []),
            output=d.get("output", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dummy_funcs": self.dummy_funcs,
            "dummy_funcs_seed": self.dummy_funcs_seed,
            "input": self.input,
            "output": self.output,
        }


@dataclass
class ExebenchSample:
    path: str
    func_def: str
    func_head: str
    func_head_types: str
    fname: str
    signature: List[str]

    asm: AsmBlock
    llvm_ir: LLVMIR
    func_info: FuncInfo
    synth_io_pairs: SynthIOPairs

    synth_deps: Optional[str] = None
    real_deps: Optional[str] = None
    real_io_pairs: Optional[Any] = None
    synth_exe_wrapper: Optional[str] = None
    real_exe_wrapper: Optional[str] = None
    ref: Optional[str] = None
    synth_iospec: Optional[str] = None
    real_iospec: Optional[str] = None
    token_length: Optional[int] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExebenchSample":
        return ExebenchSample(
            path=d["path"],
            func_def=d["func_def"],
            func_head=d["func_head"],
            func_head_types=d["func_head_types"],
            fname=d["fname"],
            signature=d.get("signature", []),
            asm=AsmBlock.from_dict(d.get("asm", {})),
            llvm_ir=LLVMIR.from_dict(d.get("llvm_ir", {})),
            func_info=FuncInfo.from_dict(d.get("func_info", {})),
            synth_io_pairs=SynthIOPairs.from_dict(d.get("synth_io_pairs", {})),
            synth_deps=d.get("synth_deps"),
            real_deps=d.get("real_deps"),
            real_io_pairs=d.get("real_io_pairs"),
            synth_exe_wrapper=d.get("synth_exe_wrapper"),
            real_exe_wrapper=d.get("real_exe_wrapper"),
            ref=d.get("ref"),
            synth_iospec=d.get("synth_iospec"),
            real_iospec=d.get("real_iospec"),
            token_length=d.get("token_length"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "func_def": self.func_def,
            "func_head": self.func_head,
            "func_head_types": self.func_head_types,
            "fname": self.fname,
            "signature": self.signature,
            "asm": self.asm.to_dict(),
            "synth_deps": self.synth_deps,
            "real_deps": self.real_deps,
            "synth_io_pairs": self.synth_io_pairs.to_dict(),
            "real_io_pairs": self.real_io_pairs,
            "synth_exe_wrapper": self.synth_exe_wrapper,
            "real_exe_wrapper": self.real_exe_wrapper,
            "ref": self.ref,
            "synth_iospec": self.synth_iospec,
            "real_iospec": self.real_iospec,
            "llvm_ir": self.llvm_ir.to_dict(),
            "func_info": self.func_info.to_dict(),
            "token_length": self.token_length,
        }