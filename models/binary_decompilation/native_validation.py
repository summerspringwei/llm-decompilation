"""Binary-only isolation: load generated PIC code and redirect native entries.

The preload gate preserves ordinary Unix process I/O while instrumentation is
installed before main. It is test infrastructure, never a standalone dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection


NATIVE_TYPES = set("void pointer int uint long ulong char uchar size_t ssize_t float double int8 uint8 int16 uint16 int32 uint32 int64 uint64 bool".split())


def validate_probe(probe: dict) -> None:
    """Reject malformed probe contracts before executing their native callbacks."""
    if not isinstance(probe, dict):
        raise ValueError("Each direct probe must be an object")
    buffers = {}

    def check_call(call):
        types, arguments = call.get("argument_types"), call.get("arguments")
        if not isinstance(call.get("return_type"), str) or call["return_type"] not in NATIVE_TYPES:
            raise ValueError("Unsupported probe return_type; use a Frida native type")
        if not isinstance(types, list) or not isinstance(arguments, list) or len(types) != len(arguments):
            raise ValueError("Probe argument_types and arguments must be matching arrays")
        for kind, argument in zip(types, arguments):
            if not isinstance(kind, str) or kind not in NATIVE_TYPES - {"void"}:
                raise ValueError("Unsupported probe argument type: " + str(kind))
            if kind != "pointer":
                if isinstance(argument, bool) or not isinstance(argument, (int, float)) or (
                        isinstance(argument, float) and not math.isfinite(argument)):
                    raise ValueError("Scalar probe arguments must be finite numbers")
                if kind not in {"float", "double"} and (int(argument) != argument or abs(argument) > 2**53 - 1):
                    raise ValueError("Integer probe arguments must be exactly representable JSON numbers")
                continue
            if not isinstance(argument, dict):
                raise ValueError("Pointer probe arguments require string, buffer, ref, or null objects")
            forms = set(argument) & {"string", "buffer", "ref", "null"}
            if len(forms) != 1:
                raise ValueError("Pointer probe argument must have exactly one storage form")
            form = next(iter(forms))
            if form == "string":
                value = argument[form]
                if not isinstance(value, str) or len(value.encode("utf-8")) > 4096:
                    raise ValueError("Probe strings must be bounded UTF-8 strings")
            elif form == "buffer":
                size, identifier, data = argument[form], argument.get("id"), argument.get("bytes", [])
                if type(size) is not int or not 1 <= size <= 4096:
                    raise ValueError('Probe buffer must be an integer from 1 to 4096; use '
                                     '{"buffer":16,"id":"buf","bytes":[0]}, not a nested buffer object')
                if not isinstance(data, list) or len(data) > size or any(type(b) is not int or not 0 <= b <= 255 for b in data):
                    raise ValueError("Probe byte initializers must fit their buffers")
                if identifier is not None:
                    if not isinstance(identifier, str) or not identifier or identifier in buffers:
                        raise ValueError("Probe buffer IDs must be nonempty and unique")
                    buffers[identifier] = size
            elif form == "ref":
                if not isinstance(argument[form], str) or argument[form] not in buffers:
                    raise ValueError("Probe ref must identify an already allocated buffer, not a global symbol")
            elif argument[form] is not True:
                raise ValueError("Null probe arguments must use {null:true}")

    setup = probe.get("setup", [])
    if not isinstance(setup, list) or len(setup) > 16:
        raise ValueError("Probe setup must be an array of at most sixteen calls")
    for call in setup:
        if not isinstance(call, dict) or call.get("symbol") not in {"stat", "lstat", "memcpy", "memset"}:
            raise ValueError("Disallowed probe setup symbol")
        check_call(call)
    check_call(probe)
    kind = probe.get("return_kind", "integer")
    if not isinstance(kind, str) or kind not in {"integer", "cstring"} or kind == "cstring" and probe["return_type"] != "pointer":
        raise ValueError("cstring probes require a pointer return type")
    outputs = probe.get("memory_outputs", [])
    if not isinstance(outputs, list) or len(outputs) > 16:
        raise ValueError("Probe memory_outputs must be a bounded array")
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("id"), str) or output["id"] not in buffers:
            raise ValueError("Probe memory output must identify an allocated buffer")
        if type(output.get("size")) is not int or not 1 <= output["size"] <= min(128, buffers[output["id"]]):
            raise ValueError("Probe memory output size exceeds its bounded allocation")

def write_json(path, value):
    from models.binary_decompilation.project_pipeline import write_json as write
    write(path, value)


def run(command):
    from models.binary_decompilation.project_pipeline import run as execute
    return execute(command)


GATE_SOURCE = '''#include <stdlib.h>
#include <unistd.h>
__attribute__((constructor)) static void wait_for_instrumentation(void) {
  const char *s = getenv("DECOMP_GATE_FD");
  if (s) { int fd = atoi(s); char byte; (void) read(fd, &byte, 1); close(fd); }
}
'''


def elf_symbols(binary: Path) -> dict:
    result = {}
    with binary.open("rb") as stream:
        elf = ELFFile(stream)
        if elf["e_machine"] != "EM_X86_64" or elf.elfclass != 64:
            raise ValueError("Native isolation currently supports Linux x86-64 ELF only")
        for table_name in (".dynsym", ".symtab"):
            table = elf.get_section_by_name(table_name)
            for symbol in table.iter_symbols() if table else []:
                if symbol.name and symbol["st_shndx"] != "SHN_UNDEF":
                    result[symbol.name] = {"address": symbol["st_value"], "size": symbol["st_size"],
                                           "kind": symbol["st_info"]["type"]}
        result["__elf_type__"] = elf["e_type"]
    return result


def build_gate(clang: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source, library = directory / "gate.c", directory / "gate.so"
    source.write_text(GATE_SOURCE)
    compiled = run([clang, "-shared", "-fPIC", str(source), "-o", str(library)])
    write_json(directory / "gate_compile.json", compiled)
    if compiled["returncode"]:
        raise ValueError(compiled["stderr"])
    return library.resolve()


def short_entry_safe(binary: Path, info: dict, symbols: dict) -> bool:
    """A five-byte redirect may consume padding, but never another function."""
    if info["size"] >= 5:
        return True
    start, end = info["address"], info["address"] + 5
    if any(start < s["address"] < end for s in symbols.values() if isinstance(s, dict)):
        return False
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64
    with binary.open("rb") as stream:
        elf = ELFFile(stream)
        for section in elf.iter_sections():
            if section["sh_addr"] <= start < section["sh_addr"] + section["sh_size"]:
                data = section.data()[start + info["size"] - section["sh_addr"]:]
                covered = 0
                for instruction in Cs(CS_ARCH_X86, CS_MODE_64).disasm(data[:32], start + info["size"]):
                    if instruction.mnemonic != "nop":
                        return False
                    covered += instruction.size
                    if covered >= 5 - info["size"]:
                        return True
    return False


def build_candidate(clang: str, binary: Path, name: str, candidate: Path, header: str,
                    directory: Path, stage: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    symbols = elf_symbols(binary)
    if name not in symbols or symbols[name]["kind"] != "STT_FUNC":
        return {"accepted": False, "feedback": "Target entry not found in ELF"}
    (directory / "project.h").write_text(header)
    source = candidate
    if stage == "c":
        source = directory / "candidate.c"
        source.write_text('#include "project.h"\n' + candidate.read_text())
    obj = directory / "candidate.o"
    args = [clang, "-fPIC", "-O1", "-D_GNU_SOURCE", "-femit-all-decls", "-fpatchable-function-entry=16,0", "-Werror=implicit-function-declaration"]
    if stage == "ir":
        opt = str(Path(clang).with_name("opt"))
        verified = run([opt, "-passes=verify", "-disable-output", str(source)])
        write_json(directory / "llvm_verify.json", verified)
        if verified["returncode"]:
            return {"accepted": False, "feedback": verified["stderr"]}
        args = [clang, "-fPIC", "-O1", "-fpatchable-function-entry=16,0", "-Wno-override-module"]
    compiled = run([*args, "-c", str(source), "-o", str(obj)])
    write_json(directory / "compile.json", compiled)
    if compiled["returncode"]:
        return {"accepted": False, "feedback": compiled["stderr"][-12000:]}
    defined = run(["nm", "--defined-only", str(obj)])["stdout"]
    definitions = {p[-1] for line in defined.splitlines() if len(p := line.split()) == 3 and p[1] in "tT"}
    if name not in definitions:
        return {"accepted": False, "feedback": "Candidate must define target " + name}
    # Application dependencies must be declarations, not silently copied bodies.
    if (definitions & set(symbols)) - {name}:
        return {"accepted": False, "feedback": "Candidate defines other original functions: " +
                str(sorted((definitions & set(symbols)) - {name}))}
    data_definitions = {p[-1] for line in defined.splitlines() if len(p := line.split()) == 3 and p[1] in "bBdDrR"}
    shadowed = {n for n in data_definitions if n in symbols and symbols[n]["kind"] == "STT_OBJECT"}
    if shadowed:
        return {"accepted": False, "feedback": "Original globals must remain external and bind to live data: " + str(sorted(shadowed))}
    exported = "__recovered_" + name
    public = directory / "candidate.public.o"
    published = run(["objcopy", "--globalize-symbol=" + name,
                     "--redefine-sym=" + name + "=" + exported, str(obj), str(public)])
    if published["returncode"]:
        return {"accepted": False, "feedback": published["stderr"]}
    undefined = [line.split()[-1] for line in run(["nm", "-u", str(public)])["stdout"].splitlines()]
    instrumented_export = "__entry_" + name
    # A tail-jump shim preserves all native argument registers and return state.
    assembly = [".text", ".p2align 4", f".globl {instrumented_export}",
                f".type {instrumented_export},@function", f".hidden {exported}",
                f"{instrumented_export}:", ".rept 16", "nop", ".endr", f"jmp {exported}",
                f".size {instrumented_export},.-{instrumented_export}"]
    bridges, globals_to_bind = [], []
    for dep in undefined:
        if dep not in symbols:
            aliases = [name for name in symbols if name != dep and re.sub(r'[^A-Za-z_0-9]', '_', name) == dep]
            if aliases:
                return {"accepted": False, "failure_kind": "symbol_identity", "repair_contract": stage == "c",
                        "feedback": f"Undefined symbol {dep!r} appears to rename ELF symbol {aliases[0]!r}. "
                                    "Preserve ELF identity with an extern __asm__(\"exact ELF name\") declaration; "
                                    "do not create replacement storage."}
            continue  # System-library imports are resolved by the native loader.
        if not re.fullmatch(r"[A-Za-z_.$][A-Za-z_0-9.$]*", dep):
            return {"accepted": False, "feedback": "Unsupported dependency symbol " + dep}
        info = symbols[dep]
        if info["kind"] == "STT_FUNC":
            slot = "__bridge_" + dep
            assembly += [f".globl {dep}", f".type {dep},@function", f"{dep}:",
                         f"jmp *{slot}(%rip)", ".data", ".p2align 3", f".globl {slot}",
                         f".hidden {slot}", f"{slot}: .quad 0", ".text"]
            bridges.append({"slot": slot, "target": dep, "address": info["address"]})
        elif info["kind"] == "STT_OBJECT":
            assembly += [f".comm {dep},{max(1, info['size'])},8"]
            globals_to_bind.append(dep)
        else:
            return {"accepted": False, "feedback": "Unsupported dependency type " + dep}
    assembly += ['.section .note.GNU-stack,"",@progbits']
    bridge_source = directory / "bridges.S"
    bridge_source.write_text("\n".join(assembly) + "\n")
    library = (directory / "candidate.so").resolve()
    linked = run([clang, "-shared", "-Wl,-z,now", str(public), str(bridge_source), "-o", str(library)])
    write_json(directory / "link.json", linked)
    if linked["returncode"]:
        return {"accepted": False, "feedback": linked["stderr"][-12000:]}
    module_symbols = elf_symbols(library)
    for bridge in bridges:
        bridge["offset"] = module_symbols[bridge["slot"]]["address"]
    bindings = []
    with library.open("rb") as stream:
        elf = ELFFile(stream)
        for section in elf.iter_sections():
            if not isinstance(section, RelocationSection):
                continue
            table = elf.get_section(section["sh_link"])
            for relocation in section.iter_relocations():
                dep = table.get_symbol(relocation["r_info_sym"]).name
                if dep in globals_to_bind:
                    if relocation["r_info_type"] != 6:  # R_X86_64_GLOB_DAT
                        return {"accepted": False, "feedback": "Unsupported data relocation for " + dep}
                    bindings.append({"offset": relocation["r_offset"], "target": dep,
                                     "address": symbols[dep]["address"]})
    if set(globals_to_bind) - {b["target"] for b in bindings}:
        return {"accepted": False, "feedback": "Missing live-global GOT binding"}
    if not short_entry_safe(binary, symbols[name], symbols):
        return {"accepted": False, "feedback": "Short function has insufficient safe redirect padding"}
    manifest = {"name": name, "entry": symbols[name]["address"], "size": symbols[name]["size"], "library": str(library),
                "export": instrumented_export, "bridges": bridges, "bindings": bindings}
    write_json(directory / "candidate_manifest.json", manifest)
    return {"accepted": True, "manifest": manifest, "feedback": "Compiled native candidate"}


AGENT = r'''
const spec = SPEC;
const app = Process.mainModule;
const base = spec.pie ? app.base : ptr(0);
const address = x => base.add(x);
const names = new Map(spec.functions.map(f => [address(f.address).toString(), f.name]));
const counts = {};
const loadedCandidates = new Map();
const executableAllocations = [];
Process.setExceptionHandler(details => {
  const context = {};
  for(const r of ['rip','rsp','rdi','rsi','rdx','rcx','r8','r9'])
    if(details.context[r]) context[r] = details.context[r].toString();
  let stack = [];
  try { stack = Thread.backtrace(details.context,Backtracer.FUZZY).map(p => DebugSymbol.fromAddress(p).toString()); } catch (_) {}
  send({event:'crash',type:details.type,address:details.address.toString(),context:context,
        memory:details.memory ? {operation:details.memory.operation,address:details.memory.address.toString()} : null, stack:stack});
  Thread.sleep(0.05);
  return false;
});
function symbol(p) {
  for(const f of spec.functions) {
    const start = address(f.address);
    if(p.compare(start) >= 0 && p.compare(start.add(f.size)) < 0) return f.name;
  }
  return names.get(p.toString()) || DebugSymbol.fromAddress(p).name || p.toString();
}
function snapshot(p, limit=128) {
  try { const r = Process.findRangeByAddress(p); if (r && r.protection.indexOf('r') >= 0)
    return Array.from(new Uint8Array(p.readByteArray(Math.min(limit, r.base.add(r.size).sub(p).toUInt32()))));
  } catch (_) {} return null;
}
for (const f of spec.functions) {
  if (f.size < 5) { send({event:'unsupported_trace',name:f.name,reason:'short entry'}); continue; }
  const p = address(f.address);
  const listener = Interceptor.attach(p, {
    onEnter(args) {
      counts[f.name] = (counts[f.name] || 0) + 1;
      this.record = counts[f.name] <= spec.sample_limit;
      if (!this.record) return;
      const regs = {};
      for (const r of ['rdi','rsi','rdx','rcx','r8','r9','rsp']) regs[r] = this.context[r].toString();
      send({event:'entry', name:f.name, thread:Process.getCurrentThreadId(), registers:regs,
        stack:[f.name].concat(Thread.backtrace(this.context, Backtracer.FUZZY).slice(0,32).map(symbol)),
        stack_method:'fuzzy', memory:[snapshot(args[0]),snapshot(args[1])],
        globals:spec.globals.map(g => ({name:g.name, bytes:snapshot(address(g.address),Math.min(128,g.size))}))});
    },
    onLeave(value) {
      if(this.record) send({event:'return',name:f.name,value:value.toString()});
      if(counts[f.name] >= spec.sample_limit) listener.detach();
    }
  });
}
for (const c of spec.candidates) {
  const m = Module.load(c.library);
  for (const b of c.bridges) m.base.add(b.offset).writePointer(address(b.address));
  for (const b of c.bindings) {
    const p = m.base.add(b.offset);
    const range = Process.findRangeByAddress(p);
    Memory.protect(p, Process.pointerSize, 'rw-');
    p.writePointer(address(b.address));
    Memory.protect(p, Process.pointerSize, range.protection);
  }
  const replacement = m.getExportByName(c.export);
  loadedCandidates.set(c.name,replacement);
  if(c.size < 5) {
    const entry = address(c.entry);
    const thunk = Memory.alloc(Process.pageSize,{near:entry,maxDistance:0x7fffffff});
    executableAllocations.push(thunk);
    Memory.patchCode(thunk,32,p => { const w = new X86Writer(p,{pc:thunk}); w.putJmpAddress(replacement); w.flush(); });
    Memory.patchCode(entry,5,p => { const w = new X86Writer(p,{pc:entry}); w.putJmpAddress(thunk); w.flush(); });
  } else Interceptor.replace(address(c.entry), replacement);
  let entries = 0;
  const listener = Interceptor.attach(replacement,{onEnter(){
    send({event:'replacement',name:c.name});
    if(++entries >= spec.sample_limit) listener.detach();
  }});
}
if(spec.direct_probe) {
  const probe = spec.direct_probe;
  const buffers = new Map();
  function argument(a) {
    if(typeof a === 'number') return a;
    if(a.string !== undefined) return Memory.allocUtf8String(a.string);
    if(a.ref) {
      if(!buffers.has(a.ref)) throw new Error('Unknown probe buffer: '+a.ref);
      return buffers.get(a.ref);
    }
    if(a.buffer) {
      if(a.buffer < 1 || a.buffer > 4096) throw new Error('Invalid probe allocation');
      const p = Memory.alloc(a.buffer);
      p.writeByteArray(new Uint8Array(a.buffer));
      if(a.bytes) p.writeByteArray(a.bytes);
      if(a.id) buffers.set(a.id,p);
      return p;
    }
    if(a.null) return ptr(0);
    throw new Error('Unknown probe argument');
  }
  const entry = app.findExportByName('main') || address(spec.main_entry);
  const dispatch = new NativeCallback((argc,argv) => {
    if(spec.set_program_name) new NativeFunction(address(spec.set_program_name),'void',['pointer'])(argv.readPointer());
    for(const setup of probe.setup || []) {
      if(!['stat','lstat','memcpy','memset'].includes(setup.symbol)) throw new Error('Disallowed probe setup');
      new NativeFunction(Module.getGlobalExportByName(setup.symbol),setup.return_type,setup.argument_types)(...setup.arguments.map(argument));
    }
    const target = loadedCandidates.get(probe.name) || address(probe.entry);
    const value = new NativeFunction(target,probe.return_type,probe.argument_types)(...probe.arguments.map(argument));
    let output = null;
    if(probe.return_kind === 'cstring') output = value.isNull() ? null : value.readUtf8String();
    else if(probe.return_type !== 'void') output = value.toString();
    const memory = (probe.memory_outputs || []).map(o => ({id:o.id,bytes:snapshot(buffers.get(o.id),o.size)}));
    const result = JSON.stringify({value:output,memory:memory})+'\n';
    const bytes = Array.from(unescape(encodeURIComponent(result))).map(c => c.charCodeAt(0));
    const text = Memory.alloc(bytes.length);
    text.writeByteArray(bytes);
    new NativeFunction(Module.getGlobalExportByName('write'),'long',['int','pointer','ulong'])(1,text,bytes.length);
    send({event:'direct_probe',name:probe.name,result:result});
    return 0;
  },'int',['int','pointer']);
  Interceptor.replace(entry,dispatch);
}
send({event:'ready'});
rpc.exports = { ping() { return true; } };
'''


def launch(spec: dict, argv: list[str]) -> int:
    import frida
    gate_read, gate_write = os.pipe()
    env = dict(os.environ)
    env["DECOMP_GATE_FD"] = str(gate_read)
    env["LD_PRELOAD"] = spec["gate"] + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    process = subprocess.Popen([Path(spec["binary"]).name, *argv], executable=spec["binary"],
                               env=env, pass_fds=(gate_read,))
    os.close(gate_read)
    session = None
    events = Path(spec["events"])
    expired = threading.Event()
    def terminate_overdue_process():
        if process.poll() is None:
            expired.set()
            process.kill()
    watchdog = threading.Timer(spec.get("runtime_timeout", 25), terminate_overdue_process)
    watchdog.daemon = True
    watchdog.start()
    def timed_out():
        with events.open("a") as stream:
            stream.write(json.dumps({"event": "timeout", "pid": process.pid, "argv": argv}) + "\n")
        print("Native validation timed out", file=sys.stderr)
        return 124
    try:
        session = frida.attach(process.pid)
        script = session.create_script(AGENT.replace("SPEC", json.dumps(spec), 1))
        ready = []
        errors = []
        def receive(message, data):
            item = message.get("payload", message)
            with events.open("a") as stream:
                stream.write(json.dumps({"pid": process.pid, "argv": argv,
                                         "test_id": os.environ.get("DECOMP_TEST_ID"), **item}) + "\n")
            if item.get("event") == "ready":
                ready.append(True)
            if message["type"] == "error":
                errors.append(message)
        script.on("message", receive)
        script.load()
        # Synchronous RPC ensures preceding send messages have been delivered.
        try:
            script.exports_sync.ping()
        except Exception:
            if errors:
                raise RuntimeError("Instrumentation setup failed: " + json.dumps(errors))
            raise
        if errors or not ready:
            raise RuntimeError("Instrumentation setup failed: " + json.dumps(errors))
        os.write(gate_write, b"1")
        os.close(gate_write)
        gate_write = -1
        code = process.wait()
        if expired.is_set():
            return timed_out()
        return code if code >= 0 else 128 - code
    except Exception as error:
        process.kill()
        process.wait()
        if expired.is_set():
            return timed_out()
        print("Binary isolation infrastructure failed: " + str(error), file=sys.stderr)
        return 125
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
        if gate_write >= 0:
            os.close(gate_write)
        if session:
            try:
                session.detach()
            except frida.InvalidOperationError:
                pass


def make_wrapper(clang: str, binary: Path, functions: list[str], candidates: list[dict],
                 directory: Path, globals_names: list[str] | None = None, direct_probe: dict | None = None,
                 runtime_timeout: float = 25) -> Path:
    if not math.isfinite(runtime_timeout) or runtime_timeout <= 0:
        raise ValueError("runtime_timeout must be positive and finite")
    directory.mkdir(parents=True, exist_ok=True)
    symbols = elf_symbols(binary)
    spec = {"binary": str(binary.resolve()), "gate": str(build_gate(clang, directory)),
            "pie": symbols["__elf_type__"] == "ET_DYN", "sample_limit": 8, "runtime_timeout": runtime_timeout,
            "functions": [{"name": n, "address": symbols[n]["address"], "size": symbols[n]["size"]} for n in functions],
            "globals": [{"name": n, "address": symbols[n]["address"], "size": symbols[n]["size"]} for n in globals_names or []],
            "candidates": candidates, "events": str((directory / "events.jsonl").resolve())}
    if direct_probe:
        validate_probe(direct_probe)
        if direct_probe.get("name") in {c["name"] for c in candidates} and direct_probe["name"] == "main":
            raise ValueError("Direct probes cannot replace main twice")
        spec["direct_probe"] = {**direct_probe, "entry": symbols[direct_probe["name"]]["address"]}
        spec["main_entry"] = symbols["main"]["address"]
        spec["set_program_name"] = symbols.get("set_program_name", {}).get("address")
    manifest = (directory / "runtime_spec.json").resolve()
    write_json(manifest, spec)
    wrapper = (directory / "instrumented").resolve()
    project = Path(__file__).resolve().parents[2]
    wrapper.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0,{str(project)!r})\n"
                       "from models.binary_decompilation.native_validation import main\n"
                       f"main({str(manifest)!r})\n")
    wrapper.chmod(0o755)
    return wrapper


def main(spec_path=None):
    if spec_path is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("spec")
        args, argv = parser.parse_known_args()
        spec_path = args.spec
    else:
        argv = sys.argv[1:]
    raise SystemExit(launch(json.loads(Path(spec_path).read_text()), argv))


if __name__ == "__main__":
    main()
