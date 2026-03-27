#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import ctypes
import dataclasses
import functools
import inspect
import os
import re
import subprocess
from pathlib import Path

import torch

from pypto import DataType, backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen
from pypto.pypto_core.ir import ConstInt, PtrType, ScalarType, TensorType, Var

_jit_functions = {}
_kernel_functions = {}
_compiled_cache = {}
_default_device = "cpu"

# torch.dtype → pl DataType (used to validate tensor args against kernel signature)
_TORCH_TO_PL_DTYPE: dict = {
    torch.float16:  DataType.FP16,
    torch.bfloat16: DataType.BF16,
    torch.float32:  DataType.FP32,
    torch.int8:     DataType.INT8,
    torch.int16:    DataType.INT16,
    torch.int32:    DataType.INT32,
    torch.int64:    DataType.INT64,
    torch.uint8:    DataType.UINT8,
    torch.bool:     DataType.BOOL,
}

# pl DataType → ctypes type (used to wrap scalar args before the ctypes call)
# Keyed by str(DataType) because nanobind DataType objects returned from IR introspection
# are distinct Python objects that compare equal (==) but have different hash values,
# making DataType-keyed dicts unreliable.
_PL_DTYPE_TO_CTYPE: dict[str, type] = {
    str(DataType.FP32):   ctypes.c_float,
    str(DataType.INT8):   ctypes.c_int8,
    str(DataType.INT16):  ctypes.c_int16,
    str(DataType.INT32):  ctypes.c_int32,
    str(DataType.INT64):  ctypes.c_int64,
    str(DataType.UINT8):  ctypes.c_uint8,
    str(DataType.UINT16): ctypes.c_uint16,
    str(DataType.UINT32): ctypes.c_uint32,
    str(DataType.UINT64): ctypes.c_uint64,
    str(DataType.BOOL):   ctypes.c_bool,
    str(DataType.INDEX):  ctypes.c_int64,
}


@dataclasses.dataclass
class ParamSpec:
    """Description of a single kernel parameter extracted from the IR."""

    name: str
    kind: str                    # "tensor" | "ptr" | "scalar"
    dtype: DataType              # element dtype
    # Tensor dims: positive int = static, str = named dynamic var, -1 = unnamed dynamic; None for ptr/scalar.
    shape: list[int | str] | None


@dataclasses.dataclass
class CompiledKernel:
    """Result of compile(): the .so path together with IR-derived parameter metadata."""

    lib_path: str
    param_specs: list[ParamSpec]

# Pattern for __global__ AICORE void kernel_name(params)
KERNEL_PATTERN = re.compile(
    r"__global__\s+AICORE\s+void\s+(\w+)\s*\((.*)\)\s*\{",
    re.DOTALL,
)

# Pattern for a single param: __gm__ type* name or similar
PARAM_PATTERN = re.compile(r"__gm__\s*(\w+)\s*\*\s*(\w+)")

# Pattern for a scalar param: type name (no pointer, no __gm__)
SCALAR_PARAM_PATTERN = re.compile(r"^(\w+)\s+(\w+)$")


def parse_kernel_signature(
    line: str, rest: str = ""
) -> tuple[str, list[tuple[str, str, bool]]] | None:
    """Match __global__ AICORE void name(...) { and return (name, [(type, name, is_ptr), ...])."""
    combined = (line + rest).strip()
    m = KERNEL_PATTERN.search(combined)  # search in case of leading whitespace
    if not m:
        return None
    kernel_name = m.group(1)
    params_str = m.group(2).strip()
    if not params_str:
        return (kernel_name, [])
    params: list[tuple[str, str, bool]] = []
    for raw_part in params_str.split(","):
        part = raw_part.strip()
        pm = PARAM_PATTERN.search(part)
        if pm:
            params.append((pm.group(1), pm.group(2), True))
            continue
        scalar = SCALAR_PARAM_PATTERN.match(part)
        if scalar:
            params.append((scalar.group(1), scalar.group(2), False))
    return (kernel_name, params)


def build_call_wrapper(kernel_name: str, params: list[tuple[str, str, bool]]) -> str:
    """Build extern "C" void call_kernel(uint32_t blockDim, void* stream, ...)."""
    param_decls = []
    cast_args = []
    for typ, name, is_ptr in params:
        if is_ptr:
            param_decls.append(f"uint8_t* {name}")
            cast_args.append(f"({typ} *){name}")
        else:
            param_decls.append(f"{typ} {name}")
            cast_args.append(name)
    return f'''extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    {", ".join(param_decls)})
{{
    {kernel_name}<<<blockDim, nullptr, stream>>>({", ".join(cast_args)});
}}'''


def convert(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines:
        return content

    kernel_name: str | None = None
    kernel_params: list[tuple[str, str, bool]] = []

    # Find kernel signature (single line or multi-line)
    for idx, rline in enumerate(lines):
        if "__global__" in rline and "AICORE" in rline:
            # Try single line first, then with following lines
            combined = "".join(lines[idx : idx + 3]).strip()
            parsed = parse_kernel_signature(combined, "")
            if parsed:
                kernel_name, kernel_params = parsed
                break

    result = "".join(lines)

    # Append call_kernel wrapper
    if kernel_name and kernel_params is not None:
        wrapper = build_call_wrapper(kernel_name, kernel_params)
        result = result.rstrip()
        if not result.endswith("\n"):
            result += "\n"
        result += "\n" + wrapper
        if not result.endswith("\n"):
            result += "\n"

    return result


def _pl_dtype_to_torch(dtype: DataType):
    """Return the torch.dtype that corresponds to a pl DataType, or None if unknown."""
    for torch_dtype, pl_dtype in _TORCH_TO_PL_DTYPE.items():
        if pl_dtype == dtype:
            return torch_dtype
    return None


def _extract_param_specs(prog) -> list[ParamSpec]:
    """Extract parameter descriptions from the first function in an ir.Program."""
    func = next(iter(prog.functions.values()))
    specs: list[ParamSpec] = []
    for var in func.params:
        t = var.type
        if isinstance(t, TensorType):
            shape = [
                s.value if isinstance(s, ConstInt)
                else (s.name if isinstance(s, Var) else -1)
                for s in t.shape
            ]
            specs.append(ParamSpec(var.name, "tensor", t.dtype, shape))
        elif isinstance(t, PtrType):
            specs.append(ParamSpec(var.name, "ptr", t.dtype, None))
        elif isinstance(t, ScalarType):
            specs.append(ParamSpec(var.name, "scalar", t.dtype, None))
    return specs


def _collect_dyn_vars(param_specs: list[ParamSpec]) -> list[str]:
    """Return dynamic shape variable names in first-occurrence order (mirrors ptoas codegen)."""
    seen: set[str] = set()
    result: list[str] = []
    for spec in param_specs:
        if spec.kind == "tensor" and spec.shape is not None:
            for dim in spec.shape:
                if isinstance(dim, str) and dim not in seen:
                    result.append(dim)
                    seen.add(dim)
    return result


def _validate_tensor_arg(
    i: int, arg, spec: ParamSpec, dyn_var_values: dict[str, int]
) -> None:
    """Validate one tensor/ptr arg; update dyn_var_values in-place for dynamic dims."""
    if not isinstance(arg, torch.Tensor):
        raise TypeError(
            f"arg[{i}] '{spec.name}': expected torch.Tensor, got {type(arg).__name__}"
        )
    if spec.kind == "ptr" or spec.shape is None:
        return

    expected_dtype = _pl_dtype_to_torch(spec.dtype)
    if expected_dtype is not None and arg.dtype != expected_dtype:
        raise TypeError(
            f"arg[{i}] '{spec.name}': dtype mismatch — "
            f"expected {expected_dtype}, got {arg.dtype}"
        )
    if len(arg.shape) != len(spec.shape):
        raise TypeError(
            f"arg[{i}] '{spec.name}': rank mismatch — "
            f"expected {len(spec.shape)}D, got {len(arg.shape)}D"
        )
    for d, (actual, expected) in enumerate(zip(arg.shape, spec.shape)):
        if isinstance(expected, int) and expected not in (-1, actual):
            raise TypeError(
                f"arg[{i}] '{spec.name}': dim[{d}] mismatch — "
                f"expected {expected}, got {actual}"
            )
        if isinstance(expected, str):
            if expected in dyn_var_values and dyn_var_values[expected] != actual:
                raise TypeError(
                    f"arg[{i}] '{spec.name}': dynamic shape variable '{expected}' "
                    f"mismatch — previously {dyn_var_values[expected]}, "
                    f"got {actual} at dim[{d}]"
                )
            dyn_var_values.setdefault(expected, actual)


def _validate_scalar_arg(i: int, arg, spec: ParamSpec) -> None:
    """Validate one scalar arg against its ParamSpec."""
    if not isinstance(arg, (int, float, bool)):
        raise TypeError(
            f"arg[{i}] '{spec.name}': expected Python scalar (int/float/bool), "
            f"got {type(arg).__name__}"
        )
    # bool is a subclass of int: valid for BOOL and integer dtypes only.
    if isinstance(arg, bool):
        if not (
            spec.dtype == DataType.BOOL
            or spec.dtype.is_signed_int()
            or spec.dtype.is_unsigned_int()
        ):
            raise TypeError(
                f"arg[{i}] '{spec.name}': bool value passed for non-boolean/non-integer "
                f"dtype {spec.dtype}"
            )
    elif isinstance(arg, float) and not spec.dtype.is_float():
        raise TypeError(
            f"arg[{i}] '{spec.name}': float value passed for non-float dtype {spec.dtype}"
        )
    elif isinstance(arg, int) and not spec.dtype.is_signed_int() and not spec.dtype.is_unsigned_int():
        raise TypeError(
            f"arg[{i}] '{spec.name}': int value passed for non-integer dtype {spec.dtype}"
        )


def _expand_tiling_args(args: tuple) -> tuple:
    """If the last arg is a tiling class instance, expand it to flat scalar values.

    Tiling fields are flattened in declaration order:
      - scalar field  → one value (int/float/bool)
      - Array[T, N]  → N individual values

    Args:
        args: Runtime arguments tuple passed to launch.

    Returns:
        New args tuple with tiling instance replaced by flat scalar values,
        or the original args if no tiling instance is present.
    """
    from pypto.language.typing.tiling import (
        is_tiling_class, get_tiling_fields, ArrayFieldInfo,
    )
    if not args:
        return args
    last_arg = args[-1]
    if not is_tiling_class(type(last_arg)):
        return args

    fields = get_tiling_fields(type(last_arg))
    flat_values: list = []
    for field_name, field_info in fields.items():
        val = getattr(last_arg, field_name)
        if isinstance(field_info, ArrayFieldInfo):
            if not hasattr(val, "__getitem__") or not hasattr(val, "__len__"):
                raise TypeError(
                    f"Tiling field '{field_name}' is Array[T, {field_info.size}]: "
                    f"expected an indexable sequence (e.g. Array[int, {field_info.size}]([...])), "
                    f"got {type(val).__name__!r}"
                )
            if len(val) != field_info.size:
                raise ValueError(
                    f"Tiling field '{field_name}' expected {field_info.size} elements, got {len(val)}"
                )
            for i in range(field_info.size):
                flat_values.append(val[i])
        else:
            flat_values.append(val)
    return args[:-1] + tuple(flat_values)


def _validate_args(args: tuple, param_specs: list[ParamSpec]) -> None:
    """Validate runtime args against the kernel's IR parameter descriptions.

    Raises:
        TypeError: On count mismatch, wrong arg type, dtype mismatch, or shape mismatch.
    """
    if len(args) != len(param_specs):
        raise TypeError(f"Expected {len(param_specs)} args, got {len(args)}")
    dyn_var_values: dict[str, int] = {}
    for i, (arg, spec) in enumerate(zip(args, param_specs)):
        if spec.kind in ("tensor", "ptr"):
            _validate_tensor_arg(i, arg, spec, dyn_var_values)
        else:
            _validate_scalar_arg(i, arg, spec)


def _args_to_ctypes(args: tuple, param_specs: list[ParamSpec]) -> list:
    """Convert runtime args to ctypes values for the call_kernel ABI.

    - tensor / ptr → ctypes.c_void_p (raw data pointer)
    - scalar       → ctypes type matching the DataType (e.g. c_float, c_int32)
    - dynamic dims → ctypes.c_int64, appended in first-occurrence order
    """
    result = []
    dyn_var_values: dict[str, int] = {}  # insertion order = first-occurrence order (Python 3.7+)

    for arg, spec in zip(args, param_specs):
        if spec.kind in ("tensor", "ptr"):
            result.append(ctypes.c_void_p(arg.data_ptr()))
            if spec.kind == "tensor" and spec.shape is not None:
                for d, dim in enumerate(spec.shape):
                    if isinstance(dim, str):
                        dyn_var_values.setdefault(dim, arg.shape[d])
        else:
            ctype = _PL_DTYPE_TO_CTYPE.get(str(spec.dtype))
            if ctype is None:
                raise TypeError(
                    f"No ctypes mapping for scalar dtype {spec.dtype} "
                    f"(param '{spec.name}'). FP16/BF16 scalars are not supported."
                )
            result.append(ctype(arg))

    result.extend(ctypes.c_int64(v) for v in dyn_var_values.values())
    return result


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())

def _generate_caller_cpp(
    kernel_params: list[tuple[str, str, bool]],
    kernel_cpp_name: str,
    kernel_name: str,
    has_cross_core_sync: bool = False,
    enable_print_debug: bool = False,
) -> str:
    """Generate extern "C" wrapper that calls the __global__ kernel.
    
    Args:
        param_specs: List of parameter specifications extracted from IR
        kernel_cpp_name: Name of the kernel .cpp file to include
        kernel_name: Name of the kernel function to call
        has_cross_core_sync: Whether the kernel uses cross-core sync ops
        enable_print_debug:
            Reserved compatibility switch from compile(); currently this wrapper
            does not need extra caller-side code changes for print debug.
    
    Returns:
        Generated caller.cpp content as string
    """
    cpp_params = []
    kernel_args = []

    for typ, name, is_ptr in kernel_params:
        if is_ptr:
            cpp_params.append(f"uint8_t* {name}")
            kernel_args.append(f"({typ} *){name}")
        else:
            cpp_params.append(f"{typ} {name}")
            kernel_args.append(name)
    sig = ", ".join(["uint32_t blockDim", "void* stream"] + cpp_params)
    call_args = ", ".join(kernel_args)

    if has_cross_core_sync:
        call_args = f"{call_args}, (int64_t*)ffts" if call_args else "(int64_t*)ffts"
        return (
            f'#include "runtime/rt_ffts.h"\n'
            f'#include "{kernel_cpp_name}"\n'
            f'extern "C" void call_kernel({sig})\n'
            "{{\n"
            "    uint64_t ffts = 0;\n"
            "    uint32_t fftsLen = 0;\n"
            "    rtGetC2cCtrlAddr(&ffts, &fftsLen);\n"
            f"    {kernel_name}<<<blockDim, nullptr, stream>>>({call_args});\n"
            "}}\n"
        )

    return (
        f'#include "{kernel_cpp_name}"\n'
        f'extern "C" void call_kernel({sig})\n'
        "{{\n"
        f"    {kernel_name}<<<blockDim, nullptr, stream>>>({call_args});\n"
        "}}\n"
    )

def _inject_set_ffts_to_mlir(mlir_code: str) -> str:
    """
    在 MLIR 代码的第一个 func.func 参数列表末尾添加 memref<?xi64> 参数，
    并在函数体开头插入 pto.set_ffts。
    使用括号计数定位参数列表的右括号，避免被类型中的尖括号干扰。
    """
    func_pattern = re.compile(r'(func\.func\s+@\w+\s*\()')
    match = func_pattern.search(mlir_code)
    if not match:
        return mlir_code
    start_pos = match.end() - 1
    pos = start_pos + 1
    depth = 1
    while depth > 0 and pos < len(mlir_code):
        if mlir_code[pos] == '(':
            depth += 1
        elif mlir_code[pos] == ')':
            depth -= 1
        pos += 1
    if depth != 0:
        return mlir_code
    right_paren_pos = pos - 1
    param_str = mlir_code[start_pos+1:right_paren_pos]
    args = re.findall(r'%arg(\d+)', param_str)
    next_idx = max(int(i) for i in args) + 1 if args else 0
    new_param = f", %arg{next_idx}: memref<?xi64>" if param_str.strip() else f"%arg{next_idx}: memref<?xi64>"
    new_param_str = param_str + new_param
    new_sig = mlir_code[:start_pos+1] + new_param_str + mlir_code[right_paren_pos:]
    brace_pos = new_sig.find('{', right_paren_pos)
    if brace_pos == -1:
        return new_sig
    # 插入 pto.set_ffts 操作
    indent = "    "
    new_op = f"\n{indent}pto.set_ffts %arg{next_idx} : memref<?xi64>"
    final_code = new_sig[:brace_pos+1] + new_op + new_sig[brace_pos+1:]
    return final_code

def _normalize_arch(arch: str | None) -> str:
    """Normalize and validate arch names to 'a2', 'a3', or 'a5'."""
    value = (arch or os.environ.get("PYPTO_JIT_ARCH") or "a3").strip().lower()
    if value not in {"a2", "a3", "a5"}:
        raise ValueError(
            f"Unsupported arch: {arch!r}, expected one of 'a2', 'a3', 'a5'"
        )
    return value


def _build_bisheng_flags(toolkit_home: str, arch: str, cpp_content: str, has_cross_sync: bool,
                         enable_print_debug: bool) -> list[str]:
    """Build bisheng flags for single-command shared-library compilation.

    Determines the npu_arch from the arch ('a2'/'a3'/'a5') and the presence
    of __DAV_CUBE__ / __DAV_VEC__ macros in the generated C++ source.
    """
    arch = _normalize_arch(arch)

    has_cube = "__DAV_CUBE__" in cpp_content
    has_vec = "__DAV_VEC__" in cpp_content

    if has_cross_sync and not (has_cube and has_vec):
        if has_cube:
            raise ValueError(
                f"Contains ffts cross sync but vector code is missing."
            )
        elif has_vec:
            raise ValueError(
                f"Contains ffts cross sync but cube code is missing."
            )
    if has_cube and has_vec:
        npu_arch = "dav-c220" if arch in ("a2", "a3") else "dav-c310"
    elif has_cube:
        npu_arch = "dav-c220-cube" if arch in ("a2", "a3") else "dav-c310-cube"
    elif has_vec:
        npu_arch = "dav-c220-vec" if arch in ("a2", "a3") else "dav-c310-vec"
    else:
        npu_arch = "dav-c220" if arch in ("a2", "a3") else "dav-c310"

    common = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        f"-I{toolkit_home}/include",
    ]
    if enable_print_debug:
        common.extend([
            "--cce-enable-print",
            "-D_DEBUG",
            "-DCCEBlockMaxSize=1048576",
            "-DPTOAS_ENABLE_CCE_PRINT=1",
        ])
    flags = [f"--cce-aicore-arch={npu_arch}"]
    if has_cube and has_vec:
        flags.append("--cce-fatobj-link")
    return [*flags, *common]


def compile(prog, clean_up=False, timeout=20, arch: str = "a3", enable_print_debug: bool | None = None):
    """Compile a PTO program to a shared library.

    Args:
        prog: The PTO program or KernelDef to compile.
            If a :class:`KernelDef`, AST parsing is triggered with *arch*
            to produce an ``ir.Program`` before code generation.
        clean_up: Whether to remove intermediate files after compilation.
        timeout: Compilation timeout in seconds.
        arch: Target architecture. Options: "a2", "a3", "a5".
        enable_print_debug:
            - None: auto-enable when PTO MLIR contains `pto.tprint` or `pto.print`
            - True: force enable device-side debug printing flags
            - False: force disable device-side debug printing flags
    """
    # Deferred compilation: parse KernelDef → ir.Program with arch info
    from pypto.frontend.kernel import KernelDef
    if isinstance(prog, KernelDef):
        prog = prog.parse(npu_arch=arch)

    arch = _normalize_arch(arch)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    Path("./build").mkdir(parents=True, exist_ok=True)
    ir_path = "./build/kernel.pto"  # TODO: use Python `tempfile` module
    raw_cpp_path = "./build/kernel.cpp"
    final_kernel = "./build/call_kernel.cpp"
    lib_path = "./build/call_kernel.so"

    if arch in ("a2", "a3"):
        os.environ["npu_arch"] = "dav-c220"
    else:
        os.environ["npu_arch"] = "dav-c310"

    # step 1, Program -> PtoAs-mlir
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(prog))

    # Auto-detect cross-core sync from IR
    has_cross_sync = "pto.sync.set" in mlir_code
    needs_print_debug = ("pto.tprint" in mlir_code) or ("pto.print" in mlir_code)
    if has_cross_sync:
        mlir_code = _inject_set_ffts_to_mlir(mlir_code)
    with open(ir_path, "w") as f:
        f.write(mlir_code)

    # step 2, IR -> CPP
    # TODO: use `ptoas --enable-insert-sync` so no need for explicit sync in frontend
    # need https://github.com/zhangstevenunity/PTOAS/issues/10
    # Note: --pto-level=level3 is required for addr operand support
    result = subprocess.run(
        ["ptoas", ir_path, "--enable-insert-sync", "--pto-level=level3", f"--pto-arch={arch}", "-o", raw_cpp_path],
        check=False, timeout=timeout, capture_output=True
    )
    if result.returncode != 0:
        print(f"ptoas failed with return code {result.returncode}")
        print(f"stderr: {result.stderr.decode()}")
        return None

    # Step 3, preprocess cpp source
    # TODO: should extend `ptoas` emitc to largely replace this ad-doc editing
    content = Path(raw_cpp_path).read_text(encoding="utf-8")
    kernel_name = None
    kernel_params = []
    for line in content.splitlines():
        if "__global__" in line and "AICORE" in line:
            parsed = parse_kernel_signature(line, "")
            if parsed:
                kernel_name, kernel_params = parsed
                break
    if kernel_name is None:
        raise RuntimeError("Could not find kernel name in generated C++ code")
    resolved_enable_print_debug = needs_print_debug if enable_print_debug is None else enable_print_debug

    if has_cross_sync:
        kernel_params = kernel_params[:-1]
    caller_content = _generate_caller_cpp(
        kernel_params=kernel_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=kernel_name,
        has_cross_core_sync=has_cross_sync,
        enable_print_debug=resolved_enable_print_debug,
    )
    Path(final_kernel).write_text(caller_content, encoding="utf-8")

    # Step 4, cpp -> so
    PTO_LIB_PATH = os.environ["ASCEND_TOOLKIT_HOME"]
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    if not ASCEND_HOME_PATH:
        raise RuntimeError("ASCEND_HOME_PATH is not set")
    LD_LIB_PATH = ASCEND_HOME_PATH + "/lib64/"

    runtime_includes = [
        f"-I{ASCEND_HOME_PATH}/include",
        f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
        f"-I{ASCEND_HOME_PATH}/pkg_inc/profiling",
        f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
        f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
    ]
    
    flags = _build_bisheng_flags(
        PTO_LIB_PATH,
        arch,
        content,
        has_cross_sync,
        enable_print_debug=resolved_enable_print_debug,
    )
    flags.extend(runtime_includes)
    result = subprocess.run(
        ["bisheng", *flags, final_kernel, "-L", LD_LIB_PATH, "-lruntime", "-lprofapi", "-o", lib_path],
        check=False, timeout=timeout, capture_output=True
    )
    if result.returncode != 0:
        print(f"bisheng compilation failed with return code {result.returncode}")
        print(f"stdout: {result.stdout.decode()}")
        print(f"stderr: {result.stderr.decode()}")
        return None

    if clean_up:
        os.remove(ir_path)
        os.remove(raw_cpp_path)
        os.remove(final_kernel)

    return CompiledKernel(lib_path=lib_path, param_specs=_extract_param_specs(prog))

def load_lib(lib_path: str, param_specs: list[ParamSpec], clean_up: bool = False):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # TODO: extend kernel to multi-core

    def func_wrapper(*args, block_dim=default_block_dim, stream=None):
        args = _expand_tiling_args(args)
        _validate_args(args, param_specs)
        if stream is None:
            stream = torch.npu.current_stream()
        ctypes_args = _args_to_ctypes(args, param_specs)
        lib.call_kernel(block_dim, stream._as_parameter_, *ctypes_args)

    if clean_up:
        os.remove(lib_path)

    return func_wrapper


def launch(stream=None, block_dim=1, compiled_result: "CompiledKernel | str" = "", *args):
    if isinstance(compiled_result, str):
        if compiled_result == "":
            raise RuntimeError("compiled_result is empty")
        lib_path: str = compiled_result
        param_specs: list[ParamSpec] = []
    else:
        lib_path = compiled_result.lib_path
        param_specs = compiled_result.param_specs
    if stream is None:
        stream = torch.npu.current_stream()
    compiled_func = load_lib(lib_path, param_specs)
    compiled_func(*args, block_dim=block_dim, stream=stream)


def jit(target=None, optimize: bool = True, cache: bool = True,
    preprocess: bool = True,
    *dargs, **kwargs):
    """Mark a function for JIT compilation.

    Args:
        target: Compilation target ('cpu', 'npu').
        optimize: Whether to enable optimizations.
        cache: Whether to cache compilation results.

    Example::

        @pto.jit
        def add(a, b):
            workspace_size = 100
            pto.launch(add_kernel)
            return workspace_size
    """

    def decorator(f):
        name = f.__name__
        signature = inspect.signature(f)

        # Strip decorator lines before storing source.
        try:
            source_lines = inspect.getsource(f).split('\n')
            source_lines = [line for line in source_lines
                          if '@pto.jit' not in line and '@pto.kernel' not in line]
            source_code = '\n'.join(source_lines).strip()
        except OSError:
            source_code = "<source unavailable>"
        # Store JIT function metadata.
        _jit_functions[name] = {
            'func': f,
            'name': name,
            'signature': signature,
            'source_code': source_code,
            'target': target or 'cpu',
            'optimize': optimize,
            'cache': cache,
            'kwargs': kwargs
        }

        @functools.wraps(f)
        def wrapper(*args, **kwargs_):
            # if name in _compiled_cache:
            #     return _compiled_cache[name](*args, **kwargs_)

            # jit cache not hit — fall back to Python execution.
            return f(*args, **kwargs_)

        return wrapper

    if callable(target):
        return decorator(target)
    else:
        return decorator


