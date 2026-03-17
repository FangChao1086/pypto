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


def _normalize_arch(arch: str | None) -> str:
    """Normalize and validate arch names."""
    value = (arch or os.environ.get("PYPTO_JIT_ARCH") or "dav-c220").strip().lower()
    if value not in {"dav-c220-vec", "dav-c220-cube", "dav-c220"}:
        raise ValueError(f"Unsupported arch: {arch}")
    return value


def _build_bisheng_compile_flags(toolkit_home: str, arch: str) -> list[str]:
    """Build bisheng compile flags for the mixed two-step compile path."""
    arch = _normalize_arch(arch)
    common = [
        "-fPIC",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        f"-I{toolkit_home}/include",
    ]
    if arch != "dav-c220":
        raise ValueError(f"_build_bisheng_compile_flags only supports mix arch, got {arch}")
    return ["--cce-aicore-arch=dav-c220", *common]


def _build_bisheng_link_flags(arch: str) -> list[str]:
    """Build bisheng link flags for shared-library output."""
    arch = _normalize_arch(arch)
    flags = ["-fPIC", "-shared"]
    if arch == "dav-c220":
        flags.append("--cce-fatobj-link")
    return flags


def _build_bisheng_flags(toolkit_home: str, arch: str) -> list[str]:
    """Build bisheng flags for the original single-command compile path."""
    arch = _normalize_arch(arch)
    common = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        f"-I{toolkit_home}/include",
    ]
    if arch == "dav-c220-vec":
        return ["--cce-aicore-arch=dav-c220-vec", *common]
    if arch == "dav-c220-cube":
        return ["--cce-aicore-arch=dav-c220-cube", *common]
    raise ValueError(f"_build_bisheng_flags only supports vec/cube arch, got {arch}")


def compile(prog, clean_up=False, timeout=20, arch: str = "dav-c220"):
    """Compile a PTO program to a shared library.

    Args:
        prog: The PTO program to compile.
        clean_up: Whether to remove intermediate files after compilation.
        timeout: Compilation timeout in seconds.
        arch: Target architecture. Options:
            - "dav-c220-vec": Vector operations only (add, mul, etc.)
            - "dav-c220-cube": Cube operations only (matmul, etc.)
            - "dav-c220": Mixed vector and cube operations (default)
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    Path("./build").mkdir(parents=True, exist_ok=True)
    ir_path = "./build/kernel.pto"  # TODO: use Python `tempfile` module
    raw_cpp_path = "./build/kernel.cpp"
    final_kernel = "./build/call_kernel.cpp"
    obj_path = "./build/call_kernel.o"
    lib_path = "./build/call_kernel.so"

    # step 1, Program -> PtoAs-mlir
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(prog))
    with open(ir_path, "w") as f:
        f.write(mlir_code)

    # step 2, IR -> CPP
    # TODO: use `ptoas --enable-insert-sync` so no need for explicit sync in frontend
    # need https://github.com/zhangstevenunity/PTOAS/issues/10
    # Note: --pto-level=level3 is required for addr operand support
    result = subprocess.run(
        ["ptoas", ir_path, "--enable-insert-sync", "--pto-level=level3", "-o", raw_cpp_path],
        check=False, timeout=timeout, capture_output=True
    )
    if result.returncode != 0:
        print(f"ptoas failed with return code {result.returncode}")
        print(f"stderr: {result.stderr.decode()}")
        return None

    # Step 3, preprocess cpp source
    # TODO: should extend `ptoas` emitc to largely replace this ad-doc editing
    content = Path(raw_cpp_path).read_text(encoding="utf-8")
    edited_content = convert(content=content)
    Path(final_kernel).write_text(edited_content, encoding="utf-8")

    # Step 4, cpp -> so
    PTO_LIB_PATH = os.environ["ASCEND_TOOLKIT_HOME"]
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    if not ASCEND_HOME_PATH:
        raise RuntimeError("ASCEND_HOME_PATH is not set")
    LD_LIB_PATH = ASCEND_HOME_PATH + "/lib64/"
    runtime_includes = [
        f"-I{ASCEND_HOME_PATH}/include",
        f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
        f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
        f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
    ]
    arch = _normalize_arch(arch)

    PTO_ISA_INCLUDE = os.environ.get("PTO_ISA_INCLUDE", "")
    if not PTO_ISA_INCLUDE:
        possible_paths = [
            os.path.expanduser("~/PTO-IR/pto-isa/include"),
            os.path.expanduser("~/pto-isa/include"),
            "/usr/local/include",
        ]
        for p in possible_paths:
            if os.path.exists(os.path.join(p, "pto", "pto-inst.hpp")):
                PTO_ISA_INCLUDE = p
                break

    if arch == "dav-c220":
        compile_flags = _build_bisheng_compile_flags(PTO_LIB_PATH, arch)
        link_flags = _build_bisheng_link_flags(arch)

        compile_flags.extend(runtime_includes)
        if PTO_ISA_INCLUDE:
            compile_flags.append(f"-I{PTO_ISA_INCLUDE}")

        result = subprocess.run(
            ["bisheng", *compile_flags, "-c", final_kernel, "-o", obj_path],
            check=False, timeout=timeout, capture_output=True
        )
        if result.returncode != 0:
            print(f"bisheng compile step failed with return code {result.returncode}")
            print(f"stdout: {result.stdout.decode()}")
            print(f"stderr: {result.stderr.decode()}")
            return None

        result = subprocess.run(
            ["bisheng", *link_flags, obj_path, "-L", LD_LIB_PATH, "-lruntime", "-o", lib_path],
            check=False, timeout=timeout, capture_output=True
        )
        if result.returncode != 0:
            print(f"bisheng link step failed with return code {result.returncode}")
            print(f"stdout: {result.stdout.decode()}")
            print(f"stderr: {result.stderr.decode()}")
            return None
    else:
        flags = _build_bisheng_flags(PTO_LIB_PATH, arch)

        flags.extend(runtime_includes)
        if PTO_ISA_INCLUDE:
            flags.append(f"-I{PTO_ISA_INCLUDE}")

        result = subprocess.run(
            ["bisheng", *flags, final_kernel, "-L", LD_LIB_PATH, "-lruntime", "-o", lib_path],
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
        if arch == "dav-c220" and Path(obj_path).exists():
            os.remove(obj_path)

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


