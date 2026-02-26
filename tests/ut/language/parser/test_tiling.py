# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tiling parameter support in the PyPTO language DSL."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from pypto.language.typing.tiling import (
    Array,
    ArrayFieldInfo,
    ScalarFieldInfo,
    get_tiling_fields,
    is_tiling_class,
)
from pypto.pypto_core import DataType


class TestTilingUtilities:
    """Tests for is_tiling_class and get_tiling_fields utilities."""

    def test_is_tiling_class_with_int_fields(self):
        class T:
            x: int
            y: int

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_float_fields(self):
        class T:
            a: float

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_bool_fields(self):
        class T:
            flag: bool

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_mixed_valid_fields(self):
        class T:
            x: int
            y: float
            enabled: bool

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_invalid_field_type(self):
        class T:
            x: int
            name: str  # str is not a valid tiling field type

        assert is_tiling_class(T) is False

    def test_is_tiling_class_with_no_annotations(self):
        class T:
            pass

        assert is_tiling_class(T) is False

    def test_is_tiling_class_with_non_class(self):
        assert is_tiling_class(42) is False
        assert is_tiling_class("string") is False
        assert is_tiling_class(None) is False

    def test_get_tiling_fields_dtype_mapping(self):
        class T:
            x: int
            y: float
            flag: bool

        fields = get_tiling_fields(T)
        assert fields == {
            "x": ScalarFieldInfo(DataType.INT32),
            "y": ScalarFieldInfo(DataType.FP32),
            "flag": ScalarFieldInfo(DataType.BOOL),
        }

    def test_get_tiling_fields_preserves_order(self):
        class T:
            c: float
            a: int
            b: bool

        fields = get_tiling_fields(T)
        assert list(fields.keys()) == ["c", "a", "b"]


class TestArrayType:
    """Tests for Array type and get_tiling_fields with array fields."""

    def test_array_type_creation(self):
        class T:
            offsets: Array[int, 4]

        assert is_tiling_class(T) is True
        fields = get_tiling_fields(T)
        assert isinstance(fields["offsets"], ArrayFieldInfo)
        assert fields["offsets"].dtype == DataType.INT32
        assert fields["offsets"].size == 4

    def test_array_type_float(self):
        class T:
            scales: Array[float, 2]

        fields = get_tiling_fields(T)
        assert isinstance(fields["scales"], ArrayFieldInfo)
        assert fields["scales"].dtype == DataType.FP32
        assert fields["scales"].size == 2

    def test_array_type_bool(self):
        class T:
            flags: Array[bool, 3]

        fields = get_tiling_fields(T)
        assert isinstance(fields["flags"], ArrayFieldInfo)
        assert fields["flags"].dtype == DataType.BOOL
        assert fields["flags"].size == 3

    def test_array_type_invalid_dtype_raises(self):
        with pytest.raises(TypeError, match="Array element type must be"):
            Array[str, 4]

    def test_array_type_zero_size_raises(self):
        with pytest.raises(ValueError, match="Array size must be a positive integer"):
            Array[int, 0]

    def test_array_type_negative_size_raises(self):
        with pytest.raises(ValueError, match="Array size must be a positive integer"):
            Array[int, -1]

    def test_array_bool_as_size_raises(self):
        with pytest.raises(ValueError, match="Array size must be a positive integer"):
            Array[int, True]

    def test_is_tiling_class_with_array_field(self):
        class T:
            offsets: Array[int, 4]

        assert is_tiling_class(T) is True

    def test_is_tiling_class_mixed_scalar_and_array(self):
        class T:
            n: int
            offsets: Array[int, 3]
            scale: float

        assert is_tiling_class(T) is True

    def test_get_tiling_fields_returns_array_field_info(self):
        class T:
            offsets: Array[int, 3]

        fields = get_tiling_fields(T)
        assert "offsets" in fields
        info = fields["offsets"]
        assert isinstance(info, ArrayFieldInfo)
        assert info.dtype == DataType.INT32
        assert info.size == 3

    def test_get_tiling_fields_mixed(self):
        class T:
            n: int
            offsets: Array[float, 2]

        fields = get_tiling_fields(T)
        assert isinstance(fields["n"], ScalarFieldInfo)
        assert fields["n"].dtype == DataType.INT32
        assert isinstance(fields["offsets"], ArrayFieldInfo)
        assert fields["offsets"].dtype == DataType.FP32
        assert fields["offsets"].size == 2


class TestTilingParameter:
    """Tests for tiling parameter parsing in @pl.function."""

    def test_tiling_only_param_flattens_to_scalar_params(self):
        class Tiling:
            x: int
            y: float

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.x
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 2
        param_names = [p.name for p in kernel.params]
        assert "tiling_x" in param_names
        assert "tiling_y" in param_names

    def test_tiling_scalar_dtypes_are_correct(self):
        class Tiling:
            n: int
            scale: float
            flag: bool

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.n
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 3
        param_map = {p.name: p for p in kernel.params}
        assert isinstance(param_map["tiling_n"].type, ir.ScalarType)
        assert param_map["tiling_n"].type.dtype == DataType.INT32
        assert param_map["tiling_scale"].type.dtype == DataType.FP32
        assert param_map["tiling_flag"].type.dtype == DataType.BOOL

    def test_tensors_plus_tiling_last(self):
        class Tiling:
            n: int
            m: int
            arr: Array[float, 3]

        @pl.function
        def kernel(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
            tiling: Tiling,
        ) -> pl.Tensor[[64], pl.FP32]:
            n = tiling.n
            m = tiling.m
            tmp1 = tiling.arr[1]
            return x

        print(kernel)

        assert isinstance(kernel, ir.Function)
        # 2 tensor params + 5 tiling fields = 7 total
        assert len(kernel.params) == 7
        param_names = [p.name for p in kernel.params]
        assert param_names[0] == "x"
        assert param_names[1] == "y"
        assert "tiling_n" in param_names
        assert "tiling_m" in param_names

    def test_tiling_name_not_registered_in_scope(self):
        """Bare tiling name used without field access raises UndefinedVariableError."""

        class Tiling:
            x: int

        with pytest.raises(UndefinedVariableError):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                return tiling  # type: ignore[return-value]

    def test_tiling_field_access_resolves_to_correct_var(self):
        """Accessing tiling.x in the body resolves to the flattened IR var."""

        class Tiling:
            x: int

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.x
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 1
        assert kernel.params[0].name == "tiling_x"

    def test_tiling_registry_reset_between_functions(self):
        """Tiling registry is reset for each new function, preventing leakage."""

        class Tiling:
            n: int

        @pl.function
        def func1(tiling: Tiling):
            x: pl.Scalar[pl.INT32] = tiling.n
            return x

        # Second function should not see tiling from first function
        @pl.function
        def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(func1, ir.Function)
        assert isinstance(func2, ir.Function)


class TestTilingErrors:
    """Tests for error cases in tiling parameter parsing."""

    def test_tiling_not_last_raises_parser_syntax_error(self):
        """Tiling parameter that is not the last param raises ParserSyntaxError."""

        class Tiling:
            x: int

        with pytest.raises(ParserSyntaxError, match="must be the last parameter"):

            @pl.function
            def kernel(
                tiling: Tiling,  # Not last!
                x: pl.Tensor[[64], pl.FP32],
            ):
                pass

    def test_multiple_tiling_params_raises_parser_syntax_error(self):
        """More than one tiling parameter raises ParserSyntaxError."""

        class TilingA:
            x: int

        class TilingB:
            y: float

        with pytest.raises(ParserSyntaxError, match="at most 1"):

            @pl.function
            def kernel(
                ta: TilingA,
                tb: TilingB,
            ):
                pass

    def test_nonexistent_tiling_field_raises_parser_type_error(self):
        """Accessing a field that doesn't exist on tiling raises ParserTypeError."""

        class Tiling:
            x: int

        with pytest.raises(ParserTypeError, match="has no field"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                result: pl.Scalar[pl.INT32] = tiling.nonexistent  # type: ignore[attr-defined]
                return result


class TestTilingArrayField:
    """Tests for array field support in tiling parameters."""

    def test_array_field_flattens_to_scalar_params(self):
        class Tiling:
            offsets: Array[int, 3]

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.offsets[0]
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 3
        param_names = [p.name for p in kernel.params]
        assert "tiling_offsets_0" in param_names
        assert "tiling_offsets_1" in param_names
        assert "tiling_offsets_2" in param_names

    def test_array_field_dtypes(self):
        class Tiling:
            ints: Array[int, 2]
            floats: Array[float, 2]
            bools: Array[bool, 2]

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.ints[0]
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 6
        param_map = {p.name: p for p in kernel.params}
        assert param_map["tiling_ints_0"].type.dtype == DataType.INT32
        assert param_map["tiling_ints_1"].type.dtype == DataType.INT32
        assert param_map["tiling_floats_0"].type.dtype == DataType.FP32
        assert param_map["tiling_floats_1"].type.dtype == DataType.FP32
        assert param_map["tiling_bools_0"].type.dtype == DataType.BOOL
        assert param_map["tiling_bools_1"].type.dtype == DataType.BOOL

    def test_mixed_scalar_and_array_fields(self):
        class Tiling:
            n: int
            offsets: Array[int, 2]
            scale: float

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.n
            return result

        assert isinstance(kernel, ir.Function)
        # 1 scalar (n) + 2 array elements (offsets) + 1 scalar (scale) = 4
        assert len(kernel.params) == 4
        param_names = [p.name for p in kernel.params]
        assert "tiling_n" in param_names
        assert "tiling_offsets_0" in param_names
        assert "tiling_offsets_1" in param_names
        assert "tiling_scale" in param_names

    def test_array_subscript_access(self):
        class Tiling:
            offsets: Array[int, 3]

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.offsets[1]
            return result

        assert isinstance(kernel, ir.Function)
        # The function body should reference the second flattened param (offsets_1)
        param_names = [p.name for p in kernel.params]
        assert "tiling_offsets_1" in param_names

    def test_array_all_indices_accessible(self):
        class Tiling:
            vals: Array[int, 3]

        @pl.function
        def kernel0(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.vals[0]
            return result

        @pl.function
        def kernel1(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.vals[1]
            return result

        @pl.function
        def kernel2(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.vals[2]
            return result

        assert isinstance(kernel0, ir.Function)
        assert isinstance(kernel1, ir.Function)
        assert isinstance(kernel2, ir.Function)

    def test_array_bare_name_raises_type_error(self):
        class Tiling:
            offsets: Array[int, 3]

        with pytest.raises(ParserTypeError, match="must be accessed with an integer index"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                result: pl.Scalar[pl.INT32] = tiling.offsets  # type: ignore[assignment]
                return result

    def test_array_out_of_bounds_raises_type_error(self):
        class Tiling:
            offsets: Array[int, 3]

        with pytest.raises(ParserTypeError, match="out of bounds"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                result: pl.Scalar[pl.INT32] = tiling.offsets[99]
                return result

    def test_array_non_literal_index_raises_error(self):
        class Tiling:
            offsets: Array[int, 3]

        with pytest.raises(UnsupportedFeatureError, match="literal integer indices"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                # Use a previously-computed value as subscript (non-literal)
                first: pl.Scalar[pl.INT32] = tiling.offsets[0]
                result: pl.Scalar[pl.INT32] = tiling.offsets[first]  # type: ignore[index]
                return result

    def test_scalar_subscript_raises_type_error(self):
        class Tiling:
            n: int

        with pytest.raises(ParserTypeError, match="does not support subscript access"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                result: pl.Scalar[pl.INT32] = tiling.n[0]  # type: ignore[index]
                return result
