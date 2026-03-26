# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for auto_sync feature with manual ops."""

import pytest
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto.pypto_core import ir


def _ir_to_str(prog: ir.Program) -> str:
    """Get the IR string representation of a program."""
    return str(prog)


class TestAutoSyncForward:
    """Forward (intra-iteration) sync insertion tests."""

    def test_load_then_add_inserts_sync(self):
        """Load (MTE2) then add (V) on same tile → sync_src + sync_dst."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            plm.load(tile_a, x, [0])             # MTE2: writes tile_a
            plm.add(tile_b, tile_a, tile_a)       # V: reads tile_a → RAW
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.sync_src" in ir_str, f"Expected sync_src in IR:\n{ir_str}"
        assert "system.sync_dst" in ir_str, f"Expected sync_dst in IR:\n{ir_str}"

    def test_same_pipe_no_sync_a5(self):
        """Two V ops on same tile, a5 arch → no sync (hardware guarantees ordering)."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            tile_c = plm.make_tile(tt, addr=256, size=128)
            plm.add(tile_b, tile_a, tile_a)       # V
            plm.neg(tile_c, tile_b)               # V - same pipe, a5 → no sync
            return x

        ir_str = _ir_to_str(k.parse(npu_arch="dav-3510"))
        assert "system.sync_src" not in ir_str, f"No sync expected on a5:\n{ir_str}"

    def test_same_pipe_no_sync_default(self):
        """Two V ops on same tile, no arch specified → no sync (default)."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            tile_c = plm.make_tile(tt, addr=256, size=128)
            plm.add(tile_b, tile_a, tile_a)       # V
            plm.neg(tile_c, tile_b)               # V - same pipe, no arch → no sync
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.sync_src" not in ir_str, f"No sync expected by default:\n{ir_str}"

    def test_same_pipe_sync_required_a3(self):
        """Two V ops on same tile, a3 arch → sync required (V→V needs software sync)."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            tile_c = plm.make_tile(tt, addr=256, size=128)
            plm.add(tile_b, tile_a, tile_a)       # V: writes tile_b
            plm.neg(tile_c, tile_b)               # V: reads tile_b → needs V→V sync
            return x

        ir_str = _ir_to_str(k.parse(npu_arch="a3"))
        assert "system.sync_src" in ir_str, f"Expected V→V sync on a3:\n{ir_str}"
        assert "system.sync_dst" in ir_str, f"Expected V→V sync on a3:\n{ir_str}"

    def test_same_pipe_sync_required_dav2201(self):
        """dav-2201 arch string also triggers same-pipe sync."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            tile_c = plm.make_tile(tt, addr=256, size=128)
            plm.add(tile_b, tile_a, tile_a)       # V
            plm.neg(tile_c, tile_b)               # V → needs sync on dav-2201
            return x

        ir_str = _ir_to_str(k.parse(npu_arch="dav-2201"))
        assert "system.sync_src" in ir_str, f"Expected V→V sync on dav-2201:\n{ir_str}"

    def test_auto_sync_disabled_no_sync(self):
        """auto_sync=False → no sync ops emitted."""

        @fe.kernel(auto_sync=False)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            plm.load(tile_a, x, [0])             # MTE2
            plm.add(tile_b, tile_a, tile_a)       # V
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.sync_src" not in ir_str, f"No sync expected:\n{ir_str}"

    def test_default_no_auto_sync(self):
        """Default kernel (no auto_sync) → no sync ops."""

        @fe.kernel
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            plm.load(tile_a, x, [0])
            plm.add(tile_b, tile_a, tile_a)
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.sync_src" not in ir_str

    def test_different_tiles_no_sync(self):
        """Ops on different tiles with no data dep → no sync."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
            y: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            plm.load(tile_a, x, [0])             # MTE2: writes tile_a
            plm.load(tile_b, y, [0])             # MTE2: writes tile_b
            return x

        ir_str = _ir_to_str(k.parse())
        # Both loads write different tiles on same pipe → no sync
        assert "system.sync_src" not in ir_str

    def test_same_kernel_different_arch(self):
        """Same KernelDef can be parsed for different architectures."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            tile_c = plm.make_tile(tt, addr=256, size=128)
            plm.add(tile_b, tile_a, tile_a)
            plm.neg(tile_c, tile_b)
            return x

        # a3: V→V needs sync
        ir_a3 = _ir_to_str(k.parse(npu_arch="a3"))
        assert "system.sync_src" in ir_a3

        # a5: V→V does not need sync
        ir_a5 = _ir_to_str(k.parse(npu_arch="a5"))
        assert "system.sync_src" not in ir_a5


class TestAutoSyncBackward:
    """Backward (cross-iteration) sync insertion tests."""

    def test_loop_backward_sync(self):
        """Loop with load (MTE2) then add (V) on same tile → backward sync."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            tile_b = plm.make_tile(tt, addr=128, size=128)
            for i in pl.range(4):
                plm.load(tile_a, x, [0])         # MTE2
                plm.add(tile_b, tile_a, tile_a)   # V → cross-pipe dep
            return x

        ir_str = _ir_to_str(k.parse())
        # Backward sync should produce sync ops both outside and inside the loop
        sync_count = ir_str.count("system.sync_src")
        # Priming (1) + body end set per dep (1) = 2 sync_src
        # Body start wait (1) + drain (1) = 2 sync_dst
        assert sync_count >= 2, f"Expected ≥2 sync_src, got {sync_count}\n{ir_str}"


class TestAutoSyncProgram:
    """Verify auto_sync produces valid KernelDefs and Programs."""

    def test_result_is_kernel_def(self):
        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a = plm.make_tile(tt, addr=0, size=128)
            plm.load(tile_a, x, [0])
            return x

        assert isinstance(k, fe.KernelDef)
        prog = k.parse()
        assert isinstance(prog, ir.Program)
        assert prog.get_function("k") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
