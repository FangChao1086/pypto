# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for auto_sync with double-buffer (DB) tile tuples."""

import pytest
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto.pypto_core import ir


def _ir_to_str(prog: ir.Program) -> str:
    return str(prog)


class TestDBForwardSync:
    """Forward sync insertion for double-buffered tile tuples."""

    def test_db_load_then_add_inserts_sync(self):
        """Load via tile_buf[buf_idx], then add on same tile → sync per slot."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a_ping = plm.make_tile(tt, addr=0, size=128)
            tile_a_pong = plm.make_tile(tt, addr=128, size=128)
            tile_b = plm.make_tile(tt, addr=256, size=128)
            tile_a_buf = (tile_a_ping, tile_a_pong)

            buf_idx = pl.const(0, pl.INDEX)
            plm.load(tile_a_buf[buf_idx], x, [0])      # MTE2: writes tile_a_ping or pong
            plm.add(tile_b, tile_a_buf[buf_idx], tile_a_buf[buf_idx])  # V: reads → RAW
            return x

        ir_str = _ir_to_str(k.parse())
        # Should contain sync for MTE2→V dependency
        assert "system.sync_src" in ir_str, f"Expected sync_src in DB IR:\n{ir_str}"
        assert "system.sync_dst" in ir_str, f"Expected sync_dst in DB IR:\n{ir_str}"

    def test_db_tile_tuple_detected(self):
        """Verify tile_tuple_registry detects (ping, pong) assignment."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_ping = plm.make_tile(tt, addr=0, size=128)
            tile_pong = plm.make_tile(tt, addr=128, size=128)
            tile_buf = (tile_ping, tile_pong)
            tile_out = plm.make_tile(tt, addr=256, size=128)

            buf_idx = pl.const(0, pl.INDEX)
            plm.load(tile_buf[buf_idx], x, [0])          # MTE2
            plm.add(tile_out, tile_buf[buf_idx], tile_buf[buf_idx])  # V
            return x

        # Just verify it parses without error and contains sync
        ir_str = _ir_to_str(k.parse())
        assert "system.sync_src" in ir_str

    def test_db_non_tuple_tile_still_works(self):
        """Non-tuple tile args continue to work as before."""

        @fe.kernel(auto_sync=True)
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
        assert "system.sync_src" in ir_str


class TestDBBackwardSync:
    """Backward sync for DB tile tuples in loops."""

    def test_db_loop_backward_deps_detected(self):
        """Loop with DB tile tuple → backward deps detected for both ping/pong."""

        @fe.kernel(auto_sync=True)
        def k(
            x: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            tile_a_ping = plm.make_tile(tt, addr=0, size=128)
            tile_a_pong = plm.make_tile(tt, addr=128, size=128)
            tile_b = plm.make_tile(tt, addr=256, size=128)
            tile_a_buf = (tile_a_ping, tile_a_pong)

            for i in pl.range(4):
                buf_idx = i % 2
                plm.load(tile_a_buf[buf_idx], x, [0])    # MTE2
                plm.add(tile_b, tile_a_buf[buf_idx], tile_a_buf[buf_idx])  # V
            return x

        ir_str = _ir_to_str(k.parse())
        # Should have backward sync for both ping and pong tiles
        sync_src_count = ir_str.count("system.sync_src")
        assert sync_src_count >= 2, f"Expected ≥2 sync_src for DB backward deps, got {sync_src_count}\n{ir_str}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
