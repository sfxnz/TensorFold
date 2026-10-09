"""TF_DSV41_QMMF: the decode tiles DeepSeek's FP8 projections take by shape, and the values it refuses."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import gemv
from tensorfold.families.deepseek_v41.cuda.gemv import KNOB, Tiles, parse, tiles_wanted


def test_entries_name_a_shape_or_every_shape():
    assert parse("") == parse("0") == parse(" ") == {}
    assert parse("16384x1280:32/8/fuse, *:64/4/cluster,5120X4096:16/6/cluster") == {
        (16384, 1280): Tiles(32, 8, True), None: Tiles(64, 4, False), (5120, 4096): Tiles(16, 6, False)}


@pytest.mark.parametrize("spec", ["16384x1280:48/8/fuse", "16384x1280:32/5/fuse", "16384x1280:32/8/both",
                                  "16384:32/8/fuse", "16384x1280x2:32/8/fuse", "16384x1280", "a:b", "*:32/8",
                                  "16384x1280:32/8/fuse,"])
def test_a_bad_entry_is_refused_naming_the_variable(spec):
    with pytest.raises(ValueError, match=KNOB):
        parse(spec)


def test_unset_takes_the_tuned_table_and_zero_turns_it_off(monkeypatch):
    monkeypatch.setattr(gemv, "TUNED", {(1, 64): Tiles(16, 4, True)})
    assert tiles_wanted({}) == {(1, 64): Tiles(16, 4, True)}
    assert tiles_wanted({KNOB: "0"}) == {}
    assert tiles_wanted({KNOB: "*:32/6/fuse"}) == {None: Tiles(32, 6, True)}


def test_the_tuned_table_names_decode_shapes_only_and_leaves_wo_a_to_its_grouped_launch():
    assert None not in gemv.TUNED and (1024, 4096) not in gemv.TUNED and (4096, 4096) not in gemv.TUNED
    assert tiles_wanted({}) == gemv.TUNED and tiles_wanted({KNOB: "0"}) == {}
    assert all(t.bn == 64 and t.fuse for t in gemv.TUNED.values())
