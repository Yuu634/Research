from dataclasses import replace
import numpy as np
import pytest

from src.common.examples import make_bar
from src.common.skeleton import FEATURE_SIZE, extract_skeleton


def test_original_positions_and_explicit_annotations():
    bar = make_bar([(0, 'kick'), (12, 'kick'), (24, 'snare'), (72, 'snare', 'Accent'), (90, 'tom_high')])
    bar = replace(bar, fill_regions=((72, 96),))
    result = extract_skeleton(bar, section='chorus')
    positions = result.report()['positions']
    assert positions['main_kick'] == [0]
    assert positions['backbeat_snare'] == [24, 72]
    assert positions['accent'] == [72]
    assert positions['fill_start'] == [72] and positions['fill_end'] == [90]
    assert result.fill_known and result.section_known
    assert result.vector().shape == (FEATURE_SIZE,)
    np.testing.assert_array_equal(result.vector(), extract_skeleton(bar, section='chorus').vector())


def test_unknown_is_distinct_from_known_empty_and_other_section():
    bar = make_bar([(0, 'kick')])
    unknown = extract_skeleton(bar)
    known_empty = extract_skeleton(bar, fill_known=True, section='Aメロ')
    assert not unknown.fill_known and not unknown.section_known
    assert known_empty.fill_known and known_empty.section[-1] == 1
    assert not np.array_equal(unknown.vector(), known_empty.vector())
    with pytest.raises(ValueError):
        extract_skeleton(replace(bar, fill_regions=((0, 24),)), fill_known=False)
