import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from peakipy.cli.main import (
    get_vclist,
    check_for_include_column_and_add_if_missing,
    select_specified_planes,
    exclude_specified_planes,
    validate_plane_selection,
    validate_sample_count,
    unpack_plotting_colors,
    get_fit_data_for_selected_peak_clusters,
)


@fixture
def actual_vclist():
    with tempfile.TemporaryFile() as fp:
        fp.write(b"1\n2\n3\n")
        fp.seek(0)
        vclist = np.genfromtxt(fp)
    return vclist


@dataclass
class PeakipyData:
    df: pd.DataFrame = pd.DataFrame()
    data: np.array = np.zeros((4, 10, 20))
    dims: list = field(default_factory=lambda: [0, 1, 2])


def test_get_vclist(actual_vclist):
    expected_vclist = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(actual_vclist, expected_vclist)


def test_get_vclist_none():
    vclist = None
    args = {}
    args = get_vclist(vclist, args)
    expected_args = dict(vclist=False)
    assert args == expected_args


def test_check_for_include_column():
    peakipy_data = PeakipyData(pd.DataFrame())
    peakipy_data = check_for_include_column_and_add_if_missing(peakipy_data)
    assert "include" in peakipy_data.df.columns


def test_select_specified_planes():
    plane = None
    expected_plane_numbers = np.arange(4)
    actual_plane_numbers, peakipy_data = select_specified_planes(plane, PeakipyData())
    np.testing.assert_array_equal(expected_plane_numbers, actual_plane_numbers)


def test_select_specified_planes_2():
    plane = [1, 2]
    expected_plane_numbers = np.array([1, 2])
    actual_plane_numbers, peakipy_data = select_specified_planes(plane, PeakipyData())
    np.testing.assert_array_equal(expected_plane_numbers, actual_plane_numbers)
    assert peakipy_data.data.shape == (2, 10, 20)


def test_exclude_specified_planes():
    plane = None
    expected_plane_numbers = np.arange(4)
    actual_plane_numbers, peakipy_data = exclude_specified_planes(plane, PeakipyData())
    np.testing.assert_array_equal(expected_plane_numbers, actual_plane_numbers)


def test_exclude_specified_planes_2():
    plane = [1, 2]
    expected_plane_numbers = np.array([0, 3])
    actual_plane_numbers, peakipy_data = exclude_specified_planes(plane, PeakipyData())
    np.testing.assert_array_equal(expected_plane_numbers, actual_plane_numbers)
    assert peakipy_data.data.shape == (2, 10, 20)


class MockPseudo3D:
    def __init__(self, n_planes):
        self.n_planes = n_planes


def test_empty_plane_selection():
    pseudo3D = MockPseudo3D(n_planes=5)
    assert validate_plane_selection([], pseudo3D) == [0, 1, 2, 3, 4]


def test_plane_selection_none():
    pseudo3D = MockPseudo3D(n_planes=5)
    assert validate_plane_selection(None, pseudo3D) == [0, 1, 2, 3, 4]


def test_valid_plane_selection():
    pseudo3D = MockPseudo3D(n_planes=5)
    assert validate_plane_selection([0, 1, 2], pseudo3D) == [0, 1, 2]


def test_invalid_plane_selection_negative():
    pseudo3D = MockPseudo3D(n_planes=5)
    with pytest.raises(ValueError):
        validate_plane_selection([-1], pseudo3D)


def test_invalid_plane_selection_too_high():
    pseudo3D = MockPseudo3D(n_planes=5)
    with pytest.raises(ValueError):
        validate_plane_selection([5], pseudo3D)


def test_invalid_plane_selection_mix():
    pseudo3D = MockPseudo3D(n_planes=5)
    with pytest.raises(ValueError):
        validate_plane_selection([-1, 3, 5], pseudo3D)


def test_valid_sample_count():
    assert validate_sample_count(10) == 10


def test_invalid_sample_count_type():
    with pytest.raises(TypeError):
        validate_sample_count("10")


def test_invalid_sample_count_float():
    with pytest.raises(TypeError):
        validate_sample_count(10.5)


def test_invalid_sample_count_list():
    with pytest.raises(TypeError):
        validate_sample_count([10])


def test_invalid_sample_count_dict():
    with pytest.raises(TypeError):
        validate_sample_count({"count": 10})


def test_invalid_sample_count_none():
    with pytest.raises(TypeError):
        validate_sample_count(None)


def test_valid_colors():
    assert unpack_plotting_colors(("red", "black")) == ("red", "black")


def test_default_colors():
    assert unpack_plotting_colors(()) == ("green", "blue")


def test_invalid_colors_type():
    assert unpack_plotting_colors("red") == ("green", "blue")


def test_invalid_colors_single():
    assert unpack_plotting_colors(("red",)) == ("green", "blue")


def test_invalid_colors_length():
    assert unpack_plotting_colors(("red", "black", "green")) == ("green", "blue")


def test_no_clusters():
    fits = pd.DataFrame({"clustid": [1, 2, 3]})
    assert get_fit_data_for_selected_peak_clusters(fits, None).equals(fits)


def test_empty_clusters():
    fits = pd.DataFrame({"clustid": [1, 2, 3]})
    assert get_fit_data_for_selected_peak_clusters(fits, []).equals(fits)


def test_valid_clusters():
    fits = pd.DataFrame({"clustid": [1, 2, 3]})
    selected_clusters = [1, 3]
    expected_result = pd.DataFrame({"clustid": [1, 3]})
    assert (
        get_fit_data_for_selected_peak_clusters(fits, selected_clusters)
        .reset_index(drop=True)
        .equals(expected_result)
    )


def test_invalid_clusters():
    fits = pd.DataFrame({"clustid": [1, 2, 3]})
    with pytest.raises(SystemExit):
        get_fit_data_for_selected_peak_clusters(fits, [4, 5, 6])
