import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pytest import fixture

from peakipy.cli.main import (
    get_vclist,
    check_for_include_column_and_add_if_missing,
    select_specified_planes,
    exclude_specified_planes,
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
