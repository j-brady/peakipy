import unittest
from unittest.mock import patch
from pathlib import Path
import json
import textwrap

import pytest
import numpy as np
import nmrglue as ng
import pandas as pd

from peakipy.io import (
    Pseudo3D,
    Peaklist,
    LoadData,
    PeaklistFormat,
    OutFmt,
    StrucEl,
    UnknownFormat,
    ClustersResult,
    get_vclist,
)
from peakipy.fitting import PeakLimits
from peakipy.utils import load_config, write_config, update_config_file


@pytest.fixture
def test_directory():
    return Path(__file__).parent

@pytest.fixture
def test_directory_protein_L(test_directory):
    return test_directory / "test_protein_L"

# test for read, edit, fit, check and spec scripts
# need to actually write proper tests
class TestBokehScript(unittest.TestCase):
    @patch("peakipy.cli.edit.BokehScript")
    def test_BokehScript(self, MockBokehScript):
        args = {"<peaklist>": "hello", "<data>": "data"}
        bokeh_plots = MockBokehScript(args)
        self.assertIsNotNone(bokeh_plots)


class TestCheckScript(unittest.TestCase):
    @patch("peakipy.cli.main.check")
    def test_main(self, MockCheck):
        args = {"<peaklist>": "hello", "<data>": "data"}
        check = MockCheck(args)
        self.assertIsNotNone(check)


class TestFitScript(unittest.TestCase):
    @patch("peakipy.cli.main.fit")
    def test_main(self, MockFit):
        args = {"<peaklist>": "hello", "<data>": "data"}
        fit = MockFit(args)
        self.assertIsNotNone(fit)



class TestReadScript(unittest.TestCase):
    test_directory = "./test/"

    @patch("peakipy.cli.main.read")
    def test_main(self, MockRead):
        args = {"<peaklist>": "hello", "<data>": "data"}
        read = MockRead(args)
        self.assertIsNotNone(read)

    def test_read_pipe_peaklist(self):
        args = {
            "path": f"{self.test_directory}/test_pipe.tab",
            "data_path": f"{self.test_directory}/test_pipe.ft2",
            "dims": [0, 1, 2],
            "fmt": PeaklistFormat.pipe,
        }
        peaklist = Peaklist(**args)
        self.assertIsNotNone(peaklist)
        self.assertIs(len(peaklist.df), 3)
        # self.assertIs(peaklist.df.X_AXISf.iloc[0], 323.019)
        self.assertIs(peaklist.fmt.value, "pipe")
        # self.assertEqual(peaklist.df.ASS.iloc[0], "None")
        # self.assertEqual(peaklist.df.ASS.iloc[1], "None_dummy_1")

def test_read_custom_csv(test_directory_protein_L, capsys):
    custom_csv_data = textwrap.dedent("""\
    ASS,X_PPM,Y_PPM
    test1,8.763,117.821
    test2,8.973,122.359
    test3,9.005,122.436
    """)
    custom_csv_data_path = test_directory_protein_L / "custom.csv"
    custom_csv_data_path.write_text(custom_csv_data)
    args = {
        "path": custom_csv_data_path,
        "data_path": f"{test_directory_protein_L}/test1.ft2",
        "dims": [0, 1, 2],
        "fmt": PeaklistFormat.csv,
    }
    peaklist = Peaklist(**args)
    assert peaklist != None
    assert peaklist.df.shape[0] == 3
    assert peaklist.fmt.value == "csv"


def test_read_pipe_peaklist_check_radius_too_small(test_directory, capsys):
    args = {
        "path": f"{test_directory}/test_pipe.tab",
        "data_path": f"{test_directory}/test_pipe.ft2",
        "dims": [0, 1, 2],
        "fmt": PeaklistFormat.pipe,
        "radii": [0.0001,0.0001],
    }
    peaklist = Peaklist(**args)
    assert peaklist != None
    assert peaklist.df.shape[0] == 3
    assert peaklist.fmt.value == "pipe"
    # assert peaklist.f1_radius == 0.0001
    peaklist.f2_radius
    assert "Warning: --x-radius-ppm" in capsys.readouterr().out
    peaklist.f1_radius
    assert "Warning: --y-radius-ppm" in capsys.readouterr().out
    # check that number of points are set to 2 if invalid input radius
    assert (peaklist.f1_radius * peaklist.pt_per_ppm_f1) == 2
    assert (peaklist.f2_radius * peaklist.pt_per_ppm_f2) == 2

def test_read_pipe_peaklist_check_radius_valid(test_directory, capsys):
    args = {
        "path": f"{test_directory}/test_pipe.tab",
        "data_path": f"{test_directory}/test_pipe.ft2",
        "dims": [0, 1, 2],
        "fmt": PeaklistFormat.pipe,
        "radii": [0.05,0.5],
    }
    peaklist = Peaklist(**args)
    assert peaklist != None
    assert peaklist.df.shape[0] == 3
    assert peaklist.fmt.value == "pipe"
    assert peaklist.f1_radius == 0.5
    assert peaklist.f2_radius == 0.05
    peaklist.f2_radius
    assert capsys.readouterr().out == ""
    peaklist.f1_radius
    assert capsys.readouterr().out == ""

def test_load_config_existing():
    config_path = Path("test_config.json")
    # Create a dummy existing config file
    with open(config_path, "w") as f:
        json.dump({"key1": "value1"}, f)

    loaded_config = load_config(config_path)

    assert loaded_config == {"key1": "value1"}

    # Clean up
    config_path.unlink()


def test_load_config_nonexistent():
    config_path = Path("test_config.json")

    loaded_config = load_config(config_path)

    assert loaded_config == {}


def test_write_config():
    config_path = Path("test_config.json")
    config_kvs = {"key1": "value1", "key2": "value2"}

    write_config(config_path, config_kvs)

    # Check if the config file is created correctly
    assert config_path.exists()

    # Check if the config file content is correct
    with open(config_path) as f:
        created_config = json.load(f)
        assert created_config == {"key1": "value1", "key2": "value2"}

    # Clean up
    config_path.unlink()


def test_update_config_file_existing():
    config_path = Path("test_config.json")
    # Create a dummy existing config file
    with open(config_path, "w") as f:
        json.dump({"key1": "value1"}, f)

    config_kvs = {"key2": "value2", "key3": "value3"}
    updated_config = update_config_file(config_path, config_kvs)

    assert updated_config == {"key1": "value1", "key2": "value2", "key3": "value3"}

    # Clean up
    config_path.unlink()


def test_update_config_file_nonexistent():
    config_path = Path("test_config.json")
    config_kvs = {"key1": "value1", "key2": "value2"}
    updated_config = update_config_file(config_path, config_kvs)

    assert updated_config == {"key1": "value1", "key2": "value2"}

    # Clean up
    config_path.unlink()


@pytest.fixture
def sample_data():
    return np.zeros((10, 10))


@pytest.fixture
def sample_peak():
    peak_data = {"X_AXIS": [5], "Y_AXIS": [5], "XW": [2], "YW": [2]}
    return pd.DataFrame(peak_data).iloc[0]


def test_peak_limits_max_min(sample_peak, sample_data):
    limits = PeakLimits(sample_peak, sample_data)

    assert limits.max_x == 8
    assert limits.max_y == 8
    assert limits.min_x == 3
    assert limits.min_y == 3


def test_peak_limits_boundary(sample_data):
    peak_data = {"X_AXIS": [8], "Y_AXIS": [8], "XW": [2], "YW": [2]}
    peak = pd.DataFrame(peak_data).iloc[0]
    limits = PeakLimits(peak, sample_data)

    assert limits.max_x == 10
    assert limits.max_y == 10
    assert limits.min_x == 6
    assert limits.min_y == 6


def test_peak_limits_at_boundary(sample_data):
    peak_data = {"X_AXIS": [0], "Y_AXIS": [0], "XW": [2], "YW": [2]}
    peak = pd.DataFrame(peak_data).iloc[0]
    limits = PeakLimits(peak, sample_data)

    assert limits.max_x == 3
    assert limits.max_y == 3
    assert limits.min_x == 0
    assert limits.min_y == 0


def test_peak_limits_outside_boundary(sample_data):
    peak_data = {"X_AXIS": [15], "Y_AXIS": [15], "XW": [2], "YW": [2]}
    peak = pd.DataFrame(peak_data).iloc[0]
    with pytest.raises(AssertionError):
        limits = PeakLimits(peak, sample_data)


def test_peak_limits_1d_data():
    data = np.zeros(10)
    peak_data = {"X_AXIS": [5], "Y_AXIS": [0], "XW": [2], "YW": [0]}
    peak = pd.DataFrame(peak_data).iloc[0]
    with pytest.raises(IndexError):
        limits = PeakLimits(peak, data)


def test_StrucEl():
    assert StrucEl.square.value == "square"
    assert StrucEl.disk.value == "disk"
    assert StrucEl.rectangle.value == "rectangle"
    assert StrucEl.mask_method.value == "mask_method"


def test_PeaklistFormat():
    assert PeaklistFormat.a2.value == "a2"
    assert PeaklistFormat.a3.value == "a3"
    assert PeaklistFormat.sparky.value == "sparky"
    assert PeaklistFormat.pipe.value == "pipe"
    assert PeaklistFormat.peakipy.value == "peakipy"


def test_OutFmt():
    assert OutFmt.csv.value == "csv"
    assert OutFmt.pkl.value == "pkl"


@pytest.fixture
def test_data_path():
    return Path("./test/test_protein_L")


@pytest.fixture
def pseudo3d_args(test_data_path):
    dic, data = ng.pipe.read(test_data_path / "test1.ft2")
    dims = [0, 1, 2]
    return dic, data, dims


@pytest.fixture
def peaklist(test_data_path):
    dims = [0, 1, 2]
    path = test_data_path / "test.tab"
    data_path = test_data_path / "test1.ft2"
    fmt = PeaklistFormat.pipe
    radii = [0.04, 0.4]
    return Peaklist(path, data_path, fmt, dims, radii)


def test_Pseudo3D_properties(pseudo3d_args):
    dic, data, dims = pseudo3d_args
    pseudo3d = Pseudo3D(dic, data, dims)
    assert pseudo3d.dic == dic
    assert np.array_equal(pseudo3d._data, data.reshape((4, 256, 546)))
    assert pseudo3d.dims == dims


def test_Peaklist_initialization(test_data_path, peaklist):

    assert peaklist.peaklist_path == test_data_path / "test.tab"
    assert peaklist.data_path == test_data_path / "test1.ft2"
    assert peaklist.fmt == PeaklistFormat.pipe
    assert peaklist.radii == [0.04, 0.4]


def test_Peaklist_a2(test_data_path):
    dims = [0, 1, 2]
    path = test_data_path / "peaks.a2"
    data_path = test_data_path / "test1.ft2"
    fmt = PeaklistFormat.a2
    radii = [0.04, 0.4]
    peaklist = Peaklist(path, data_path, fmt, dims, radii)
    peaklist.update_df()


def test_Peaklist_a3(test_data_path):
    dims = [0, 1, 2]
    path = test_data_path / "ccpnTable.tsv"
    data_path = test_data_path / "test1.ft2"
    fmt = PeaklistFormat.a3
    radii = [0.04, 0.4]
    peaklist = Peaklist(path, data_path, fmt, dims, radii)
    peaklist.update_df()


def test_Peaklist_sparky(test_data_path):
    dims = [0, 1, 2]
    path = test_data_path / "peaks.sparky"
    data_path = test_data_path / "test1.ft2"
    fmt = PeaklistFormat.sparky
    radii = [0.04, 0.4]
    Peaklist(path, data_path, fmt, dims, radii)


@pytest.fixture
def loaddata(test_data_path):
    dims = [0, 1, 2]
    path = test_data_path / "test.csv"
    data_path = test_data_path / "test1.ft2"
    fmt = PeaklistFormat.peakipy
    radii = [0.04, 0.4]
    return LoadData(path, data_path, fmt, dims, radii)


def test_LoadData_initialization(test_data_path, loaddata):
    assert loaddata.peaklist_path == test_data_path / "test.csv"
    assert loaddata.data_path == test_data_path / "test1.ft2"
    assert loaddata.fmt == PeaklistFormat.peakipy
    assert loaddata.radii == [0.04, 0.4]
    loaddata.check_data_frame()
    loaddata.check_assignments()
    loaddata.check_peak_bounds()
    loaddata.update_df()


def test_LoadData_with_Edited_column(loaddata):
    loaddata.df["Edited"] = "yes"
    loaddata.check_data_frame()


def test_LoadData_without_include_column(loaddata):
    loaddata.df.drop(columns=["include"], inplace=True)
    loaddata.check_data_frame()
    assert "include" in loaddata.df.columns
    assert np.all(loaddata.df.include == "yes")


def test_LoadData_with_X_DIAMETER_PPM_column(loaddata):
    loaddata.df["X_DIAMETER_PPM"] = 0.04
    loaddata.check_data_frame()
    assert "X_DIAMETER_PPM" in loaddata.df.columns


def test_UnknownFormat():
    with pytest.raises(UnknownFormat):
        raise UnknownFormat("This is an unknown format")


def test_update_df(peaklist):
    peaklist.update_df()

    df = peaklist.df

    # Check that X_AXIS and Y_AXIS columns are created and values are set correctly
    assert "X_AXIS" in df.columns
    assert "Y_AXIS" in df.columns

    # Check that X_AXISf and Y_AXISf columns are created and values are set correctly
    assert "X_AXISf" in df.columns
    assert "Y_AXISf" in df.columns

    # Check that XW_HZ and YW_HZ columns are converted to float correctly
    assert df["XW_HZ"].dtype == float
    assert df["YW_HZ"].dtype == float

    # Check that XW and YW columns are created
    assert "XW" in df.columns
    assert "YW" in df.columns

    # Check the assignment column
    assert "ASS" in df.columns

    # Check radii columns
    assert "X_RADIUS_PPM" in df.columns
    assert "Y_RADIUS_PPM" in df.columns
    assert "X_RADIUS" in df.columns
    assert "Y_RADIUS" in df.columns

    # Check 'include' column is created and set to 'yes'
    assert "include" in df.columns
    assert all(df["include"] == "yes")

    # Check that the peaks are within bounds
    assert all(
        (df["X_PPM"] < peaklist.f2_ppm_max) & (df["X_PPM"] > peaklist.f2_ppm_min)
    )
    assert all(
        (df["Y_PPM"] < peaklist.f1_ppm_max) & (df["Y_PPM"] > peaklist.f1_ppm_min)
    )


def test_update_df_with_excluded_peaks(peaklist):
    peaklist._df.loc[1, "X_PPM"] = 100.0  # This peak should be out of bounds
    peaklist.update_df()

    df = peaklist.df

    # Check that out of bounds peak is excluded
    assert len(df) == 62
    assert not ((df["X_PPM"] == 100.0).any())


def test_clusters_result_initialization():
    labeled_array = np.array([[1, 2], [3, 4]])
    num_features = 5
    closed_data = np.array([[5, 6], [7, 8]])
    peaks = [(1, 2), (3, 4)]

    clusters_result = ClustersResult(labeled_array, num_features, closed_data, peaks)

    assert np.array_equal(clusters_result.labeled_array, labeled_array)
    assert clusters_result.num_features == num_features
    assert np.array_equal(clusters_result.closed_data, closed_data)
    assert clusters_result.peaks == peaks


def test_get_vclist_None():
    assert get_vclist(None, {})["vclist"] == False


def test_get_vclist_exists(test_data_path):
    vclist = test_data_path / "vclist"
    assert get_vclist(vclist, {})["vclist"] == True


def test_get_vclist_not_exists(test_data_path):
    vclist = test_data_path / "vclistbla"
    with pytest.raises(Exception):
        get_vclist(vclist, {})["vclist"] == True


if __name__ == "__main__":
    unittest.main(verbosity=2)
