from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime
import json
import os
import tempfile
from pathlib import Path

import pytest
import pandas as pd

# Assuming the run_log function is defined in a module named 'log_module'
from peakipy.utils import (
    run_log,
    update_args_with_values_from_config_file,
    update_peak_positions_from_ppm_to_points,
    update_linewidths_from_hz_to_points,
    save_data,
)


@patch("peakipy.utils.open", new_callable=mock_open)
@patch("peakipy.utils.datetime")
@patch("peakipy.utils.sys")
def test_run_log(mock_sys, mock_datetime, mock_open_file):
    # Mocking sys.argv
    mock_sys.argv = ["test_script.py", "arg1", "arg2"]

    # Mocking datetime to return a fixed timestamp
    fixed_timestamp = datetime(2024, 5, 20, 15, 45)
    mock_datetime.now.return_value = fixed_timestamp

    # Expected timestamp string
    expected_time_stamp = fixed_timestamp.strftime("%A %d %B %Y at %H:%M")

    # Run the function
    run_log("mock_run_log.txt")

    # Prepare the expected log content
    expected_log_content = (
        f"# Script run on {expected_time_stamp}:\ntest_script.py arg1 arg2\n"
    )

    # Assert that the file was opened correctly
    mock_open_file.assert_called_once_with("mock_run_log.txt", "a")

    # Assert that the correct content was written to the file
    mock_open_file().write.assert_called_once_with(expected_log_content)

    # Assert that the script name is correctly set to the basename
    assert mock_sys.argv[0] == "test_script.py"


# Mock configuration loader function (you need to replace 'config_module.load_config' with the actual path if different)
@patch("peakipy.utils.load_config")
@patch("peakipy.utils.Path.exists")
def test_update_args_with_config(mock_path_exists, mock_load_config):
    # Test setup
    mock_path_exists.return_value = True  # Pretend the config file exists
    mock_load_config.return_value = {
        "dims": [1, 2, 3],
        "noise": "0.05",
        "colors": ["#ff0000", "#00ff00"],
    }

    args = {"dims": (0, 1, 2), "noise": False, "colors": ["#5e3c99", "#e66101"]}

    # Run the function
    updated_args, config = update_args_with_values_from_config_file(args)

    # Check the updates to args
    assert updated_args["dims"] == [1, 2, 3]
    assert updated_args["noise"] == 0.05
    assert updated_args["colors"] == ["#ff0000", "#00ff00"]

    # Check the returned config
    assert config == {
        "dims": [1, 2, 3],
        "noise": "0.05",
        "colors": ["#ff0000", "#00ff00"],
    }


@patch("peakipy.utils.Path.exists")
def test_update_args_with_no_config_file(mock_path_exists):
    # Test setup
    mock_path_exists.return_value = False  # Pretend the config file does not exist

    args = {"dims": (0, 1, 2), "noise": False, "colors": ["#5e3c99", "#e66101"]}

    # Run the function
    updated_args, config = update_args_with_values_from_config_file(args)

    # Check the updates to args
    assert updated_args["dims"] == (0, 1, 2)
    assert updated_args["noise"] == False
    assert updated_args["colors"] == ["#5e3c99", "#e66101"]

    # Check the returned config (should be empty)
    assert config == {}


@patch("peakipy.utils.load_config")
@patch("peakipy.utils.Path.exists")
def test_update_args_with_corrupt_config_file(mock_path_exists, mock_load_config):
    # Test setup
    mock_path_exists.return_value = True  # Pretend the config file exists
    mock_load_config.side_effect = json.decoder.JSONDecodeError(
        "Expecting value", "", 0
    )  # Simulate corrupt JSON

    args = {"dims": (0, 1, 2), "noise": False, "colors": ["#5e3c99", "#e66101"]}

    # Run the function
    updated_args, config = update_args_with_values_from_config_file(args)

    # Check the updates to args
    assert updated_args["dims"] == (0, 1, 2)
    assert updated_args["noise"] == False
    assert updated_args["colors"] == ["#5e3c99", "#e66101"]

    # Check the returned config (should be empty due to error)
    assert config == {}

    # Mock class to simulate the peakipy_data object


class MockPeakipyData:
    def __init__(self, df, pt_per_hz_f2, pt_per_hz_f1, uc_f2, uc_f1):
        self.df = df
        self.pt_per_hz_f2 = pt_per_hz_f2
        self.pt_per_hz_f1 = pt_per_hz_f1
        self.uc_f2 = uc_f2
        self.uc_f1 = uc_f1


# Test data
@pytest.fixture
def mock_peakipy_data():
    df = pd.DataFrame(
        {
            "XW_HZ": [10, 20, 30],
            "YW_HZ": [5, 15, 25],
            "X_PPM": [1.0, 2.0, 3.0],
            "Y_PPM": [0.5, 1.5, 2.5],
        }
    )

    pt_per_hz_f2 = 2.0
    pt_per_hz_f1 = 3.0

    uc_f2 = MagicMock()
    uc_f1 = MagicMock()
    uc_f2.side_effect = lambda x, unit: x * 100.0 if unit == "PPM" else x
    uc_f1.side_effect = lambda x, unit: x * 200.0 if unit == "PPM" else x
    uc_f2.f = MagicMock(side_effect=lambda x, unit: x * 1000.0 if unit == "PPM" else x)
    uc_f1.f = MagicMock(side_effect=lambda x, unit: x * 2000.0 if unit == "PPM" else x)

    return MockPeakipyData(df, pt_per_hz_f2, pt_per_hz_f1, uc_f2, uc_f1)


def test_update_linewidths_from_hz_to_points(mock_peakipy_data):
    peakipy_data = update_linewidths_from_hz_to_points(mock_peakipy_data)

    expected_XW = [20.0, 40.0, 60.0]
    expected_YW = [15.0, 45.0, 75.0]

    pd.testing.assert_series_equal(
        peakipy_data.df["XW"], pd.Series(expected_XW, name="XW")
    )
    pd.testing.assert_series_equal(
        peakipy_data.df["YW"], pd.Series(expected_YW, name="YW")
    )


def test_update_peak_positions_from_ppm_to_points(mock_peakipy_data):
    peakipy_data = update_peak_positions_from_ppm_to_points(mock_peakipy_data)

    expected_X_AXIS = [100.0, 200.0, 300.0]
    expected_Y_AXIS = [100.0, 300.0, 500.0]
    expected_X_AXISf = [1000.0, 2000.0, 3000.0]
    expected_Y_AXISf = [1000.0, 3000.0, 5000.0]

    pd.testing.assert_series_equal(
        peakipy_data.df["X_AXIS"], pd.Series(expected_X_AXIS, name="X_AXIS")
    )
    pd.testing.assert_series_equal(
        peakipy_data.df["Y_AXIS"], pd.Series(expected_Y_AXIS, name="Y_AXIS")
    )
    pd.testing.assert_series_equal(
        peakipy_data.df["X_AXISf"], pd.Series(expected_X_AXISf, name="X_AXISf")
    )
    pd.testing.assert_series_equal(
        peakipy_data.df["Y_AXISf"], pd.Series(expected_Y_AXISf, name="Y_AXISf")
    )


@pytest.fixture
def sample_dataframe():
    data = {"A": [1, 2, 3], "B": [4.5678, 5.6789, 6.7890]}
    return pd.DataFrame(data)


def test_save_data_csv(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        output_name = Path(tmpfile.name)

    try:
        save_data(sample_dataframe, output_name)

        assert output_name.exists()

        # Load the CSV and compare with the original dataframe
        loaded_df = pd.read_csv(output_name)
        pd.testing.assert_frame_equal(
            loaded_df, sample_dataframe, check_exact=False, rtol=1e-4
        )
    finally:
        os.remove(output_name)


def test_save_data_tab(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".tab", delete=False) as tmpfile:
        output_name = Path(tmpfile.name)

    try:
        save_data(sample_dataframe, output_name)

        assert output_name.exists()

        # Load the tab-separated file and compare with the original dataframe
        loaded_df = pd.read_csv(output_name, sep="\t")
        pd.testing.assert_frame_equal(
            loaded_df, sample_dataframe, check_exact=False, rtol=1e-4
        )
    finally:
        os.remove(output_name)


def test_save_data_pickle(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmpfile:
        output_name = Path(tmpfile.name)

    try:
        save_data(sample_dataframe, output_name)

        assert output_name.exists()

        # Load the pickle file and compare with the original dataframe
        loaded_df = pd.read_pickle(output_name)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    finally:
        os.remove(output_name)
