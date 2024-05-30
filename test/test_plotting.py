from dataclasses import dataclass

import numpy as np

from peakipy.plotting import (
    construct_surface_legend_string,
    plot_data_is_valid,
    df_to_rich_table,
)


@dataclass
class Row:
    assignment: str


def test_construct_surface_legend_string():
    row = Row("assignment")
    expected_legend = "assignment"
    actual_legend = construct_surface_legend_string(row)
    assert expected_legend == actual_legend


import pytest
from unittest.mock import MagicMock, patch


# Mock PlottingDataForPlane class for testing purposes
class PlottingDataForPlane:
    def __init__(
        self, x_plot, y_plot, masked_data, plane_lineshape_parameters, pseudo3D, plane
    ):
        self.x_plot = x_plot
        self.y_plot = y_plot
        self.masked_data = masked_data
        self.plane_lineshape_parameters = plane_lineshape_parameters
        self.pseudo3D = pseudo3D
        self.plane = plane


@pytest.fixture
def valid_plot_data():
    return PlottingDataForPlane(
        x_plot=np.array([[1, 2, 3]]),
        y_plot=np.array([[4, 5, 6]]),
        masked_data=np.array([[7, 8, 9]]),
        plane_lineshape_parameters=MagicMock(clustid=1),
        pseudo3D=MagicMock(
            f1_ppm_limits=[0, 1], f2_ppm_limits=[0, 1], f1_label="F1", f2_label="F2"
        ),
        plane=MagicMock(clustid=1),
    )


@pytest.fixture
def invalid_plot_data_empty_x():
    return PlottingDataForPlane(
        x_plot=np.array([]),
        y_plot=np.array([[4, 5, 6]]),
        masked_data=np.array([[7, 8, 9]]),
        plane_lineshape_parameters=MagicMock(clustid=1),
        pseudo3D=MagicMock(
            f1_ppm_limits=[0, 1], f2_ppm_limits=[0, 1], f1_label="F1", f2_label="F2"
        ),
        plane=MagicMock(clustid=1),
    )


@pytest.fixture
def invalid_plot_data_empty_masked():
    return PlottingDataForPlane(
        x_plot=np.array([[1, 2, 3]]),
        y_plot=np.array([[4, 5, 6]]),
        masked_data=np.array([[]]),
        plane_lineshape_parameters=MagicMock(clustid=1),
        pseudo3D=MagicMock(
            f1_ppm_limits=[0, 1], f2_ppm_limits=[0, 1], f1_label="F1", f2_label="F2"
        ),
        plane=MagicMock(clustid=1),
    )


def test_plot_data_is_valid(valid_plot_data):
    assert plot_data_is_valid(valid_plot_data) == True


@patch("peakipy.plotting.print")
@patch("peakipy.plotting.plt.close")
def test_plot_data_is_invalid_empty_x(
    mock_close, mock_print, invalid_plot_data_empty_x
):
    assert plot_data_is_valid(invalid_plot_data_empty_x) == False
    assert mock_print.call_count == 3
    mock_close.assert_called_once()


@patch("peakipy.plotting.print")
@patch("peakipy.plotting.plt.close")
def test_plot_data_is_invalid_empty_masked(
    mock_close, mock_print, invalid_plot_data_empty_masked
):
    assert plot_data_is_valid(invalid_plot_data_empty_masked) == False
    assert mock_print.call_count == 4
    mock_close.assert_called_once()
