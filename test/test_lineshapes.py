from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from peakipy.io import Peaklist, PeaklistFormat
from peakipy.constants import tiny
from peakipy.lineshapes import (
    gaussian,
    gaussian_lorentzian,
    pv_g,
    pv_l,
    voigt2d,
    pvoigt2d,
    pv_pv,
    get_lineshape_function,
    Lineshape,
    calculate_height_for_voigt_lineshape,
    calculate_fwhm_for_voigt_lineshape,
    calculate_fwhm_for_pseudo_voigt_lineshape,
    calculate_height_for_pseudo_voigt_lineshape,
    calculate_height_for_gaussian_lineshape,
    calculate_height_for_lorentzian_lineshape,
    calculate_height_for_pv_pv_lineshape,
    calculate_lineshape_specific_height_and_fwhm,
    calculate_peak_linewidths_in_hz,
    calculate_peak_centers_in_ppm,
)


def test_gaussian_typical_values():
    x = np.array([0, 1, 2])
    center = 0.0
    sigma = 1.0
    expected = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x - center) ** 2) / (2 * sigma**2)
    )
    result = gaussian(x, center, sigma)
    assert_almost_equal(result, expected, decimal=7)


def test_gaussian_center_nonzero():
    x = np.array([0, 1, 2])
    center = 1.0
    sigma = 1.0
    expected = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x - center) ** 2) / (2 * sigma**2)
    )
    result = gaussian(x, center, sigma)
    assert_almost_equal(result, expected, decimal=7)


def test_gaussian_sigma_nonzero():
    x = np.array([0, 1, 2])
    center = 0.0
    sigma = 2.0
    expected = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x - center) ** 2) / (2 * sigma**2)
    )
    result = gaussian(x, center, sigma)
    assert_almost_equal(result, expected, decimal=7)


def test_gaussian_zero_center():
    x = np.array([0, 1, 2])
    center = 0.0
    sigma = 1.0
    expected = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x - center) ** 2) / (2 * sigma**2)
    )
    result = gaussian(x, center, sigma)
    assert_almost_equal(result, expected, decimal=7)


def test_calculate_height_for_voigt_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_voigt_lineshape(df)

    assert np.allclose(result_df["height"], [0.435596, 0.217798])
    assert np.allclose(result_df["height_err"], [0.04356, 0.02178])


def test_calculate_fwhm_for_voigt_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
    }
    df = pd.DataFrame(data)
    result_df = calculate_fwhm_for_voigt_lineshape(df)

    assert np.allclose(result_df["fwhm_l_x"], [2.0, 4.0])
    assert np.allclose(result_df["fwhm_l_y"], [2.0, 4.0])
    assert np.allclose(result_df["fwhm_g_x"], [2.35482, 4.70964])
    assert np.allclose(result_df["fwhm_g_y"], [2.35482, 4.70964])
    assert np.allclose(result_df["fwhm_x"], [3.601309, 7.202619])
    assert np.allclose(result_df["fwhm_y"], [3.601309, 7.202619])


def test_calculate_height_for_pseudo_voigt_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_pseudo_voigt_lineshape(df)

    assert np.allclose(result_df["height"], [1.552472, 0.776236])
    assert np.allclose(result_df["height_err"], [0.155247, 0.077624])


def test_calculate_fwhm_for_pseudo_voigt_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    result_df = calculate_fwhm_for_pseudo_voigt_lineshape(df)

    assert np.allclose(result_df["fwhm_x"], [2.0, 4.0])
    assert np.allclose(result_df["fwhm_y"], [2.0, 4.0])


def test_calculate_height_for_gaussian_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_gaussian_lineshape(df)

    assert np.allclose(result_df["height"], [2.206356, 1.103178])
    assert np.allclose(result_df["height_err"], [0.220636, 0.110318])


def test_calculate_height_for_lorentzian_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_lorentzian_lineshape(df)

    assert np.allclose(result_df["height"], [1.013212, 0.506606])
    assert np.allclose(result_df["height_err"], [0.101321, 0.050661])


def test_calculate_height_for_pv_pv_lineshape():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction_x": [0.5, 0.5],
        "fraction_y": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_pv_pv_lineshape(df)

    assert np.allclose(result_df["height"], [1.552472, 0.776236])
    assert np.allclose(result_df["height_err"], [0.155247, 0.077624])


def test_calculate_height_for_pv_pv_lineshape_fraction_y():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction_x": [0.5, 0.5],
        "fraction_y": [1.0, 1.0],
    }
    df = pd.DataFrame(data)
    result_df = calculate_height_for_pv_pv_lineshape(df)

    assert np.allclose(result_df["height"], [1.254186, 0.627093])
    assert np.allclose(result_df["height_err"], [0.125419, 0.062709])


def test_calculate_lineshape_specific_height_and_fwhm():
    data = {
        "sigma_x": [1.0, 2.0],
        "sigma_y": [1.0, 2.0],
        "gamma_x": [1.0, 2.0],
        "gamma_y": [1.0, 2.0],
        "amp": [10.0, 20.0],
        "amp_err": [1.0, 2.0],
        "fraction": [0.5, 0.5],
        "fraction_x": [0.5, 0.5],
        "fraction_y": [0.5, 0.5],
    }
    df = pd.DataFrame(data)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.G, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.L, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.V, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.PV, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.PV_PV, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.PV_G, df)
    calculate_lineshape_specific_height_and_fwhm(Lineshape.PV_L, df)


def test_get_lineshape_function():
    assert get_lineshape_function(Lineshape.PV) == pvoigt2d
    assert get_lineshape_function(Lineshape.L) == pvoigt2d
    assert get_lineshape_function(Lineshape.G) == pvoigt2d
    assert get_lineshape_function(Lineshape.G_L) == gaussian_lorentzian
    assert get_lineshape_function(Lineshape.PV_G) == pv_g
    assert get_lineshape_function(Lineshape.PV_L) == pv_l
    assert get_lineshape_function(Lineshape.PV_PV) == pv_pv
    assert get_lineshape_function(Lineshape.V) == voigt2d


def test_get_lineshape_function_exception():
    with pytest.raises(Exception):
        get_lineshape_function("bla")


@pytest.fixture
def peakipy_data():
    test_data_path = Path("./test/test_protein_L/")
    return Peaklist(
        test_data_path / "test.tab", test_data_path / "test1.ft2", PeaklistFormat.pipe
    )


def test_calculate_peak_linewidths_in_hz():
    # Sample data for testing
    data = {
        "sigma_x": [1.0, 2.0, 3.0],
        "sigma_y": [1.5, 2.5, 3.5],
        "fwhm_x": [0.5, 1.5, 2.5],
        "fwhm_y": [0.7, 1.7, 2.7],
    }
    df = pd.DataFrame(data)

    # Mock peakipy_data object
    peakipy_data = Mock()
    peakipy_data.ppm_per_pt_f2 = 0.01
    peakipy_data.ppm_per_pt_f1 = 0.02
    peakipy_data.hz_per_pt_f2 = 10.0
    peakipy_data.hz_per_pt_f1 = 20.0

    # Expected results
    expected_sigma_x_ppm = [0.01, 0.02, 0.03]
    expected_sigma_y_ppm = [0.03, 0.05, 0.07]
    expected_fwhm_x_ppm = [0.005, 0.015, 0.025]
    expected_fwhm_y_ppm = [0.014, 0.034, 0.054]
    expected_fwhm_x_hz = [5.0, 15.0, 25.0]
    expected_fwhm_y_hz = [14.0, 34.0, 54.0]

    # Run the function
    result_df = calculate_peak_linewidths_in_hz(df, peakipy_data)

    # Assertions
    pd.testing.assert_series_equal(
        result_df["sigma_x_ppm"], pd.Series(expected_sigma_x_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["sigma_y_ppm"], pd.Series(expected_sigma_y_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["fwhm_x_ppm"], pd.Series(expected_fwhm_x_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["fwhm_y_ppm"], pd.Series(expected_fwhm_y_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["fwhm_x_hz"], pd.Series(expected_fwhm_x_hz), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["fwhm_y_hz"], pd.Series(expected_fwhm_y_hz), check_names=False
    )


def test_calculate_peak_centers_in_ppm():
    # Sample data for testing
    data = {
        "center_x": [10, 20, 30],
        "center_y": [15, 25, 35],
        "init_center_x": [12, 22, 32],
        "init_center_y": [18, 28, 38],
    }
    df = pd.DataFrame(data)

    # Mock peakipy_data object
    peakipy_data = Mock()
    peakipy_data.uc_f2.ppm = Mock(side_effect=lambda x: x * 0.1)
    peakipy_data.uc_f1.ppm = Mock(side_effect=lambda x: x * 0.2)

    # Expected results
    expected_center_x_ppm = [1.0, 2.0, 3.0]
    expected_center_y_ppm = [3.0, 5.0, 7.0]
    expected_init_center_x_ppm = [1.2, 2.2, 3.2]
    expected_init_center_y_ppm = [3.6, 5.6, 7.6]

    # Run the function
    result_df = calculate_peak_centers_in_ppm(df, peakipy_data)

    # Assertions
    pd.testing.assert_series_equal(
        result_df["center_x_ppm"], pd.Series(expected_center_x_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["center_y_ppm"], pd.Series(expected_center_y_ppm), check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["init_center_x_ppm"],
        pd.Series(expected_init_center_x_ppm),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result_df["init_center_y_ppm"],
        pd.Series(expected_init_center_y_ppm),
        check_names=False,
    )
