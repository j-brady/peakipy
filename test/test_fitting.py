import unittest
from unittest.mock import MagicMock
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from pytest import fixture
import nmrglue as ng
from numpy.testing import assert_array_equal
from lmfit import Model, Parameters
from lmfit.model import ModelResult

from peakipy.io import Pseudo3D
from peakipy.fitting import (
    FitDataModel,
    validate_fit_data,
    validate_fit_dataframe,
    select_reference_planes_using_indices,
    slice_peaks_from_data_using_mask,
    select_planes_above_threshold_from_masked_data,
    get_limits_for_axis_in_points,
    deal_with_peaks_on_edge_of_spectrum,
    estimate_amplitude,
    make_mask,
    make_mask_from_peak_cluster,
    make_meshgrid,
    get_params,
    fix_params,
    make_param_dict,
    to_prefix,
    make_models,
    PeakLimits,
    update_params,
    make_masks_from_plane_data,
    get_fit_peaks_result_validation_model,
    FitPeaksResultRowPVPV,
    FitPeaksResultRowVoigt,
    FitPeaksResultRowGLPV,
    filter_peak_clusters_by_max_cluster_size,
    set_parameters_to_fix_during_fit,
    unpack_fitted_parameters_for_lineshape,
    get_default_lineshape_param_names,
    split_parameter_sets_by_peak,
    get_prefix_from_parameter_names,
    create_parameter_dict,
    perform_initial_lineshape_fit_on_cluster_of_peaks,
    merge_unpacked_parameters_with_metadata,
    add_vclist_to_df,
    update_cluster_df_with_fit_statistics,
    rename_columns_for_compatibility,
    FitPeaksArgs,
    FitPeaksInput,
    FitResult,
)
from peakipy.lineshapes import Lineshape, pvoigt2d, pv_pv


@pytest.fixture
def fitdatamodel_dict():
    return FitDataModel(
        plane=1,
        clustid=1,
        assignment="assignment",
        memcnt=1,
        amp=10.0,
        height=10.0,
        center_x_ppm=0.0,
        center_y_ppm=0.0,
        fwhm_x_hz=10.0,
        fwhm_y_hz=10.0,
        lineshape="PV",
        x_radius=0.04,
        y_radius=0.4,
        center_x=0.0,
        center_y=0.0,
        sigma_x=1.0,
        sigma_y=1.0,
    ).model_dump()


def test_validate_fit_data_PVGL(fitdatamodel_dict):
    fitdatamodel_dict.update(dict(fraction=0.5))
    validate_fit_data(fitdatamodel_dict)

    fitdatamodel_dict.update(dict(lineshape="G"))
    validate_fit_data(fitdatamodel_dict)

    fitdatamodel_dict.update(dict(lineshape="L"))
    validate_fit_data(fitdatamodel_dict)

    fitdatamodel_dict.update(
        dict(lineshape="V", fraction=0.5, gamma_x=1.0, gamma_y=1.0)
    )
    validate_fit_data(fitdatamodel_dict)

    fitdatamodel_dict.update(dict(lineshape="PVPV", fraction_x=0.5, fraction_y=1.0))
    validate_fit_data(fitdatamodel_dict)


def test_validate_fit_dataframe(fitdatamodel_dict):
    fitdatamodel_dict.update(dict(fraction=0.5))
    df = pd.DataFrame([fitdatamodel_dict] * 5)
    validate_fit_dataframe(df)


def test_select_reference_planes_using_indices():
    data = np.zeros((6, 100, 200))
    indices = []
    np.testing.assert_array_equal(
        select_reference_planes_using_indices(data, indices), data
    )
    indices = [1]
    assert select_reference_planes_using_indices(data, indices).shape == (1, 100, 200)
    indices = [1, -1]
    assert select_reference_planes_using_indices(data, indices).shape == (2, 100, 200)


def test_select_reference_planes_using_indices_min_index_error():
    data = np.zeros((6, 100, 200))
    indices = [-7]
    with pytest.raises(IndexError):
        select_reference_planes_using_indices(data, indices)


def test_select_reference_planes_using_indices_max_index_error():
    data = np.zeros((6, 100, 200))
    indices = [6]
    with pytest.raises(IndexError):
        select_reference_planes_using_indices(data, indices)


def test_slice_peaks_from_data_using_mask():
    data = np.array(
        [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
                    [0, 0, 1, 2, 3, 3, 2, 1, 0, 0],
                    [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
                    [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
                    [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
                    [0, 0, 1, 2, 3, 3, 2, 1, 0, 0],
                    [0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
            for i in range(5)
        ]
    )
    mask = data[0] > 0
    assert data.shape == (5, 11, 10)
    assert mask.shape == (11, 10)
    peak_slices = slice_peaks_from_data_using_mask(data, mask)
    # array is flattened by application of mask
    assert peak_slices.shape == (5, 50)


def test_select_planes_above_threshold_from_masked_data():
    peak_slices = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [-1, -1, -1, -1, -1, -1],
            [-2, -2, -2, -2, -2, -2],
        ]
    )
    assert peak_slices.shape == (4, 6)
    threshold = -1
    assert select_planes_above_threshold_from_masked_data(
        peak_slices, threshold
    ).shape == (
        4,
        6,
    )
    threshold = 2
    assert_array_equal(
        select_planes_above_threshold_from_masked_data(peak_slices, threshold),
        peak_slices,
    )
    threshold = 1
    assert select_planes_above_threshold_from_masked_data(
        peak_slices, threshold
    ).shape == (2, 6)

    threshold = None
    assert_array_equal(
        select_planes_above_threshold_from_masked_data(peak_slices, threshold),
        peak_slices,
    )
    threshold = 10
    assert_array_equal(
        select_planes_above_threshold_from_masked_data(peak_slices, threshold),
        peak_slices,
    )


def test_make_param_dict():
    selected_planes = [1, 2]
    data = np.ones((4, 10, 5))
    expected_shape = (2, 10, 5)
    actual_shape = data[np.array(selected_planes)].shape
    assert expected_shape == actual_shape


def test_make_param_dict_sum():
    data = np.ones((4, 10, 5))
    expected_sum = 200
    actual_sum = data.sum()
    assert expected_sum == actual_sum


def test_make_param_dict_selected():
    selected_planes = [1, 2]
    data = np.ones((4, 10, 5))
    data = data[np.array(selected_planes)]
    expected_sum = 100
    actual_sum = data.sum()
    assert expected_sum == actual_sum


def test_update_params_normal_case():
    params = Parameters()
    params.add("center_x", value=0)
    params.add("center_y", value=0)
    params.add("sigma", value=1)
    params.add("gamma", value=1)
    params.add("fraction", value=0.5)

    param_dict = {
        "center_x": 10,
        "center_y": 20,
        "sigma": 2,
        "gamma": 3,
        "fraction": 0.8,
    }

    xy_bounds = (5, 5)

    update_params(params, param_dict, Lineshape.PV, xy_bounds)

    assert params["center_x"].value == 10
    assert params["center_y"].value == 20
    assert params["sigma"].value == 2
    assert params["gamma"].value == 3
    assert params["fraction"].value == 0.8
    assert params["center_x"].min == 5
    assert params["center_x"].max == 15
    assert params["center_y"].min == 15
    assert params["center_y"].max == 25
    assert params["sigma"].min == 0.0
    assert params["sigma"].max == 1e4
    assert params["gamma"].min == 0.0
    assert params["gamma"].max == 1e4
    assert params["fraction"].min == 0.0
    assert params["fraction"].max == 1.0
    assert params["fraction"].vary is True


def test_update_params_lineshape_G():
    params = Parameters()
    params.add("fraction", value=0.5)

    param_dict = {"fraction": 0.7}

    update_params(params, param_dict, Lineshape.G)

    assert params["fraction"].value == 0.7
    assert params["fraction"].min == 0.0
    assert params["fraction"].max == 1.0
    assert params["fraction"].vary is False


def test_update_params_lineshape_L():
    params = Parameters()
    params.add("fraction", value=0.5)

    param_dict = {"fraction": 0.7}

    update_params(params, param_dict, Lineshape.L)

    assert params["fraction"].value == 0.7
    assert params["fraction"].min == 0.0
    assert params["fraction"].max == 1.0
    assert params["fraction"].vary is False


def test_update_params_lineshape_PV_PV():
    params = Parameters()
    params.add("fraction", value=0.5)

    param_dict = {"fraction": 0.7}

    update_params(params, param_dict, Lineshape.PV_PV)

    assert params["fraction"].value == 0.7
    assert params["fraction"].min == 0.0
    assert params["fraction"].max == 1.0
    assert params["fraction"].vary is True


def test_update_params_no_bounds():
    params = Parameters()
    params.add("center_x", value=0)
    params.add("center_y", value=0)

    param_dict = {
        "center_x": 10,
        "center_y": 20,
    }

    update_params(params, param_dict, Lineshape.PV, None)

    assert params["center_x"].value == 10
    assert params["center_y"].value == 20
    assert params["center_x"].min == -np.inf
    assert params["center_x"].max == np.inf
    assert params["center_y"].min == -np.inf
    assert params["center_y"].max == np.inf


def test_peak_limits_normal_case():
    peak = pd.DataFrame({"X_AXIS": [5], "Y_AXIS": [5], "XW": [2], "YW": [2]}).iloc[0]
    data = np.zeros((10, 10))
    pl = PeakLimits(peak, data)
    assert pl.min_x == 3
    assert pl.max_x == 8
    assert pl.min_y == 3
    assert pl.max_y == 8


def test_peak_limits_at_edge():
    peak = pd.DataFrame({"X_AXIS": [1], "Y_AXIS": [1], "XW": [2], "YW": [2]}).iloc[0]
    data = np.zeros((10, 10))
    pl = PeakLimits(peak, data)
    assert pl.min_x == 0
    assert pl.max_x == 4
    assert pl.min_y == 0
    assert pl.max_y == 4


def test_peak_limits_exceeding_bounds():
    peak = pd.DataFrame({"X_AXIS": [9], "Y_AXIS": [9], "XW": [2], "YW": [2]}).iloc[0]
    data = np.zeros((10, 10))
    pl = PeakLimits(peak, data)
    assert pl.min_x == 7
    assert pl.max_x == 10
    assert pl.min_y == 7
    assert pl.max_y == 10


def test_peak_limits_small_data():
    peak = pd.DataFrame({"X_AXIS": [2], "Y_AXIS": [2], "XW": [5], "YW": [5]}).iloc[0]
    data = np.zeros((5, 5))
    pl = PeakLimits(peak, data)
    assert pl.min_x == 0
    assert pl.max_x == 5
    assert pl.min_y == 0
    assert pl.max_y == 5


def test_peak_limits_assertion_error():
    peak = pd.DataFrame({"X_AXIS": [11], "Y_AXIS": [11], "XW": [2], "YW": [2]}).iloc[0]
    data = np.zeros((10, 10))
    with pytest.raises(AssertionError):
        pl = PeakLimits(peak, data)


def test_estimate_amplitude():
    peak = namedtuple("peak", ["X_AXIS", "XW", "Y_AXIS", "YW"])
    p = peak(5, 2, 3, 2)
    data = np.ones((20, 10))
    expected_result = 25
    actual_result = estimate_amplitude(p, data)
    assert expected_result == actual_result


def test_estimate_amplitude_invalid_indices():
    peak = namedtuple("peak", ["X_AXIS", "XW", "Y_AXIS", "YW"])
    p = peak(1, 2, 3, 2)
    data = np.ones((20, 10))
    expected_result = 20
    actual_result = estimate_amplitude(p, data)
    assert expected_result == actual_result


def test_make_mask_from_peak_cluster():
    data = np.ones((10, 10))
    group = pd.DataFrame(
        {"X_AXISf": [3, 6], "Y_AXISf": [3, 6], "X_RADIUS": [2, 3], "Y_RADIUS": [2, 3]}
    )
    mask, peak = make_mask_from_peak_cluster(group, data)
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert_array_equal(expected_mask, mask)


# get_limits_for_axis_in_points
def test_positive_points():
    group_axis_points = np.array([1, 2, 3, 4, 5])
    mask_radius_in_points = 2
    expected = (8, -1)  # ceil(5+1+1), floor(1-1)
    assert (
        get_limits_for_axis_in_points(group_axis_points, mask_radius_in_points)
        == expected
    )


def test_single_point():
    group_axis_points = np.array([5])
    mask_radius_in_points = 3
    expected = (9, 2)
    assert (
        get_limits_for_axis_in_points(group_axis_points, mask_radius_in_points)
        == expected
    )


def test_no_radius():
    group_axis_points = np.array([1, 2, 3])
    mask_radius_in_points = 0
    expected = (4, 1)
    assert (
        get_limits_for_axis_in_points(group_axis_points, mask_radius_in_points)
        == expected
    )


# deal_with_peaks_on_edge_of_spectrum
def test_min_y_less_than_zero():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 50, 30, 10, -10) == (
        50,
        30,
        10,
        0,
    )


def test_min_x_less_than_zero():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 50, -5, 70, 20) == (
        50,
        0,
        70,
        20,
    )


def test_max_y_exceeds_data_shape():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 50, 30, 110, 20) == (
        50,
        30,
        100,
        20,
    )


def test_max_x_exceeds_data_shape():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 250, 30, 70, 20) == (
        200,
        30,
        70,
        20,
    )


def test_values_within_range():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 50, 30, 70, 20) == (
        50,
        30,
        70,
        20,
    )


def test_all_edge_cases():
    assert deal_with_peaks_on_edge_of_spectrum((100, 200), 250, -5, 110, -10) == (
        200,
        0,
        100,
        0,
    )


def test_make_meshgrid():
    data_shape = (4, 5)
    expected_x = np.array(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    )
    expected_y = np.array(
        [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
    )
    XY = make_meshgrid(data_shape)
    np.testing.assert_array_equal(XY[0], expected_x)
    np.testing.assert_array_equal(XY[1], expected_y)


class TestCoreFunctions(unittest.TestCase):
    test_directory = Path(__file__).parent
    test_directory = "./test"

    def test_make_mask(self):
        data = np.ones((10, 10))
        c_x = 5
        c_y = 5
        r_x = 3
        r_y = 2

        expected_result = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        result = np.array(make_mask(data, c_x, c_y, r_x, r_y), dtype=int)
        test = result - expected_result
        # print(test)
        # print(test.sum())
        # print(result)
        self.assertEqual(test.sum(), 0)

    def test_make_mask_2(self):
        data = np.ones((10, 10))
        c_x = 5
        c_y = 8
        r_x = 3
        r_y = 2

        expected_result = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )

        result = np.array(make_mask(data, c_x, c_y, r_x, r_y), dtype=int)
        test = result - expected_result
        # print(test)
        # print(test.sum())
        # print(result)
        self.assertEqual(test.sum(), 0)

    def test_fix_params(self):
        mod = Model(pvoigt2d)
        pars = mod.make_params()
        to_fix = ["center", "sigma", "fraction"]
        fix_params(pars, to_fix)

        self.assertEqual(pars["center_x"].vary, False)
        self.assertEqual(pars["center_y"].vary, False)
        self.assertEqual(pars["sigma_x"].vary, False)
        self.assertEqual(pars["sigma_y"].vary, False)
        self.assertEqual(pars["fraction"].vary, False)

    def test_get_params(self):
        mod = Model(pvoigt2d, prefix="p1_")
        pars = mod.make_params(p1_center_x=20.0, p1_center_y=30.0)
        pars["p1_center_x"].stderr = 1.0
        pars["p1_center_y"].stderr = 2.0
        ps, ps_err, names, prefixes = get_params(pars, "center")
        #  get index of values
        cen_x = names.index("p1_center_x")
        cen_y = names.index("p1_center_y")

        self.assertEqual(ps[cen_x], 20.0)
        self.assertEqual(ps[cen_y], 30.0)
        self.assertEqual(ps_err[cen_x], 1.0)
        self.assertEqual(ps_err[cen_y], 2.0)
        self.assertEqual(prefixes[cen_y], "p1_")

    def test_make_param_dict(self):
        peaks = pd.DataFrame(
            {
                "ASS": ["one", "two", "three"],
                "X_AXISf": [5.0, 10.0, 15.0],
                "X_AXIS": [5, 10, 15],
                "Y_AXISf": [15.0, 10.0, 5.0],
                "Y_AXIS": [15, 10, 5],
                "XW": [2.5, 2.5, 2.5],
                "YW": [2.5, 2.5, 2.5],
            }
        )
        data = np.ones((20, 20))

        for ls, frac in zip([Lineshape.PV, Lineshape.G, Lineshape.L], [0.5, 0.0, 1.0]):
            params = make_param_dict(peaks, data, ls)
            self.assertEqual(params["_one_fraction"], frac)
            self.assertEqual(params["_two_fraction"], frac)
            self.assertEqual(params["_three_fraction"], frac)

        self.assertEqual(params["_one_center_x"], 5.0)
        self.assertEqual(params["_two_center_x"], 10.0)
        self.assertEqual(params["_two_sigma_x"], 1.25)
        self.assertEqual(params["_two_sigma_y"], 1.25)

        voigt_params = make_param_dict(peaks, data, Lineshape.V)
        self.assertEqual(
            voigt_params["_one_sigma_x"], 2.5 / (2.0 * np.sqrt(2.0 * np.log(2)))
        )
        self.assertEqual(voigt_params["_one_gamma_x"], 2.5 / 2.0)

    def test_to_prefix(self):
        names = [
            (1, "_1_"),
            (1.0, "_1_0_"),
            (" one", "_one_"),
            (" one/two", "_oneortwo_"),
            (" one?two", "_onemaybetwo_"),
            (r" [{one?two\}][", "___onemaybetwo____"),
            ("hel'lo", "_hel_lo_"),
        ]
        for test, expect in names:
            prefix = to_prefix(test)
            # print(prefix)
            self.assertEqual(prefix, expect)

    def test_make_models(self):
        peaks = pd.DataFrame(
            {
                "ASS": ["one", "two", "three"],
                "X_AXISf": [5.0, 10.0, 15.0],
                "X_AXIS": [5, 10, 15],
                "Y_AXISf": [15.0, 10.0, 5.0],
                "Y_AXIS": [15, 10, 5],
                "XW": [2.5, 2.5, 2.5],
                "YW": [2.5, 2.5, 2.5],
                "CLUSTID": [1, 1, 1],
            }
        )

        group = peaks.groupby("CLUSTID")

        data = np.ones((20, 20))

        lineshapes = [Lineshape.PV, Lineshape.L, Lineshape.G, Lineshape.PV_PV]

        for lineshape in lineshapes:
            match lineshape:
                case lineshape.PV:
                    mod, p_guess = make_models(pvoigt2d, peaks, data, lineshape)
                    self.assertEqual(p_guess["_one_fraction"].vary, True)
                    self.assertEqual(p_guess["_one_fraction"].value, 0.5)

                case lineshape.G:
                    mod, p_guess = make_models(pvoigt2d, peaks, data, lineshape)
                    self.assertEqual(p_guess["_one_fraction"].vary, False)
                    self.assertEqual(p_guess["_one_fraction"].value, 0.0)

                case lineshape.L:
                    mod, p_guess = make_models(pvoigt2d, peaks, data, lineshape)
                    self.assertEqual(p_guess["_one_fraction"].vary, False)
                    self.assertEqual(p_guess["_one_fraction"].value, 1.0)

                case lineshape.PV_PV:
                    mod, p_guess = make_models(pv_pv, peaks, data, lineshape)
                    self.assertEqual(p_guess["_one_fraction_x"].vary, True)
                    self.assertEqual(p_guess["_one_fraction_x"].value, 0.5)
                    self.assertEqual(p_guess["_one_fraction_y"].vary, True)
                    self.assertEqual(p_guess["_one_fraction_y"].value, 0.5)

    def test_Pseudo3D(self):
        datasets = [
            (f"{self.test_directory}/test_protein_L/test1.ft2", [0, 1, 2]),
            (f"{self.test_directory}/test_protein_L/test_tp.ft2", [2, 1, 0]),
            (f"{self.test_directory}/test_protein_L/test_tp2.ft2", [1, 2, 0]),
        ]

        # expected shape
        data_shape = (4, 256, 546)
        test_nu = 1
        for dataset, dims in datasets:
            with self.subTest(i=test_nu):
                dic, data = ng.pipe.read(dataset)
                pseudo3D = Pseudo3D(dic, data, dims)
                self.assertEqual(dims, pseudo3D.dims)
                self.assertEqual(pseudo3D.data.shape, data_shape)
                self.assertEqual(pseudo3D.f1_label, "15N")
                self.assertEqual(pseudo3D.f2_label, "HN")
                self.assertEqual(pseudo3D.dims, dims)
                self.assertEqual(pseudo3D.f1_size, 256)
                self.assertEqual(pseudo3D.f2_size, 546)
            test_nu += 1


def test_get_fit_peaks_result_validation_model_PVPV():
    validation_model = get_fit_peaks_result_validation_model(Lineshape.PV_PV)
    assert validation_model == FitPeaksResultRowPVPV


def test_get_fit_peaks_result_validation_model_G():
    validation_model = get_fit_peaks_result_validation_model(Lineshape.G)
    assert validation_model == FitPeaksResultRowGLPV


def test_get_fit_peaks_result_validation_model_L():
    validation_model = get_fit_peaks_result_validation_model(Lineshape.L)
    assert validation_model == FitPeaksResultRowGLPV


def test_get_fit_peaks_result_validation_model_PV():
    validation_model = get_fit_peaks_result_validation_model(Lineshape.PV)
    assert validation_model == FitPeaksResultRowGLPV


def test_get_fit_peaks_result_validation_model_V():
    validation_model = get_fit_peaks_result_validation_model(Lineshape.V)
    assert validation_model == FitPeaksResultRowVoigt


def test_filter_groups_by_max_cluster_size():
    groups = pd.DataFrame(
        dict(
            col1=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7],
            col2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7],
        )
    ).groupby("col1")
    max_cluster_size = 3
    filtered_groups = filter_peak_clusters_by_max_cluster_size(groups, max_cluster_size)
    filtered_group_names = filtered_groups.col1.unique()
    expected_group_names = np.array([3, 4, 5, 6, 7])
    np.testing.assert_array_equal(filtered_group_names, expected_group_names)


def test_set_parameters_to_fix_during_fit():
    parameter_set = Parameters()
    parameter_set.add("test1", vary=True)
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameter_set, ["test1"]
    )
    assert modified_parameter_set["test1"].vary == False


@fixture
def parameters_set_with_two_variables():
    parameter_set = Parameters()
    parameter_set.add("prefix1_test1", vary=True)
    parameter_set.add("prefix1_test2", vary=True)
    return parameter_set


def test_set_parameters_to_fix_during_fit_2(parameters_set_with_two_variables):
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameters_set_with_two_variables, ["prefix1_test1", "prefix1_test2"]
    )
    assert (
        modified_parameter_set["prefix1_test2"].vary
        == modified_parameter_set["prefix1_test1"].vary
        == False
    )


def test_set_parameters_to_fix_during_fit_3():
    parameter_set = Parameters()
    parameter_set.add("test1", vary=True)
    parameter_set.add("test2", vary=True)
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameter_set, ["test2"]
    )
    assert (
        modified_parameter_set["test1"].vary
        != modified_parameter_set["test2"].vary
        == False
    )


def test_set_parameters_to_fix_during_fit_None():
    parameter_set = Parameters()
    parameter_set.add("test1", vary=True)
    parameter_set.add("test2", vary=True)
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameter_set, None
    )
    assert (
        modified_parameter_set["test1"].vary
        == modified_parameter_set["test2"].vary
        == True
    )


def test_set_parameters_to_fix_during_fit_None_str():
    parameter_set = Parameters()
    parameter_set.add("test1", vary=True)
    parameter_set.add("test2", vary=True)
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameter_set, ["None"]
    )
    assert (
        modified_parameter_set["test1"].vary
        == modified_parameter_set["test2"].vary
        == True
    )


def test_update_cluster_df_with_fit_statistics():
    result = ModelResult(Model(pvoigt2d), None, None)
    result.aic = None
    result.bic = None
    data = [
        dict(
            chisqr=None,
            redchi=None,
            residual_sum=None,
            aic=None,
            bic=None,
            nfev=0,
            ndata=0,
        )
    ]
    expected_cluster_df = pd.DataFrame(data)
    actual_cluster_df = update_cluster_df_with_fit_statistics(
        expected_cluster_df, result
    )
    pd.testing.assert_frame_equal(actual_cluster_df, expected_cluster_df)


def test_rename_columns_for_compatibility():
    df = pd.DataFrame(
        [
            dict(
                amplitude=1,
                amplitude_stderr=1,
                X_AXIS=1,
                Y_AXIS=1,
                ASS="None",
                MEMCNT=1,
                X_RADIUS=1,
                Y_RADIUS=1,
            )
        ]
    )
    expected_columns = [
        "amp",
        "amp_err",
        "init_center_x",
        "init_center_y",
        "assignment",
        "memcnt",
        "x_radius",
        "y_radius",
    ]
    actual_columns = rename_columns_for_compatibility(df).columns
    assert all([i == j for i, j in zip(actual_columns, expected_columns)])


def test_get_default_param_names_pseudo_voigt():
    assert get_default_lineshape_param_names(Lineshape.PV) == [
        "amplitude",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "fraction",
    ]


def test_get_default_param_names_gaussian():
    assert get_default_lineshape_param_names(Lineshape.G) == [
        "amplitude",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "fraction",
    ]


def test_get_default_param_names_lorentzian():
    assert get_default_lineshape_param_names(Lineshape.L) == [
        "amplitude",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "fraction",
    ]


def test_get_default_param_names_pv_pv():
    assert get_default_lineshape_param_names(Lineshape.PV_PV) == [
        "amplitude",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "fraction_x",
        "fraction_y",
    ]


def test_get_default_param_names_voigt():
    assert get_default_lineshape_param_names(Lineshape.V) == [
        "amplitude",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "gamma_x",
        "gamma_y",
        "fraction",
    ]


def test_split_parameter_sets_by_peak(default_pseudo_voigt_parameter_names):
    # the second element of each tuple actually contains an
    # lmfit.Parameter object
    params = [
        ("p1_amplitude", "amplitude"),
        ("p1_center_x", "center_x"),
        ("p1_center_y", "center_y"),
        ("p1_sigma_x", "sigma_x"),
        ("p1_sigma_y", "sigma_y"),
        ("p1_fraction", "fraction"),
        ("p2_amplitude", "amplitude"),
        ("p2_center_x", "center_x"),
        ("p2_center_y", "center_y"),
        ("p2_sigma_x", "sigma_x"),
        ("p2_sigma_y", "sigma_y"),
        ("p2_fraction", "fraction"),
        ("p3_amplitude", "amplitude"),
        ("p3_center_x", "center_x"),
        ("p3_center_y", "center_y"),
        ("p3_sigma_x", "sigma_x"),
        ("p3_sigma_y", "sigma_y"),
        ("p3_fraction", "fraction"),
    ]
    expected_result = [
        [
            ("p1_amplitude", "amplitude"),
            ("p1_center_x", "center_x"),
            ("p1_center_y", "center_y"),
            ("p1_sigma_x", "sigma_x"),
            ("p1_sigma_y", "sigma_y"),
            ("p1_fraction", "fraction"),
        ],
        [
            ("p2_amplitude", "amplitude"),
            ("p2_center_x", "center_x"),
            ("p2_center_y", "center_y"),
            ("p2_sigma_x", "sigma_x"),
            ("p2_sigma_y", "sigma_y"),
            ("p2_fraction", "fraction"),
        ],
        [
            ("p3_amplitude", "amplitude"),
            ("p3_center_x", "center_x"),
            ("p3_center_y", "center_y"),
            ("p3_sigma_x", "sigma_x"),
            ("p3_sigma_y", "sigma_y"),
            ("p3_fraction", "fraction"),
        ],
    ]
    expected_result_parameter_names = [[j[0] for j in i] for i in expected_result]
    split_parameter_names = [
        [j[0] for j in i]
        for i in split_parameter_sets_by_peak(
            default_pseudo_voigt_parameter_names, params
        )
    ]
    assert split_parameter_names == expected_result_parameter_names


@fixture
def default_pseudo_voigt_parameter_names():
    return Model(pvoigt2d).param_names


def test_get_prefix_from_parameter_names(default_pseudo_voigt_parameter_names):
    parameter_items_with_prefixes = [
        ("p1_amplitude", "amplitude"),
        ("p1_center_x", "center_x"),
        ("p1_center_y", "center_y"),
        ("p1_sigma_x", "sigma_x"),
        ("p1_sigma_y", "sigma_y"),
        ("p1_fraction", "fraction"),
    ]
    expected_result = "p1_"
    actual_result = get_prefix_from_parameter_names(
        default_pseudo_voigt_parameter_names, parameter_items_with_prefixes
    )
    assert expected_result == actual_result


@fixture
def pseudo_voigt_model_result():
    m1 = Model(pvoigt2d, prefix="p1_")
    m2 = Model(pvoigt2d, prefix="p2_")
    model = m1 + m2
    params = model.make_params()
    model_result = ModelResult(model, params)
    return model_result


def test_create_parameter_dict(pseudo_voigt_model_result):
    prefix = "p1_"
    params = list(pseudo_voigt_model_result.params.items())[:6]
    expected_result = dict(
        prefix="p1_",
        amplitude=1.0,
        amplitude_stderr=None,
        center_x=0.5,
        center_x_stderr=None,
        center_y=0.5,
        center_y_stderr=None,
        sigma_x=1.0,
        sigma_x_stderr=None,
        sigma_y=1.0,
        sigma_y_stderr=None,
        fraction=0.5,
        fraction_stderr=None,
    )
    actual_result = create_parameter_dict(prefix, params)
    assert expected_result == actual_result


def test_unpack_fitted_parameters_for_lineshape_PV(pseudo_voigt_model_result):
    expected_params = [
        dict(
            prefix="p1_",
            plane=0,
            amplitude=1.0,
            amplitude_stderr=None,
            center_x=0.5,
            center_x_stderr=None,
            center_y=0.5,
            center_y_stderr=None,
            sigma_x=1.0,
            sigma_x_stderr=None,
            sigma_y=1.0,
            sigma_y_stderr=None,
            fraction=0.5,
            fraction_stderr=None,
        ),
        dict(
            prefix="p2_",
            plane=0,
            amplitude=1.0,
            amplitude_stderr=None,
            center_x=0.5,
            center_x_stderr=None,
            center_y=0.5,
            center_y_stderr=None,
            sigma_x=1.0,
            sigma_x_stderr=None,
            sigma_y=1.0,
            sigma_y_stderr=None,
            fraction=0.5,
            fraction_stderr=None,
        ),
    ]
    unpacked_params = unpack_fitted_parameters_for_lineshape(
        Lineshape.PV, list(pseudo_voigt_model_result.params.items()), plane_number=0
    )
    assert expected_params == unpacked_params


def test_merge_unpacked_parameters_with_metadata():
    cluster_fit_df = pd.DataFrame(
        dict(
            plane=[0, 1, 2, 3, 0, 1, 2, 3],
            prefix=["_p1_", "_p1_", "_p1_", "_p1_", "_p2_", "_p2_", "_p2_", "_p2_"],
        )
    )
    peak_df = pd.DataFrame(dict(ASS=["p1", "p2"], data=["p1_data", "p2_data"]))
    expected_result = pd.DataFrame(
        dict(
            plane=[0, 1, 2, 3, 0, 1, 2, 3],
            prefix=["_p1_", "_p1_", "_p1_", "_p1_", "_p2_", "_p2_", "_p2_", "_p2_"],
            ASS=["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            data=[
                "p1_data",
                "p1_data",
                "p1_data",
                "p1_data",
                "p2_data",
                "p2_data",
                "p2_data",
                "p2_data",
            ],
        )
    )
    actual_result = merge_unpacked_parameters_with_metadata(cluster_fit_df, peak_df)
    assert expected_result.equals(actual_result)


def test_add_vclist_to_df():
    args = FitPeaksArgs(
        noise=0, uc_dics={}, lineshape=Lineshape.PV, vclist_data=np.array([1, 2, 3])
    )
    fit_peaks_input = FitPeaksInput(
        args=args, data=None, config=None, plane_numbers=None
    )
    df = pd.DataFrame(dict(plane=[0, 1, 2]))
    expected_df = pd.DataFrame(dict(plane=[0, 1, 2], vclist=[1, 2, 3]))
    actual_df = add_vclist_to_df(fit_peaks_input, df)
    assert actual_df.equals(expected_df)


def test_add_vclist_to_df_plane_order():
    args = FitPeaksArgs(
        noise=0, uc_dics={}, lineshape=Lineshape.PV, vclist_data=np.array([1, 2, 3])
    )
    fit_peaks_input = FitPeaksInput(
        args=args, data=None, config=None, plane_numbers=None
    )
    df = pd.DataFrame(dict(plane=[2, 1, 0]))
    expected_df = pd.DataFrame(dict(plane=[2, 1, 0], vclist=[3, 2, 1]))
    actual_df = add_vclist_to_df(fit_peaks_input, df)
    assert actual_df.equals(expected_df)


# def test_perform_initial_lineshape_fit_on_cluster_of_peaks(pseudo_voigt_model_result):
#     expected_result = pseudo_voigt_model_result
#     actual_result = perform_initial_lineshape_fit_on_cluster_of_peaks()
#     assert expected_result == actual_result
# Mock FitPeakClusterInput class for testing purposes
class MockFitPeakClusterInput:
    def __init__(
        self,
        mod,
        peak_slices,
        XY_slices,
        p_guess,
        weights,
        fit_method,
        mask,
        XY,
        first_plane_data,
        last_peak,
        group,
        min_x,
        min_y,
        max_x,
        max_y,
        verbose,
        uc_dics,
    ):
        self.mod = mod
        self.peak_slices = peak_slices
        self.XY_slices = XY_slices
        self.p_guess = p_guess
        self.weights = weights
        self.fit_method = fit_method
        self.mask = mask
        self.XY = XY
        self.first_plane_data = first_plane_data
        self.last_peak = last_peak
        self.group = group
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.verbose = verbose
        self.uc_dics = uc_dics


@pytest.fixture
def fit_peak_cluster_input():
    mod = MagicMock()
    mod.fit = MagicMock(
        return_value=MagicMock(
            params="params", fit_report=MagicMock(return_value="fit_report")
        )
    )
    mod.eval = MagicMock(return_value=np.array([2.0, 1.0, 2.0]))

    return MockFitPeakClusterInput(
        mod=mod,
        peak_slices="peak_slices",
        XY_slices="XY_slices",
        p_guess="p_guess",
        weights="weights",
        fit_method="fit_method",
        mask=np.array([True, False, True]),
        XY=(np.array([0, 1, 2]), np.array([0, 1, 2])),
        first_plane_data=np.array([2.0, 1.0, 2.0]),
        last_peak="last_peak",
        group="group",
        min_x="min_x",
        min_y="min_y",
        max_x="max_x",
        max_y="max_y",
        verbose=True,
        uc_dics="uc_dics",
    )


def test_perform_initial_lineshape_fit_on_cluster_of_peaks(fit_peak_cluster_input):

    result = perform_initial_lineshape_fit_on_cluster_of_peaks(fit_peak_cluster_input)

    # Check if result is an instance of FitResult
    assert isinstance(result, FitResult)

    # Verify returned values
    assert result.out.params == "params"
    np.testing.assert_array_equal(result.mask, np.array([True, False, True]))
    assert result.fit_str == ""
    assert result.log == ""
    assert result.group == "group"
    assert result.uc_dics == "uc_dics"
    assert result.min_x == "min_x"
    assert result.min_y == "min_y"
    assert result.max_x == "max_x"
    assert result.max_y == "max_y"
    np.testing.assert_array_equal(result.X, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.Y, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.Z, np.array([2.0, np.nan, 2.0]))
    np.testing.assert_array_equal(result.Z_sim, np.array([2.0, np.nan, 2.0]))
    assert result.peak_slices == "peak_slices"
    assert result.XY_slices == "XY_slices"
    assert result.weights == "weights"
    assert result.mod == fit_peak_cluster_input.mod

    # Check if mod.fit and mod.eval were called with correct arguments
    fit_peak_cluster_input.mod.fit.assert_called_once_with(
        "peak_slices",
        XY="XY_slices",
        params="p_guess",
        weights="weights",
        method="fit_method",
    )
    # fit_peak_cluster_input.mod.eval.assert_called_once_with(
    #     XY=(np.array([0,1,2]), np.array([0,1,2])), params='params'
    # )
