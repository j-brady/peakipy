import unittest
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import nmrglue as ng
from numpy.testing import assert_array_equal
from lmfit import Model, Parameters

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
