from pytest import fixture

import pandas as pd
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult

from peakipy.cli.fit import (
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
)
from peakipy.core import Lineshape, pvoigt2d


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
