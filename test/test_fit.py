from pytest import fixture

import pandas as pd
import numpy as np
from lmfit import Parameters

from peakipy.cli.fit import (
    get_fit_peaks_result_validation_model,
    FitPeaksResultRowPVPV,
    FitPeaksResultRowVoigt,
    FitPeaksResultRowGLPV,
    filter_peak_clusters_by_max_cluster_size,
    set_parameters_to_fix_during_fit,
)
from peakipy.core import Lineshape


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


def test_set_parameters_to_fix_during_fit_2():
    parameter_set = Parameters()
    parameter_set.add("test1", vary=True)
    parameter_set.add("test2", vary=True)
    modified_parameter_set, float_str = set_parameters_to_fix_during_fit(
        parameter_set, ["test1", "test2"]
    )
    assert (
        modified_parameter_set["test2"].vary
        == modified_parameter_set["test1"].vary
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
