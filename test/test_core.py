import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import nmrglue as ng
from lmfit import Model

from peakipy.core import (
    make_mask,
    fix_params,
    pvoigt2d,
    get_params,
    make_param_dict,
    to_prefix,
    make_models,
    Pseudo3D,
)


class TestCoreFunctions(unittest.TestCase):
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

        mod = Model(pvoigt2d)
        pars = mod.make_params(center_x=20.0, center_y=30.0)
        pars["center_x"].stderr = 1.0
        pars["center_y"].stderr = 2.0
        ps, ps_err, names = get_params(pars, "center")
        #  get index of values
        cen_x = names.index("center_x")
        cen_y = names.index("center_y")

        self.assertEqual(ps[cen_x], 20.0)
        self.assertEqual(ps[cen_y], 30.0)
        self.assertEqual(ps_err[cen_x], 1.0)
        self.assertEqual(ps_err[cen_y], 2.0)

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

        for ls, frac in zip(["PV", "G", "L"], [0.5, 0.0, 1.0]):

            params = make_param_dict(peaks, data, ls)
            self.assertEqual(params["_one_fraction"], frac)
            self.assertEqual(params["_two_fraction"], frac)
            self.assertEqual(params["_three_fraction"], frac)

        self.assertEqual(params["_one_center_x"], 5.0)
        self.assertEqual(params["_two_center_x"], 10.0)
        self.assertEqual(params["_two_sigma_x"], 1.25)
        self.assertEqual(params["_two_sigma_y"], 1.25)

    def test_to_prefix(self):

        names = [
            (" one", "_one_"),
            (" one/two", "_oneortwo_"),
            (" one?two", "_onemaybetwo_"),
            (" [{one?two\}][", "___onemaybetwo____"),
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

        lineshapes = ["PV", "L", "G"]

        for lineshape in lineshapes:

            mod, p_guess = make_models(pvoigt2d, peaks, data, lineshape)

            if lineshape == "PV":
                self.assertEqual(p_guess["_one_fraction"].vary, True)
                self.assertEqual(p_guess["_one_fraction"].value, 0.5)

            if lineshape == "G":
                self.assertEqual(p_guess["_one_fraction"].vary, False)
                self.assertEqual(p_guess["_one_fraction"].value, 0.0)

            if lineshape == "L":
                self.assertEqual(p_guess["_one_fraction"].vary, False)
                self.assertEqual(p_guess["_one_fraction"].value, 1.0)

    def test_Pseudo3D(self):

        datasets = [("test/test_protein_L/test1.ft2",[0, 1, 2]),

                    ("test/test_protein_L/test_tp.ft2", [2, 1, 0]),
                ("test/test_protein_L/test_tp2.ft2",[1, 2, 0])]

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
