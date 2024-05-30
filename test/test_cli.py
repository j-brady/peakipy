import os
from pathlib import Path

import pytest

import peakipy.cli.main
from peakipy.cli.main import PeaklistFormat, Lineshape
from peakipy.io import StrucEl


@pytest.fixture
def protein_L():
    path = Path("test/test_protein_L")
    return path


def test_read_main_with_default_pipe(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.tab"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.pipe,
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_default_analysis(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.a2"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.a2,
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_default_sparky(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.sparky"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.sparky,
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_strucel_square(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.sparky"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.sparky,
        struc_el=StrucEl.square,
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_strucel_rectangle(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.sparky"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.sparky,
        struc_el=StrucEl.rectangle,
        struc_size=(3, 3),
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_mask_method(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.sparky"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.sparky,
        struc_el=StrucEl.mask_method,
        struc_size=(1, 1),
    )
    peakipy.cli.main.read(**args)


def test_read_main_with_mask_method_fuda(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("peaks.sparky"),
        data_path=protein_L / Path("test1.ft2"),
        peaklist_format=PeaklistFormat.sparky,
        struc_el=StrucEl.mask_method,
        struc_size=(1, 1),
        fuda=True,
    )
    peakipy.cli.main.read(**args)


def test_fit_main_with_default(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV.csv"),
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_centers_floated(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV_centers_floated.csv"),
        fix=["fraction", "sigma"],
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_centers_bounded(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV_centers_bounded.csv"),
        fix=["fraction", "sigma"],
        xy_bounds=[0.01, 0.1],
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_sigmas_floated(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV_sigmas_floated.csv"),
        fix=["fraction", "center"],
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_vclist(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV.csv"),
        vclist=protein_L / Path("vclist"),
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_gaussian(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_G.csv"),
        lineshape=Lineshape.G,
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_lorentzian(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_L.csv"),
        lineshape=Lineshape.L,
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_voigt(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_V.csv"),
        lineshape=Lineshape.V,
    )
    peakipy.cli.main.fit(**args)


def test_fit_main_with_pv_pv(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV_PV.csv"),
        lineshape=Lineshape.PV_PV,
    )
    peakipy.cli.main.fit(**args)


def test_check_main_with_default(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_PV.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=True,
        test=True,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_gaussian(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_G.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=True,
        test=True,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_lorentzian(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_L.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=True,
        test=True,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_voigt(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_V.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=True,
        test=True,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_pv_pv(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_PV_PV.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=True,
        test=True,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_panel_PVPV(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_PV_PV.csv"),
        data_path=protein_L / Path("test1.ft2"),
        test=True,
        panel=True,
    )
    peakipy.cli.main.check(**args)


def test_check_panel_PV(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_PV.csv"),
        data_path=protein_L / Path("test1.ft2"),
        test=True,
        panel=True,
    )
    peakipy.cli.main.check(**args)


def test_check_panel_V(protein_L):
    args = dict(
        fits_path=protein_L / Path("fits_V.csv"),
        data_path=protein_L / Path("test1.ft2"),
        test=True,
        panel=True,
    )
    peakipy.cli.main.check(**args)


def test_edit_panel(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        test=True,
    )
    peakipy.cli.main.edit(**args)
