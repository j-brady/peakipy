import os
from pathlib import Path

import pytest

import peakipy.cli.main
from peakipy.cli.main import PeaklistFormat, Lineshape

os.chdir("test")


@pytest.fixture
def protein_L():
    path = Path("test_protein_L")
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


def test_fit_main_with_default(protein_L):
    args = dict(
        peaklist_path=protein_L / Path("test.csv"),
        data_path=protein_L / Path("test1.ft2"),
        output_path=protein_L / Path("fits_PV.csv"),
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


def test_check_main_with_default(protein_L):
    args = dict(
        fits=protein_L / Path("fits_PV.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=False,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_gaussian(protein_L):
    args = dict(
        fits=protein_L / Path("fits_G.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=False,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_lorentzian(protein_L):
    args = dict(
        fits=protein_L / Path("fits_L.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=False,
        individual=True,
    )
    peakipy.cli.main.check(**args)


def test_check_main_with_voigt(protein_L):
    args = dict(
        fits=protein_L / Path("fits_V.csv"),
        data_path=protein_L / Path("test1.ft2"),
        clusters=[1],
        first=True,
        label=True,
        show=False,
        individual=True,
    )
    peakipy.cli.main.check(**args)

def test_edit_with_default(protein_L):
    args = dict(
        peaklist_path=protein_L/Path("peaks.csv"),
        data_path=protein_L/Path("test1.ft2"),
        test=True,
    )
    peakipy.cli.main.edit(**args)


# if __name__ == "__main__":

#     unittest.TestLoader.sortTestMethodsUsing = None
#     unittest.main(verbosity=2)
#     to_clean = ["test.csv", "peakipy.config", "run_log.txt", "fits.csv"]
#     for i in to_clean:
#         print(f"Deleting: {i}")
#         shutil.rmtree(i)
