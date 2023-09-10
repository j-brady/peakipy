import unittest
import shutil
import os
from pathlib import Path

import peakipy.cli.main
from peakipy.cli.main import PeaklistFormat, Lineshape

os.chdir("test")

class TestPeakipyCLI(unittest.TestCase):
    def test_read_main_with_default_pipe(self):
        args = dict(
            peaklist_path=Path("test_protein_L/test.tab"),
            data_path=Path("test_protein_L/test1.ft2"),
            peaklist_format=PeaklistFormat.pipe,
        )
        peakipy.cli.main.read(**args)

    def test_read_main_with_default_analysis(self):
        args = dict(
            peaklist_path=Path("test_protein_L/peaks.a2"),
            data_path=Path("test_protein_L/test1.ft2"),
            peaklist_format=PeaklistFormat.a2,
        )
        peakipy.cli.main.read(**args)

    def test_read_main_with_default_sparky(self):
        args = dict(
            peaklist_path=Path("test_protein_L/peaks.sparky"),
            data_path=Path("test_protein_L/test1.ft2"),
            peaklist_format=PeaklistFormat.sparky,
        )
        peakipy.cli.main.read(**args)

    def test_fit_main_with_default(self):
        args = dict(
            peaklist_path=Path("test_protein_L/test.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            output_path=Path("test_protein_L/fits_PV.csv"),
        )
        peakipy.cli.main.fit(**args)

    def test_fit_main_with_gaussian(self):
        args = dict(
            peaklist_path=Path("test_protein_L/test.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            output_path=Path("test_protein_L/fits_G.csv"),
            lineshape=Lineshape.G,
        )
        peakipy.cli.main.fit(**args)

    def test_fit_main_with_lorentzian(self):
        args = dict(
            peaklist_path=Path("test_protein_L/test.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            output_path=Path("test_protein_L/fits_L.csv"),
            lineshape=Lineshape.L,
        )
        peakipy.cli.main.fit(**args)

    def test_fit_main_with_voigt(self):
        args = dict(
            peaklist_path=Path("test_protein_L/test.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            output_path=Path("test_protein_L/fits_V.csv"),
            lineshape=Lineshape.V,
        )
        peakipy.cli.main.fit(**args)

    def test_check_main_with_default(self):
        args = dict(
            fits=Path("test_protein_L/fits_PV.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            clusters=[1],
            first=True,
            label=True,
            show=False,
            individual=True,
        )
        peakipy.cli.main.check(**args)

    def test_check_main_with_gaussian(self):
        args = dict(
            fits=Path("test_protein_L/fits_G.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            clusters=[1],
            first=True,
            label=True,
            show=False,
            individual=True,
        )
        peakipy.cli.main.check(**args)

    def test_check_main_with_lorentzian(self):
        args = dict(
            fits=Path("test_protein_L/fits_L.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            clusters=[1],
            first=True,
            label=True,
            show=False,
            individual=True,
        )
        peakipy.cli.main.check(**args)

    def test_check_main_with_voigt(self):
        args = dict(
            fits=Path("test_protein_L/fits_V.csv"),
            data_path=Path("test_protein_L/test1.ft2"),
            clusters=[1],
            first=True,
            label=True,
            show=False,
            individual=True,
        )
        peakipy.cli.main.check(**args)

    # def test_edit_with_default(self):
    #     args = dict(
    #         peaklist_path=Path("test_protein_L/peaks.csv"),
    #         data_path=Path("test_protein_L/test1.ft2"),
    #     )
    #     peakipy.cli.main.edit(**args)


# if __name__ == "__main__":

#     unittest.TestLoader.sortTestMethodsUsing = None
#     unittest.main(verbosity=2)
#     to_clean = ["test.csv", "peakipy.config", "run_log.txt", "fits.csv"]
#     for i in to_clean:
#         print(f"Deleting: {i}")
#         shutil.rmtree(i)
