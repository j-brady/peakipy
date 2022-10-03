import unittest
import shutil

import peakipy.commandline.edit
import peakipy.commandline.check
import peakipy.commandline.fit
import peakipy.commandline.read
import peakipy.commandline.spec


class TestPeakipyCommandline(unittest.TestCase):
    def test_read_main_with_default_pipe(self):
        argv = ["test_protein_L/test.tab", "test_protein_L/test1.ft2", "--pipe"]
        peakipy.commandline.read.main(argv)

    def test_read_main_with_default_analysis(self):
        argv = ["test_protein_L/peaks.a2", "test_protein_L/test1.ft2", "--a2"]
        peakipy.commandline.read.main(argv)

    def test_read_main_with_default_sparky(self):
        argv = ["test_protein_L/peaks.sparky", "test_protein_L/test1.ft2", "--sparky"]
        peakipy.commandline.read.main(argv)
        # args = docopt(peakipy.commandline.read.__doc__, argv=argv)
        # print(args)

    # def test_read_main_args(self):
    #     argv = ["test_protein_L/test.tab", "test_protein_L/test1.ft2", "--pipe", ]
    #     peakipy.commandline.read.main(argv)

    def test_fit_main_with_default(self):
        argv = [
            "test_protein_L/test.csv",
            "test_protein_L/test1.ft2",
            "test_protein_L/fits_PV.csv",
        ]
        peakipy.commandline.fit.main(argv)

    def test_fit_main_with_gaussian(self):
        argv = [
            "test_protein_L/test.csv",
            "test_protein_L/test1.ft2",
            "test_protein_L/fits_G.csv",
            "--lineshape=G",
        ]
        peakipy.commandline.fit.main(argv)

    def test_fit_main_with_lorentzian(self):
        argv = [
            "test_protein_L/test.csv",
            "test_protein_L/test1.ft2",
            "test_protein_L/fits_L.csv",
            "--lineshape=L",
        ]
        peakipy.commandline.fit.main(argv)

    def test_fit_main_with_voigt(self):
        argv = [
            "test_protein_L/test.csv",
            "test_protein_L/test1.ft2",
            "test_protein_L/fits_V.csv",
            "--lineshape=V",
        ]
        peakipy.commandline.fit.main(argv)

    def test_check_main_with_default(self):
        argv = [
            "test_protein_L/fits_PV.csv",
            "test_protein_L/test1.ft2",
            "--first",
            # "--clusters=1",
            "-l",
            "-s",
            "-i",
        ]
        peakipy.commandline.check.main(argv)

    def test_check_main_with_gaussian(self):
        argv = [
            "test_protein_L/fits_G.csv",
            "test_protein_L/test1.ft2",
            "--first",
            # "--clusters=1",
            "-l",
            "-s",
            "-i",
        ]
        peakipy.commandline.check.main(argv)

    def test_check_main_with_lorentzian(self):
        argv = [
            "test_protein_L/fits_L.csv",
            "test_protein_L/test1.ft2",
            "--first",
            # "--clusters=1",
            "-l",
            "-s",
            "-i",
        ]
        peakipy.commandline.check.main(argv)

    def test_check_main_with_voigt(self):
        argv = [
            "test_protein_L/fits_V.csv",
            "test_protein_L/test1.ft2",
            "--first",
            # "--clusters=1",
            "-l",
            "-s",
            "-i",
        ]
        peakipy.commandline.check.main(argv)

    def test_edit_with_default(self):
        argv = ["test_protein_L/peaks.csv", "test_protein_L/test1.ft2"]
        peakipy.commandline.edit.main(argv)


if __name__ == "__main__":

    unittest.main(verbosity=2)
    to_clean = ["test.csv", "peakipy.config", "run_log.txt", "fits.csv"]
    for i in to_clean:
        print(f"Deleting: {i}")
        shutil.rmtree(i)
