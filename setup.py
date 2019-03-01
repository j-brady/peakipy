import os
from glob import glob
from distutils.core import setup

if os.path.exists("build"):
    os.system("rm -r build")

# scripts = glob("scripts/*.py")
scripts = [
    "fit_peaks.py",
    "run_check_fits.py",
    "check_fits.py",
    "read_peaklist.py",
    "spec.py",
]
scripts = [f"scripts/{script}" for script in scripts]
[os.system(f"chmod +x {script}") for script in scripts]

setup(
    name="peakipy",
    version="0.1",
    description="Some functions and scripts for deconvoluting nmrpeaks interactively",
    author="Jacob Peter Brady",
    author_email="jacob.brady0449@gmail.com",
    packages=["peakipy"],
    scripts=scripts,
)
