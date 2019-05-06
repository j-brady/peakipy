import os
from glob import glob
from setuptools import setup

if os.path.exists("build"):
    os.system("rm -r build")

scripts = [
    "fit_peaks",
    "check_fits",
    "read_peaklist",
    "spec",
    "edit_fits",
    "edit_fits_script.py",
]
scripts = [f"scripts/{script}" for script in scripts]
[os.system(f"chmod +x {script}") for script in scripts]

requirements = [
        "pandas>=0.24.0",
        "numpy>=1.16",
        "matplotlib>=3.0",
        "PyYAML>=3.13",
        "nmrglue>=0.6.0",
        "scipy>=1.2",
        "docopt>=0.6.2",
        "lmfit>=0.9.12",
        "scikit-image>=0.14.2",
        "bokeh>=1.0.4",
        "schema>=0.7.0",
        ]

setup(
    name="peakipy",
    version="0.1.7",
    description="Some functions and scripts for deconvoluting NMR peaks interactively",
    author="Jacob Peter Brady",
    author_email="jacob.brady0449@gmail.com",
    url="https://j-brady.github.io/peakipy",
    packages=["peakipy"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
    license="GNU GPLv3",
    scripts=scripts,
    install_requires=requirements,
    include_package_data=True,
)
