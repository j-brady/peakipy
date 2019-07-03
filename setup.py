import os
from glob import glob
from setuptools import setup, find_packages

scripts = [
#    "fit_peaks",
#    "check_fits",
#    "read_peaklist",
    "spec",
#    "edit_fits",
    "edit_fits_script.py",
]

scripts_path = "scripts"
scripts = [os.path.join(scripts_path, script) for script in scripts]
#[os.system(f"chmod +x {script}") for script in scripts]

requirements = [
        "pandas>=0.24.0",
        "numpy>=1.16",
        "matplotlib>=3.0",
        "PyYAML>=5.1",
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
    version="0.1.17",
    description="Some functions and scripts for deconvoluting NMR peaks interactively",
    author="Jacob Peter Brady",
    author_email="jacob.brady0449@gmail.com",
    url="https://j-brady.github.io/peakipy",
    packages=find_packages(),
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
    entry_points={'console_scripts': ['peakipy = peakipy.__main__:main',], },
    install_requires=requirements,
    include_package_data=True,
)
