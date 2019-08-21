from setuptools import setup, find_packages

long_description = open("README.md").read()

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
        "numba>=0.44.1",
        "tabulate>=0.8.3",
        "colorama>=0.4.1",
        "numdifftools>=0.9.39",
        ]

setup(
    name="peakipy",
    version="0.1.24",
    description="Some functions and scripts for deconvoluting NMR peaks interactively",
    long_description = long_description,
    long_description_content_type="text/markdown",
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
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
    license="GNU GPLv3",
    entry_points={'console_scripts': ['peakipy = peakipy.__main__:main',], },
    install_requires=requirements,
    include_package_data=True,
)
