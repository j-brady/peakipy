import os
from glob import glob
from distutils.core import setup

scripts = glob("scripts/*.py")
[os.system(f"chmod +x {script}") for script in scripts]

setup(name='peak_deconvolution',
      version='0.1',
      description='Some functions and scripts for deconvoluting nmrpeaks interactively',
      author='Jacob Peter Brady',
      author_email='jacob.brady0449@gmail.com',
      packages=['peak_deconvolution'],
      scripts=scripts,
      )
