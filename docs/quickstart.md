# Quickstart Instructions
This guide provides a quick overview of how to get started with `peakipy`. Follow these instructions to set up your environment, install the necessary components, and execute basic commands.

## Inputs
To work with `peakipy`, you will need the following input files:
1. **Peak List:** A formatted file that describes the peaks of interest. Refer to the [instructions](./instructions.md) for formatting guidance.
2. **NMRPipe Frequency Domain Dataset:** This can be a 2D or pseudo 3D dataset for processing.

## Main Commands
`peakipy` provides four main commands to facilitate peak analysis:

1. **`peakipy read`**  
   Converts your peak list into a `.csv` file while selecting clusters of peaks.  
   **Usage:**  
   ```bash
   peakipy read [PEAK_LIST_FILE] [NMR_DATA_FILE] [OUTPUT_FORMAT] [OPTIONS]
   ```
   
2. **`peakipy edit`**  
   Interactively checks and adjusts fit parameters (i.e., clusters and mask radii) if the initial clustering is not satisfactory.  
   **Usage:**  
   ```bash
   peakipy edit [PEAK_LIST_FILE] [NMR_DATA_FILE] [OPTIONS]
   ```

3. **`peakipy fit`**  
   Fits clusters of peaks using the `.csv` peak list generated (or edited) by the `read` (or `edit`) command(s).  
   **Usage:**  
   ```bash
   peakipy fit [PEAK_LIST_FILE] [NMR_DATA_FILE] [OUTPUT_FILE] [OPTIONS]
   ```
   
4. **`peakipy check`**  
   Checks individual fits or groups of fits and generates plots.  
   **Usage:**  
   ```bash
   peakipy check [OUTPUT_FILE] [NMR_DATA_FILE] [OPTIONS]
   ```

For more details on how to run these scripts, check the [instructions](./instructions.md). You can also use the `--help` flag for further guidance on running each command (e.g., `peakipy read --help`).

## How to Install `peakipy`

### Using `uv` (recommended)

Either of the following approaches should work:

1. Clone the `peakipy` repository:
   ```bash
   git clone https://github.com/j-brady/peakipy.git
   cd peakipy
   uv sync
   ```

   The `uv sync` command will automatically create a virtual environment for you in `.venv` which you can then activate with the usual `source .venv/bin/activate`.
   
2. Install from PyPI:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install peakipy
   ```

### Using pip
1. Install `peakipy` using pip:
   ```bash
   pip install peakipy
   ```

#### Example Bash Script
Below is an example of an installation script and a basic use case:

```bash
#!/bin/bash
# Create a virtual environment and activate it
uv venv --python 3.12
source .venv/bin/activate

# Install peakipy
uv pip install peakipy

# Process some data using peakipy
peakipy read peaks.a2 test.ft2 a2 --y-radius-ppm 0.213 --show
peakipy edit peaks.csv test.ft2  # Adjust fitting parameters
peakipy fit peaks.csv test.ft2 fits.csv --vclist vclist --max-cluster-size 15
peakipy check fits.csv test.ft2 --clusters 1 --clusters 2 --clusters 3 --colors purple green --show --outname tmp.pdf
```

Run this script by sourcing the file:
```bash
source file_containing_commands
```

**Note:** It is recommended to use a virtual environment:
```bash
uv venv --python 3.12
source .venv/bin/activate
# or for csh:
source .venv/bin/activate.csh
```

## Requirements
### Latest
The latest version (2.1.1) of `peakipy` requires Python 3.11 or above (see `pyproject.toml` for details).

### Legacy Version (0.2.0)
`peakipy` version 0.2.0, which runs on Python 3.8, can be installed as follows:
```bash
git clone --branch v0.2 https://github.com/j-brady/peakipy.git
cd peakipy
poetry install
```
Or:
```bash
pip install peakipy==0.2.0
```
