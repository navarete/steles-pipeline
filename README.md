# STELES Data Reduction Pipeline

**Version 2.3.0**

Python-based data reduction pipeline for the **STELES spectrograph** installed at the SOAR Telescope.

The pipeline processes raw STELES observations and produces wavelength-calibrated, spatially corrected, blaze-corrected one-dimensional spectra.

## 1. Overview

The STELES Data Reduction Pipeline provides a modular and reproducible workflow for reducing observations obtained with the STELES optical echelle spectrograph.

The pipeline is written in Python and organized into two sequential processing stages that can be executed separately:

- **Part I – CCD Reduction and Order Extraction:** Performs detector preprocessing, overscan correction, amplifier merging, cosmic-ray removal, master-frame creation, spectral-order identification and tracing, extraction, and geometric rectification.
- **Part II – Spectroscopic Calibration and Extraction:** Determines the wavelength solution, characterizes and corrects spatial distortions, calculates the blaze function, and extracts the final one-dimensional spectra.

The two stages can be executed independently. Part I generates intermediate products that serve as input to Part II, allowing users to repeat wavelength calibration and spectral extraction without repeating the initial detector processing.

The pipeline also generates diagnostic plots and quality-control (QC) reports to help users evaluate the reduction.


## 2. Repository Structure

The repository is organized into the following directories and files:

```text
steles-pipeline/
│
├── configs/
│   ├── obs_2026_10_09/
│   ├── obs_2026_11_11/
│   └── obs_2026_11_19/
│
├── db/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   ├── ThAr_atlas_3000_1060_75k.fits
│   ├── ThAr_atlas_3000_1060_75k.txt
│   └── ThAr_atlas_UVES.dat
│
├── examples/
│   ├── example_data_blue/
│   └── example_data_red/
│
├── notebooks/
│   ├── STELES_pipeline_v2.3.ipynb
│   ├── STELES_pipeline_v2.3_UNC.ipynb
│   └── STELES_pipeline_wavesol_debug.ipynb
│
├── steles_pipeline/
│   ├── __init__.py
│   ├── steles_base.py
│   ├── steles_ccd.py
│   ├── steles_imports.py
│   ├── steles_part_I.py
│   ├── steles_part_II.py
│   ├── steles_qc.py
│   └── steles_spec.py
│
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-tested.txt
└── requirements.txt
```

### Main directories

**`configs/`**

Contains Python configuration files used to define the input observations and reduction parameters for individual observing runs. Configuration files are organized by observation date and detector channel (blue and red). Some configurations also include full processing settings, while others use the pipeline default parameters.

**`db/`**

Contains instrument-specific calibration resources and reference data required by the pipeline.

The `v1/`, `v2/`, `v3/`, and `v4/` subdirectories contain different versions of the STELES calibration databases. Each version includes spectral-order tracing reference files and wavelength calibration resources for the blue and red channels. The corresponding date files identify the validity period of each database version.

The directory also contains ThAr reference atlases used during wavelength calibration.

**`examples/`**

Contains example datasets and configuration files intended to demonstrate the complete reduction workflow. The current example datasets are being updated to match pipeline version 2.3 and should not yet be used as reference datasets.

**`notebooks/`**

Contains the Jupyter notebooks used to configure and execute the pipeline:

- `STELES_pipeline_v2.3.ipynb`: Main notebook for the standard data-reduction workflow.
- `STELES_pipeline_v2.3_UNC.ipynb`: Alternative notebook for the uncertainty-aware reduction workflow.
- `STELES_pipeline_wavesol_debug.ipynb`: Notebook for wavelength-solution inspection and debugging.

**`steles_pipeline/`**

Contains the Python package implementing the reduction pipeline:

| Module | Description |
|:---|:---|
| `steles_base.py` | Base class with shared functionality, configuration handling, and instrument-specific parameters. |
| `steles_part_I.py` | CCD preprocessing, calibration frames, spectral-order tracing, extraction, and rectification. |
| `steles_part_II.py` | Wavelength calibration, spatial-distortion correction, blaze modeling, and final spectral extraction. |
| `steles_ccd.py` | CCD processing and detector-level routines. |
| `steles_spec.py` | Spectroscopic processing and calibration algorithms. |
| `steles_qc.py` | Quality-control plots and diagnostic routines. |
| `steles_imports.py` | Shared imports, constants, and compatibility definitions. |
| `__init__.py` | Package initialization and public class definitions. |

### Main repository files

- `pyproject.toml`: Package metadata, installation configuration, and dependencies.
- `requirements.txt`: List of required Python packages.
- `requirements-tested.txt`: Reference package versions used during pipeline testing.
- `CITATION.cff`: Citation metadata for the software.
- `LICENSE`: Software license.


## 3. Requirements

The pipeline requires:

- Python 3.7 or newer.
- Jupyter Notebook or JupyterLab.
- The Python packages listed in `requirements.txt`.

Main dependencies include:

- Astropy
- CCDProc
- LACosmic
- Matplotlib
- NumPy
- OpenPyXL
- Pandas
- PyYAML
- SciPy

The pipeline has been tested on Windows, macOS, and Linux.


## 4. Installation

### 4.1. Download the repository

Clone the GitHub repository:

```bash
git clone https://github.com/navarete/steles-pipeline.git
cd steles-pipeline
```

To use the development version 2.3, check out the corresponding branch:

```bash
git checkout development/v2.3
```

Alternatively, download the repository as a ZIP archive from GitHub and extract its contents.

### 4.2. Create a Python environment

It is recommended to use a dedicated Python environment to avoid conflicts with other installed packages.

Create a virtual environment:

```bash
python -m venv steles_env
```

Activate the environment.

**Windows:**

```bash
steles_env\Scripts\activate
```

**Linux/macOS:**

```bash
source steles_env/bin/activate
```


### 4.3. Install the dependencies

Two dependency installation options are available.

**Option A – General installation**

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

**Option B – Reference Windows environment**

To reproduce the package versions used during development and testing under Windows (Python 3.7.4), install:

```bash
python -m pip install -r requirements-tested.txt
```

The tested dependency versions correspond to the reference environment and may not be suitable for other Python versions or operating systems.

### 4.4. Install the STELES package

From the repository root directory, install the pipeline in editable mode:

```bash
python -m pip install -e . --no-deps
```

The `--no-deps` option prevents pip from attempting to resolve or replace the dependencies already installed in the previous step.

Editable installation allows changes to the pipeline source code to take effect without reinstalling the package.

### 4.5. Install and launch Jupyter

If Jupyter Notebook or JupyterLab is not already installed, install JupyterLab:

```bash
python -m pip install jupyterlab
```

Launch JupyterLab from the repository root directory:

```bash
jupyter lab
```

Open the appropriate notebook from the `notebooks/` directory to configure and execute the reduction.



## 5. Running the Pipeline

The data reduction is performed using Jupyter notebooks located in the `notebooks/` directory.

### 5.1. Standard Data Reduction

Open:

`notebooks/STELES_pipeline_v2.3.ipynb`

This notebook contains the complete data-reduction workflow, including both processing stages.

The Part I and Part II pipeline classes are imported using:

```python
from steles_pipeline import STELES_Pipeline
from steles_pipeline import STELES_Pipeline_PartII
```

Configure the input observations, calibration exposures, processing parameters, and output filenames before executing the reduction.

The notebook guides the user through the following stages:

1. **Part I – CCD Reduction and Order Extraction:** Performs detector preprocessing, calibration-frame preparation, spectral-order tracing, extraction, and geometric rectification.
2. **Part II – Spectroscopic Calibration and Extraction:** Determines the wavelength solution, evaluates and applies spatial-distortion corrections, calculates the blaze function, and extracts the final one-dimensional spectra.

### 5.2. Uncertainty-aware Reduction

The alternative notebook:

`notebooks/STELES_pipeline_v2.3_UNC.ipynb`

provides the uncertainty-aware reduction workflow.

Refer to the notebook documentation for details about its implementation and differences from the standard reduction.

### 5.3. Wavelength Solution Inspection

The notebook:

`notebooks/STELES_pipeline_wavesol_debug.ipynb`

provides tools for inspecting and debugging wavelength calibration solutions.

### 5.4. Processing Workflow

A typical reduction follows these steps:

1. Organize the raw STELES FITS files.
2. Configure the observations and processing parameters.
3. Execute Part I.
4. Inspect the master frames, spectral-order traces, extraction apertures, and rectified products.
5. Execute Part II using the intermediate products generated by Part I.
6. Inspect the wavelength solution, spatial-distortion corrections, blaze function, and final spectra.
7. Review the QC reports and verify the quality of the final products.

The blue and red detector channels must be processed independently.


## 6. Input Data

The pipeline requires the following observations:

- **Quartz flat-field frames:** Used for spectral-order tracing, detector-response characterization, and blaze-function determination.
- **ThAr arc frames:** Used to determine the wavelength solution and spatial-distortion model.
- **Science frames:** Target observations to be processed.

Raw observations must be provided in the standard STELES multi-extension FITS format, with one extension per amplifier.

The blue and red channels must be reduced independently. The pipeline identifies the detector channel and instrument configuration from the FITS headers.

Dedicated bias frames are not required by the standard reduction procedure because overscan correction is applied. Dark-frame subtraction and sky subtraction are not currently implemented.

## 7. Output Products

The pipeline generates several intermediate and final data products, including:

- Overscan-corrected and merged FITS frames.
- Cosmic-ray-cleaned frames.
- Master calibration and science frames.
- Spectral-order tracing solutions.
- Extracted and geometrically rectified two-dimensional spectral products.
- Wavelength calibration solutions.
- Spatial-distortion models.
- Blaze-function products.
- Final wavelength-calibrated one-dimensional spectra.
- Diagnostic plots and machine-readable QC reports.

Refer to the STELES User Manual for a detailed description of the output products and their corresponding FITS formats.

## 8. Calibration Resources

The pipeline includes instrument-specific calibration resources and configuration files required for data processing.

The appropriate calibration database is selected according to the observation date and detector channel, as identified from the STELES FITS headers.

Users should avoid modifying the distributed calibration resources unless they are intentionally updating the instrument calibration database.

## 9. Quality Control

The pipeline generates diagnostic plots and reports at different stages of the reduction.

Users should inspect the intermediate and final products, paying particular attention to:

- Overscan correction and amplifier merging.
- Spectral-order identification and tracing.
- Extraction apertures and rectification.
- Wavelength calibration residuals.
- Spatial-distortion correction.
- Blaze-function behavior.
- Final spectral quality and signal-to-noise ratio.

Successful execution does not necessarily guarantee that the resulting spectra are suitable for every scientific application.

## 10. Current Limitations

The current implementation does not provide:

- Absolute flux calibration.
- Telluric correction.
- Sky subtraction.
- Continuum normalization.
- Automatic merging of adjacent spectral orders.
- Combination of extracted one-dimensional spectra.

These operations must be performed during subsequent post-processing when required.


## 11. Documentation and References

The following resources provide complementary information about the STELES instrument, its capabilities, and observing procedures.

- **STELES Instrument Paper:**

  The instrument design, control systems, measured performance, and initial scientific results are described in Navarete et al. (2026), available through the [SPIE Digital Library](https://doi.org/10.1117/12.3104767).

- **STELES Science Verification Call for Proposals (2026V):**

  The original call for proposals for the STELES Science Verification campaign provides information about the initial observing opportunities, available observing modes, technical constraints, and proposal submission procedures. It is available on the [LNA website](https://www.gov.br/lna/pt-br/composicao-1/coast/obs/soar/chamadas/chamada-extra-para-envio-de-propostas-de-observacao-soar-2026v).

A dedicated STELES User Manual is under preparation and will provide detailed instructions for pipeline installation, configuration, data reduction, and quality control.



## 12. Contributing and Development

The pipeline is developed and maintained by the STELES instrument team.

The development/v2.3 branch contains the ongoing development version of the pipeline. Stable releases will be made available through the main branch.

Contributions, bug reports, and suggestions for improvements are welcome. Please contact the maintainer or submit an issue through the GitHub repository.

When modifying the pipeline, users are encouraged to preserve the modular organization of the two processing stages and verify that changes do not compromise the reproducibility of existing reductions.


## 13. Citation and Acknowledgements

If the STELES pipeline is used in scientific publications, please acknowledge the use of the STELES spectrograph and its data reduction software.

The software citation metadata is provided in [`CITATION.cff`](CITATION.cff).

### Python Dependencies

The reference package versions used during development and testing are documented in [`requirements-tested.txt`](requirements-tested.txt).

The dependency versions may differ between operating systems and Python environments.

### Tested Platforms

The pipeline has been tested under the following environments:

| Operating System | Python Version |
|:-----------------|:---------------|
| Windows 10 | 3.7.4 |
| macOS | Tested (Intel) |
| Linux Mint 22.2 Cinnamon | Tested |

The pipeline has been successfully executed on all three platforms.

### Pipeline Validation

The following table summarizes the validation status of the pipeline versions:

| Pipeline Version | Part I | Part II | Tested Date |
|:-----------------|:-------|:--------|:------------|
| 2.2 | Passed | Passed | 2026-09-30 |
| 2.3 | Pending | Pending | — |

## License

This project is distributed under the BSD 3-Clause License.
See the [LICENSE](LICENSE) file for details.


---

**Repository:** https://github.com/navarete/steles-pipeline