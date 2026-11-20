# steles_imports.py

# ============================================================
# Standard library
# ============================================================

import glob
import json
import logging
import os
import time
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

# ============================================================
# Third-party packages
# ============================================================

import lacosmic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import CCDData
from astropy.stats import sigma_clip, mad_std
from astropy.table import Table

from ccdproc import combine

from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import center_of_mass, map_coordinates, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter

# ============================================================
# NumPy compatibility
# ============================================================

try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
    
# workaround for more recent numpy versions
#if not hasattr(np, 'RankWarning'):
#    np.RankWarning = np.exceptions.RankWarning
if not hasattr(np, 'warnings'):
    np.warnings = warnings
if not hasattr(np, 'in1d'):
    np.in1d = np.isin


# ============================================================
# Constants / configuration
# ============================================================

UTC = timezone.utc

FITS_EXTENSION  = ".fits"
TABLE_EXTENSION = ".txt"
PLOT_EXTENSION  = ".png"
CSV_EXTENSION   = ".csv"
JSON_EXTENSION  = ".json"
HTML_EXTENSION  = ".html"

# ============================================================
# Warnings
# ============================================================

# Ignore polynomial fitting rank warnings
warnings.simplefilter("ignore", np.RankWarning)

# Ignore poorly-conditioned fit warnings
warnings.filterwarnings("ignore", message="The fit may be poorly conditioned.*")
