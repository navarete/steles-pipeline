# configs/obs_2026_10_09/config_blue.py

# ============================================================
# General configuration
# ============================================================

suffix_b = 'B'

config_blue = {
    # General execution control
    "dry_run": False,
    
    # =========================================================
    # PART I
    # =========================================================

    # Input / Output
    "raw_directory": "/home/fnavarete/astro/data/SOAR/Steles/2026-10-09/test_bench/",
    "save_plots": True,
    "show_plots": True,
    
    # Input calibration files
    "flat_list": [f"q20w_{suffix_b}_{i:04d}" for i in range(16, 26)],   # list of files with suffix 0016-0025
    "arc_list":  [f"thar_{suffix_b}_{i:04d}" for i in range(103, 105)], # list of files with suffix 0103-0104
    
    # Master calibration frames
    "fileout_flat": f"master_flat_{suffix_b}",
    "fileout_arc": f"master_thar_{suffix_b}_0103",

    # Input science files
    "sci_list":  [f"sci_{suffix_b}_0102"],
    # Master science frames
    "fileout_sci": f"master_HD216956_tracing_{suffix_b}_0102", 
    
    # Arc scaling
    "saturation_threshold": 40000,
    "combine_sigma_clip": 5,
    
    # Tracing parameters
    "trace_source":                      "sci", # "sci" or flat", default='sci', 'flat' not implemented due to faint flux on UV.
    "trace_from_observations":            True, # True if refiting trace from observations instead of using the trace from the database
    "refit_trace_ndeg":                      2, # 2: poor weather, 3-4: good weather
    "refit_trace_skip_orders":          [None],
    "refit_trace_use_reference_fallback": True, # enable the fallback in case the flux is insufficient for tracing
    "refit_trace_divergence_threshold":    0.0, # pixels: max allowed shift at y_center before fallback; 0.0 -> always trigger fallback!
    "refit_trace_fallback_half_width":      10, # half-width in x for peak search at y_center
    "fail_on_trace_intersection":        False, # or True for strict mode
    
    # QC visualization
    "aperture_orders": [0, 14, 29, 44, 59],
    "aperture_vmax": {"flat": 4.0, "arc": 3.5, "sci": 4.0},
    
    # Extraction
    "aperture_extract": [-15,+11], 
    
    # =========================================================
    # PART II
    # =========================================================
    
    # Wavelength solution
    "window_fit_peak":           11,   # pixel window for peak refinement
    "max_fwhm_threshold":      10.0,   # exclude peaks broader than this threshold (in pixels) 
    "wavelength_pixel_offset":   -2,   # offset the reference wavelength peak lines by a number of pixels (helps improving the wavelength solution)  
    "wavesolution_ndeg":          3,   # polynomial degree
    "wavesolution_niter":         3,   # number of iterations to refine wavelength solution
    "wavesolution_rms_threshold": 2.0, # sigma-threshold for removing lines from the wavelength solution
    
    # Spatial distortion
    "ndeg_distortion":           2, # polynomial degree for distortion
    "n_pix_step":                1, # sampling along spatial axis
    "saturation_threshold_for_spatial_distortion": 1000000,
    "sigma_threshold":         3.0,
    "fitting_sigma_threshold": 3.0,
    "fitting_niter":             3,
    "distortion_sigma_init":   3.0,
    "distortion_sigma_min":    1.0,
    "distortion_sigma_step":   0.5,
    
    # QC visualization parameters
    "wave_pixel_reference":  [2, 1024, 2045],  # typical for 2048 pix detector
    "spectral_pixel_limits": [1500, 2000],
    
    # Blaze function
    "blaze_mode":   "simple", # "simple" or "continuum", default="simple"
    "blaze_min_deg":      15,
    "blaze_max_deg":      55,
    "blaze_sigma":         5,
    "blaze_ncols":         7,
    "qc_blaze_n_panels":  10,
    "qc_snr_n_panels":     7,
    "qc_final1d_n_panels": 7,
    
    # Optional behavior
    "overwrite": True,   # overwrite existing products
}
