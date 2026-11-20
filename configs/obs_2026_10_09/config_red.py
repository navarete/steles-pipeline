# configs/obs_2026_10_09/config_blue.py

# ============================================================
# General configuration
# ============================================================

suffix_r = 'R'

config_red = {
    # =========================================================
    # PART I
    # =========================================================
    # Input / Output
    "raw_directory": "/home/fnavarete/astro/data/SOAR/Steles/2026-10-09/test_bench/",
    # Input calibration files
    "flat_list": [f"q20w_{suffix_r}_{i:04d}" for i in range(17, 26)],   # list of files with suffix 0017-0025
    "arc_list":  [f"thar_{suffix_r}_{i:04d}" for i in range(103, 105)], # list of files with suffix 0103-0104
    
    # Master calibration frames
    "fileout_flat": f"master_flat_{suffix_r}",
    "fileout_arc": f"master_thar_{suffix_r}_0103",

    # Input science files
    "sci_list":  [f"sci_{suffix_r}_0102"],
    # Master science frames
    "fileout_sci": f"master_HD216956_tracing_{suffix_r}_0102",
    
    # Tracing parameters
    "refit_trace_ndeg":                      2, # 2: poor weather, 3-4: good weather
    "refit_trace_skip_orders":          [None],
    # QC visualization
    "aperture_orders": [0, 9, 19, 29, 38],
    "aperture_vmax": {"flat": 4.0, "arc": 3.5, "sci": 4.0},
    # Extraction
    "aperture_extract": [-10,+5],
    
    # =========================================================
    # PART II
    # =========================================================
    # Wavelength solution
    "wavelength_pixel_offset":    0,   # offset the reference wavelength peak lines by a number of pixels (helps improving the wavelength solution)  
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
    
    # Blaze function
    "blaze_mode":   "simple", # "simple" or "continuum", default="simple"
    "blaze_min_deg":      15,
    "blaze_max_deg":      55,
    
    # Optional behavior
    "overwrite": True,   # overwrite existing products
}


