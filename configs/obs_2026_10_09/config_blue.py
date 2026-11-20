# configs/obs_2026_10_09/config_blue.py

# ============================================================
# General configuration
# ============================================================

suffix_b = 'B'

config_blue = {
    # =========================================================
    # PART I
    # =========================================================
    # Input / Output
    "raw_directory": "/home/fnavarete/astro/data/SOAR/Steles/2026-10-09/test_bench/",
    
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
    
    # Tracing parameters
    "refit_trace_ndeg":                      2, # 2: poor weather, 3-4: good weather
    "refit_trace_skip_orders":          [None],
    # QC visualization
    "aperture_vmax": {"flat": 4.0, "arc": 3.5, "sci": 4.0},
    # Extraction
    "aperture_extract": [-15,+11], 
    
    # =========================================================
    # PART II
    # =========================================================    
    # Wavelength solution
    "wavelength_pixel_offset":   -2,   # offset the reference wavelength peak lines by a number of pixels (helps improving the wavelength solution)  
    "wavesolution_ndeg":          3,   # polynomial degree
    "wavesolution_niter":         3,   # number of iterations to refine wavelength solution
    "wavesolution_rms_threshold": 2.0, # sigma-threshold for removing lines from the wavelength solution
    # Spatial distortion
    "ndeg_distortion":           2, # polynomial degree for distortion
    "n_pix_step":                1, # sampling along spatial axis
    # Blaze function
    "blaze_mode":   "simple", # "simple" or "continuum", default="simple"
    "blaze_min_deg":      15,
    "blaze_max_deg":      55,
}
