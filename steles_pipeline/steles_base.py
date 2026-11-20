# steles_base.py
from .steles_imports import *
from . import DB_DIR

class STELES_BasePipeline:

    DEFAULT_CONFIG = {
        # General execution control
        "dry_run": False,
        # =========================================================
        # PART I
        # =========================================================
        # Input / Output
        "save_plots": True,
        "show_plots": True,
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
        "aperture_vmax": {"flat": 4.0, "arc": 3.5, "sci": 4.0},
        # =========================================================
        # PART II
        # =========================================================
        # Wavelength solution
        "window_fit_peak":           11,   # pixel window for peak refinement
        "max_fwhm_threshold":      10.0,   # exclude peaks broader than this threshold (in pixels) 
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
        # QC visualization parameters
        "wave_pixel_reference":  [2, 1024, 2045],  # typical for 2048 pix detector
        "spectral_pixel_limits": [1500, 2000],
        # Blaze function
        "blaze_mode":   "simple", # "simple" or "continuum", default="simple"
        "blaze_min_deg":      15,
        "blaze_max_deg":      55,
        "blaze_sigma":         5,
        "qc_blaze_n_panels":  10,
        # Optional behavior
        "overwrite": True,   # overwrite existing products  
    }
    
    CHANNEL_DEFAULT_CONFIG = {
        "blue": {
            # Extraction
            "aperture_extract": [-13, +13],
            # QC visualization
            "aperture_orders": [0, 14, 29, 44, 59],
            
            # Blaze function
            "blaze_ncols":         7,
            "qc_snr_n_panels":     7,
            "qc_final1d_n_panels": 7,
        },
        "red": {
            # Extraction
            "aperture_extract": [-9, +9],
            # QC visualization
            "aperture_orders": [0, 9, 19, 29, 38],
            # Blaze function
            "blaze_ncols":         4,
            "qc_snr_n_panels":     4,
            "qc_final1d_n_panels": 4,
        },
    }

    REQUIRED_CONFIG = ["raw_directory", "flat_list", "arc_list", "sci_list", "fileout_flat", "fileout_arc", "fileout_sci"]

    def _validate_config(self, channel_defaults=False):
        # Add default values for missing parameters
        for key, value in self.DEFAULT_CONFIG.items():
            self.cfg.setdefault(key, value)
            
        # Add channel-specific defaults
        if channel_defaults:
            if self.channel not in self.CHANNEL_DEFAULT_CONFIG:
                raise ValueError(f"Unknown channel: {self.channel}")

            for key, value in self.CHANNEL_DEFAULT_CONFIG[self.channel].items():
                self.cfg.setdefault(key, value)

        # Check mandatory parameters
        missing = [key for key in self.REQUIRED_CONFIG
                   if key not in self.cfg or self.cfg[key] is None]

        if missing:
            raise ValueError(f"Missing required configuration parameters: {missing}")
            
    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, config):
        """ init for both PartI and PartII """
        
        self.cfg = config.copy()
        self._validate_config()
        
        self._setup_logging()
        self._identify_channel()
        self._get_steles_version()
        self._get_channel_parameters()
        self._validate_config(channel_defaults=True)
        
        self._setup_directories()
        
        self._print_summary()
        
        self.trace_poly_coeffs = None
        
        # Data containers
        self.arc_cube  = None
        self.flat_cube = None
        self.sci_cube  = None

        self.wave1d  = None
        self.blaze1d = None
        
        # extensions of the final 1d products
        self.ext_flux       = 0
        self.ext_wave       = 1
        self.ext_noise      = 2
        self.ext_blaze      = 3
        self.ext_blaze_flux = 4       
        
        # initialize containers
        self.qa_metrics = {}
        #PartI
        self.trace_intersections_qc = {}
        #PartII
        self.wavelength_rms_qc = {}
        self.line_residual_qc = {}
        self.dispersion_qc = {}
        self.rms_dispersion_qc = {}
        self.snr_qc = {}
        self.blaze_qc = {}
        self.blaze_qc_stats = {}
        self.snr_qc_stats = {}
        self.final1d_qc_stats = {}
        
        self.wavelength_info = pd.read_excel(self.wavelength_file)     

    def _setup_logging(self, logger_name="STELES"):

        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        self.log = logging.getLogger(self.logger_name)


    def _identify_channel(self):
        """ get channel based on the scientific data """
    
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: _identify_channel()")
            return
        
        self.log.info("Identifying channel")
        
        # check the first raw science frame
        sci_files = self._return_raw_file_list("sci_list")
        sci_file = sci_files[0]

        with fits.open(sci_file) as hdul:
            steles_header = hdul[0].header

        try:
            self.channel = 'red' if steles_header['FPA'] == "RED" else 'blue'
        except:
            self.channel = 'red' if ( "RED" in steles_header['FIRMWARE'] ) else 'blue'
            
        # also, store the 'obs-date' information
        self.obsdate = steles_header['DATE']

    def _get_steles_version(self):
        """ get version based on OBS-DATE of the scientific data """

        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: _get_steles_version()")
            return
        
        self.log.info("Identifying STELES Version")
        
        if type(self.obsdate) is str:
            obs_date = date.fromisoformat(self.obsdate)
        version = "v0"

        PIPELINE_VERSIONS = [(date(2025, 8,  7), "v1", "First Light"),
                             (date(2026, 1, 28), "v2", "Mission 14"),
                             (date(2026, 6,  1), "v3", "End 2026A"),
                             (date(2026, 8, 28), "v4", "Pre-SV"),
                            ]
        
        for start_date, ver, description in PIPELINE_VERSIONS:
            if obs_date >= start_date:
                version = ver
            else:
                break
                
        self.version = version

    def _load_instrument_config(self, file_yaml=False):

        #filename = f"{DB_DIR}/{self.version}.txt"
        filename = DB_DIR / f"{self.version}.txt"
        
        # if using YAML file:
        if file_yaml:
            #filename = f"{DB_DIR}/{self.version}.yaml"
            filename = DB_DIR / f"{self.version}.yaml"
        
        with open(filename, "r") as f:
            return yaml.safe_load(f)

    def _get_channel_parameters(self):
        """
          get default parameters based on the channel - simplifies the config dictionary
        
            gain_per_amplifier (list): relative gain values for each amplifier to apply before merging them
            gain              (float): average gain of the detector (in e-/ADU)
            rdnoise           (float): read noise of the detector (in e-)
            transpose          (bool): transpose frame axis (True for Blue channel, False for Red channel)
            aperture_orders    (list): orders used for creating QC plot showing the apertures before extraction
            y_range            (list): pixel interval in the y-axis (dispersion) to extract the spectrum
            wavelength_file     (str): full path of the wavelength database for the corresponding channel
            order_centers      (list): central position in the x-axis of each spectral order for each channel
        
        """

        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: get_channel_parameters()")
            return
        
        self.log.info(f"Retrieving parameters for the {self.channel} channel")
        
        instrument_parameters = self._load_instrument_config()
        channel_parameters    = instrument_parameters[self.channel]
        
        self.gain_per_amplifier = channel_parameters["gain_per_amplifier"]
        self.gain               = channel_parameters["gain"]
        self.rdnoise            = channel_parameters["rdnoise"]
        self.transpose          = channel_parameters["transpose"]
        self.y_range            = channel_parameters["y_range"]
        
        #self.wavelength_file = f"{DB_DIR}/{self.version}/{self.channel}_wavelength_solution.xlsx"
        self.wavelength_file = DB_DIR / self.version  / f"{self.channel}_wavelength_solution.xlsx"
        
        self.order_centers = channel_parameters["order_centers"]

        self.n_orders = len(self.order_centers)
        print_results=False
        if print_results:
            print("===instrument_parameters===")
            print("wavelength_file   =",self.wavelength_file)
            print("gain_per_amplifier=",self.gain_per_amplifier)
            print("gain              =",self.gain)
            print("rdnoise           =",self.rdnoise)
            print("transpose         =",self.transpose)
            print("y_range           =",self.y_range)
            print("n_orders          =",self.n_orders)
            print("order_centers     =",self.order_centers)

    def _print_summary(self):
        self.log.info("")
        self.log.info("==== STELES PIPELINE CONFIG ====")
        # values from _get_channel_parameters()
        self.log.info(f"version: {self.version}")
        self.log.info(f"channel: {self.channel}")
        self.log.info(f"n_orders: {self.n_orders}")
        self.log.info(f"gain_per_amplifier: {self.gain_per_amplifier}")
        self.log.info(f"gain: {self.gain}")
        self.log.info(f"rdnoise: {self.rdnoise}")
        self.log.info(f"transpose: {self.transpose}")
        self.log.info(f"y_range: {self.y_range}")
        self.log.info(f"order_centers: {self.order_centers}")
        self.log.info(f"wavelength_file: {self.wavelength_file}")
        self.log.info(f"trace_default_dir: {self.trace_default_dir}")
        self.log.info(f"proc_dir: {self.proc_dir}")
        self.log.info(f"trace_dir: {self.trace_dir}")
        self.log.info(f"plot_dir: {self.plot_dir}")
        #
        self.log.info("")
        self.log.info("==== USER INPUT CONFIG ====")
        # values from the input user config file
        for k, v in self.cfg.items():
            self.log.info(f"{k}: {v}")
        self.log.info("================================")   
        self.log.info("")
        
    # ============================================================
    # FITS I/O
    # ============================================================
    
    def _create_directory(self, directory):
        

        directory = Path(directory)

        if not directory.exists():
            self.log.info(f"Creating new directory at {directory}")
            directory.mkdir(parents=True)
        else:
            self.log.info(f"Directory '{directory}' already exists.")
            
        #if not os.path.isdir(directory):
        #    self.log.info(f"Creating new directory at {directory}")
        #    os.mkdir(directory)
        #else:
        #    self.log.info(f"Directory '{directory}' already exists.")

    def _setup_directories(self):

        self.raw_dir = Path(self.cfg["raw_directory"])

        if not os.path.exists(self.raw_dir):
            raise ValueError("<ERROR> Raw directory does not exist.")

        self.trace_default_dir = DB_DIR / self.version / f"db_tracing_{self.channel}"

        self.proc_dir  = self.raw_dir  / f"PROC_{self.channel}"
        self.plot_dir  = self.proc_dir / "plots"
        self.trace_dir = self.proc_dir / "db_tracing"
        self.wave_dir  = self.proc_dir / "db_wavelength"
        self.dist_dir  = self.proc_dir / "db_distortion"

        self._create_directory(self.proc_dir)
        self._create_directory(self.plot_dir)
        self._create_directory(self.trace_dir)
        self._create_directory(self.wave_dir)
        self._create_directory(self.dist_dir)
        
    def _write_fits(self, data, header, output_path, overwrite=True):
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite) 
        

    def _read_fits(self, fits_file, level="master"):
        """
        Read a STELES FITS file.

        Parameters
        ----------
        fits_file : str
            Path to the FITS file.

        level : str
            Processing level of the file. The red-channel raw data are
            flipped when level='raw'.
            
        Returns
        -------
        image_data : numpy.ndarray
            Primary HDU data.

        image_header : astropy.io.fits.Header
            Primary HDU header.
        """

        levels = [ "raw", "master", "rectified", "rectified_1d", "wavelength_solution", "blaze_solution", "extracted" ]
        if level not in levels:
            self.log.info("Level not specified. Assuming 'master' frames.")
            level = "master"
                 
        if not os.path.isfile(fits_file):
            raise FileNotFoundError(f"FITS file not found: {fits_file}")

        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

        except Exception as e:
            raise ValueError(f"Could not read FITS file: {fits_file}") from e

        if self.channel == "red" and level == "raw":
            data = np.flip(data, axis=1)

        return data, header

    def _save_cube(self, data, header, filename):
        
        hdr = header.copy()
        hdr['ST_TRACE'] = True
        hdr['ST_LINEA'] = True
        
        # --- Write output ---
        #self._write_fits(data, hdr, os.path.join(self.proc_dir, filename), overwrite=self.cfg["overwrite"])
        self._write_fits(data, hdr, self.proc_dir / filename, overwrite=self.cfg["overwrite"])
        
    # ============================================================
    # Product management
    # ============================================================
    
    def _master_files_exist(self):
        #files = [os.path.join(self.proc_dir, self.cfg["fileout_flat"] + ".fits"),
        #         os.path.join(self.proc_dir, self.cfg["fileout_sci"] + ".fits"),
        #         os.path.join(self.proc_dir, self.cfg["fileout_arc"] + ".fits")]
        files = [self.proc_dir / ( self.cfg["fileout_flat"] + FITS_EXTENSION ),
                 self.proc_dir / ( self.cfg["fileout_sci"]  + FITS_EXTENSION ),
                 self.proc_dir / ( self.cfg["fileout_arc"]  + FITS_EXTENSION )]
        return all(os.path.exists(f) for f in files)    
       

    def _load_products(self, products, level="master"):
        """
        Load one or more pipeline products.

        Parameters
        ----------
        products : dict
            Mapping of attribute name to filename.
            Example:
                {"quartz_data": "master_flat.fits",
                 "arc_data": "master_arc.fits"}

        level : str
            Descriptive label used for logging.

        Returns
        -------
        dict
            Loaded data and headers.
        """

        self.log.info(f"Loading {level} products")

        loaded = {}

        for attr, filename in products.items():

            #path = os.path.join(self.proc_dir, filename)
            path = self.proc_dir / filename

            data, header = self._read_fits(path, level=level)

            setattr(self, attr, data)

            if attr.endswith("_data"):
                header_attr = attr.replace("_data", "_header")
            elif attr.endswith("_cube"):
                header_attr = attr.replace("_cube", "_header")
            else:
                header_attr = attr + "_header"

            setattr(self, header_attr, header)

            loaded[attr] = data
            loaded[header_attr] = header

        return loaded
        
    def _products_exist(self, products):
        return all( os.path.exists(os.path.join(self.proc_dir, filename)) for filename in products.values() )   
        
    def load_master_frames(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: load_master_frames()")
            return

        self._load_products({"quartz_data": self.cfg["fileout_flat"] + FITS_EXTENSION,
                             "arc_data":    self.cfg["fileout_arc"]  + FITS_EXTENSION,
                             "sci_data":    self.cfg["fileout_sci"]  + FITS_EXTENSION,},
                            level="master")

        self.log.info("Master frames successfully loaded.")
        
            
    def _rectified_files_exist(self):
        products = {"flat": self.cfg["fileout_flat"] + f"_rectified{FITS_EXTENSION}",
                    "arc":  self.cfg["fileout_arc"]  + f"_rectified{FITS_EXTENSION}",
                    "sci":  self.cfg["fileout_sci"]  + f"_rectified{FITS_EXTENSION}",}
        
        return self._products_exist(products)
    
    def load_rectified_products(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: load_rectified_products()")
            return

        self.log.info("Loading linearized products")

        self._load_products({"flat_cube": self.cfg["fileout_flat"] + f"_rectified{FITS_EXTENSION}",
                             "arc_cube":  self.cfg["fileout_arc"]  + f"_rectified{FITS_EXTENSION}",
                             "sci_cube":  self.cfg["fileout_sci"]  + f"_rectified{FITS_EXTENSION}",},
                             level="rectified")
        
        if not (self.flat_cube.shape == self.arc_cube.shape == self.sci_cube.shape):
            raise RuntimeError("Rectified products have inconsistent dimensions: "
                                f"flat={self.flat_cube.shape}, "
                                f"arc={self.arc_cube.shape}, "
                                f"sci={self.sci_cube.shape}")
        
        self.n_orders, self.n_spat, self.n_spec = (self.arc_cube.shape)        
        
        self.log.info("Rectified products successfully loaded")
    
    def load_arc1d(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: load_spec1d()")
            return
        
        self.log.info("Loading spec1d frame")
        self._load_products({"arc1d": self.cfg["fileout_arc"] + f"_rectified_1d{FITS_EXTENSION}"}, level="rectified_1d")
    
        
    def load_wavelength_qc_table(self):

        #out = os.path.join(self.proc_dir, self.cfg["fileout_arc"]+"_wavelength_qc.csv")
        out = self.proc_dir / ( self.cfg["fileout_arc"] + f"_wavelength_qc{CSV_EXTENSION}" )
                
        df = pd.read_csv(out)
        
        self.wavesol_stats = df.to_dict(orient="records")

        self.log.info(f"Loaded wavelength QC table: {out}") 
        
        
    def load_wavelength_solution(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: load_wavelength_solution()")
            return

        self.log.info("Loading wavelength solution")

        self.load_wavelength_qc_table()

        self._load_products({"wave1d": self.cfg["fileout_arc"] + f"_wavelength_solution{FITS_EXTENSION}"}, level="wavelength_solution")

    
    def load_blaze_solution(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: load_blaze_solution()")
            return

        self.log.info("Loading blaze solution")

        self._load_products({"blaze1d": self.cfg["fileout_flat"] + f"_blaze_function{FITS_EXTENSION}"}, level="blaze_solution")    
    
    def load_extracted_products(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: load_extracted_products()")
            
        self.log.info("Loading extracted products")
        
        self._load_products({"sci_extracted": self.cfg["fileout_sci"] + f"_final_1d{FITS_EXTENSION}",},
                            level="extracted")
        
        self.log.info("Extracted 1d products successfully loaded")
    

    # ============================================================
    # Input management
    # ============================================================

    def _return_raw_file_list(self, list_key):
        files = []

        for name in self.cfg[list_key]:
            #files.append(os.path.join(self.cfg["raw_directory"], name + ".fits"))
            files.append(Path(self.cfg["raw_directory"]) / ( name + FITS_EXTENSION) )

        return files   
    
    def _collect_input_files(self, groups=("flat", "arc", "sci")):
        """
        Collect raw files from one or more groups.

        Parameters
        ----------
        groups : iterable of str
            Any combination of:
                "flat"
                "arc"
                "sci"
                "raw"
        """

        all_files = []

        mapping = {"flat": "flat_list",
                   "arc":  "arc_list",
                   "sci":  "sci_list",}

        for group in groups:
            all_files.extend(self._return_raw_file_list(mapping[group]))

        all_files = sorted(set(all_files))

        if not all_files:
            raise RuntimeError("No input files found.")

        return all_files      
        

    # ============================================================
    # Common pipeline utilities
    # ============================================================

    def _fname(self, base, level="raw"):
        prefix = {
            "raw":     "",
            "merged":  "m_",
            "cleaned": "cm_"
        }[level]

        return prefix + base
        
    def _timeit(self, func, name, *args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        dt = time.time() - t0
        self.log.info(f"{name} completed in {dt:.2f}s")

    def _get_bool(self, key, default=False):
        val = self.cfg.get(key, default)

        if isinstance(val, bool):
            return val

        if isinstance(val, str):
            return val.lower() in ["true", "1", "yes"]

        return bool(val)

    def _check_done(self, outputs):
        """Check if all output files exist"""
        return all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in outputs)

    def _resolve(self, list_key, prefix_key):
        files = []

        # --- list case ---
        if self.cfg.get(list_key):
            for name in self.cfg[list_key]:
                #files.append(os.path.join(self.cfg["raw_directory"], name + ".fits"))
                files.append(self.raw_dir / name + FITS_EXTENSION)

        # --- prefix case ---
        elif self.cfg.get(prefix_key):
            #pattern = os.path.join(self.cfg["raw_directory"], self.cfg[prefix_key] + "*.fits")
            pattern = self.raw_dir / ( self.cfg[prefix_key] + "*" + FITS_EXTENSION)
            files.extend(sorted(glob.glob(pattern)))

        return files    
    
    # =========================================================
    # Report system
    # =========================================================

    def save_report(self, report_name, outputs, qa_metrics=None, summary=None, qc_plots=None):

        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: save_report()")
            return

        # Build report
        report = {"timestamp": datetime.now(UTC).isoformat(),
                  "raw_directory": self.raw_dir,
                  "channel": self.channel,
                  "parameters": self.cfg,
                  "outputs": outputs,
                  "summary": summary or {"n_orders": self.n_orders, "status": "SUCCESS"}
                 }

        if qa_metrics is not None:
            report["qa_metrics"] = qa_metrics
        
        # Output filenames
        report_name = (f"steles_pipeline_report_{report_name}_{self.channel}_{self.cfg['fileout_sci']}")

        #json_file = os.path.join(self.proc_dir,report_name + ".json")
        #html_file = os.path.join(self.proc_dir,report_name + ".html")
        
        json_file = self.proc_dir / ( report_name + JSON_EXTENSION )
        html_file = self.proc_dir / ( report_name + HTML_EXTENSION )

        # JSON
        with open(json_file, "w") as f:
            json.dump(report, f, indent=4, default=str)

        self.log.info(f"JSON report saved: {json_file}")

        # HTML
        self._generate_html_report(report, html_file, qc_plots=qc_plots)

        self.log.info(f"HTML report saved: {html_file}")

        # Optional: auto-open
        if self.cfg.get("open_report", False):
            webbrowser.open(html_file)

    def _generate_html_report(self, report, html_file, qc_plots=None):

        def fmt_value(value):
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        def fmt_table(rows):
            if not rows:
                return "<p>No data available.</p>"

            if "order" in rows[0]:
                orders = [int(row["order"]) for row in rows if row.get("order") is not None]

                if orders and min(orders) == 0:
                    rows = [dict(row, order=int(row["order"]) + 1) for row in rows]

                rows = sorted(rows, key=lambda x: int(x.get("order", -1)), reverse=True)

            columns = list(rows[0].keys())

            html = "<table class='metrics'><thead><tr>"
            html += "".join(f"<th>{c}</th>" for c in columns)
            html += "</tr></thead><tbody>"

            for row in rows:
                html += "<tr>"
                html += "".join(f"<td>{fmt_value(row.get(c, ''))}</td>" for c in columns)
                html += "</tr>"

            return html + "</tbody></table>"
        
        def dict_table(d):
            rows = "".join(f"<tr><td>{k}</td><td>{fmt_value(v)}</td></tr>" for k, v in d.items())
            return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"

        def list_table(data):
            if not data:
                return "<p>None</p>"

            keys = list(data[0].keys())
            header = "".join(f"<th>{k}</th>" for k in keys)

            rows = ""
            for item in data:
                rows += "<tr>" + "".join(f"<td>{fmt_value(item.get(k, ''))}</td>" for k in keys) + "</tr>"

            return f"<table><tr>{header}</tr>{rows}</table>"

        def fmt_qa_metrics(d):
            html = ""

            simple = {}
            nested = {}

            for key, value in d.items():
                if isinstance(value, dict):
                    nested[key] = value
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    nested[key] = value
                else:
                    simple[key] = value

            if simple:
                html += dict_table(simple)

            for key, value in nested.items():
                html += f"<h3>{key.replace('_', ' ').title()}</h3>"

                if isinstance(value, dict):
                    html += dict_table(value)
                elif isinstance(value, list):
                    html += list_table(value)

            return html    
        
        
        # Header
        html = f"""
        <html><head><title>STELES Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1, h2 {{ color: #2c3e50; }}
                .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .code {{ background: #ecf0f1; padding: 10px; border-radius: 5px; font-family: monospace; }}
                img {{ max-width: 100%; border: 1px solid #ccc; margin-top: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: left; }}
                th {{ background-color: #ecf0f1; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h3 {{ margin-top: 20px; color: #34495e; }}
                .metrics {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
                .metrics th {{ background-color: #ecf0f1; font-weight: bold; }}
                .metrics tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>

        <body>

        <h1>STELES Pipeline Report</h1>

        <div class="section">
            <h2>General Information</h2>
            <p><b>Timestamp:</b> {report['timestamp']}</p>
            <p><b>Raw Directory:</b> {report['raw_directory']}</p>
            <p><b>Channel:</b> {report['channel']}</p>
        </div>

        <div class="section">
            <h2>Parameters</h2>
            <div class="code">
                {dict_table(report['parameters'])}
            </div>
        </div>

        <div class="section">
            <h2>Outputs</h2>
            <div class="code">
                {dict_table(report['outputs'])}
            </div>
        </div>
        """

        # QA metrics
        if "qa_metrics" in report:

            html += "<div class='section'><h2>QA Metrics</h2>"
            qa = report["qa_metrics"]

            # Trace summary
            if "n_orders" in qa and "n_trace_fallback" in qa:

                n_orders = qa["n_orders"]
                n_fallback = qa["n_trace_fallback"]
                n_automatic = n_orders - n_fallback

                html += (f"<p><b>Automatic tracing:</b> "
                         f"{n_automatic}/{n_orders} orders</p>")

            if "trace_fallback_orders" in qa:

                orders = qa["trace_fallback_orders"]

                if orders:
                    order_text = ", ".join(str(o) for o in orders)

                    html += (f"<p><b>Default trace used:</b> "
                             f"order{'s' if len(orders) > 1 else ''} "
                             f"{order_text}</p>")
                else:
                    html += ("<p><b>Default trace used:</b> None</p>")

            # Other QA metrics
            for key, value in qa.items():

                title = key.replace("_", " ").title()

                if isinstance(value, dict):

                    html += f"<h3>{title}</h3>"

                    simple = {}
                    tables = {}

                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], dict):
                            tables[subkey] = subvalue
                        else:
                            simple[subkey] = subvalue

                    # Scalar metrics
                    if simple:
                        html += dict_table(simple)

                    # Per-order tables
                    for subkey, subvalue in tables.items():
                        subtitle = subkey.replace("_", " ").title()
                        html += f"<h4>{subtitle}</h4>"
                        html += fmt_table(subvalue)

                elif isinstance(value, list) and value and isinstance(value[0], dict):

                    html += f"<h3>{title}</h3>"
                    html += fmt_table(value)

                else:
                    html += f"<p><b>{title}</b>: {fmt_value(value)}</p>"

            html += "</div>"

        # Summary
        if "summary" in report:

            html += f"""
            <div class="section">
                <h2>Summary</h2>
                <div class="code">
                    {dict_table(report['summary'])}
                </div>
            </div>
            """

        # QC plots
        if qc_plots:

            html += """
            <div class="section">
                <h2>Quality Control Plots</h2>
            """

            for plot in qc_plots:

                #plot_path = os.path.join(self.plot_dir,plot)
                plot_path = self.plot_dir / plot

                if os.path.exists(plot_path):

                    #rel_path = os.path.relpath(plot_path,self.proc_dir)
                    rel_path = plot_path.relative_to(self.proc_dir)

                    html += ( f"<h3>{plot}</h3>"
                              f"<img src='{rel_path}'>" )

            html += "</div>"

        # Close HTML
        html += """
        </body>
        </html>
        """

        with open(html_file, "w") as f:
            f.write(html)

# END