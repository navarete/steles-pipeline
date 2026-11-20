import os
import numpy as np

from astropy.io import fits

from .steles_base import STELES_BasePipeline
from .steles_imports import FITS_EXTENSION, PLOT_EXTENSION
from . import steles_ccd, steles_qc

class STELES_Pipeline(STELES_BasePipeline):
    
    # =========================================================
    # Stage 1: Preprocessing
    # =========================================================
    logger_name = "STELES-PartI"

    # =========================================================
    # Stage 1: Preprocessing
    # =========================================================

    def merge_extensions(self, input_files):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: merge_extensions()")
            return

        self.log.info("Merging extensions")
        
        n_files = len(input_files)

        for i, file in enumerate(input_files):

            filename = os.path.basename(file)

            self.log.info(f"[{i+1}/{n_files}] Merging {filename}")   
            
            filename = os.path.basename(file)
            output_filename = "m_" + filename
            #output_path = os.path.join(self.proc_dir, output_filename)
            output_path = self.proc_dir / output_filename

            # --- Checkpoint ---
            if os.path.exists(output_path) and not self.cfg["overwrite"]:
                self.log.info(f"Skipping already merged: {output_filename}")
                continue
                
            # --- Run merge ---
            fits_out, hdr_out = steles_ccd.steles_merge_extensions(
                file,
                master_bias=None,
                overscan_correction=True,
                gain_per_amplifier=self.gain_per_amplifier,
                remove_astrometry=True,
                show_results=False,
                figure_out=None
            )
            
            # --- Write output ---
            self._write_fits(fits_out, hdr_out, output_path, overwrite=self.cfg["overwrite"])
            
    def cosmic_ray_clean(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: cosmic_ray_clean()")
            return
        
        self.log.info("Running cosmic ray cleaning")

        # apply only on science targets, not calibration files.
        input_files = self._collect_input_files(("sci",))
        
        #for raw_file in input_files:
        n_files = len(input_files)

        for i, raw_file in enumerate(input_files):

            filename = os.path.basename(raw_file)

            self.log.info(f"[{i+1}/{n_files}] CR cleaning {filename}")        
            merged_file = self.proc_dir / self._fname(filename, level="merged")
            output_file = self.proc_dir / self._fname(filename, level="cleaned")
            
            if not os.path.exists(merged_file):
                self.log.warning(f"Merged file not found, skipping: {merged_file}")
                continue

            if os.path.exists(output_file) and not self.cfg["overwrite"]:
                self.log.info(f"Skipping CR-cleaned file: {output_file}")
                continue

            img, hdr = steles_ccd.steles_cosmicray_removal(
                merged_file,
                channel=self.channel,
                lacosmic_gain=self.gain,
                lacosmic_rdnoise=self.rdnoise,
                show_plot=False,
                save_fits=False
            )

            hdr['ST_CRCLN'] = (True, "STELES cosmic ray clean")

            self._write_fits(img, hdr, output_file, overwrite=self.cfg["overwrite"])
    
    # =========================================================
    # Stage 2: Calibration Preparation
    # =========================================================
    def scale_and_combine(self, frame_type):

        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: scale_and_combine({frame_type})")
            return

        frame_config = {"arc": {
                            "list_key": "arc_list",
                            "output_key": "fileout_arc",
                            "threshold_key": "saturation_threshold",
                            "label": "ThAr",
                        },
                        "flat": {
                            "list_key": "flat_list",
                            "output_key": "fileout_flat",
                            "threshold_key": "saturation_threshold",
                            "label": "flat",
                        },}

        if frame_type not in frame_config:
            raise ValueError(f"Unknown frame type '{frame_type}'. Expected one of: {list(frame_config)}")

        config = frame_config[frame_type]

        #output_path = os.path.join(self.proc_dir, self.cfg[config["output_key"]] + ".fits")
        output_path = self.proc_dir / ( self.cfg[config["output_key"]] + FITS_EXTENSION )

        # --- Checkpoint ---
        if self._check_done([output_path]) and not self.cfg["overwrite"]:
            self.log.info(f"Skipping {config['label']} scaling/combination")
            return

        self.log.info(f"Scaling and combining {config['label']} frames")

        # --- Input files ---
        files = [self._fname(f + FITS_EXTENSION, level="merged") for f in self.cfg[config["list_key"]] ]

        if len(files) == 0:
            raise RuntimeError(f"No {frame_type} files found.")

        # --- Combine/scale ---
        if len(files) > 1:
            scaled_frame = steles_ccd.scale_saturated_thar_spectrum(self.proc_dir, files, saturation_threshold=self.cfg[config["threshold_key"]], return_hdu=True)

        # --- Single frame ---
        else:
            #with fits.open(os.path.join(self.proc_dir, files[0]), memmap=False) as hdul:
            with fits.open(self.proc_dir / files[0], memmap=False) as hdul:
                data   = hdul[0].data.astype(np.float64).copy()
                header = hdul[0].header.copy()

            scaled_frame = fits.PrimaryHDU(data=data, header=header)

        # --- Transpose ---
        #if self.transpose:
        scaled_frame = steles_ccd.transpose_frame(scaled_frame, channel=self.channel)

        # --- Write output ---
        self._write_fits(scaled_frame.data, scaled_frame.header, output_path, overwrite=self.cfg["overwrite"])

    # =========================================================
    # Stage 3: Master Frames
    # =========================================================
    def _load_frames(self, list_key=None, level="raw"):

        if not list_key or not self.cfg.get(list_key):
            return []

        self.log.info(f"Loading frames from list: {list_key} (level={level})")

        names = [self._fname(f, level=level) for f in self.cfg[list_key]]

        return steles_ccd.load_frames(self.proc_dir, names)
    
    def create_master_frame(self, frame_type):

        if self._get_bool("dry_run"):
                self.log.info(f"[DRY RUN] Would execute: create_master_frames()")
                return

        if self._master_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Master frames exist - loading instead of recomputing")
            self.load_master_frames()
            return

        self.log.info("Creating master frame")

        mapping = {"flat": ("flat_list", "fileout_flat"),
                   "sci":  ("sci_list",  "fileout_sci"),}

        list_key, output_key = mapping[frame_type]

        frames = self._load_frames(list_key, level="merged")

        master = steles_ccd.process_frames(
            frames,
            sigma_clip=self.cfg["combine_sigma_clip"],
            interpolate_bad_columns=self.cfg.get("interpolate_bad_columns", False),
            interpolate_bad_columns_sigma=self.cfg.get("interpolate_bad_columns_sigma", 2),
            interpolate_bad_columns_iterations=self.cfg.get("interpolate_bad_columns_iterations", 10),
        )

        #if self.transpose:
        master = steles_ccd.transpose_frame(master, channel=self.channel)
            
        # make orders to increase from left to right
        #if self.channel == "red":
        #    img = np.flip(img, axis=1)

        self._write_fits(master.data, master.header,
                         #os.path.join(self.proc_dir, self.cfg[output_key] + ".fits"),
                         self.proc_dir / ( self.cfg[output_key] + FITS_EXTENSION ),
                         overwrite=self.cfg["overwrite"]) 
    
    # =========================================================
    # Stage 4: Load Data
    # =========================================================
        
    # =========================================================
    # Stage 4b: Quality control data
    # =========================================================
        

        
    # =========================================================
    # Stage 5: Tracing
    # =========================================================
    
    def save_trace(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: save_trace()")
            return

        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Skipping trace save (already exists)")
            return
        
        self.log.info("Saving trace coefficients")

        for i, coeffs in enumerate(self.trace_poly_coeffs):
            #file_csv = os.path.join(self.trace_dir, f"poly_tracing_n{i+1:02d}.csv")
            file_csv = self.trace_dir / f"poly_tracing_n{i+1:02d}.csv"

            steles_ccd.save_polyfits_tracing_csv(i + 1, coeffs, file_csv, show_messages=False)

    def trace_orders(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: trace_orders()")
            return
 
        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Trace files found - loading existing solution")
            self.load_trace()
            return
 
        self.log.info("Tracing spectral orders")
 
        trace_data = self.sci_data if self.cfg["trace_source"] == "sci" else self.quartz_data
 
        if self.cfg["trace_source"] == "sci":
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_sci']}_order_retracing")
        else:
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_flat']}_order_retracing")
 
        self.trace_from_observations = self.cfg["trace_from_observations"]
 
        if self.trace_from_observations:
 
            self.trace_fallback_orders = []
 
            # --- Load DB reference tables once, before the per-order loop.
            # ref_tables[order] = (ref_y, ref_x) arrays covering the full CCD y-range.
            # Used to:
            #   (a) pad hybrid tables outside the detected y-range, and
            #   (b) provide reference polynomial coefficients via np.polyfit
            #       directly on the raw table points — avoiding any systematic
            #       shift that read_master_tracing's astropy fitter may introduce.
            ref_tables = {}
            for n in range(self.n_orders):
                order = n + 1
                ref_file = self.trace_default_dir / f"trace_xy_{order:02d}.txt"
                ref_tables[order] = steles_ccd.load_trace_table(ref_file)   # (ref_y, ref_x)
 
            # Reference polynomials fitted directly with np.polyfit on the raw
            # reference table points.  These are used as reference_poly_coeffs in
            # refit_trace so that the Gaussian search window in step 6b is centred
            # accurately on the reference position.
            ndeg_ref = self.cfg.get("refit_trace_ndeg", 4)
            ref_poly_coeffs = []
            for n in range(self.n_orders):
                order = n + 1
                ref_y, ref_x = ref_tables[order]
                coeffs = np.polyfit(ref_y, ref_x, ndeg_ref)   # highest degree first
                ref_poly_coeffs.append(coeffs)
 
            # DB reference polynomials from read_master_tracing — used only as
            # trace_init fallback; the directly-fitted ref_poly_coeffs above are
            # what get passed to refit_trace as reference_poly_coeffs.
            trace_ref = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            # --- Per-order tracing loop ---
            y_ranges_valid = {}
 
            for n in range(self.n_orders):
 
                order = n + 1
                ref_y, ref_x = ref_tables[order]
                cref = ref_poly_coeffs[n]   # np.polyfit coefficients for this order
 
                self.log.info(f"[{order}/{self.n_orders}] Tracing order")
 
                try:
                    trace_y, trace_x = steles_ccd.trace_stellar_profile(
                        trace_data,
                        self.order_centers[n],
                        y_center=int(0.5 * trace_data.shape[0]),
                        y_range=self.y_range,
                        half_width=15,
                        sigma_threshold=1.0, step=25, poly_order=2,
                        collapse_rows=5, max_shift=5.0, max_failures=5,
                        use_gaussian=True, use_fwhm_fallback=True,
                        show_plot=False, displ_max_noise=10, debug=False)
 
                    # Record the y-range actually covered by the observation
                    y_obs_min = int(trace_y.min())
                    y_obs_max = int(trace_y.max())
                    y_ranges_valid[order] = (y_obs_min, y_obs_max)
 
                    # --- Build hybrid trace table ---
                    # Keep observed points within the detected range; pad with
                    # reference points outside it so the saved table spans the
                    # full order and read_master_tracing fits a well-constrained
                    # polynomial.
                    #
                    # Boundary offsets are measured between the observed polynomial
                    # and the reference polynomial at the 10% of observed points
                    # nearest each border, using np.polyfit on both tables so
                    # both polynomials are on the same footing.
                    cobs = np.polyfit(trace_y, trace_x, ndeg_ref)
 
                    n_bnd = max(1, len(trace_y) // 10)
 
                    y_bnd_lo = np.mean(trace_y[:n_bnd])
                    offset_lo = np.polyval(cobs, y_bnd_lo) - np.polyval(cref, y_bnd_lo)
 
                    y_bnd_hi = np.mean(trace_y[-n_bnd:])
                    offset_hi = np.polyval(cobs, y_bnd_hi) - np.polyval(cref, y_bnd_hi)
 
                    # sanity-check: warn if boundary offsets are suspiciously large
                    max_expected_offset = self.cfg.get("refit_trace_divergence_threshold", 10.0) * 2
                    for label, off in [("low", offset_lo), ("high", offset_hi)]:
                        if abs(off) > max_expected_offset:
                            self.log.warning(
                                f"  Order {order:02d}: {label}-boundary offset={off:+.2f}px "
                                f"exceeds {max_expected_offset:.1f}px — hybrid padding may be unreliable"
                            )
 
                    # reference points below the observed range — shift by offset_lo
                    mask_ref_lo = ref_y < y_obs_min
                    pad_y_lo = ref_y[mask_ref_lo]
                    pad_x_lo = ref_x[mask_ref_lo] + offset_lo
 
                    # reference points above the observed range — shift by offset_hi
                    mask_ref_hi = ref_y > y_obs_max
                    pad_y_hi = ref_y[mask_ref_hi]
                    pad_x_hi = ref_x[mask_ref_hi] + offset_hi
 
                    # concatenate: pad_lo | observed | pad_hi, sorted by y
                    hybrid_y = np.concatenate([pad_y_lo, trace_y, pad_y_hi])
                    hybrid_x = np.concatenate([pad_x_lo, trace_x, pad_x_hi])
                    sort_idx = np.argsort(hybrid_y)
                    hybrid_y = hybrid_y[sort_idx]
                    hybrid_x = hybrid_x[sort_idx]
 
                    steles_ccd.save_trace_table(hybrid_y, hybrid_x, order, self.trace_dir)
 
                    self.log.info(
                        f"  Order {order:02d}: obs y=[{y_obs_min},{y_obs_max}], "
                        f"padded with {mask_ref_lo.sum()} ref pts below "
                        f"(offset={offset_lo:+.2f}px) and "
                        f"{mask_ref_hi.sum()} ref pts above "
                        f"(offset={offset_hi:+.2f}px)"
                    )
 
                except (TypeError, ValueError, RuntimeError) as e:
                    self.log.warning(f"Tracing failed for order {order}: {e}")
                    self.log.info(f"Loading default trace for order {order}")
                    self.trace_fallback_orders.append(order)
 
                    # Complete tracing failure — use full reference table as-is;
                    # mark full y-range as valid so refit_trace applies no restriction
                    y_ranges_valid[order] = (int(ref_y.min()), int(ref_y.max()))
                    steles_ccd.save_trace_table(ref_y, ref_x, order, self.trace_dir)
 
            # --- Fit polynomials to the saved hybrid tables ---
            # Every saved table now spans the full order y-range, so
            # read_master_tracing fits a well-constrained polynomial for all orders.
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_dir, self.n_orders)
 
            # --- Refit with directly-fitted reference polynomials ---
            # ref_poly_coeffs (np.polyfit on raw reference tables) are used as
            # reference_poly_coeffs so the Gaussian search window in step 6b is
            # centred accurately, free of any astropy-fitter systematic offset.
            self.trace_poly_coeffs, fallback_orders = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_dir,
                trace_window=15,
                trace_ndeg=self.cfg["refit_trace_ndeg"],
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
                use_reference_fallback=self.cfg["refit_trace_use_reference_fallback"],
                reference_poly_coeffs=ref_poly_coeffs,
                divergence_threshold=self.cfg["refit_trace_divergence_threshold"],
                fallback_y_center=None,
                fallback_half_width=self.cfg["refit_trace_fallback_half_width"],
                y_ranges_valid=y_ranges_valid,
            )
 
            self.qa_metrics = {
                "n_orders":              self.n_orders,
                "n_trace_fallback":      len(fallback_orders),
                "trace_fallback_orders": fallback_orders,
            }
 
        else:
 
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            self.trace_poly_coeffs, _ = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_default_dir,
                trace_window=15,
                trace_ndeg=4,
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
            )
 
        # save trace
        self.save_trace()

    def trace_orders_last_last(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: trace_orders()")
            return
 
        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Trace files found - loading existing solution")
            self.load_trace()
            return
 
        self.log.info("Tracing spectral orders")
 
        trace_data = self.sci_data if self.cfg["trace_source"] == "sci" else self.quartz_data
 
        if self.cfg["trace_source"] == "sci":
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_sci']}_order_retracing")
        else:
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_flat']}_order_retracing")
 
        self.trace_from_observations = self.cfg["trace_from_observations"]
 
        if self.trace_from_observations:
 
            self.trace_fallback_orders = []
 
            # --- Load DB reference tables once, before the per-order loop.
            # ref_tables[n] = (ref_y, ref_x) arrays covering the full CCD y-range.
            # Used to pad the observed tables outside the detected y-range so that
            # read_master_tracing always fits a polynomial over the full order extent.
            ref_tables = {}
            for n in range(self.n_orders):
                order = n + 1
                ref_file = self.trace_default_dir / f"trace_xy_{order:02d}.txt"
                ref_tables[order] = steles_ccd.load_trace_table(ref_file)   # (ref_y, ref_x)
 
            # DB reference polynomials — fitted to the reference tables.
            # Passed to refit_trace as an independent shape reference.
            trace_ref = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            # --- Per-order tracing loop ---
            y_ranges_valid = {}
 
            for n in range(self.n_orders):
 
                order = n + 1
                ref_y, ref_x = ref_tables[order]
 
                self.log.info(f"[{order}/{self.n_orders}] Tracing order")
 
                try:
                    trace_y, trace_x = steles_ccd.trace_stellar_profile(
                        trace_data,
                        self.order_centers[n],
                        y_center=int(0.5 * trace_data.shape[0]),
                        y_range=self.y_range,
                        half_width=15,
                        sigma_threshold=1.0, step=25, poly_order=2,
                        collapse_rows=5, max_shift=5.0, max_failures=5,
                        use_gaussian=True, use_fwhm_fallback=True,
                        show_plot=False, displ_max_noise=10, debug=False)
 
                    # Record the y-range actually covered by the observation
                    y_obs_min = int(trace_y.min())
                    y_obs_max = int(trace_y.max())
                    y_ranges_valid[order] = (y_obs_min, y_obs_max)
 
                    # --- Build hybrid trace table ---
                    # Keep observed points within the detected range; pad with
                    # reference points outside it so the saved table spans the
                    # full order and read_master_tracing fits a well-constrained
                    # polynomial.  Reference points used for padding are shifted
                    # by the median x-offset measured at the overlap boundary so
                    # the two segments join smoothly.
                    #
                    # Boundary offset: median difference between the observed
                    # polynomial and the reference polynomial evaluated at the
                    # 10 % of observed points nearest each border.
                    n_bnd = max(1, len(trace_y) // 10)
 
                    # fit temporary polynomials to the observed and reference tables
                    cobs = np.polyfit(trace_y, trace_x, 4)
                    cref = np.polyfit(ref_y,   ref_x,   4)
 
                    # offset at the low-y boundary (bottom of observed range)
                    y_bnd_lo = np.mean(trace_y[:n_bnd])
                    offset_lo = np.polyval(cobs, y_bnd_lo) - np.polyval(cref, y_bnd_lo)
 
                    # offset at the high-y boundary (top of observed range)
                    y_bnd_hi = np.mean(trace_y[-n_bnd:])
                    offset_hi = np.polyval(cobs, y_bnd_hi) - np.polyval(cref, y_bnd_hi)
 
                    # reference points below the observed range — shift by offset_lo
                    mask_ref_lo = ref_y < y_obs_min
                    pad_y_lo = ref_y[mask_ref_lo]
                    pad_x_lo = ref_x[mask_ref_lo] + offset_lo
 
                    # reference points above the observed range — shift by offset_hi
                    mask_ref_hi = ref_y > y_obs_max
                    pad_y_hi = ref_y[mask_ref_hi]
                    pad_x_hi = ref_x[mask_ref_hi] + offset_hi
 
                    # concatenate: pad_lo | observed | pad_hi, sorted by y
                    hybrid_y = np.concatenate([pad_y_lo, trace_y, pad_y_hi])
                    hybrid_x = np.concatenate([pad_x_lo, trace_x, pad_x_hi])
                    sort_idx = np.argsort(hybrid_y)
                    hybrid_y = hybrid_y[sort_idx]
                    hybrid_x = hybrid_x[sort_idx]
 
                    steles_ccd.save_trace_table(hybrid_y, hybrid_x, order, self.trace_dir)
 
                    self.log.info(
                        f"  Order {order:02d}: obs y=[{y_obs_min},{y_obs_max}], "
                        f"padded with {mask_ref_lo.sum()} ref pts below "
                        f"(offset={offset_lo:+.2f}px) and "
                        f"{mask_ref_hi.sum()} ref pts above "
                        f"(offset={offset_hi:+.2f}px)"
                    )
 
                except (TypeError, ValueError, RuntimeError) as e:
                    self.log.warning(f"Tracing failed for order {order}: {e}")
                    self.log.info(f"Loading default trace for order {order}")
                    self.trace_fallback_orders.append(order)
 
                    # Complete tracing failure — use full reference table as-is;
                    # mark full y-range as valid so refit_trace applies no restriction
                    y_ranges_valid[order] = (int(ref_y.min()), int(ref_y.max()))
                    steles_ccd.save_trace_table(ref_y, ref_x, order, self.trace_dir)
 
            # --- Fit polynomials to the saved hybrid tables ---
            # Every saved table now spans the full order y-range, so
            # read_master_tracing fits a well-constrained polynomial for all orders.
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_dir, self.n_orders)
 
            # --- Refit with DB reference polynomials and per-order valid y-ranges ---
            # y_ranges_valid tells refit_trace which rows were directly observed;
            # only those rows are used for the sigma-clip and polynomial refit.
            # Outside the valid range the reference shape (trace_ref) is used.
            self.trace_poly_coeffs, fallback_orders = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_dir,
                trace_window=15,
                trace_ndeg=self.cfg["refit_trace_ndeg"],
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
                use_reference_fallback=self.cfg["refit_trace_use_reference_fallback"],
                reference_poly_coeffs=trace_ref,
                divergence_threshold=self.cfg["refit_trace_divergence_threshold"],
                fallback_y_center=None,
                fallback_half_width=self.cfg["refit_trace_fallback_half_width"],
                y_ranges_valid=y_ranges_valid,
            )
 
            self.qa_metrics = {
                "n_orders":              self.n_orders,
                "n_trace_fallback":      len(fallback_orders),
                "trace_fallback_orders": fallback_orders,
            }
 
        else:
 
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            self.trace_poly_coeffs, _ = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_default_dir,
                trace_window=15,
                trace_ndeg=4,
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
            )
 
        # save trace
        self.save_trace()
 

    
    def trace_orders_last(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: trace_orders()")
            return
 
        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Trace files found - loading existing solution")
            self.load_trace()
            return
 
        self.log.info("Tracing spectral orders")
 
        trace_data = self.sci_data if self.cfg["trace_source"] == "sci" else self.quartz_data
 
        if self.cfg["trace_source"] == "sci":
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_sci']}_order_retracing")
        else:
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_flat']}_order_retracing")
 
        # add option for tracing the orders from scratch and not loading the database solution
        self.trace_from_observations = self.cfg["trace_from_observations"]
 
        if self.trace_from_observations:
 
            self.trace_fallback_orders = []
 
            # --- Load DB reference polynomials once, before the per-order loop.
            # These are used as:
            #   (a) the fallback shape in refit_trace (reference_poly_coeffs), and
            #   (b) the x-search anchor for rows outside the observed y-range.
            # Loading from trace_default_dir keeps them independent of the
            # (possibly noisy) polynomials fitted to the current observation.
            trace_ref = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            # --- Per-order tracing loop ---
            # For each order, trace_stellar_profile walks up/down from y_center
            # and stops when flux drops below the S/N threshold.  We record the
            # actual y-range covered so that refit_trace can restrict its refit
            # to that region and avoid extrapolating into noisy borders.
            y_ranges_valid = {}
 
            for n in range(self.n_orders):
 
                order = n + 1
 
                self.log.info(f"[{order}/{self.n_orders}] Tracing order")
 
                try:
                    trace_y, trace_x = steles_ccd.trace_stellar_profile(
                        trace_data,
                        self.order_centers[n],
                        y_center=int(0.5 * trace_data.shape[0]),
                        y_range=self.y_range,
                        half_width=15,
                        sigma_threshold=1.0, step=25, poly_order=2,
                        collapse_rows=5, max_shift=5.0, max_failures=5,
                        use_gaussian=True, use_fwhm_fallback=True,
                        show_plot=False, displ_max_noise=10, debug=False)
 
                    # Record the y-range actually covered by this observation
                    y_ranges_valid[order] = (int(trace_y.min()), int(trace_y.max()))
 
                    steles_ccd.save_trace_table(trace_y, trace_x, order, self.trace_dir)
 
                except (TypeError, ValueError, RuntimeError) as e:
                    self.log.warning(f"Tracing failed for order {order}: {e}")
                    self.log.info(f"Loading default trace for order {order}")
                    self.trace_fallback_orders.append(order)
 
                    # Fall back to the full reference table; mark the full
                    # y-range as valid so refit_trace uses it without restriction
                    trace_file = (self.trace_default_dir / f"trace_xy_{order:02d}.txt")
                    trace_y, trace_x = steles_ccd.load_trace_table(trace_file)
                    y_ranges_valid[order] = (int(trace_y.min()), int(trace_y.max()))
 
                    steles_ccd.save_trace_table(trace_y, trace_x, order, self.trace_dir)
 
            # --- Fit polynomials to the saved (possibly partial) tables ---
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_dir, self.n_orders)
 
            # --- Refit with DB reference polynomials and per-order valid y-ranges ---
            self.trace_poly_coeffs, fallback_orders = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_dir,
                trace_window=15,
                trace_ndeg=self.cfg["refit_trace_ndeg"],
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
                use_reference_fallback=self.cfg["refit_trace_use_reference_fallback"],
                reference_poly_coeffs=trace_ref,                              # DB reference, not observed fit
                divergence_threshold=self.cfg["refit_trace_divergence_threshold"],
                fallback_y_center=None,
                fallback_half_width=self.cfg["refit_trace_fallback_half_width"],
                y_ranges_valid=y_ranges_valid,                                # per-order valid y-range
            )
 
            self.qa_metrics = {
                "n_orders":              self.n_orders,
                "n_trace_fallback":      len(fallback_orders),
                "trace_fallback_orders": fallback_orders,
            }
 
        else:
 
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
 
            self.trace_poly_coeffs, _ = steles_ccd.refit_trace(
                trace_data,
                self.channel,
                trace_init,
                self.trace_default_dir,
                trace_window=15,
                trace_ndeg=4,
                skip_orders=self.cfg["refit_trace_skip_orders"],
                output_file_prefix=output_file_prefix,
            )
 
        # save trace
        self.save_trace()
 
    
    def trace_orders_prev(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: trace_orders()")
            return

        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Trace files found - loading existing solution")
            self.load_trace()
            return        

        self.log.info("Tracing spectral orders")

        trace_data = self.sci_data if self.cfg["trace_source"] == "sci" else self.quartz_data

        if self.cfg["trace_source"] == "sci":
            #output_file_prefix = os.path.join(self.trace_dir, f"{self.cfg['fileout_sci']}_order_retracing")
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_sci']}_order_retracing")
        else:
            #output_file_prefix = os.path.join(self.trace_dir, f"{self.cfg['fileout_flat']}_order_retracing")
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_flat']}_order_retracing")
        
        # add option for tracing the orders from scratch and not loading the database solution
        self.trace_from_observations = self.cfg["trace_from_observations"]
        
        if self.trace_from_observations:

            # 1. Load DB reference polynomials once, before the loop
            trace_ref = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)

            # 2. Per-order tracing loop - save hybrid tables
            y_ranges_valid = {}

            self.trace_fallback_orders = []
            
            for n in range(self.n_orders):
                
                order = n + 1
                
                self.log.info(f"[{order}/{self.n_orders}] Tracing order")
                
                try:
                    trace_y, trace_x = steles_ccd.trace_stellar_profile(trace_data, 
                                                                        self.order_centers[n],
                                                                        y_center=int(0.5*trace_data.shape[0]),
                                                                        y_range=self.y_range,
                                                                        half_width=15,
                                                                        sigma_threshold=1.0, step=25, poly_order=2,
                                                                        collapse_rows=5, max_shift=5.0, max_failures=5,
                                                                        use_gaussian=True, use_fwhm_fallback=True, 
                                                                        show_plot=False, displ_max_noise=10, debug=False)
                    
                    # record the y-range actually covered by the observation
                    y_ranges_valid[order] = (int(trace_y.min()), int(trace_y.max()))
                    
                    steles_ccd.save_trace_table(trace_y, trace_x, order, self.trace_dir)
                
                except (TypeError, ValueError, RuntimeError) as e:
                    self.log.warning(f"Tracing failed for order {order}: {e}")
                    self.trace_fallback_orders.append(order)
                    
                    # use full reference table, mark full y-range as valid
                    trace_file = self.trace_default_dir / f"trace_xy_{order:02d}.txt"
                    trace_y, trace_x = steles_ccd.load_trace_table(trace_file)
                    y_ranges_valid[order] = (int(trace_y.min()), int(trace_y.max()))
                    steles_ccd.save_trace_table(trace_y, trace_x, order, self.trace_dir)

                # 3. Fit polynomials to the saved (possibly partial) tables
                trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_dir, self.n_orders)


                # 4. Refit with DB reference — offset measured only where flux is reliable
                self.trace_poly_coeffs, fallback_orders = steles_ccd.refit_trace(trace_data,
                                                                                 self.channel,
                                                                                 trace_init,
                                                                                 self.trace_dir,
                                                                                 trace_window=15,
                                                                                 trace_ndeg=self.cfg["refit_trace_ndeg"],
                                                                                 skip_orders=self.cfg["refit_trace_skip_orders"],
                                                                                 output_file_prefix=output_file_prefix,
                                                                                 use_reference_fallback=self.cfg["refit_trace_use_reference_fallback"],
                                                                                 reference_poly_coeffs=trace_ref,                              # DB reference
                                                                                 divergence_threshold=self.cfg["refit_trace_divergence_threshold"],
                                                                                 fallback_y_center=None,
                                                                                 fallback_half_width=self.cfg["refit_trace_fallback_half_width"],
                                                                                 y_ranges_valid=y_ranges_valid,                                # new: per-order valid y-range
                                                                                )
                                                                        
            self.qa_metrics = {"n_orders": self.n_orders,
                               "n_trace_fallback": len(fallback_orders),
                               "trace_fallback_orders": fallback_orders,}

                
        else:
            
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
            
            self.trace_poly_coeffs = steles_ccd.refit_trace(trace_data,
                                                            self.channel,
                                                            trace_init,
                                                            self.trace_default_dir,
                                                            trace_window=15,
                                                            trace_ndeg=4,
                                                            skip_orders=self.cfg["refit_trace_skip_orders"],
                                                            output_file_prefix=output_file_prefix)
        # save trace
        self.save_trace()

    def trace_orders_prev(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: trace_orders()")
            return

        if self._trace_files_exist() and not self.cfg["overwrite"]:
            self.log.info("Trace files found - loading existing solution")
            self.load_trace()
            return        

        self.log.info("Tracing spectral orders")

        trace_data = self.sci_data if self.cfg["trace_source"] == "sci" else self.quartz_data

        if self.cfg["trace_source"] == "sci":
            #output_file_prefix = os.path.join(self.trace_dir, f"{self.cfg['fileout_sci']}_order_retracing")
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_sci']}_order_retracing")
        else:
            #output_file_prefix = os.path.join(self.trace_dir, f"{self.cfg['fileout_flat']}_order_retracing")
            output_file_prefix = (self.trace_dir / f"{self.cfg['fileout_flat']}_order_retracing")
        
        # add option for tracing the orders from scratch and not loading the database solution
        self.trace_from_observations = self.cfg["trace_from_observations"]
        
        if self.trace_from_observations:
            
            self.trace_fallback_orders = []
            
            for n in range(self.n_orders):
                
                order = n + 1
                
                self.log.info(f"[{order}/{self.n_orders}] Tracing order")
                
                
                try:
                    trace_y, trace_x = steles_ccd.trace_stellar_profile(trace_data, 
                                                                        self.order_centers[n],
                                                                        y_center=int(0.5*trace_data.shape[0]),
                                                                        y_range=self.y_range,
                                                                        half_width=15,
                                                                        sigma_threshold=1.0, step=25, poly_order=2,
                                                                        collapse_rows=5, max_shift=5.0, max_failures=5,
                                                                        use_gaussian=True, use_fwhm_fallback=True, 
                                                                        show_plot=False, displ_max_noise=10, debug=False)
                
                    steles_ccd.save_trace_table(trace_y, trace_x, order, self.trace_dir)
                    
                except (TypeError, ValueError, RuntimeError) as e:
                    self.log.warning(f"Tracing failed for order {order}: {e}")
                    self.log.info(f"Loading default trace for order {order}")
                    self.trace_fallback_orders.append(order)
                    
                    #trace_file = os.path.join(self.trace_default_dir, f"trace_xy_{order:02d}.txt")
                    trace_file = (self.trace_default_dir / f"trace_xy_{order:02d}.txt")
                    trace_y, trace_x = steles_ccd.load_trace_table(trace_file)
                    
                    steles_ccd.save_trace_table(trace_y, trace_x, n+1, self.trace_dir)
                    
            # load reference polynomials from the DB, independent of the observation
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_dir, self.n_orders)

            self.trace_poly_coeffs, fallback_orders = steles_ccd.refit_trace(trace_data,
                                                                             self.channel,
                                                                             trace_init,
                                                                             self.trace_dir,
                                                                             trace_window=15,
                                                                             trace_ndeg=self.cfg["refit_trace_ndeg"],
                                                                             skip_orders=self.cfg["refit_trace_skip_orders"],
                                                                             output_file_prefix=output_file_prefix,
                                                                             use_reference_fallback = self.cfg["refit_trace_use_reference_fallback"], # enable the fallback
                                                                             reference_poly_coeffs  = None,                                   # the trace_poly_coeffs_init list (reference fits)
                                                                             divergence_threshold   = self.cfg["refit_trace_divergence_threshold"],   # pixels: max allowed shift at y_center before fallback
                                                                             fallback_y_center      = None,                                   # y pixel to measure offset (None → midpoint of y_vals)
                                                                             fallback_half_width    = self.cfg["refit_trace_fallback_half_width"],        # half-width in x for peak search at y_center
                                                                            )

            self.qa_metrics = {"n_orders": self.n_orders,
                               "n_trace_fallback": len(fallback_orders),
                               "trace_fallback_orders": fallback_orders,}

                
        else:
            
            trace_init = steles_ccd.read_master_tracing(trace_data, self.trace_default_dir, self.n_orders)
            
            self.trace_poly_coeffs = steles_ccd.refit_trace(trace_data,
                                                            self.channel,
                                                            trace_init,
                                                            self.trace_default_dir,
                                                            trace_window=15,
                                                            trace_ndeg=4,
                                                            skip_orders=self.cfg["refit_trace_skip_orders"],
                                                            output_file_prefix=output_file_prefix)
        # save trace
        self.save_trace()


    
    def load_trace(self):
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: load_trace()")
            return
        
        self.log.info("Loading trace coefficients from disk")

        try:
            self.trace_poly_coeffs = steles_ccd.read_polyfits_tracing_csv(
                self.trace_dir,
                "poly_tracing_n",
                self.n_orders)
            self.log.info("Trace coefficients successfully loaded")

        except Exception as e:
            raise RuntimeError(f"Failed to load trace coefficients: {e}")    
            
    def _trace_files_exist(self):
        """ check if CSV files exist inside the 'trace_dir', if not, derived them """

        for i in range(self.n_orders):
            #file_csv = os.path.join(self.trace_dir, f"poly_tracing_n{i+1:02d}.csv")
            file_csv = (self.trace_dir / f"poly_tracing_n{i+1:02d}.csv")
            if not os.path.exists(file_csv):
                return False
        return True
    
    def extract_orders(self, frame="sci"):

        if self._get_bool("dry_run"):
            self.log.info(f"Would execute: extract_orders(frame='{frame}')")
            return

        frames = {"sci":  (self.sci_data,    self.sci_header,    self.cfg["fileout_sci"]),
                  "flat": (self.quartz_data, self.quartz_header, self.cfg["fileout_flat"]),
                  "arc":  (self.arc_data,    self.arc_header,    self.cfg["fileout_arc"]),}

        if frame not in frames:
            raise ValueError(f"Unknown frame '{frame}'. Choose from: {list(frames)}")

        data, header, fileout = frames[frame]

        #output_file = os.path.join(self.proc_dir, fileout + "_rectified.fits")
        output_file = self.proc_dir / ( fileout + "_rectified" + FITS_EXTENSION)
        
        if os.path.exists(output_file) and not self.cfg["overwrite"]:
            self.log.info(f"Linearized {frame} product exists - skipping")
            return

        self.log.info(f"Extracting and linearizing {frame} orders")

        cube = []

        for order, coeffs in enumerate(self.trace_poly_coeffs):

            #self.log.info(f"[{order + 1}/{self.n_orders}] Extracting {frame} order")

            extracted = steles_ccd.linearize_trace(data,
                                                   coeffs,
                                                   x_left=self.cfg["aperture_extract"][0],
                                                   x_right=self.cfg["aperture_extract"][1],
                                                   y_range=self.y_range,
                                                   flip_axis=True,
                                                   show_plot=False,
                                                   output_file=None)

            cube.append(extracted)

        cube = np.asarray(cube)

        self._save_cube(cube, header, fileout + "_rectified" + FITS_EXTENSION)

        return cube

    # =========================================================
    # Quality Control plots
    # =========================================================
    def quality_control_master_frames(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: quality_control_master_frames()")
            return

        if (not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False)):
            self.log.info("Skipping QC plots (disabled in config)")
            return

        self.log.info("Generating QC plots for master frames")

        #output_path = os.path.join(self.plot_dir, self.cfg["fileout_sci"] + "_dataset.png")
        output_path = self.plot_dir / ( self.cfg["fileout_sci"] + f"_dataset{PLOT_EXTENSION}")

        steles_qc.quality_control_master_frames(quartz=self.quartz_data,
                                                arc=self.arc_data,
                                                sci=self.sci_data,
                                                quartz_header=self.quartz_header,
                                                arc_header=self.arc_header,
                                                sci_header=self.sci_header,
                                                channel=self.channel, transpose=self.transpose,
                                                output_path=output_path,
                                                save_plot=self.cfg.get("save_plots", True),
                                                show_plot=self.cfg.get("show_plots", False) )

        if self.cfg.get("save_plots", True):
            self.log.info(f"QC plot saved: {output_path}")
            
    def quality_control_tracing(self):

        if self._get_bool("dry_run"):
            self.log.info( "Would execute: quality_control_tracing()" )
            return

        if ( not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False) ):
            self.log.info("Skipping tracing QC plots (disabled)")
            return

        self.log.info("Generating tracing QC plot")

        #output_path = os.path.join(self.plot_dir, self.cfg["fileout_sci"] + "_tracing.png")
        output_path = self.plot_dir / ( self.cfg["fileout_sci"] + f"_tracing{PLOT_EXTENSION}")

        intersections = steles_qc.quality_control_tracing(trace_data=self.sci_data, trace_poly_coeffs=self.trace_poly_coeffs,
                                                          y_range=self.y_range, output_path=output_path, 
                                                          save_plot=self.cfg.get("save_plots", True), show_plot=self.cfg.get("show_plots", False))

        self.trace_intersections_qc = intersections
        self.qa_metrics["trace_intersections"] = len(intersections)
        
        # Report intersections
        if intersections:
            for intersection in intersections:
                self.log.warning(f"Trace intersection: orders {intersection['order_1']} and {intersection['order_2']} at "
                                 f"(x={intersection['x']:.2f}, y={intersection['y']:.2f})")

            # Optional: fail pipeline
            if self.cfg.get("fail_on_trace_intersection", False):
                raise RuntimeError("Trace intersections detected - aborting pipeline.")

        if self.cfg.get("save_plots", True):
            self.log.info(f"Tracing QC plot saved: {output_path}")            

    def quality_control_apertures(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: quality_control_apertures()")
            return

        if ( not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False) ):
            self.log.info("Skipping aperture QC plots (disabled)")
            return

        self.log.info("Generating aperture QC plots")

        if self.trace_poly_coeffs is None:
            raise RuntimeError("Trace coefficients not available")

        plot_orders = self.cfg.get( "aperture_orders", [])

        if len(plot_orders) == 0:
            raise ValueError( "No aperture_orders defined in config")

        datasets = {"flat": (self.quartz_data, self.cfg["fileout_flat"]),
                    "arc":  (self.arc_data,    self.cfg["fileout_arc"]),
                    "sci":  (self.sci_data,    self.cfg["fileout_sci"]),}

        steles_qc.quality_control_apertures(datasets=datasets, trace_poly_coeffs=self.trace_poly_coeffs,
                                            aperture_extract=self.cfg["aperture_extract"], y_range=self.y_range,
                                            plot_orders=plot_orders, aperture_vmax=self.cfg.get("aperture_vmax",{}),
                                            output_dir=self.plot_dir,
                                            save_plot=self.cfg.get("save_plots",True),
                                            show_plot=self.cfg.get("show_plots",False))

        if self.cfg.get("save_plots", True):
            self.log.info("Aperture QC plots saved")
            
    # =========================================================
    # Report system
    # =========================================================
    def save_report_partI(self):
        
        qc_plots=[self.cfg["fileout_sci"]  + f"_dataset{PLOT_EXTENSION}",
                  self.cfg["fileout_sci"]  + f"_tracing{PLOT_EXTENSION}",
                  self.cfg["fileout_sci"]  + f"_aperture{PLOT_EXTENSION}",
                  self.cfg["fileout_flat"] + f"_aperture{PLOT_EXTENSION}",
                  self.cfg["fileout_arc"]  + f"_aperture{PLOT_EXTENSION}",]
        
        self.save_report(report_name='PartI',
                         outputs={"science_linear": self.proc_dir / ( self.cfg["fileout_sci"]  + "_rectified" + FITS_EXTENSION),
                                  "arc_linear":     self.proc_dir / ( self.cfg["fileout_arc"]  + "_rectified" + FITS_EXTENSION),
                                  "flat_linear":    self.proc_dir / ( self.cfg["fileout_flat"] + "_rectified" + FITS_EXTENSION),},
                         qa_metrics=self.qa_metrics,
                         summary={"n_orders": self.n_orders, "status": "SUCCESS"},
                         qc_plots=qc_plots)
        
    # =========================================================
    # Run Full Pipeline
    # =========================================================
    def run(self):
        
        if self._get_bool("dry_run"):
            self.log.info("DRY RUN MODE - no files will be written")     
        
        self._timeit(self.merge_extensions, "Merge Extensions [Flat]", self._collect_input_files(("flat",)))
        self._timeit(self.merge_extensions, "Merge Extensions [Sci]",  self._collect_input_files(("sci",)))
        self._timeit(self.merge_extensions, "Merge Extensions [Arc]",  self._collect_input_files(("arc",)))
        #
        self._timeit(self.cosmic_ray_clean, "Cosmic Ray Cleaning")
        self._timeit(self.scale_and_combine, "scale and combine [Arcs]", "arc")
        #    
        self._timeit(self.create_master_frame, "Create Master Frame [Flat]", "flat")
        self._timeit(self.create_master_frame, "Create Master Frame [Sci]",  "sci")
        self.load_master_frames()
        self.quality_control_master_frames()
        #
        self._timeit(self.trace_orders, "Trace Orders")
        self.quality_control_tracing()
        #
        self.quality_control_apertures()
        self._timeit(self.extract_orders, "Extract Orders [Flat]", frame='flat')
        self._timeit(self.extract_orders, "Extract Orders [Arc]",  frame='arc')
        self._timeit(self.extract_orders, "Extract Orders [Sci]",  frame='sci')
        #
        self.save_report_partI()
        #
        self.log.info("Pipeline completed successfully.")
        
    def run_trace(self):
        
        if self._get_bool("dry_run"):
            self.log.info("DRY RUN MODE - no files will be written")     
        #    
        self.load_master_frames()
        #
        self._timeit(self.trace_orders, "Trace Orders")
        self.quality_control_tracing()
        #
        self.quality_control_apertures()
        self._timeit(self.extract_orders, "Extract Orders [Flat]", frame='flat')
        self._timeit(self.extract_orders, "Extract Orders [Arc]",  frame='arc')
        self._timeit(self.extract_orders, "Extract Orders [Sci]",  frame='sci')
        #
        self.save_report_partI()
        #
        self.log.info("Pipeline completed successfully.")
        
    def run_extract_all(self):
        
        if self._get_bool("dry_run"):
            self.log.info("DRY RUN MODE - no files will be written")     
        #    
        self.load_master_frames()
        self.quality_control_master_frames()
        #
        self.load_trace()
        #
        self.quality_control_apertures()
        self._timeit(self.extract_orders, "Extract Orders [Flat]", frame='flat')
        self._timeit(self.extract_orders, "Extract Orders [Arc]",  frame='arc')
        self._timeit(self.extract_orders, "Extract Orders [Sci]",  frame='sci')
        #
        self.save_report_partI()
        #
        self.log.info("Pipeline completed successfully.")
        
        
    def run_extract_arc_science(self):
        #
        self.log.info("Using previous tracing and extracting arc and science frames.")
        #
        self._timeit(self.merge_extensions, "Merge Extensions [Sci]",  self._collect_input_files(("sci",)))
        self._timeit(self.merge_extensions, "Merge Extensions [Arc]",  self._collect_input_files(("arc",)))
        self._timeit(self.cosmic_ray_clean, "Cosmic Ray Cleaning")
        self._timeit(self.scale_and_combine, "scale and combine [Arcs]", "arc")
        self._timeit(self.create_master_frame, "Create Master Frame [Sci]",  "sci")
        #
        self.load_master_frames()
        self.quality_control_master_frames()
        self.load_trace()
        #
        self.quality_control_apertures()
        self._timeit(self.extract_orders, "Extract Orders [Arc]",  frame='arc')
        self._timeit(self.extract_orders, "Extract Orders [Sci]",  frame='sci')
        #
        self.save_report_partI()
        #
        self.log.info("Pipeline completed successfully.")
            
    def run_extract_science(self):
        #
        self.log.info("Using previous tracing and extracting science frames only.")
        #
        self._timeit(self.merge_extensions, "Merge Extensions [Sci]",  self._collect_input_files(("sci",)))
        self._timeit(self.cosmic_ray_clean, "Cosmic Ray Cleaning")
        self._timeit(self.create_master_frame, "Create Master Frame [Sci]",  "sci")
        #
        self.load_master_frames()
        self.quality_control_master_frames()
        self.load_trace()
        #
        self.quality_control_apertures()
        self._timeit(self.extract_orders, "Extract Orders [Sci]",  frame='sci')
        #
        self.save_report_partI()
        #
        self.log.info("Pipeline completed successfully.")
