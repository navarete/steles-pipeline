import os
import numpy as np
import pandas as pd

from astropy.io import fits
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from .steles_base import STELES_BasePipeline
from .steles_imports import CSV_EXTENSION, FITS_EXTENSION, PLOT_EXTENSION
from . import steles_spec, steles_qc, DB_DIR

class STELES_Pipeline_PartII(STELES_BasePipeline):
        
    # =========================================================
    # Checkpointing System
    # =========================================================
    logger_name = "STELES-PartII"
    
    def create_arc1d(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: create_arc1d()")
            return
        
        #output = os.path.join(self.proc_dir, self.cfg["fileout_arc"] + "_rectified_1d.fits")
        output = self.proc_dir / ( self.cfg["fileout_arc"] + f"_rectified_1d{FITS_EXTENSION}" )
        
        if self._check_done([output]) and not self.cfg["overwrite"]:
            self.log.info("Skipping creating 1d arc spectrum")
            return        

        self.log.info("Creating the 1d arc spectrum for wavelength solution")

        # Collapse arc
        ymid = int(0.5 * self.n_spat)
        #y_range = [0, self.n_spat] # use full range
        y_range = [ymid-2, ymid+3]   # use the central region of the slit +/-2 pixels width

        arc1d = steles_spec.collapse_spectrum2d(self.arc_cube,
                                           row_range=y_range,
                                           return_average=True)

        # --- Write output ---
        self._write_fits(arc1d, None, output, overwrite=True)
        
        self.arc1d = arc1d
    

    def save_wavelength_qc_table(self):

        rows = []
        for s in self.wavesol_stats:
            rows.append({
                "order":      s["order"],
                "n_lines":    s["n_lines"],
                "dispersion": s["dispersion"],
                "rms":        s["rms"],
                "residuals":  s["residuals"],
                "min_wave":   s["min_wave"],
                "mid_wave":   s["mid_wave"],
                "max_wave":   s["max_wave"]
            })

        df = pd.DataFrame(rows)

        out = self.proc_dir / ( self.cfg["fileout_arc"] + f"_wavelength_qc{CSV_EXTENSION}" )
        
        df.to_csv(out, index=False)

        self.log.info(f"Saved wavelength QC table: {out}")                   

    def wavelength_solution(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: wavelength_solution()")
            return
        
        #output = os.path.join(self.proc_dir, self.cfg["fileout_arc"] + f"_wavelength_solution{FITS_EXTENSION}")
        output = self.proc_dir / ( self.cfg["fileout_arc"] + f"_wavelength_solution{FITS_EXTENSION}" )
        
        if self._check_done([output]) and not self.cfg["overwrite"]:
            self.log.info("Skipping computing wavelength solution")
            return
        
        self.log.info("Computing wavelength solution")

        self.load_arc1d()
        wave1d = self.arc1d.copy()
        log_lines = []
        wavelength_info = self.wavelength_info.copy()
        
        self.wavesol_stats = []

        for order in range(1, self.n_orders + 1):

            spec = self.arc1d[order-1]

            info = wavelength_info[wavelength_info['order'] == order]
            wl = info['wavelength'].to_numpy().copy()
            px = info['pixel'].to_numpy().copy()
            
            px += self.cfg["wavelength_pixel_offset"]

            if not np.isnan(px).any():
                
                file_out = self.wave_dir / ( self.cfg["fileout_arc"] + f"_n{order:02d}_" )

                w_axis, rms, poly, refined_peak_positions, wl = steles_spec.fit_wavelength_solution(
                    spec, wl, px, order,
                    fit_window=self.cfg["window_fit_peak"],
                    max_fwhm_threshold=self.cfg["max_fwhm_threshold"],
                    wavesolution_ndeg=self.cfg["wavesolution_ndeg"],
                    wavesolution_niter=self.cfg["wavesolution_niter"],
                    wavesolution_rms_threshold=self.cfg["wavesolution_rms_threshold"],
                    wave_peak_threshold=self.cfg["wavesolution_peak_offset_threshold"],
                    show_plots=self.cfg["show_plots"],
                    save_plots=True,
                    plot_directory=self.wave_dir,
                    plot_prefix=file_out,
                    displ_percent_max=99,
                    ref_thar_directory=DB_DIR,
                    ref_thar_file="ThAr_atlas_3000_10600_75k"
                )
                
                # model prediction at peak pixels
                model_wavelength = poly(refined_peak_positions)
                residuals = model_wavelength - wl
                
                # compute dispersion per order
                dispersion = np.abs(np.diff(w_axis))
                disp_med = np.nanmedian(dispersion)
                n_lines =  len(refined_peak_positions)
                # central, min and max wavelength (in A)
                mid_wave = w_axis[int(0.5*len(w_axis))]
                min_wave = np.nanmin(w_axis)
                max_wave = np.nanmax(w_axis)

            else:
                w_axis = np.arange(len(spec))
                rms = np.nan
                residuals = np.array([])
                disp_med = np.nan
                n_lines =  0
                # central, min and max wavelength (in A)
                mid_wave = np.nan
                min_wave = np.nan
                max_wave = np.nan
  
            wave1d[order-1] = w_axis
            log_lines.append((order, rms))
            
            self.wavesol_stats.append({"order": order,
                                       "min_wave": min_wave,
                                       "mid_wave": mid_wave,
                                       "max_wave": max_wave,
                                       "rms": rms,
                                       "dispersion": disp_med,
                                       "n_lines": n_lines,
                                       "residuals": residuals})
        
        
        # show orders with relatively high RMS > 0.5 - should be set by the user?
        bad_orders = [s for s in self.wavesol_stats if s["rms"] > 0.5]
        if len(bad_orders) > 0:
            self.log.warning(f"{len(bad_orders)} orders have large RMS")

        # save the wavelength solution
        self.wave1d = wave1d
        
        self._write_fits(wave1d, None, output, overwrite=True)
        
        self.save_wavelength_qc_table()

    # plots
    def qc_wavelength_rms(self):

        if self._get_bool("dry_run"):
            self.log.info("[DRY RUN] Would execute: qc_wavelength_rms()")
            return

        if (not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False)):
            self.log.info("Skipping wavelength RMS QC plot (disabled)")
            return

        stats, output_path = steles_qc.qc_wavelength_rms(wavesol_stats=self.wavesol_stats,
                                                         output_dir=self.plot_dir,
                                                         output_name=self.cfg["fileout_arc"],
                                                         save_plot=self.cfg.get("save_plots", True),
                                                         show_plot=self.cfg.get("show_plots", False),
                                                         figsize=(10, 4),
                                                         nbins=20,)
        
        self.wavelength_rms_qc = stats
        self.qa_metrics["wavelength_rms"] = stats
        

        if output_path is not None:
            self.log.info(f"Saved RMS histogram: {output_path}")
        
    def qc_line_residuals(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_line_residuals()")
            return

        if not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False):
            self.log.info("Skipping line residual QC plot (disabled)")
            return

        stats, output_path = steles_qc.qc_line_residuals(wavesol_stats=self.wavesol_stats,
                                                         n_orders=self.n_orders,
                                                         output_dir=self.plot_dir,
                                                         output_name=self.cfg["fileout_arc"],
                                                         save_plot=self.cfg.get("save_plots", True),
                                                         show_plot=self.cfg.get("show_plots", False))

        self.line_residual_qc = stats
        self.qa_metrics["line_residuals"] = stats
        
        if output_path is not None:
            self.log.info(f"Saved residuals plot: {output_path}")  

    def qc_dispersion_stability(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_dispersion_stability()")
            return

        if not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False):
            self.log.info("Skipping dispersion QC plot (disabled)")
            return

        stats, output_path = steles_qc.qc_dispersion_stability(wavesol_stats=self.wavesol_stats,
                                                               n_orders=self.n_orders,
                                                               output_dir=self.plot_dir,
                                                               output_name=self.cfg["fileout_arc"],
                                                               save_plot=self.cfg.get("save_plots", True),
                                                               show_plot=self.cfg.get("show_plots", False))

        self.dispersion_qc = stats
        self.qa_metrics["dispersion"] = stats
        
        if output_path is not None:
            self.log.info(f"Saved dispersion plot: {output_path}")
            
    def qc_rms_over_dispersion(self, velocity_scale=True):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_rms_over_dispersion()")
            return

        if not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False):
            self.log.info("Skipping RMS/dispersion QC plot (disabled)")
            return

        stats, output_path = steles_qc.qc_rms_over_dispersion(wavesol_stats=self.wavesol_stats,
                                                              n_orders=self.n_orders,
                                                              output_dir=self.plot_dir,
                                                              output_name=self.cfg["fileout_arc"],
                                                              velocity_scale=velocity_scale,
                                                              save_plot=self.cfg.get("save_plots", True),
                                                              show_plot=self.cfg.get("show_plots", False))

        self.rms_dispersion_qc = stats
        self.qa_metrics["rms_dispersion"] = stats

        if output_path is not None:
            self.log.info(f"Saved RMS/dispersion plot: {output_path}")       
        
    #=========================================================
    # 4. Fit Spatial Distortion
    #========================================================= 

    def _save_polyfits_csv(self, ThAr_peak_pixels, polyfits_x_along_y, output_csv, write_ThAr_peak_pixels=False): #, show_messages=False):

        """
        Save Polynomial1D coefficients to CSV.
        Handles missing/invalid fits by filtering them out.
        """

        # --- Ensure array ---
        polyfits = np.array(polyfits_x_along_y)

        # =====================================================
        # Case 1: 2D (multiple spatial rows × spectral lines)
        # =====================================================
        if polyfits.ndim > 1:

            valid_cols = []

            # --- detect valid spectral lines ---
            for j in range(polyfits.shape[1]):
                try:
                    coeffs = polyfits[0, j].parameters
                    if not np.isnan(coeffs).any():
                        valid_cols.append(j)
                except:
                    continue

            if len(valid_cols) == 0:
                raise RuntimeError("No valid spatial distortion fits to save.")

            # --- filter ---
            pixel_array = ThAr_peak_pixels[valid_cols]
            polyfits = polyfits[:, valid_cols]

            # --- allocate string array ---
            string_array = np.empty(polyfits.shape, dtype=object)

            for i in range(polyfits.shape[0]):   # spatial
                for j in range(polyfits.shape[1]):  # spectral lines
                    coeffs = polyfits[i, j].parameters
                    string_array[i, j] = np.array2string(coeffs, separator=',')

            if write_ThAr_peak_pixels:
                # --- reshape pixel row properly ---
                pixel_array = pixel_array.reshape(1, -1)

                # --- stack ---
                string_array = np.vstack((pixel_array, string_array))

        # =====================================================
        # Case 2: 1D (single fit)
        # =====================================================
        else:

            valid_idx = []

            for i, p in enumerate(polyfits):
                try:
                    if not np.isnan(p.parameters).any():
                        valid_idx.append(i)
                except:
                    continue

            if len(valid_idx) == 0:
                raise RuntimeError("No valid polynomial fits to save.")

            polyfits = polyfits[valid_idx]

            string_array = np.empty(len(polyfits), dtype=object)

            for i in range(len(polyfits)):
                coeffs = polyfits[i].parameters
                string_array[i] = np.array2string(coeffs, separator=',')

            if write_ThAr_peak_pixels:
                pixel_array = np.array([1])
                string_array = np.concatenate((pixel_array, string_array))

        # Save
        np.savetxt(output_csv, string_array, fmt='%s', delimiter=';')

    def _write_spatial_distortion_polynomials(self, order, px, polyfits):
        
        # --- Write output ---
        output_csv = self.dist_dir / ( self.cfg["fileout_arc"] + f"_n{order:02d}_spatdist_polynomials{CSV_EXTENSION}")
        
        # store the distortion coefficients on the database directory
        self._save_polyfits_csv(px, polyfits, output_csv, write_ThAr_peak_pixels=True) #, show_messages=False)
        
    def fit_spatial_distortion(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: fit_spatial_distortion()")
            return

        self.log.info("Fitting spatial distortion")

        wavelength_info = self.wavelength_info.copy()

        for order in range(1, self.n_orders + 1):
            self.log.info(f"[{order}/{self.n_orders}] Fitting spatial distortion")

            data = self.arc_cube[order-1]

            # Skip spatial distortion if there are insufficient spatial rows
            min_rows = max(8, 2 * self.cfg["ndeg_distortion"] + 2)

            if data.shape[0] < min_rows:
                self.log.warning(f"Order {order:02d}: only {data.shape[0]} spatial rows available "
                                 f"(minimum {min_rows}). Skipping spatial distortion correction.")

                # Remove any previous solution to avoid applying a stale correction
                output_csv = self.dist_dir / ( self.cfg["fileout_arc"] + f"_n{order:02d}_spatdist_polynomials{CSV_EXTENSION}")
                if output_csv.exists():
                    output_csv.unlink()
                continue
                
            info = wavelength_info[wavelength_info['order'] == order]
            px = info['pixel'].to_numpy().copy()

            if np.isnan(px).any():
                continue
                
            # correct pixel offset
            px += self.cfg["wavelength_pixel_offset"]

            file_out = self.dist_dir / ( self.cfg["fileout_arc"] + f"_n{order:02d}_spatdist{PLOT_EXTENSION}" )
            
            return_average_distortion=True  # use "average" fitting considering all features.

            spat_rows, polyfits = steles_spec.fit_spatial_distortion(data,
                                                                     px,
                                                                     order,
                                                                     self.channel,
                                                                     window=30,
                                                                     n_pix_step=1,
                                                                     ndeg=self.cfg["ndeg_distortion"],
                                                                     fitting_sigma_threshold=self.cfg["fitting_sigma_threshold"],
                                                                     fitting_niter=self.cfg["fitting_niter"],
                                                                     saturation_threshold=self.cfg["saturation_threshold_for_spatial_distortion"],
                                                                     sigma_threshold = self.cfg["distortion_sigma_init"],
                                                                     sigma_min       = self.cfg["distortion_sigma_min"],
                                                                     sigma_step      = self.cfg["distortion_sigma_step"],
                                                                     single_fit=return_average_distortion,
                                                                     n_col=6,
                                                                     output=file_out,
                                                                     show_plots=self.cfg["show_plots"])

            # if the fitting was successful, save csv and apply correction
            if ( ~np.isnan(polyfits[0][0].c0.value).any() ):
                self._write_spatial_distortion_polynomials(order, px, polyfits)
            else:
                self.log.info(f"No polynomials fitted for order n={order}, no csv file saved.")

    def _read_spatial_distortion_polynomials(self, order):
        
        # --- Write output ---
        output_csv = self.dist_dir / ( self.cfg["fileout_arc"] + f"_n{order:02d}_spatdist_polynomials{CSV_EXTENSION}" )

        if os.path.isfile(output_csv):
            px, polyfits = steles_spec.read_polyfits_csv(output_csv)
        else:
            self.log.info(f"No csv file to load for order n={order}")
            px, polyfits = np.nan, np.nan
        
        return px, polyfits
        
    #=========================================================
    # 5. Apply Wavelength and Distortion corrections
    #========================================================= 

    def _collapse_spectrum2d(self, image, row_range=None, return_average=False, compute_noise_background=False):
        
        # read a 3d image, collapse the spatial pixels and returns a 2d frame
        n_orders, n_spat, n_spec = image.shape
        image_out = np.empty((n_orders, n_spec))
        noise_out = image_out.copy()
        
        # start the loop
        for order in np.arange(1,n_orders+1):
            if row_range is None:
                flux_per_order   = np.sum(image[order-1,:,:], axis=0)
                if compute_noise_background:
                    bg_per_order = stats.mode(image[order-1,:,:], axis=0)[0][0] # scipy.stats.mode[0][0] to access the mode values.
                n_rows = n_spat
            else:
                flux_per_order   = np.sum(image[order-1,row_range[0]:row_range[1],:], axis=0)
                if compute_noise_background:
                    bg_per_order = stats.mode(image[order-1,row_range[0]:row_range[1],:], axis=0)[0][0]
                n_rows = row_range[1]-row_range[0]+1
                
            image_out[order-1,:] = flux_per_order

            if compute_noise_background: # star + sky case
                # noise = sqrt( signal + sky + dark + rdnoise**2 ), sky=mode(flux), dark=0
                noise_per_order = np.sqrt( (        flux_per_order * self.gain ) + 
                                           ( n_rows * bg_per_order * self.gain ) +
                                           ( n_rows * ( self.rdnoise )**2 )
                                         ) / self.gain # result is in ADU
                
            else: # flats and arcs case
                noise_per_order = np.sqrt( (        flux_per_order * self.gain ) + 
                                           ( n_rows * ( self.rdnoise )**2 )
                                         ) / self.gain # result is in ADU

            # add noise per order to the output array
            noise_out[order-1,:] = noise_per_order
            if return_average:
                noise_out[order-1,:] /= n_rows
                    
        if return_average:
            image_out /= n_rows

        return image_out, noise_out

    def _apply_wave_spat_correction(self,
                                    data2d,
                                    data2d_header,
                                    n_orders,
                                    wave1d,
                                    wavelength_info,
                                    spat_rows,
                                    wave_pixel_reference, 
                                    spatdist_file, 
                                    file_out_prefix, 
                                    wavelength_pixel_offset=0,
                                    compute_noise_background=False,
                                    file_suffix="_spatdist", 
                                    show_distortion_correction_plot=False
                                   ):

        """ Wrapper for applying the wavelength solution and spatial distortion correct the data in a single shot! """

        # create new array for saving the corrected data
        data2d_distcor = np.empty(data2d.shape)

        #file_distcor_2d_out = file_out_prefix.replace(FITS_EXTENSION, file_suffix +         FITS_EXTENSION)
        #file_distcor_1d_out = file_out_prefix.replace(FITS_EXTENSION, file_suffix + "_1d" + FITS_EXTENSION)
        
        file_distcor_2d_out = file_out_prefix.with_name(file_out_prefix.stem + file_suffix +         file_out_prefix.suffix)
        file_distcor_1d_out = file_out_prefix.with_name(file_out_prefix.stem + file_suffix + "_1d" + file_out_prefix.suffix)
        
        # start the loop
        spat_corrected_any = False
        for order in np.arange(1,n_orders+1):

            #print(f"Working on order={order}...")

            # now extract the data
            spectrum = data2d[order-1,:]

            # get information for wavelength solution 
            wave_info_order = wavelength_info[wavelength_info['order'] == order]

            # get central wavelength of the lines and the corresponding pixel position
            ThAr_peak_pixels = wave_info_order['pixel'].to_numpy().copy()
            ThAr_peak_pixels += wavelength_pixel_offset # add a constant offset to match the pixel position

            # store the distortion coefficients on the database directory

            # 6.3 apply spatial distortion correction
            px, polyfits = self._read_spatial_distortion_polynomials(order)

            if show_distortion_correction_plot is True:
                show_spatdist_plot = True if order == 1 else False
            else:
                show_spatdist_plot = False
            
            # use the first pixels and middle pixel to compare the spectra
            wave_pix_ref = [2,int(spectrum.shape[0]*0.5),spectrum.shape[0]-3]
            
            if np.isfinite(px):
                
                spec_corrected = steles_spec.correct_spatial_distortion(spectrum, spat_rows, polyfits, ThAr_peak_pixels,
                                                                        order, y_ref=None, show_plot=show_spatdist_plot, 
                                                                        figsize=(20,12), wave_pixel_reference = wave_pix_ref,
                                                                        max_displ_percent=99, output=None)
                data2d_distcor[order-1,:,:] = spec_corrected
                
                spat_corrected_any = True
                
            else:
                print(f"No spatial correction solution for order n={order:02d}.")
                data2d_distcor[order-1,:,:] = spectrum

        # add header keywords - 2d data
        data2d_distcor_header = data2d_header.copy()
        
        if spat_corrected_any:
            data2d_distcor_header.set("ST_SPATD", value=True, comment="Spatial distortion corrected")
            data2d_distcor_header.set("ST_FSPAT", value=spatdist_file, comment="Ref file for spatial distortion")
        else:
            data2d_distcor_header.set("ST_SPATD", value=False, comment="Spatial distortion corrected")
            data2d_distcor_header.set("ST_FSPAT", value="NaN", comment="Ref file for spatial distortion")

        # save the extracted orders after correcting for spatial distortion
        self._write_fits(data2d_distcor, data2d_distcor_header, file_distcor_2d_out, overwrite=True)

        # add header keywords - 1d data
        data1d_distcor_header = data2d_distcor_header.copy()
        data1d_distcor_header.set("ST_EXT1", value="FluxInt",    comment="Extension 1")
        data1d_distcor_header.set("ST_EXT2", value="Wavelength", comment="Extension 2")
        data1d_distcor_header.set("ST_EXT3", value="Noise",      comment="Extension 3")
        # sum the flux along the slit to create a 1d spectrum
        data1d_distcor, noise1d = self._collapse_spectrum2d(data2d_distcor, row_range=None, return_average=False, 
                                                            compute_noise_background=compute_noise_background)
        
        # save the extracted orders after correcting for spatial distortion
        data1d_distcor = np.asarray([data1d_distcor,wave1d,noise1d], dtype=np.float64)
        
        self._write_fits(data1d_distcor, data1d_distcor_header, file_distcor_1d_out, overwrite=True)        
        
    def apply_calibration(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: apply_calibration()")
            return

        self.log.info("Applying wavelength + distortion solution")

        spat_rows = np.arange(self.cfg["n_pix_step"],
                              self.n_spat - self.cfg["n_pix_step"],
                              self.cfg["n_pix_step"])

        datasets = [("flat", self.cfg["fileout_flat"]),
                    ("arc",  self.cfg["fileout_arc"]),
                    ("sci",  self.cfg["fileout_sci"])]

        for kind, name in datasets:

            self.log.info(f"Applying calibration to {kind}")

            #file_in = os.path.join(self.proc_dir, name + "_rectified.fits")
            file_in = self.proc_dir / ( name + f"_rectified{FITS_EXTENSION}" )
            
            with fits.open(file_in) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            show_distortion_correction_plot = True if kind == "arc" else False
            
            self._apply_wave_spat_correction(data,
                                             header,
                                             self.n_orders,
                                             self.wave1d,
                                             self.wavelength_info,
                                             spat_rows,
                                             self.cfg["wave_pixel_reference"],
                                             self.cfg["fileout_arc"],
                                             file_in,
                                             wavelength_pixel_offset=self.cfg["wavelength_pixel_offset"],
                                             compute_noise_background=False,
                                             file_suffix="_spatdist",
                                             show_distortion_correction_plot=show_distortion_correction_plot)


    def apply_calibration_solution(self, science_only=False):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: apply_calibration_solution()")
            return

        self.log.info("Applying wavelength + distortion solution")

        spat_rows = np.arange(self.cfg["n_pix_step"],
                              self.n_spat - self.cfg["n_pix_step"],
                              self.cfg["n_pix_step"])
        if science_only is False:
            datasets = [("flat", self.cfg["fileout_flat"]),
                        ("arc",  self.cfg["fileout_arc"]),
                        ("sci",  self.cfg["fileout_sci"])]
        else:
            datasets = [("sci",  self.cfg["fileout_sci"])]

        for kind, name in datasets:

            self.log.info(f"Applying calibration to {kind}")

            file_name = name + "_rectified" # no ".fits"

            #file_in = os.path.join(self.proc_dir, file_name + ".fits")
            file_in = self.proc_dir / ( file_name + FITS_EXTENSION)
            
            with fits.open(file_in) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            compute_noise_background = True if kind == "sci" else False # compute bg for sci; flats+arcs fill the entire slit 
            dq_dir = self.plot_dir          if kind == "arc" else None  # only create DQ plot for arcs
            
            show_distortion_correction_plot = True if kind == "arc" else False
            
            self._apply_wave_spat_correction(data,
                                             header,
                                             self.n_orders,
                                             self.wave1d,
                                             self.wavelength_info,
                                             spat_rows,
                                             self.cfg["wave_pixel_reference"],
                                             self.cfg["fileout_arc"],
                                             file_in,
                                             wavelength_pixel_offset=self.cfg["wavelength_pixel_offset"],
                                             compute_noise_background=compute_noise_background,
                                             file_suffix="_spatdist",
                                             show_distortion_correction_plot=show_distortion_correction_plot)
                    
    #=========================================================
    # 6. Blaze Function
    #========================================================= 

    def compute_blaze_function(self):   
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: compute_blaze_function()")
            return
        
        #output = os.path.join(self.proc_dir, self.cfg["fileout_flat"] + "_blaze_function.fits")
        output = self.proc_dir / ( self.cfg["fileout_flat"] + F"_blaze_function{FITS_EXTENSION}" )
        
        
        if self._check_done([output]) and not self.cfg["overwrite"]:
            self.log.info("Skipping computing blaze function")
            return
        
        self.log.info("Computing blaze function")
        
        if not hasattr(self, "wave1d") or self.wave1d is None:
            self.log.info("Loading wavelength solution")
            self.load_wavelength_solution() 

        #flat_file = os.path.join(self.proc_dir, self.cfg["fileout_flat"] + "_rectified_spatdist_1d.fits")
        flat_file = self.proc_dir / ( self.cfg["fileout_flat"] + f"_rectified_spatdist_1d{FITS_EXTENSION}" )

        blaze = steles_spec.fit_quartz(flat_file,
                                       nmin=self.cfg["blaze_min_deg"],
                                       nmax=self.cfg["blaze_max_deg"],
                                       sigma_clip_thresh=self.cfg["blaze_sigma"],
                                       n_cols = self.cfg["blaze_ncols"],
                                       output_blaze =  self.plot_dir / ( self.cfg["fileout_flat"] + f"_blaze_function{PLOT_EXTENSION}" )
                                      )
        
        self._write_fits(np.asarray([blaze, self.wave1d], dtype=np.float64), None, output, overwrite=True)
          
    def apply_blaze(self,
                    file_in,
                    blaze_orders,
                    smooth_blaze=True,
                    smooth_window=31,
                    smooth_poly=3,
                    epsilon=1e-6,
                    renormalize=True):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: apply_blaze()")
            return

        # --- Load data ---
        try:
            with fits.open(file_in) as hdul:
                data = hdul[0].data
        except:
            data = file_in

        # --- Shape handling ---
        if data.ndim == 2:
            n_orders, nx = data.shape
            ny = None
        elif data.ndim == 3:
            n_orders, ny, nx = data.shape
        else:
            raise ValueError("Data format is not allowed.")

        data_corr = data.copy()

        xarr = np.arange(nx)

        # --- Loop over orders ---
        for n in range(n_orders):

            blaze = blaze_orders[0,n,:].copy()
            
            # --------------------------------------------------
            # 1) Normalize blaze (CRITICAL)
            # --------------------------------------------------
            med = np.nanmedian(blaze)
            if med == 0 or np.isnan(med):
                raise RuntimeError(f"Invalid blaze normalization for order {n}")
            blaze /= med

            # --------------------------------------------------
            # 2) Smooth blaze (optional but recommended)
            # --------------------------------------------------
            if smooth_blaze:
                try:

                    # Ensure window is valid (odd and < nx)
                    win = min(smooth_window, nx - 1 if nx % 2 == 0 else nx)
                    if win % 2 == 0:
                        win -= 1

                    if win > smooth_poly:
                        blaze = savgol_filter(blaze, win, smooth_poly)

                #except Exception:
                #    pass  # fail silently → safe fallback
                except Exception as e:
                    self.log.warning(f"Blaze smoothing failed: {e}")

            # --------------------------------------------------
            # 3) Avoid division by small values
            # --------------------------------------------------
            blaze_safe = blaze.copy()
            blaze_safe[blaze_safe < epsilon] = np.nan

            # --------------------------------------------------
            # 4) Apply correction
            # --------------------------------------------------
            if data.ndim == 2:

                yvals = data[n, :]
                ycorr = yvals / blaze_safe

                # --------------------------------------------------
                # 5) Renormalize per order (stabilizes flux scale)
                # --------------------------------------------------
                if renormalize:
                    med_corr = np.nanmedian(ycorr)
                    if med_corr > 0 and not np.isnan(med_corr):
                        ycorr /= med_corr

                data_corr[n, :] = ycorr

            else:
                for y in range(ny):

                    yvals = data[n, y, :]
                    ycorr = yvals / blaze_safe

                    if renormalize:
                        med_corr = np.nanmedian(ycorr)
                        if med_corr > 0 and not np.isnan(med_corr):
                            ycorr /= med_corr

                    data_corr[n, y, :] = ycorr

        return data_corr        

    def apply_blaze_continuum(self,
        data_in,
        blaze_orders,
        smooth_blaze=True,
        smooth_window=31,
        smooth_poly=3,
        epsilon=1e-6,
        fit_method="spline",   # "poly" or "spline"
        poly_degree=5,
        spline_smooth=0.001,
        sigma_clip=3.0,
        n_iter=3
    ):

        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: apply_blaze_continuum()")
            return
        
        # --- Load data ---
        if isinstance(data_in, str):
            with fits.open(data_in) as hdul:
                data = hdul[0].data
        else:
            data = data_in

        if data.ndim == 2:
            n_orders, nx = data.shape
            ny = None
        elif data.ndim == 3:
            n_orders, ny, nx = data.shape
        else:
            raise ValueError("Invalid data format")

        data_corr = data.copy()
        x = np.arange(nx)

        # =====================================================
        # Helper: continuum fitting with sigma clipping
        # =====================================================
        def fit_continuum(y):

            mask = np.isfinite(y)

            for _ in range(n_iter):

                if np.sum(mask) < 10:
                    break

                if fit_method == "poly":
                    coeffs = np.polyfit(x[mask], y[mask], poly_degree)
                    model = np.poly1d(coeffs)(x)

                elif fit_method == "spline":
                    spline = UnivariateSpline(
                        x[mask], y[mask],
                        s=spline_smooth * len(x)
                    )
                    model = spline(x)

                else:
                    raise ValueError("fit_method must be 'poly' or 'spline'")

                resid = y - model
                std = np.nanstd(resid[mask])

                new_mask = np.abs(resid) < sigma_clip * std

                mask = mask & new_mask

            return model

        # =====================================================
        # Main loop
        # =====================================================
        for n in range(n_orders):

            blaze = blaze_orders[0,n,:].copy()

            # --- Normalize blaze ---
            med = np.nanmedian(blaze)
            if not np.isfinite(med) or med == 0:
                self.log.warning(f"Invalid blaze in order {n}, skipping normalization")
                continue
            blaze /= med

            # --- Smooth blaze ---
            if smooth_blaze:
                try:
                    win = min(smooth_window, nx - 1 if nx % 2 == 0 else nx)
                    if win % 2 == 0:
                        win -= 1
                    if win > smooth_poly:
                        blaze = savgol_filter(blaze, win, smooth_poly)
                #except:
                #    pass
                except Exception as e:
                    self.log.warning(f"Blaze smoothing failed: {e}")

            # --- Safe blaze ---
            blaze_safe = blaze.copy()
            blaze_safe[blaze_safe < epsilon] = np.nan

            # --------------------------------------------------
            # Apply correction
            # --------------------------------------------------
            if data.ndim == 2:

                y = data[n]
                y_corr = y / blaze_safe

                # --- Continuum fit ---
                cont = fit_continuum(y_corr)

                # --- Final correction ---
                y_final = y_corr / cont

                data_corr[n] = y_final

            else:
                for j in range(ny):

                    y = data[n, j, :]
                    y_corr = y / blaze_safe

                    cont = fit_continuum(y_corr)

                    y_final = y_corr / cont

                    data_corr[n, j, :] = y_final

        return data_corr    
    
    #=========================================================
    # 7. Final Extraction
    #========================================================= 
    def extract_1d_spectra(self):
        
        if self._get_bool("dry_run"):
            self.log.info(f"[DRY RUN] Would execute: extract_1d_spectra()")
            return
        
        #output = os.path.join(self.proc_dir, self.cfg["fileout_sci"] + "_final_1d.fits")
        output = self.proc_dir / ( self.cfg["fileout_sci"] + f"_final_1d{FITS_EXTENSION}" )
        
        if self._check_done([output]) and not self.cfg["overwrite"]:
            self.log.info("Skipping computing blaze function")
            return

        self.log.info("Extracting final spectra")

        if not hasattr(self, "blaze1d") or self.blaze1d is None:
            self.log.info("Loading blaze solution")
            self.load_blaze_solution() 
        
        #sci_file = os.path.join(self.proc_dir, self.cfg["fileout_sci"] + "_rectified_spatdist_1d.fits")
        sci_file = self.proc_dir / ( self.cfg["fileout_sci"] + f"_rectified_spatdist_1d{FITS_EXTENSION}")

        with fits.open(sci_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        flux  = data[self.ext_flux]
        wave  = data[self.ext_wave]
        noise = data[self.ext_noise]

        # use the blaze + continuum normalization option or not
        if self.cfg.get("blaze_mode", "simple") == "simple":
            flux_corr = self.apply_blaze(flux,
                                         self.blaze1d,
                                         smooth_blaze=True,
                                         smooth_window=31,
                                         smooth_poly=3,
                                         epsilon=1e-6,
                                         renormalize=True)
        else:
            flux_corr = self.apply_blaze_continuum(flux, 
                                                   self.blaze1d,
                                                   smooth_blaze=True,
                                                   smooth_window=31,
                                                   smooth_poly=3,
                                                   epsilon=1e-6,
                                                   fit_method="spline",   # "poly" or "spline"
                                                   poly_degree=5,
                                                   spline_smooth=0.001,
                                                   sigma_clip=3.0,
                                                   n_iter=3
                                                  )

        out_data = np.asarray([flux, wave, noise, self.blaze1d[0,:,:], flux_corr], dtype=np.float64)
        
        header_out = header.copy()
        # added extensions
        header_out.set("ST_EXT4", value="Blaze",      comment="Extension 4")
        header_out.set("ST_EXT5", value="FluxBlaze",  comment="Extension 5")
        header_out.set("ST_FINAL", value=True,  comment="End of processing")

        #save fits file
        self._write_fits(out_data, header_out, output, overwrite=True)

    def qc_extracted_products(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_extracted_products()")
            return

        if not self.cfg.get("save_plots", True) and not self.cfg.get("show_plots", False):
            self.log.info("Skipping extracted-product QC plot (disabled)")
            return

        if not hasattr(self, "sci_extracted") or self.sci_extracted is None:
            self.log.info("Loading extracted spectrum")
            self.load_extracted_products()

        stats, output_path = steles_qc.qc_extracted_products(sci_extracted=self.sci_extracted,
                                                             sci_header=self.sci_extracted_header,
                                                             channel=self.channel,
                                                             ext_flux=self.ext_flux,
                                                             ext_wave=self.ext_wave,
                                                             ext_noise=self.ext_noise,
                                                             ext_blaze=self.ext_blaze,
                                                             ext_blaze_flux=self.ext_blaze_flux,
                                                             output_dir=self.plot_dir,
                                                             output_name=self.cfg["fileout_sci"],
                                                             save_plot=self.cfg.get("save_plots", True),
                                                             show_plot=self.cfg.get("show_plots", False))

        self.snr_qc = stats
        self.qa_metrics["signal_to_noise"] = stats
        
        if output_path is not None:
            self.log.info(f"Saved extracted-product QC plot: {output_path}")

             
    def qc_blaze(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_blaze()")
            return

        if not hasattr(self, "sci_extracted") or self.sci_extracted is None:
            self.log.info("Loading extracted products")
            self.load_extracted_products()

        stats, output_paths = steles_qc.qc_blaze(sci_extracted=self.sci_extracted,
                                                 ext_flux=self.ext_flux,
                                                 ext_wave=self.ext_wave,
                                                 ext_blaze=self.ext_blaze,
                                                 n_panels=self.cfg["qc_blaze_n_panels"],
                                                 output_dir=self.plot_dir,
                                                 output_name=self.cfg["fileout_sci"],
                                                 save_plot=self.cfg.get("save_plots", True),
                                                 show_plot=self.cfg.get("show_plots", False))

        self.blaze_qc = stats
        
        self.qa_metrics["blaze"] = stats

        for output_path in output_paths:
            self.log.info(f"Saved blaze QC plot: {output_path}")

    def qc_snr(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_snr()")
            return

        if not hasattr(self, "sci_extracted") or self.sci_extracted is None:
            self.log.info("Loading extracted products")
            self.load_extracted_products()

        stats, output_paths = steles_qc.qc_snr(sci_extracted=self.sci_extracted,
                                               ext_flux=self.ext_flux,
                                               ext_wave=self.ext_wave,
                                               ext_noise=self.ext_noise,
                                               n_orders=self.n_orders,
                                               n_panels=self.cfg["qc_snr_n_panels"],
                                               output_dir=self.plot_dir,
                                               output_name=self.cfg["fileout_sci"],
                                               save_plot=self.cfg.get("save_plots", True),
                                               show_plot=self.cfg.get("show_plots", False))

        self.snr_qc = {"n_orders": stats["n_orders"],
                       "n_valid_orders": stats["n_valid_orders"]}

        self.snr_qc_stats = stats["snr_stats"]
        
        self.qa_metrics["snr"] = stats

        for output_path in output_paths:
            self.log.info(f"Saved SNR QC plot: {output_path}")           
            
    def qc_final_1d_spectra(self):

        if self._get_bool("dry_run"):
            self.log.info("Would execute: qc_final_1d_spectra()")
            return

        if not hasattr(self, "sci_extracted") or self.sci_extracted is None:
            self.log.info("Loading extracted products")
            self.load_extracted_products()

        stats, output_paths = steles_qc.qc_final_1d_spectra(sci_extracted=self.sci_extracted,
                                                            ext_wave=self.ext_wave,
                                                            ext_blaze_flux=self.ext_blaze_flux,
                                                            n_orders=self.n_orders,
                                                            n_panels=self.cfg["qc_final1d_n_panels"],
                                                            output_dir=self.plot_dir,
                                                            output_name=self.cfg["fileout_sci"],
                                                            save_plot=self.cfg.get("save_plots", True),
                                                            show_plot=self.cfg.get("show_plots", False))

        self.final1d_qc = {"n_orders": stats["n_orders"],
                           "n_valid_orders": stats["n_valid_orders"],
                           "n_panels": stats["n_panels"]}
        self.final1d_qc_stats = stats["flux_stats"]
        self.qa_metrics["final1d"] = stats

        for output_path in output_paths:
            self.log.info(f"Saved final 1D QC plot: {output_path}")       
            
    # =========================================================
    # Report system
    # =========================================================
    
    def save_report_partII(self):
        
        wavelength_stats = [{"order":      s["order"],
                             "min_wave":   s["min_wave"],
                             "mid_wave":   s["mid_wave"],
                             "max_wave":   s["max_wave"],
                             "wave_range": s["max_wave"] - s["min_wave"],
                             "dispersion": s["dispersion"],
                             "rms":        s["rms"],
                             "n_lines":    s["n_lines"]}
                             for s in self.wavesol_stats
                            ]
        
        self.qa_metrics["wavelength_solution"] = wavelength_stats
        
        # Reorder based on a custom list
        desired_order = ["wavelength_solution", "snr", "signal_to_noise", "blaze", "final1d"]

        self.qa_metrics = {key: self.qa_metrics[key] for key in desired_order}

        qc_plots = [self.cfg["fileout_arc"] + f"_qc_rms_hist{PLOT_EXTENSION}",
                    self.cfg["fileout_arc"] + f"_qc_dispersion{PLOT_EXTENSION}",
                    self.cfg["fileout_arc"] + f"_qc_residuals{PLOT_EXTENSION}",
                    self.cfg["fileout_arc"] + f"_qc_rms_over_disp{PLOT_EXTENSION}",
                    self.cfg["fileout_sci"] + f"_final_1d{PLOT_EXTENSION}",
                    self.cfg["fileout_sci"] + f"_qc_snr{PLOT_EXTENSION}",
                   ]
        
        for i in range(1, self.cfg["qc_blaze_n_panels"]):
            qc_plots.append(f"{self.cfg['fileout_sci']}_blaze_qc_panel{i}{PLOT_EXTENSION}")
            
        self.save_report(report_name="PartII",
                         #outputs={"wavelength_solution": os.path.join( self.proc_dir, self.cfg["fileout_arc"] + "_wavelength_solution.fits"),
                         #         "blaze_solution": os.path.join( self.proc_dir, self.cfg["fileout_flat"] + "_blaze_function.fits"),
                         #         "final_spectrum": os.path.join(self.proc_dir, self.cfg["fileout_sci"] + "_final_1d.fits")},
                         outputs={"wavelength_solution": self.proc_dir / ( self.cfg["fileout_arc"]  + f"_wavelength_solution{FITS_EXTENSION}"),
                                  "blaze_solution":      self.proc_dir / ( self.cfg["fileout_flat"] + f"_blaze_function{FITS_EXTENSION}"),
                                  "final_spectrum":      self.proc_dir / ( self.cfg["fileout_sci"]  + f"_final_1d{FITS_EXTENSION}")},
                         qa_metrics=self.qa_metrics,
                         summary={"n_orders": self.n_orders,
                                  "status": "SUCCESS"},
                         qc_plots=qc_plots)        
        
        
    #=========================================================
    # 8. Run controller
    #========================================================= 
    def run(self):
        
        self.load_rectified_products()
        self.create_arc1d()
        #
        self._timeit(self.wavelength_solution, "Evaluating wavelength solution")
        self.qc_wavelength_rms()
        self.qc_dispersion_stability()
        self.qc_line_residuals()
        self.qc_rms_over_dispersion()
        #
        self.load_wavelength_solution()
        self._timeit(self.fit_spatial_distortion, "Evaluating spatial distortion solution")
        #    
        self.apply_calibration()
        #
        self._timeit(self.compute_blaze_function, "Evaluating blaze function")
        # 
        self._timeit(self.extract_1d_spectra, "Extracting one-dimensional spectrum")
        self.qc_extracted_products()
        self.qc_blaze()
        self.qc_snr()
        self.qc_final_1d_spectra()
        #
        self.save_report_partII()
        #
        self.log.info("Part II completed successfully") 
        
    def run_wavesol(self):
        # do not run blaze fitting but apply blaze correction
        
        self.load_rectified_products()
        self.create_arc1d()
        #
        self._timeit(self.wavelength_solution, "Evaluating wavelength solution")
        self.qc_wavelength_rms()
        self.qc_dispersion_stability()
        self.qc_line_residuals()
        self.qc_rms_over_dispersion()
        #   
        self.load_rectified_products()
        self.load_wavelength_solution()
        self._timeit(self.fit_spatial_distortion, "Evaluating spatial distortion solution")
        #    
        self.apply_calibration()
        #
        self._timeit(self.extract_1d_spectra, "Extracting one-dimensional spectrum")
        self.qc_extracted_products()
        self.qc_blaze()
        self.qc_snr()
        self.qc_final_1d_spectra()
        #  
        self.save_report_partII()
        #
        self.log.info("Part II completed successfully") 
      

    def run_distortion(self):
        # do not run blaze fitting but apply blaze correction
        
        self.load_rectified_products()
        self.load_wavelength_solution()
        self._timeit(self.fit_spatial_distortion, "Evaluating spatial distortion solution")
        #    
        self.apply_calibration()
        #
        self._timeit(self.extract_1d_spectra, "Extracting one-dimensional spectrum")
        self.qc_extracted_products()
        self.qc_blaze()
        self.qc_snr()
        self.qc_final_1d_spectra()
        #  
        self.save_report_partII()
        #
        self.log.info("Part II completed successfully") 
      
    def run_blaze(self):
        
        self.load_rectified_products()
        self.load_wavelength_solution()
        self.load_wavelength_solution()
        self.apply_calibration()
        #
        self._timeit(self.compute_blaze_function, "Evaluating blaze function")
        # 
        self._timeit(self.extract_1d_spectra, "Extracting one-dimensional spectrum")
        self.qc_extracted_products()
        self.qc_blaze()
        self.qc_snr()
        self.qc_final_1d_spectra()
        #
        self.save_report_partII()
        #
        self.log.info("Part II completed successfully") 

    def run_calibrate_science(self):
        # apply wavelength solution, spatial distortion and blaze correction
        
        self.load_rectified_products()
        self.load_wavelength_solution()
        #
        self.apply_calibration_solution(science_only=True)
        #
        self.extract_1d_spectra()
        self.qc_extracted_products()
        self.qc_blaze()
        self.qc_snr()
        self.qc_final_1d_spectra()
        #
        self.save_report_partII()
        #
        self.log.info("Part II completed successfully") 