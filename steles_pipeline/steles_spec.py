# steles_spec.py
from .steles_imports import *

from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
import numpy as np

# wavelength solution
def collapse_spectrum2d(image, row_range=None, return_average=False):
    # read a 3d image, collapse the spatial pixels and returns a 2d frame
    n_orders, n_spat, n_spec = image.shape
    image_out = np.empty((n_orders, n_spec))
    # start the loop
    for order in np.arange(1,n_orders+1):
        if row_range is None:
            image_out[order-1,:] = np.sum(image[order-1,:,:], axis=0)
            n_rows = n_spat
        else:
            image_out[order-1,:] = np.sum(image[order-1,row_range[0]:row_range[1],:], axis=0)
            n_rows = row_range[1]-row_range[0]+1
            
    if return_average:
        image_out /= n_rows
        
    return image_out


# wavelength calibration functions
def gaussian_kernel(x, amp, mu, sigma, offset):
    """ Gaussian kernel to search for arc lines """
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

def refine_peak_positions(spectrum, initial_guesses, window=10, show_plots=False, n_col=None, output=None):
    """
    Refine peak positions from initial pixel guesses by fitting Gaussians.
    
    Parameters:
        spectrum (np.ndarray): 1D array of arc lamp flux.
        initial_guesses (list): Approximate pixel locations of arc peaks.
        window (int): Half-width of region to fit around each guess.
        show_plots (bool): If True, plots the fitted region.

    Returns:
        refined_positions (np.ndarray): Subpixel centroid positions of each arc line.
    """
    refined_positions = []
    peak_fluxes = []
    
    n_peaks = len(initial_guesses)
    if n_col is None:
        n_col = n_peaks if n_peaks <= 4 else 4
    n_row = int(n_peaks / n_col)
    if n_col * n_row < n_peaks:
        n_row += 1
    
    plt.figure(figsize=(n_col * 4, n_row * 3))
    
    for n, guess in enumerate(initial_guesses):
        
        guess = int(guess) # force int
        
        # Define local fitting window        
        #x_fit = np.arange(guess - window, guess + window + 1)
        #ensure that x_fit is always within the size of 'spectrum'
        x_fit = np.clip( np.arange(guess - window, guess + window + 1), 0, len(spectrum) - 1)
        y_fit = spectrum[x_fit]

        # Initial parameters: amplitude, center, width, offset
        amp0 = y_fit.max() - y_fit.min()
        mu0 = guess
        sigma0 = 3.0
        offset0 = y_fit.min()

        try:
            popt, _ = curve_fit(gaussian_kernel, x_fit, y_fit, p0=[amp0, mu0, sigma0, offset0])
            
            # maybe add a max_fwhm_threshold parameter and exclude negative flux_peak values from the results...
            # the data is already cleaned by rms threshold filtering on the fit_wavelength_solution()
            
            flux_peak  = popt[0]
            refined_mu = popt[1]
            fwhm       = abs(popt[2]) * 2.355
            
            refined_positions.append(refined_mu)
            peak_fluxes.append(flux_peak)

            #if show_plots:
            plt.subplot(n_row,n_col,n+1)
            plt.plot(x_fit, y_fit, 'k.')
            plt.plot(x_fit, gaussian_kernel(x_fit, *popt), 'r-', label='Fit')
            plt.axvline(refined_mu, color='blue', linestyle='--', label=r'x$_{peak}$, FWHM='+f'{refined_mu:.2f}, {fwhm:.2f}')
            plt.title(f'Line near x={guess}')
            plt.grid(True)
            if n == 0:
                plt.ylabel('Flux')
            plt.xlabel('Pixel')
            plt.legend()
                    
        #except RuntimeError:
        except:
            print(f"Fit failed near x={guess}")
            refined_positions.append(np.nan)  # fallback to original guess
            peak_fluxes.append(np.nan)  # fallback to original guess

    plt.suptitle("Refine peak positions for the Arc lamp spectrum",y=1.0)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    return np.array(refined_positions), np.array(peak_fluxes)


def extract_spectrum1d(arc_frame, row_range=(100, 110)):
    """
    Extract 1D spectrum by summing rows across a spatial region.
    
    Parameters:
        arc_frame (CCDData): The 2D arc frame.
        row_range (tuple): Tuple (y1, y2) defining rows to extract.

    Returns:
        1D numpy array: Extracted arc spectrum (flux vs pixel).
    """
    data = arc_frame
    spectrum_1d = np.sum(data[row_range[0]:row_range[1], :], axis=0)
    return spectrum_1d

def wavelength_calibration(x, y, order, x_range=None, ndeg=3, show_plot=True, figsize=(12,5), output=None):

    # start the fitter
    linfitter = LinearLSQFitter()
    spatial_model = Polynomial1D(degree=ndeg)
    fit_wavelength = linfitter(model=spatial_model, x=x, y=y)
    
    wave_xi = fit_wavelength(x)
    
    x_axis = x if x_range is None else np.arange(x_range[0],x_range[1],1)
    wave_x = fit_wavelength(x_axis)
    
    # compute RMS of the fitting
    residuals = y-wave_xi
    rms = np.sqrt(np.sum(abs(residuals)**2)/len(residuals))
    
    plt.figure(figsize=figsize)
        
    plt.subplot(2,1,1)
    plt.plot(x,y,'+',label='Measurements')
    plt.plot(x_axis,wave_x,color='red', alpha=0.5, label=f'Model (deg={ndeg})')
    plt.ylabel(r'Wavelength ($\AA$)')
    plt.xlabel('Dispersion axis (pixel)')
    if x_range is not None:
        plt.xlim(x_range)
    plt.legend()
    # plot residuals
    plt.subplot(2,1,2)
    plt.plot(y,residuals,'+',label=fr'RMS={rms:.6f} $\AA$' )
    plt.axhline(0,ls='--',color='grey')
    plt.legend()
    if x_range is not None:
        plt.xlim(min(wave_x),max(wave_x))
    plt.ylabel(r'Residuals ($\AA$)')
    plt.xlabel(r'Wavelength ($\AA$)')
    # make y-axis centered at 0.
    ymax = np.max(abs(residuals))*1.1
    plt.ylim(-ymax,ymax)
    plt.suptitle(f"Wavelength calibration (n={order:02d})", va='bottom')
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=300)
            
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fit_wavelength, rms

def wavelength_calibration_iter(x, y, order, x_range=None, ndeg=3, n_iter=3,
                                max_residual=None, rms_threshold=3.0,
                                show_plot=True, figsize=(12, 5),
                                output=None, return_mask=False):

    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    if len(x) < ndeg + 1:
        raise RuntimeError(f"Order {order}: only {len(x)} valid calibration lines.")

    mask = np.ones(len(x), dtype=bool)
    fitter = LinearLSQFitter()

    def fit_current():
        x_fit, y_fit = x[mask], y[mask]
        x0, scale = np.mean(x_fit), np.ptp(x_fit)
        if scale == 0:
            raise RuntimeError(f"Order {order}: all calibration pixels are identical.")

        model = fitter(Polynomial1D(degree=ndeg), (x_fit-x0)/scale, y_fit)

        def evaluate(xx):
            return model((np.asarray(xx)-x0)/scale)

        residuals = y - evaluate(x)
        rms = np.sqrt(np.mean(residuals[mask]**2))
        return evaluate, residuals, rms

    # Initial fit sets the rejection threshold.
    fit_model, residuals, rms = fit_current()
    threshold = max_residual if max_residual is not None else rms_threshold*rms

    # Reject at most one worst outlier per iteration.
    for iteration in range(n_iter):
        worst = np.argmax(np.where(mask, np.abs(residuals), -np.inf))
        worst_residual = abs(residuals[worst])

        if worst_residual <= threshold:
            break

        if np.sum(mask) - 1 < ndeg + 1:
            print(f"Order {order}: cannot reject another line; insufficient measurements.")
            break

        print(f"Order {order}: rejecting line at pixel {x[worst]:.2f} "
              f"(residual={residuals[worst]:.4f} A; pass {iteration+1}/{n_iter})")

        mask[worst] = False
        fit_model, residuals, rms = fit_current()

    # Final fit and residuals.
    fit_model, residuals, rms = fit_current()

    x_axis = x if x_range is None else np.arange(x_range[0], x_range[1])
    y_axis = fit_model(x_axis)

    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.plot(x[mask], y[mask], '+', label='Used')
    if np.any(~mask):
        plt.plot(x[~mask], y[~mask], 'rx', ms=10, label='Rejected')
    plt.plot(x_axis, y_axis, 'r-', label=f'Polynomial (deg={ndeg})')
    plt.ylabel(r"Wavelength ($\AA$)")
    if x_range is not None:
        plt.xlim(x_range)
    plt.legend()

    plt.subplot(212)
    plt.plot(y[mask], residuals[mask], '+', label=fr'RMS = {rms:.5f} $\AA$')
    if np.any(~mask):
        plt.plot(y[~mask], residuals[~mask], 'rx')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Residual ($\AA$)")
    ymax = 1.2*np.max(np.abs(residuals))
    plt.ylim(-ymax, ymax)
    plt.legend()
    plt.suptitle(f"Wavelength calibration (Order {order})")
    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_mask:
        return fit_model, rms, mask
    return fit_model, rms

def wavelength_calibration_iter_previous(x, y, order,
                                x_range=None, ndeg=3, n_iter=3, max_residual=None, rms_threshold=3.0,
                                show_plot=True, figsize=(12, 5), output=None, return_mask=False):
    
    "n_inter=3 does not do anything..."

        #max_residual=None,      # e.g. 0.05 or 0.10 Angstrom
        #rms_threshold=4.0,         # used only if max_residual is None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if len(x) < ndeg + 1:
        raise RuntimeError(
            f"Order {order}: only {len(x)} valid calibration lines."
        )

    mask = np.ones(len(x), dtype=bool)

    fitter = LinearLSQFitter()

    # ==================================================
    # FIRST FIT
    # ==================================================

    x0 = np.mean(x)
    scale = np.ptp(x)

    if scale == 0:
        raise RuntimeError(f"Order {order}: all calibration pixels are identical.")

    x_norm = (x - x0) / scale

    model = Polynomial1D(degree=ndeg)

    fit_model = fitter(model, x_norm, y)

    y_fit = fit_model(x_norm)

    residuals = y - y_fit

    rms = np.sqrt(np.mean(residuals**2))

    # --------------------------------------------------
    # Determine rejection threshold
    # --------------------------------------------------

    if max_residual is None:
        threshold = rms_threshold * rms

    else:
        threshold = max_residual

    # --------------------------------------------------
    # Reject only the single worst point
    # --------------------------------------------------

    worst = np.argmax(np.abs(residuals))

    if np.abs(residuals[worst]) > threshold:

        print(f"Order {order}: rejecting line {worst} "
              fr"(residual={residuals[worst]:.4f} A)")

        mask[worst] = False

        if np.sum(mask) >= ndeg + 1:

            x_fit = x[mask]
            y_fit = y[mask]

            x0 = np.mean(x_fit)
            scale = np.ptp(x_fit)

            x_fit_norm = (x_fit - x0)/scale
            x_norm = (x - x0)/scale

            model = Polynomial1D(degree=ndeg)

            fit_model = fitter(model, x_fit_norm, y_fit)

            residuals = y - fit_model(x_norm)

            rms = np.sqrt(np.mean(residuals[mask]**2))

    else:

        print(f"Order {order}: no outliers found "
              f"(largest residual = {np.max(np.abs(residuals)):.4f} A)")

    # --------------------------------------------------
    # Final callable
    # --------------------------------------------------
    def final_fit(xx):

        xx = np.asarray(xx)

        return fit_model((xx - x0)/scale)

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------

    x_axis = (x if x_range is None else np.arange(x_range[0], x_range[1]))

    y_axis = final_fit(x_axis)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.plot(x[mask], y[mask], '+', label='Used')
    if np.any(~mask):
        plt.plot(x[~mask], y[~mask], 'rx', ms=10, label='Rejected')
    plt.plot(x_axis, y_axis, 'r-', label=f'Polynomial (deg={ndeg})')
    plt.ylabel(r"Wavelength ($\AA$)")
    if x_range is not None:
        plt.xlim(x_range)
    plt.legend()

    plt.subplot(212)
    plt.plot(y[mask], residuals[mask], '+', label=fr'RMS = {rms:.5f} $\AA$')
    if np.any(~mask):
        plt.plot(y[~mask], residuals[~mask], 'rx')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Residual ($\AA$)")
    #ymax = max(0.05, 1.1*np.max(np.abs(residuals)))
    ymax = 1.2 * np.max(np.abs(residuals))
    plt.ylim(-ymax, ymax)
    plt.legend()
    plt.suptitle(f"Wavelength calibration (Order {order})")
    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_mask:
        return final_fit, rms, mask

    return final_fit, rms

def fit_wavelength_solution(arc_spectrum, ThAr_peak_wavelengths, ThAr_peak_pixels, order,
                            fit_window=11, max_fwhm_threshold=None,
                            wavesolution_ndeg=3, wavesolution_niter=3, wavesolution_rms_threshold=3.0,
                            wave_peak_threshold=1.5,
                            ref_thar_directory='../db/', ref_thar_file='ThAr_atlas_3000_10600_75k',
                            show_plots=True, save_plots=False, plot_directory=None, plot_prefix='file', displ_percent_max=99):
    """ a wrapper for performing wavelength solution
    arc_spectrum (np.array): 1d spectrum
    ThAr_peak_wavelengths (np.array): list of peak position wavelengths (in Angstrom)
    ThAr_peak_pixels (np.array): list of peak position (in pixels)
    fit_window (int): set window to search for refined pixel positions (in pixels)
    wavesolution_ndeg (int): polynomial degree for wavelength solution
    save_plots (bool): save plots
    plot_directory (str): directory for saving the plots
    
    returns:
    wavelength_axis (np.array): wavelength axis containing the same number of elements of 'arc_spectrum'
    rms_wavesol (float): RMS of the wavelength solution (in Angstrom)
    
    """
    
    if save_plots:
        output_peak     = plot_prefix.parent / ( plot_prefix.name + f"_peak_positions{PLOT_EXTENSION}" )
        output_cal      = plot_prefix.parent / ( plot_prefix.name + f"_wavelength_solution{PLOT_EXTENSION}" )
        output_thar_ref = plot_prefix.parent / ( plot_prefix.name + f"_refThAr{PLOT_EXTENSION}" )
    else:
        output_peak=None
        output_cal =None
    
    # read the ThAr spectrum
    ref_flux, ref_wavelength = read_ref_thar_simple(ref_thar_directory, ref_thar_file)
    
    # refine center positions for the HeAr lines used for wavelength calibration
    ThAr_peak_pixels_refined, _ = refine_peak_positions(arc_spectrum, ThAr_peak_pixels,
                                                        window=fit_window,
                                                        show_plots=show_plots, n_col=5, output=output_peak)

    # check difference between refined and reference peak positions
    delta_peak = ThAr_peak_pixels_refined - ThAr_peak_pixels
    avg_delta = np.average(delta_peak)
    std_delta = np.std(delta_peak)
    sigma_delta_peak = abs( delta_peak - avg_delta ) / std_delta

    good_detections = ( sigma_delta_peak < wave_peak_threshold )
    
    if np.sum(good_detections) > wavesolution_ndeg+1:
        print(f"Order {order}: a total of {np.sum(good_detections)}/{len(ThAr_peak_pixels_refined)} spectral features are available for deriving an accurate wavelength solution")
        ThAr_peak_pixels_refined = ThAr_peak_pixels_refined[good_detections]
        ThAr_peak_pixels         = ThAr_peak_pixels[good_detections]
        ThAr_peak_wavelengths    = ThAr_peak_wavelengths[good_detections]
    else:
        print(f"Order {order}: not enough spectral features ({np.sum(good_detections)}) to derive an accurate wavelength solution")
    
    # 7. wavelength calibration
    fit_wavelength, rms_wavesol = wavelength_calibration_iter(ThAr_peak_pixels_refined, ThAr_peak_wavelengths, order,
                                                              n_iter=wavesolution_niter, rms_threshold=wavesolution_rms_threshold,
                                                              x_range=[0,arc_spectrum.shape[0]], ndeg=wavesolution_ndeg,
                                                              show_plot=show_plots, figsize=(15,5), output=output_cal)
    
    # get wavelength axis
    x_axis = np.arange(len(arc_spectrum.data))
    wavelength_axis = fit_wavelength(x_axis)

    # scale atlas with observed ThAr lines
    ymax = np.percentile(arc_spectrum,displ_percent_max)
    idx_wave = ( ref_wavelength > min(wavelength_axis) ) *  ( ref_wavelength < max(wavelength_axis) )
    ymax_ref = np.percentile(ref_flux[idx_wave],displ_percent_max+0.5)
    ref_scalefactor = ymax / ymax_ref

    # compare lines with reference atlas
    plt.figure(figsize=(15,3))
    plt.plot(ref_wavelength,ref_flux*ref_scalefactor, lw=1, color='orange',   alpha=0.75, label='reference ThAr')
    plt.plot(wavelength_axis, arc_spectrum-np.percentile(arc_spectrum,0.5), lw=1, color='black', alpha=0.5,
             label=f'STELES (dlambda={abs(wavelength_axis[-2]-wavelength_axis[-1]):.4f}A/pix)')
    txt_label = 'Reference ThAr lines'
    for wave in ThAr_peak_wavelengths:
        plt.axvline(wave,ls='--', color='red', lw=1, label=txt_label)
        txt_label=''
    plt.xlim(min(wavelength_axis),max(wavelength_axis))
    plt.ylim(0,ymax)
    plt.xlabel('Wavelength (A)')
    plt.legend()
    plt.suptitle(f"Reference ThAr spectrum and STELES (n={order:02d})")
    plt.tight_layout()
    plt.savefig(output_thar_ref, format=PLOT_EXTENSION.replace(".",""), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    return wavelength_axis, rms_wavesol, fit_wavelength, ThAr_peak_pixels_refined, ThAr_peak_wavelengths

def plot_wavesolution_ThAr_atlas(wavelength_axis, arc_spectrum, ThAr_peak_wavelengths, ref_wavelength, ref_flux, 
                                 ref_scalefactor = 0.0250, figsize=(15,3), dpi=300, output=None):
    """ overlay the ThAr spectrum with the reference atlas to check wavelength solution"""

    plt.figure(figsize=(figsize))
    plt.plot(ref_wavelength,ref_flux*ref_scalefactor, lw=1, color='red',   alpha=0.9, label='reference ThAr')
    plt.plot(wavelength_axis, arc_spectrum,           lw=1, color='black', alpha=0.9, 
             label=f'STELES (dlambda={abs(wavelength_axis[1]-wavelength_axis[0]):.4f}A/pix)')
    txt_label = 'Reference ThAr lines'
    for wave in ThAr_peak_wavelengths:
        plt.axvline(wave,ls='--', color='green', label=txt_label)
        txt_label=''
    plt.xlim(min(wavelength_axis),max(wavelength_axis))
    plt.ylim(y_range)
    plt.xlabel('Wavelength (A)')
    plt.legend()
    plt.title(f"ThAr spectrum - order={order}")
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=dpi, format='png')
    plt.show()

def interpolate_wavelength_axis(flux, wavelength, header, dispersion=None, show_plot=False, figsize=(12,4)):
    """
    Create a linear wavelength grid from a non-linear wavelength array.
    """
    wave_min = wavelength.min()
    wave_max = wavelength.max()

    # Estimate the mean delta lambda from original
    deltas = np.diff(wavelength)
    
    if deltas[0] < 0:
        deltas = abs(deltas)
        
    if dispersion is None:
        delta_lambda = np.nanmin(deltas)
        print(f"Dispersion={delta_lambda:.4f} A/pix")
    else:
        delta_lambda = dispersion
        
    wave_linear = np.arange(wave_min, wave_max, delta_lambda)
    
    f_interp = interp1d(wavelength, flux, kind='cubic', fill_value='extrapolate')
    flux_interp = f_interp(wave_linear)
    
    header = header
    
    # Wavelength solution keywords
    crpix1 = 1  # Reference pixel (1-based in FITS)
    crval1 = wave_linear[0]  # Wavelength at reference pixel
    cdelt1 = wave_linear[1] - wave_linear[0]  # Dispersion (assumes linear)
    
    header.set("CRPIX1", value=crpix1, comment='Coordinate reference pixel' )
    header.set("CRVAL1", value=crval1, comment='Coordinate reference value' )
    header.set("CDELT1", value=cdelt1, comment='Coordinate interval' )
    header.set("CTYPE1", value='Wavelength', comment='Axis type' )
    header.set("CUNIT1", value='Angstrom', comment='Axis unit' )
    header.set("ST_WVDSP", value=delta_lambda, comment='Dispersion of the order (A/pix)')
    
    if show_plot:
        plt.figure(figsize=figsize)
        plt.plot(wavelength,  flux.data,   alpha=0.75, label='original')
        plt.plot(wave_linear, flux_interp, alpha=0.75, label='interpolated')
        plt.xlim(np.nanmin(wavelength),np.nanmax(wavelength))
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel(r'Flux (ADU)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return flux_interp, wave_linear, header

def read_ref_thar_line_list(file_path):
    """
    Reads an ASCII file containing two columns: wavelength and line name.

    Parameters
    ----------
    file_path : str
        Path to the ASCII file.

    Returns
    -------
    wavelengths : ndarray
        Array of central wavelengths (floats).

    line_names : list of str
        List of line names (strings).
    """
    wavelengths = []
    line_names = []

    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines or comment lines
            if line.strip() == "" or line.strip().startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) >= 2:
                wl = float(parts[0])
                name = parts[1]
                wavelengths.append(wl)
                line_names.append(name)

    return np.array(wavelengths), line_names
    
def read_ref_thar(infolder, fits_file):

    # Data from https://noirlab.edu/science/data-services/other/spectral-atlas
    with fits.open(infolder / ( fits_file + FITS_EXTENSION ) ) as hdul:
        ref_flux   = hdul[0].data
        ref_header = hdul[0].header

    # read the line list
    l_wave, l_name = read_ref_thar_line_list(infolder / ( fits_file + TABLE_EXTENSION ) )
   
    # Read wavelength calibration info from header (standard keywords)
    crval1 = ref_header.get('CRVAL1')  # Starting wavelength
    cdelt1 = ref_header.get('CDELT1')  # Wavelength increment per pixel
    crpix1 = ref_header.get('CRPIX1', 1)  # Reference pixel (default to 1 if missing)

    # Number of pixels
    npix = ref_flux.shape[0]

    # Compute wavelength array
    pixel_indices = np.arange(1, npix + 1)  # FITS convention: first pixel = 1
    ref_wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

    return ref_flux, ref_wavelength, l_wave, l_name


def read_ref_thar_simple(infolder, fits_file):

    # Data from https://noirlab.edu/science/data-services/other/spectral-atlas
    with fits.open(infolder / ( fits_file + FITS_EXTENSION ) ) as hdul:
        ref_flux   = hdul[0].data
        ref_header = hdul[0].header
   
    # Read wavelength calibration info from header (standard keywords)
    crval1 = ref_header.get('CRVAL1')  # Starting wavelength
    cdelt1 = ref_header.get('CDELT1')  # Wavelength increment per pixel
    crpix1 = ref_header.get('CRPIX1', 1)  # Reference pixel (default to 1 if missing)

    # Number of pixels
    npix = ref_flux.shape[0]

    # Compute wavelength array
    pixel_indices = np.arange(1, npix + 1)  # FITS convention: first pixel = 1
    ref_wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

    return ref_flux, ref_wavelength

def plot_thar_reference_spectrum(infolder, thar_fits, outfolder, wave_min=3000, wave_max=9400, dwave=200, display_percentile=[0.1,99.5]):
    """
      infolder(str): directory containing the input data
      thar_fits (str): name of the fits file containing the ThAr spectrum
      outfolder (str): directory to store the plots
      wave_min (float, default=3000): minimum wavelength to plot (in Angstrom)
      wave_max (float, default=9400): maximum wavelength to plot (in Angstrom)
      dwave (float, default=200): wavelength interval for each plot (in Angstrom)
      display_percentile(float, default=[0.1,99.5]): percentile range to show the plot

    """

    # read the ThAr spectrum
    ref_flux, ref_wavelength, l_wave, l_name = read_ref_thar(infolder, thar_fits)

    # set number of plots
    n_plot = int( ( wave_max - wave_min ) / dwave )

    # loop into wavelength interval
    for i in np.arange(n_plot):

        # get wavelength interval
        wave_min_n = wave_min + dwave * i
        wave_max_n = wave_min + dwave * (i+1)

        # set x-axis range
        xlim = (wave_min_n-dwave*0.05,wave_max_n+dwave*0.05)

        # set y-axis range
        idx_wave = (ref_wavelength > wave_min_n) * (ref_wavelength < wave_max_n)
        ylim = np.percentile(ref_flux[idx_wave], display_percentile)

        # now plot
        plt.figure(figsize=(15,4))
        plt.plot(ref_wavelength, ref_flux, lw=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Wavelength (A)')

        # add spectral lines
        plot_thar_lines(l_wave, l_name, wave_min_n, wave_max_n, linestyle='--', linecolor='black', linewidth=0.25, fontsize=6)

        plt.tight_layout()
        plt.savefig(outfolder + f'w_{wave_min_n}_{wave_max_n}{PLOT_EXTENSION}', format='png', dpi=450)
        plt.show()
    
def plot_thar_lines(l_wave, l_name, wave_min, wave_max, linestyle='--', linecolor='black', linewidth=1, fontcolor='black', fontsize=7):
    
    idx = (l_wave > wave_min ) * (l_wave < wave_max)
        
    n_lines = np.sum(idx)
    print("n_lines=",n_lines)
    
    if n_lines > 0:
    
        p_wave = l_wave[idx]

        for i,w in enumerate(p_wave):
            #p_name = l_name[idx[i]] + f'[{w:.3f}]'
            p_name = f'[{w:.3f}]'
            plt.axvline(w, ls=linestyle, color=linecolor, lw=linewidth)
            plt.text(w, plt.ylim()[1]*1.02, p_name, rotation=90,
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
                color=fontcolor)
    else:
        print("No lines to plot.")
        
# fit spatial distortion along the slit
def fit_distortion_line(rows, x_peak, degree=2, allow_fallback=False):
    """
    Fit the spatial distortion of a single calibration line.

    Parameters
    ----------
    rows : ndarray
        Spatial pixel positions.

    x_peak : ndarray
        Measured centroid position of the line.

    degree : int
        Polynomial degree.

    Returns
    -------
    result : dict
        Dictionary containing the fitted model and diagnostics.

        result["model"]          Polynomial1D model
        result["row0"]           normalization offset
        result["row_scale"]      normalization scale
        result["rms"]            RMS residual
        result["max_residual"]   maximum absolute residual
        result["npoints"]        number of fitted points
    """
    rows = np.asarray(rows, dtype=float).ravel()
    x_peak = np.asarray(x_peak, dtype=float).ravel()

    # Check input dimensions and matching lengths
    if rows.size != x_peak.size:
        if not allow_fallback:
            return None
        rows = np.asarray(rows, dtype=float).ravel()
        x_peak = np.full(rows.shape, np.nan)

    valid = np.isfinite(rows) & np.isfinite(x_peak)
    rows = rows[valid]
    x_peak = x_peak[valid]
    
    
    #rows = np.asarray(rows, dtype=float)
    #x_peak = np.asarray(x_peak, dtype=float)

    #valid = np.isfinite(rows) & np.isfinite(x_peak)

    #rows = rows[valid]
    #x_peak = x_peak[valid]

    if len(rows) < max(8, 2 * degree + 2):
        
        if allow_fallback:
            
            identity_model = Polynomial1D(degree=degree)
            identity_model.parameters[:] = 0.0

            return {"model": identity_model,
                    "row0": 0.,
                    "row_scale": 1.,
                    "rms": np.nan,
                    "max_residual": np.nan,
                    "npoints": len(rows),
                    "fallback": True,}
        
        return None

    # --------------------------------------------------------
    # Normalize independent variable
    # --------------------------------------------------------

    row0 = np.mean(rows)

    row_scale = np.ptp(rows)

    if row_scale == 0:
        row_scale = 1.0

    rows_norm = (rows - row0) / row_scale

    # --------------------------------------------------------
    # Polynomial fit
    # --------------------------------------------------------

    fitter = LinearLSQFitter()

    model = fitter(Polynomial1D(degree=degree), rows_norm, x_peak)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    residual = x_peak - model(rows_norm)

    rms = np.sqrt(np.mean(residual**2))

    max_residual = np.max(np.abs(residual))

    return {"model": model,
            "row0": row0,
            "row_scale": row_scale,
            "rms": rms,
            "max_residual": max_residual,
            "npoints": len(rows)}


def denormalize_polynomial(model, row0, row_scale):
    """
    Convert a Polynomial1D fitted in normalized coordinates
    back into detector coordinates.

    The returned polynomial satisfies

        poly(y) = model((y-row0)/row_scale)
    """

    degree = model.degree

    # Only degree+1 samples are needed to recover a polynomial exactly.
    y = np.linspace(row0-row_scale, row0+row_scale, degree+1)

    x = model((y-row0)/row_scale)

    fitter = LinearLSQFitter()

    return fitter( Polynomial1D(degree=degree), y, x)

 
# fit spatial distortion along the slit
def fit_spatial_distortion(image, peak_positions_in_pixel, order, channel, window=30, n_pix_step=1,
                           ndeg=2, fitting_sigma_threshold=3.0, fitting_niter=3, 
                           saturation_threshold=50000, sigma_threshold=3., sigma_min= 1.0, sigma_step= 0.5,
                           single_fit=False, n_col=None, output=None, show_plots=True):
    
    """
       Compute spatial distortion along the slit using selected spectral lines in the spectrum.
       image (numpy.ndarray): input 2D spectrum to fit
       peak_positions_in_pixel (numpy.array): list of spectral lines to fit
       window (int, default=30): pixel range to refine the line position
       n_pix_step (int, default=1): step size in pixels to compute the distortion along the slit.
       ndeg (int, default=2): polynomial degree to fit the distortion
       single_fit (boolean, default=False): use all line information to compute a single function (set to True to avoid outliers, specially when saturated lines are included)
       n_col (int, default=None): set number of columns for plotting
       output (str, default, None): name of the output file for the plot. 
    """

    ####################################################################
    # Image preparation
    ####################################################################

    n_spat, n_spec = image.shape

    # set a new grid of spatial pixels, avoiding the borders
    spat_rows = np.arange(n_pix_step, n_spat-n_pix_step, n_pix_step)

    # for the red channel, the spec2d image is flipped.
    #if channel.lower() == "red":
    #    image = np.flip(image, axis=0)

    # get number of spatial rows
    n_spat_rows = len(spat_rows)

    ####################################################################
    # Estimate image RMS
    ####################################################################

    # estimate the RMS from the image using the Median Absolute Deviation (MAD)
    image_rms = np.median(np.abs(image - np.median(image)))
    if image_rms == 0:
        image_rms = np.nanstd(image)
        
    ####################################################################
    # Smooth image along spatial direction
    ####################################################################

    image_smooth = uniform_filter1d(image, size=3, axis=0, mode="nearest")
    

    ####################################################################
    # Trace every calibration line
    ####################################################################

    # now get the refined peak positions for every line and across the rows in 'spat_rows'
    peak_refined_rows = []
    peak_fluxes = []

    current_positions = peak_positions_in_pixel.copy()

    for row in spat_rows:

        arc_1d = image_smooth[row]

        peak_refined_row, peak_flux = refine_peak_positions(arc_1d, peak_positions_in_pixel, window=window, show_plots=False)

        # update peak information from previous fit - only if it returns a finite value (avoid adding NaN)
        idx = np.isfinite(peak_refined_row)
        current_positions[idx] = peak_refined_row[idx]

        # store the data into a list
        peak_refined_rows.append(peak_refined_row)
        peak_fluxes.append(peak_flux)

    # convert list to numpy array
    peak_refined_rows = np.asarray(peak_refined_rows, dtype=np.float32)
    peak_fluxes       = np.asarray(peak_fluxes,      dtype=np.float32)    
    

    ####################################################################
    # Reject saturated / weak measurements
    ####################################################################

    # now exclude saturated lines
    idx_saturated = (peak_fluxes >= saturation_threshold)

    print(f" Excluding {np.sum(idx_saturated)} saturated measurements (>{saturation_threshold:.0f})")

    # --- Adaptive sigma threshold ---
    sigma_current = sigma_threshold
    # set number of minimum lines for fitting
    min_lines_required = max(8, 2 * ndeg + 2)

    #while True:
    for sigma_current in np.arange(sigma_threshold, sigma_min-sigma_step, -sigma_step):

        idx_weak = (peak_fluxes <= sigma_current * image_rms)

        # GOOD lines = not saturated AND not weak
        idx_good = (~idx_saturated) & (~idx_weak)

        if np.sum(idx_good) >= min_lines_required:
            break

        sigma_current -= sigma_step

        if sigma_current < sigma_min:
            break

    idx_weak = ( peak_fluxes <= sigma_current * image_rms)

    idx_out = ~(~idx_saturated & ~idx_weak)

    print(f" Excluding {np.sum(idx_weak)} measurements below {sigma_current:.2f}-sigma (<{sigma_current*image_rms:.1f})" )
    print(f" Using {np.sum(~idx_out)}/{peak_fluxes.size} measurements.")

    ####################################################################
    # Flag rejected measurements
    ####################################################################

    peak_refined_rows_sat = peak_refined_rows.copy()
    peak_refined_rows_weak = peak_refined_rows.copy()

    peak_refined_rows_sat[idx_saturated] = np.nan
    peak_refined_rows_weak[idx_weak] = np.nan


    ####################################################################
    # Prepare plotting
    ####################################################################

    # get number of spectral lines to fit
    n_lines = len(peak_positions_in_pixel)

    n_plots = n_lines + 1

    if n_col is None:
        n_col = min(4, n_plots)

    n_row = int(np.ceil(n_plots / n_col))

    # now plot
    plt.figure(figsize=(3 * n_col, 3 * n_row))    
    
    ####################################################################
    # Fit each spectral line independently
    ####################################################################

    line_fit = []
    line_valid = []

    line_rms = []
    line_max_residual = []
    line_npoints = []

    for n in range(n_lines):    

        #---------------------------------------------------------------
        # Measurements for this line
        #---------------------------------------------------------------
        rows = spat_rows
        # set (x,y) for each line - we want f(y)=x!
        x = peak_refined_rows[:, n]
        # read saturated and weak measurements
        xsat  = peak_refined_rows_sat[:,  n]
        xweak = peak_refined_rows_weak[:, n]
        # if y has "NaN" values, the line is probably saturated, so skip
        valid = ( np.isfinite(xsat) & np.isfinite(xweak))
        
        result = fit_distortion_line(rows[valid], x[valid], degree=ndeg)

        plt.subplot(n_row, n_col, n + 1)

        if result is None:

            plt.plot(x, rows, "+")
            plt.xlabel("Dispersion (pixel)")
            if n == 0:
                plt.ylabel("Along slit (pixel)")
            if np.any(np.isnan(xsat)):
                title = "saturated"
            elif np.any(np.isnan(xweak)):
                title = "weak"
            else:
                title = "insufficient"
            plt.title(f"x={peak_positions_in_pixel[n]:.1f}\n({title})")

            line_fit.append(None)
            line_valid.append(False)
            line_rms.append(np.nan)
            line_max_residual.append(np.nan)
            line_npoints.append(0)

            continue

        #---------------------------------------------------------------
        # Valid measurements
        #---------------------------------------------------------------

        rows_valid = rows[valid]
        x_valid = x[valid]
        
        #---------------------------------------------------------------
        # Normalize spatial coordinate
        #---------------------------------------------------------------

        row0 = np.mean(rows_valid)
        row_scale = np.ptp(rows_valid)

        if row_scale == 0:
            row_scale = 1.0

        rows_norm = (rows_valid - row0) / row_scale
        rows_all_norm = (rows - row0) / row_scale

        #---------------------------------------------------------------
        # Polynomial fit
        #---------------------------------------------------------------

        fitter = LinearLSQFitter()
        model = Polynomial1D(degree=ndeg)
        fit_spat = fitter(model, rows_norm, x_valid)        
        
        #---------------------------------------------------------------
        # Diagnostics
        #---------------------------------------------------------------

        x_model_valid = fit_spat(rows_norm)
        residual = x_valid - x_model_valid
        rms = np.sqrt(np.mean(residual ** 2))
        max_residual = np.max(np.abs(residual))

        #---------------------------------------------------------------
        # Store model and normalization
        #---------------------------------------------------------------

        line_fit.append(result)

        line_valid.append(True)
        line_rms.append(result["rms"])
        line_max_residual.append(result["max_residual"])
        line_npoints.append(result["npoints"])

        #---------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------
        rows_norm = ( rows - result["row0"] ) / result["row_scale"]
        x_model = result["model"](rows_norm)
        plt.plot(x_model, rows, color="red", lw=2)
        plt.plot(x[valid], rows[valid], "+", ms=6)

        plt.xlabel("Dispersion (pixel)")
        if n == 0:
            plt.ylabel("Along slit (pixel)")
        plt.title(f"x={peak_positions_in_pixel[n]:.1f}")
        
    ####################################################################
    # Reject globally bad polynomial fits
    ####################################################################

    rms_array = np.asarray(line_rms)

    good = np.isfinite(rms_array)

    if np.sum(good):
        median_rms = np.median(rms_array[good])
        mad_rms = np.median(np.abs(rms_array[good] - median_rms))

        if mad_rms == 0:
            mad_rms = np.nanstd(rms_array[good])

        rms_limit = median_rms + 3 * mad_rms

    else:
        rms_limit = np.inf

    print(" Line     Npts    RMS(px)   Max(px)   Status")

    print("----------------------------------------------")

    for n in range(n_lines):

        if not line_valid[n]:
            print(f"{n:4d}      ---      ---       ---     SKIPPED")
            continue

        if line_rms[n] > rms_limit:
            line_valid[n] = False
            status = "  REJECT"
        else:
            status = "  OK"

        print(f"{n+1:4d}{line_npoints[n]:9d}{line_rms[n]:11.4f}{line_max_residual[n]:11.4f}{status}")

    print()
    
    ########################################################################
    # Build one distortion curve for every accepted line
    ########################################################################

    master_curves = []

    center_row = spat_rows[n_spat_rows // 2]

    for result, valid in zip(line_fit, line_valid):

        if not valid:
            continue

        rows_norm = ( spat_rows - result["row0"] ) / result["row_scale"]

        curve = result["model"](rows_norm)

        # reference to slit center
        center_value = result["model"]( (center_row - result["row0"]) / result["row_scale"])

        curve = curve - center_value

        master_curves.append(curve)    
    
    #---------------------------------------------------------------
    # Average the curves
    #---------------------------------------------------------------
    
    master_curves = np.asarray(master_curves)
    master_curve = np.nanmean(master_curves, axis=0)
    
    #---------------------------------------------------------------
    # Fit the master curve
    #---------------------------------------------------------------
    
    master_result = fit_distortion_line(spat_rows, master_curve, degree=ndeg, allow_fallback=True)

    if master_result is None:
        print()
        print(f"WARNING: Could not fit spatial distortion for order {order}.")
        print("         Using zero-distortion model.")
        print()
        
        identity_model = Polynomial1D(degree=ndeg)
        identity_model.parameters[:] = 0.0

        master_result = {"model": identity_model,"row0": 0.0,
                         "row_scale": 1.0,
                         "rms": np.nan,
                         "max_residual": np.nan,
                         "npoints": 0,
                         "fallback": True,}

    else:
        master_result["fallback"] = False
        
        
    ########################################################################
    # Evaluate final distortion model
    ########################################################################

    rows_norm = (spat_rows - master_result["row0"]) / master_result["row_scale"]

    distortion_fit = master_result["model"](rows_norm)        
        
    ########################################################################
    # Master distortion plot
    ########################################################################

    plt.subplot(n_row, n_col, n_plots)

    for curve in master_curves:
        plt.plot(curve, spat_rows, color="0.8", lw=1)

    plt.plot(master_curve, spat_rows, "k.", ms=3, label="Average")
    plt.plot(distortion_fit, spat_rows, "r", lw=2, label="Polynomial")

    plt.xlabel("Distortion (pixel)")
    plt.ylabel("Along slit (pixel)")
    plt.title("Master distortion")
    plt.legend(fontsize=8)        

    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
    
    ####################################################################
    # Prepare output (keep original API)
    ####################################################################

    if single_fit:
        # --------------------------------------------------------------
        # Convert master model back to detector coordinates
        # --------------------------------------------------------------

        polyfit = denormalize_polynomial(master_result["model"], master_result["row0"], master_result["row_scale"])

        # --------------------------------------------------------------
        # Preserve the original API:
        # one identical polynomial for every spatial row
        # --------------------------------------------------------------

        n_rows = len(spat_rows)

        polyfits_x_along_y = np.empty((n_rows, 1), dtype=object)

        for i in range(n_rows):
            polyfits_x_along_y[i, 0] = polyfit

    else:
        # --------------------------------------------------------------
        # One polynomial per ThAr line
        # --------------------------------------------------------------
        polyfits_x_along_y = np.empty((1, n_lines),dtype=object)

        for j, result in enumerate(line_fit):

            # Invalid fit → return NaN polynomial
            if result is None:
                poly = Polynomial1D(degree=ndeg)
                poly.parameters[:] = np.nan

            else:
                poly = denormalize_polynomial(result["model"],result["row0"],result["row_scale"])

            polyfits_x_along_y[0, j] = poly

    ####################################################################
    # Return (same API as original implementation)
    ####################################################################

    return spat_rows, polyfits_x_along_y 
    
    

def read_polyfits_csv(input_csv):
    """
    Read CSV file containing:
    - First line: reference pixel (float)
    - Remaining lines: polynomial coefficients per row

    Returns:
        pixels   : float
        polyfits : array of Polynomial1D objects (shape = n_rows)
    """

    with open(input_csv, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise RuntimeError(f"Invalid CSV format: {input_csv}")

    # --- First line = reference pixel ---
    try:
        pixels = float(lines[0])
    except ValueError:
        raise RuntimeError(f"Could not parse reference pixel in {input_csv}")

    # --- Remaining lines = polynomial coefficients ---
    polyfits = []

    for line in lines[1:]:
        # Remove brackets
        coeffs_txt = line.replace("[", "").replace("]", "")

        # Convert to float list
        try:
            coeffs = [float(x) for x in coeffs_txt.split(",")]
        except ValueError:
            raise RuntimeError(f"Invalid coefficient line: {line}")

        degree = len(coeffs) - 1

        poly = Polynomial1D(
            degree=degree,
            **{f"c{k}": v for k, v in enumerate(coeffs)}
        )

        polyfits.append(poly)

    polyfits = np.array(polyfits, dtype=object)

    return pixels, polyfits


def correct_spatial_distortion(image, y_positions, x_y_fits, reference_peak_positions, order, y_ref=None, 
                               show_plot=True, figsize=(12,12), wave_pixel_reference = [3,11,19], 
                               max_displ_percent = 99, plot_x_limit=(900,1500), output=None):
    """
    Rectify 2D image using a set of x(y) distortion curves at given x positions.
    
    Parameters:
        image : 2D numpy array ()
            The distorted input image [ny, nx].
        y_positions : array-like
            The x positions at which the x(y) functions were measured.
        x_y_fits : list of Polynomial1D models
            The distortion curves (x as a function of y) for each x_position.
        y_ref : float
            Reference y value to align all x(y) to (default: center of image).
        wave_pixel_reference : float (list
            Reference line positions (in pixel) to display over the distorted-corrected images.
    
    Returns:
        rectified_ccddata : 2D numpy array (CCDData)
            The distortion-corrected image.
    """
    ny, nx = image.shape
    if y_ref is None:
        y_ref = ny // 2
    
    # this version of the code only uses one polynomial fit for the entire order, no matter how many lines are stored into 'x_y_fits'.
    # TODO: add all the measurements from 'x_y_fits' to the fit
    if x_y_fits.ndim > 1:
        print("More than one fit to consider...")
        # first, get the center of the dispersion axis
        center_pixel = int(nx // 2)
        # now, find the line closest to the center of the dispersion axis
        idx_center_pixel = np.abs(reference_peak_positions - center_pixel).argmin()
        # set the x_y_fits to the coefficients corresponding to the line closest to the center of the dispersion axis
        x_y_fits = x_y_fits[:,idx_center_pixel]
    
    y_vals = np.arange(ny)
    x_vals = np.arange(nx)

    # Step 1: Evaluate x(y) for each given x_position
    distortion_curves = np.zeros((len(y_positions), ny))  # [n_sample_y, ny]
    
    for i, fit in enumerate(x_y_fits):
        distortion_curves[i, :] = fit(y_vals)

    # Step 2: Interpolate across all x positions to make a full 2D distortion map
    x_shift_map = np.zeros((ny, nx))
    for y in range(ny):
        interp = interp1d(y_positions, distortion_curves[:, y], kind='cubic', fill_value='extrapolate')
        distorted_x = interp(x_vals)
        # Reference: what x value should this trace have at y_ref?
        ref_interp = interp1d(y_positions, [fit(y_ref) for fit in x_y_fits], kind='cubic', fill_value='extrapolate')
        ref_x = ref_interp(x_vals)
        x_shift_map[y, :] = ref_x - distorted_x

    # Step 3: Apply the shift to resample the image
    y_coords, x_coords = np.indices(image.shape)
    corrected_x = x_coords - x_shift_map
    
    rectified_image = map_coordinates(image, [y_coords, corrected_x], order=1, mode='nearest')

    if show_plot:
        xmin=0
        xmax=image.shape[1]

        #fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"hspace": 0.3}, sharex=True, layout='constrained')
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
        #plt.rcParams['figure.constrained_layout.use'] = True

        #print('input_image:',np.nanmin(image),np.nanmax(image),max_displ)
        max_displ = np.percentile(image, max_displ_percent)
        
        ##################################        
        #plt.subplot(4,1,1)
        im_ref = axes[0].imshow(image, aspect='auto', origin='lower', vmin=0, vmax=max_displ)

        # draw vertical lines showing the spectral reference positions
        for x in reference_peak_positions:
            axes[0].axvline(x, ls='--', color='black')
        
        # draw horizontal lines showing the spatial reference positions
        for ypp in wave_pixel_reference:
            axes[0].axhline(ypp, ls='--', label=f"y={ypp} (original)")
            
        #plt.xlim(xmin,xmax)
        axes[0].set_xlim(plot_x_limit)
        axes[0].set_title(f"Original 2D spectrum (order={order})")
        
        ##################################        
        #plt.subplot(4,1,2)
        axes[1].imshow(rectified_image, aspect='auto', origin='lower', vmin=0, vmax=max_displ)
        for x in reference_peak_positions:
            axes[1].axvline(x, ls='--', color='black')

            
        for ypp in wave_pixel_reference:
            axes[1].axhline(ypp, ls='--', label=f"y={ypp} (original)")            
            
        #plt.xlim(xmin,xmax)
        axes[1].set_xlim(plot_x_limit)
        axes[1].set_title("Distortion corrected 2D spectrum")

        # Add one colorbar for the first two
        cbar = fig.colorbar(im_ref, ax=[axes[0], axes[1]], location='right', shrink=1.0, fraction=0.05, pad=0.005, aspect=30, label='Flux (ADU)')
        
        ##################################
        #axes[2].subplot(4,1,3)
        y = image
        y_avg = np.nanmean(image,axis=0)
        
        axes[2].plot(np.arange(len(y_avg)),y_avg, color='black', lw=1.5, label=f"Average (original)")
        for ypp in wave_pixel_reference:
            yp = y[ypp,:]
            axes[2].plot(np.arange(len(yp)),yp, lw=1, alpha=0.8, label=f"y={ypp} (original)")
        for x in reference_peak_positions:
            axes[2].axvline(x, ls='--', lw=0.75, color='black')
        axes[2].set_xlim(plot_x_limit)
        axes[2].set_ylim(0,max_displ)
        axes[2].legend()
        axes[2].set_title("Original 1D spectrum")

        ##################################
        #plt.subplot(4,1,4)
        y = rectified_image
        y_avg = np.nanmean(rectified_image,axis=0)
        axes[3].plot(np.arange(len(y_avg)),y_avg, color='black', lw=1.5, label=f"Average (corrected)")
        for ypp in wave_pixel_reference:
            yp = y[ypp,:]
            axes[3].plot(np.arange(len(yp)),yp, lw=1, alpha=0.8, label=f"y={ypp} (corrected)")
        for x in reference_peak_positions:
            axes[3].axvline(x, ls='--', lw=0.75, color='black')
        axes[3].set_xlim(plot_x_limit)
        axes[3].set_ylim(0,max_displ)
        axes[3].legend()
        axes[3].set_title("Distortion corrected 1D spectrum")
        
        #plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    return rectified_image


def polynomialfit(x, y, nmin=1, nmax=3, debug=False):
    """
    Select the minimum chi squared polynomial fit between nmin and nmax

    :param x: x axis array
    :param y: y axis array
    :param nmin: minimum order to fit
    :param nmax: maximum order to fit

    Note could improve this to weight it by pixel uncertainties
    """
    chis, ps = [], []
    nrange = range(nmin, nmax+1)
    
    for n in nrange:
        p = np.polyfit(x, y, n)
        yfit = np.polyval(p, x)
        
        # get the reduced chi2
        chi_squared = np.sum((y-yfit)**2)
        
        ps.append(p)
        chis.append(chi_squared)
        if debug:
            print(n,chi_squared)
    
    argmin = np.argmin(chis)
    return ps[argmin], nrange[argmin]

def eval_polynomialfit(x, y, nmin=1, nmax=3, chi2_threshold=0.1, debug=False):
    """
    Select the minimum chi squared polynomial fit between nmin and nmax

    :param x: x axis array
    :param y: y axis array
    :param nmin: minimum order to fit
    :param nmax: maximum order to fit

    Note could improve this to weight it by pixel uncertainties
    """
    chis, ps = [], []
    nrange = range(nmin, nmax+1)
    
    for n in nrange:
        p = np.polyfit(x, y, n)
        yfit = np.polyval(p, x)
        
        # evaluate the degrees of freedom of the plot
        dof = len(x) - n - 1
        # get the reduced chi2
        chi_squared = np.sum((y-yfit)**2) / dof
        
        ps.append(p)
        chis.append(chi_squared)
        if debug:
            print(n,chi_squared)
    
    best_degree = None
    best_chi2 = np.inf
    for i,n in enumerate(nrange):
        # Check for the simplest model within the threshold
        
        #if best_degree is not None:
        #    print(i,chis[i], best_chi2, chis[i] < best_chi2, best_chi2 - chis[i], best_chi2 - chis[i] < chi2_threshold)
        
        if chis[i] < best_chi2:
            if best_degree is not None and best_chi2 - chis[i] < chi2_threshold:
                #best_degree = min(best_degree, n)  # Choose simpler model
                best_degree = min(best_degree, i)  # Choose simpler model
            else:
                #best_degree = n
                best_degree = i
                best_chi2 = chis[i]
        #print(i,best_degree)
        
    argmin = best_degree #np.where(nrange == best_degree)[0]
    #print('best_degree=',nrange[argmin])
    return ps[argmin], nrange[argmin]


def fit_quartz(quartz_1d_extracted,
               nmin=1,
               nmax=30,
               sigma_clip_thresh=None,
               n_cols=4,
               output_blaze=None):

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from scipy.signal import savgol_filter

    warnings.filterwarnings('ignore', message="overflow encountered in", category=RuntimeWarning)

    try:
        warnings.simplefilter('ignore', np.RankWarning)
    except:
        warnings.simplefilter('ignore', np.exceptions.RankWarning)

    # Read input
    try:
        with fits.open(quartz_1d_extracted) as hdul:
            quartz1d_data = hdul[0].data
    except:
        quartz1d_data = quartz_1d_extracted

    if quartz1d_data.ndim > 2:
        quartz1d_data = quartz1d_data[0, :, :]

    fit = quartz1d_data.copy()

    plt.figure(figsize=(18,18))

    n_rows = int(np.ceil(quartz1d_data.shape[0] / n_cols))

    for n in range(quartz1d_data.shape[0]):

        yarr = quartz1d_data[n,:].astype(float)

        # -----------------------------------------
        # Normalize order
        # -----------------------------------------

        scale = np.percentile(yarr[np.isfinite(yarr)],99)

        if scale == 0:
            fit[n,:] = np.nan
            continue

        yarr /= scale

        # -----------------------------------------
        # Normalized x coordinates
        # -----------------------------------------

        xarr = np.linspace(-1.0,1.0,len(yarr))

        # -----------------------------------------
        # Valid pixels
        # -----------------------------------------

        idx_finite = np.isfinite(yarr) & (yarr > 0)

        if np.sum(idx_finite) < 10:
            fit[n,:] = np.nan
            continue

        # -----------------------------------------
        # Reject extremely low S/N pixels
        # -----------------------------------------

        signal_threshold = 0.01*np.nanmax(yarr)

        idx_signal = yarr > signal_threshold

        idx_fit = idx_finite & idx_signal

        # -----------------------------------------
        # Optional sigma clipping
        # -----------------------------------------

        if sigma_clip_thresh is not None:

            clipped = sigma_clip(yarr[idx_fit],
                                 sigma=sigma_clip_thresh,
                                 maxiters=3)

            tmp = np.where(idx_fit)[0]

            idx_fit[:] = False
            idx_fit[tmp[~clipped.mask]] = True

        if np.sum(idx_fit) < 10:
            idx_fit = idx_finite

        # -----------------------------------------
        # Polynomial fit
        # -----------------------------------------

        ypfit, n_fit = polynomialfit(
            xarr[idx_fit],
            yarr[idx_fit],
            nmin=nmin,
            nmax=nmax,
            debug=False)

        yfit = np.polyval(ypfit,xarr)

        # -----------------------------------------
        # Smooth final blaze slightly
        # -----------------------------------------

        window = min(31, len(yfit)//2*2-1)

        if window >= 7:
            yfit = savgol_filter(
                yfit,
                window_length=window,
                polyorder=3,
                mode='interp'
            )

        # -----------------------------------------
        # Remove unphysical values
        # -----------------------------------------

        yfit[yfit <= 0] = np.nan

        # Preserve invalid detector regions
        yfit[~idx_finite] = np.nan

        fit[n,:] = yfit

        # -----------------------------------------
        # Plot
        # -----------------------------------------

        plt.subplot(n_rows,n_cols,n+1)

        plt.plot(xarr,
                 yarr,
                 color='black',
                 alpha=0.5,
                 lw=0.8)

        plt.plot(xarr,
                 yfit,
                 color='red',
                 lw=2)

        plt.ylim(0,1.1)
        plt.xlim(-1,1)

        plt.title(f"order={n+1}, n_fit={n_fit}")

    plt.tight_layout()

    if output_blaze is not None:
        plt.savefig(output_blaze,dpi=300)

    plt.show()

    return fit