# steles_ccd.py
from .steles_imports import *

def header2python(str_hdr, pixel=True):
    """
    Parse a string containing [XX:XX,YY:YY] to pixels readable by python.

    Parameter
    ---------
        str_hdr : str
        pixel (bool, default = True): if dealing with pixels, subtract 1
        
    """
    str_hdr = str_hdr.replace('[', '')
    str_hdr = str_hdr.replace(']', '')
    str_hdr = str_hdr.replace(',', ' ')
    str_hdr = str_hdr.replace(':', ' ')
    str_out  = np.array(str_hdr.split(' '), dtype=int)
    
    # if pixel is True, subtract 1.
    if pixel: str_out  -= 1
        
    return str_out

def steles_merge_extensions(file, master_bias=None, overscan_correction=False, gain_per_amplifier=[2.22, 2.27], 
                            remove_astrometry=True, show_results=False, figure_out=None):
    """
    Open a FITS image and join its extensions in a single array.

    Args:

        file (str) : full path of the file for merging
        
        master_bias (str, default = None) : name of the master bias used for correction, if None, do overscan
        
        overscan_correction (bool, default = False) : model overscan and remove it
        
        gain_per_amplifier (list, default = [1.0,1.0]) : gain of each amplifier (in e-/ADU)
        
        remove_astrometry (bool, default = True) : remove astrometric keywords from header
        
        show_results (bool, default = False) : print the overscan fit model and
            the merged file on screen
            
    output:
    
        fits_out (ndarray): the merged array
        
        hdr_out (astropy.io.fits.header): the output header

    """
    
    #hdu = fits.open(file)
    with fits.open(file) as hdu:

        if master_bias is not None:
            # store each bias section into a list
            with fits.open(master_bias) as hdul_bias:
                bias_frames = []
                for i in range(0, 2):
                    hdu_i  = hdul_bias[i+1]
                    bias_frames.append(hdu_i.data)
                # store result as a numpy array
                bias_frames = np.array(bias_frames)

        # get gain per amplifier
        gain = np.array(gain_per_amplifier)

        if show_results:
            plt.figure(figsize=(8,8))

        # Process each extension
        for i in range(0, 2):
            hdu_i  = hdu[i+1]
            hdr_i  = hdu_i.header
            data_i = hdu_i.data

            if show_results:
                plt.subplot(2,3,2 + 3 * i)
                img = plt.imshow(data_i, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(data_i,99.5))
                plt.title(f"Amplifier #{i+1}")


                plt.subplot(2,3,1 + 3 * i)
                plt.plot(np.sum(data_i[50:4095,:],axis=0)/data_i.shape[0], label='original')
                plt.title(f"Mean profile")

            if master_bias is not None:
                data_sub = data_i - bias_frames[i,:,:]
            else:
                data_sub = overscan_subtraction(data_i, header2python(hdr_i['BIASSEC']))

            data_sub = data_sub
            data_sub *= gain[i]

            if show_results:
                plt.plot(np.sum(data_sub[50:4095,:],axis=0)/data_sub.shape[0], label='bias sub')
                plt.legend()
                plt.ylim(0,)
                plt.xlim(0,data_sub.shape[1])

                plt.subplot(2,3,3 + 3 * i)
                img2 = plt.imshow(data_sub, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(data_i,99.5)* gain[i])
                if master_bias is not None:
                    plt.title(f"Bias correction")
                else:
                    plt.title(f"Overscan correction")
                plt.colorbar(img2)
                if i == 1:
                    plt.tight_layout()
                    plt.show()

            # stores the new data using single precision float (doubles the size of the data)
            data_type = 'single'

            # extract header information in a readable way
            trim = header2python(hdr_i['TRIMSEC'])
            bias = header2python(hdr_i['BIASSEC'])
            dsec = header2python(hdr_i['DETSEC'])

            if i == 0:
                # extract header information in a readable way
                det_size = header2python(hdr_i['DETSIZE'])
                ccdsum   = header2python(hdr_i['CCDSUM'], pixel=False)

                # Save the area that corresponds to each amplifier
                bin_size = np.array(hdr_i['CCDSUM'].split(' '), dtype=int)

                # adjust for binning
                det_size[1] = det_size[1] / bin_size[0]
                det_size[3] = det_size[3] / bin_size[1] 

                # final image size
                nnx = det_size[1] - det_size[0] + 1
                nny = det_size[3] - det_size[2] + 1

                # create output image
                fits_out = np.empty((nny,nnx), dtype=data_type)

                # define new header
                hdr_out = hdr_i

                # update keywords
                hdr_out.set('NAXIS1', value=nnx, comment='length of data axis 1')
                hdr_out.set('NAXIS2', value=nny, comment='length of data axis 2')
                hdr_out.set('BZERO',  value=0,   comment='offset data range to that of unsigned short')
                hdr_out.set('EXTEND', value='F', comment='correct extensions')
                #ct = str(datetime.datetime.now())
                ct = str(datetime.now())

                filename = os.path.basename(file)
                hdr_out.set('ST_INPUT', value=filename,            comment="Original raw file")
                hdr_out.set('ST_MEXEC', value=ct,                  comment='STELES_Merge_Extensions execution time')
                hdr_out.set('ST_MERGE', value=True,                comment='STELES extensions were merged')
                hdr_out.set('ST_OVERS', value=overscan_correction, comment='STELES overscan correction')

            # trim the data
            data_out = data_sub[:,trim[0]:trim[1]+1] # add +1 at the last element to match output from IDL

            nx = data_out.shape[1]            

            # set DETSEC
            dsec[0:1] =  dsec[0:1] / bin_size[0] 
            dsec[2:3] =  dsec[2:3] / bin_size[1] 

            # add the extension to the output file 
            fits_out[dsec[2]:dsec[3]+1,dsec[0]:dsec[1]+1] = data_out
    # end HDU
        
    if show_results:
        plt.figure(figsize=(8,4))
        
        plt.subplot(121)
        plt.axvline(1022, ls=':', lw=1, color='black', alpha=0.25, label='Ampl #1 <-> Ampl #2')
        plt.plot(np.sum(fits_out[50:4095,:],axis=0)/fits_out.shape[0])
        plt.xlim(0,fits_out.shape[1])
        plt.legend(fontsize=9)
        plt.title(f"Mean profile")
        
        plt.subplot(122)
        img = plt.imshow(fits_out, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(fits_out,99.5))
        plt.colorbar(img)
        plt.title("Merged frame")
        plt.tight_layout()
        if figure_out is not None:
            plt.savefig(figure_out, format='png', dpi=300)
        plt.show()

    # remove keywords from the output header
    #del hdr_out['NEXTEND']
    #del hdr_out['DATASEC']
    #del hdr_out['BIASSEC']
    #del hdr_out['CCDSEC']
    #del hdr_out['CCDSIZE']
    #del hdr_out['AMPSEC']
    #del hdr_out['TRIMSEC']
    #del hdr_out['DETSEC']
    hdr_out.pop('NEXTEND', None)
    hdr_out.pop('DATASEC', None)
    hdr_out.pop('BIASSEC', None)
    hdr_out.pop('CCDSEC',  None)
    hdr_out.pop('CCDSIZE', None)
    hdr_out.pop('AMPSEC',  None)
    hdr_out.pop('TRIMSEC', None)
    hdr_out.pop('DETSEC',  None)
    if remove_astrometry:
        #del hdr_out['PIXSIZE1']
        #del hdr_out['PIXSIZE2']
        #del hdr_out['PIXSCAL1']
        #del hdr_out['PIXSCAL2']
        hdr_out.pop('PIXSIZE1', None)
        hdr_out.pop('PIXSIZE2', None)
        hdr_out.pop('PIXSCAL1', None)
        hdr_out.pop('PIXSCAL2', None)
    
    return fits_out, hdr_out


def overscan_subtraction(data, biassec):
    """
    Subtract row-wise overscan using median statistics.
    data:    2D ndarray (raw frame)
    biassec: tuple/list (x1, x2) overscan columns
    """
    overscan = data[:, biassec[0]+1:biassec[1]-1]
    row_bias = np.median(overscan, axis=1)
    
    return data - row_bias[:, None]


def steles_cosmicray_removal(fits_file, channel=None,
                             lacosmic_contrast=3, lacosmic_cr_threshold=10, lacosmic_neighbor_threshold=10,
                             lacosmic_gain=2.36, lacosmic_rdnoise=8.47, lacosmic_maxiter=4, 
                             save_fits=True, show_plot=False, return_cosmicray_mask=False):
    """
      A wrapper for lacosmic (https://lacosmic.readthedocs.io/en/stable/index.html) to be used on STELES data
      input parameters are:
        fits_file (str): full path and file name of the original fits file to be cleaned
        gain (float, default=2.36): gain value for the input data (red channel=2.36, blue channel=2.22, in e-/ADU units)
        rdnoise (float, default=8.47): readnoise of the input data (red channel=8.47, blue channel=6.38, in e- units)
        maxiter (int, default=4): maximum number of iterations of lacosmic().
        save_fits (bool, default=True): save results as a new fits file
        show_plot (bool, default=False): if True, show plot on screen of the input file, cleaned file and cosmic ray mask.
        return_cosmicray_mask (bool, default=False): if True, return the cosmic ray mask
        
        future implementations should replace lacosmic.lacosmic() by lacosmic.remove_cosmics()
        
    """
    
    try:
        with fits.open(fits_file) as hdul:
            img   = hdul[0].data
            hdr = hdul[0].header
    except:
        raise ValueError("Check input fits file name and path.")
    
    # make orders to increase from left to right
    if channel == "red":
        img = np.flip(img, axis=1)

    # if using older version of lacosmic
    try:
        img_clean, cr_mask = lacosmic.lacosmic(img, lacosmic_contrast, lacosmic_cr_threshold, lacosmic_neighbor_threshold,
                                               effective_gain=lacosmic_gain, readnoise=lacosmic_rdnoise, maxiter=lacosmic_maxiter)
    except:
        img_clean, cr_mask = lacosmic.remove_cosmics(img, lacosmic_contrast, lacosmic_cr_threshold, lacosmic_neighbor_threshold,
                                               effective_gain=lacosmic_gain, readnoise=lacosmic_rdnoise, maxiter=lacosmic_maxiter)

    # number of cleaned pixels
    n_cr = np.sum(cr_mask)
    
    if show_plot:
        
        plt.figure(figsize=(15,10))
        plt.subplot(131)
        if channel == 'blue':
            plt.imshow(img.T, origin='lower', vmin=0, vmax=100)
        else:
            plt.imshow(img, origin='lower', vmin=0, vmax=100)
        plt.subplot(132)
        if channel == 'blue':
            plt.imshow(img_clean.T, origin='lower', vmin=0, vmax=100)
        else:
            plt.imshow(img_clean, origin='lower', vmin=0, vmax=100)
        plt.subplot(133)
        if channel == 'blue':
            plt.imshow(cr_mask.T, origin='lower', cmap='hot')
        else:
            plt.imshow(cr_mask, origin='lower', cmap='hot')
        plt.show()
        
    # define new header
    hdr_out = hdr

    # update keywords
    hdr_out.set('ST_CRAY',  value=True, comment='STELES Cosmic Ray removal')
    hdr_out.set('ST_NCRAY', value=n_cr, comment='number of pixels affected by cosmic rays')
    hdr_out
    
    #avoiding flipping the array
    #if channel == 'red':
    #    img_clean = np.flip(img_clean,1)
    
    if save_fits:
        fits.writeto(fits_file.replace(FITS_EXTENSION,"_cleancr"+FITS_EXTENSION), img_clean, hdr_out, overwrite=True)
        
    if return_cosmicray_mask:
        return img_clean, hdr_out, cr_mask
    else:
        return img_clean, hdr_out
        
def transpose_frame(ccd, channel=None):
    """
        If working with BLUE channel, transpose the array so the orders increase along the x-axis
        If working with RED channel, flip the array so the orders increase along the x-axis
    
    """
    
    # Transpose the data
    if channel == "blue":
        ccd.data = ccd.data.T
    # Flip the data
    ccd.data = np.flip(ccd.data,axis=1)

    # Swap WCS / header axis-related keywords if they exist
    header = ccd.header

    # Common FITS keywords that need swapping for a transpose
    for key_prefix in ["NAXIS", "CRPIX", "CRVAL", "CDELT", "CTYPE", "CUNIT", "CD", "PC"]:
        # Handle the ones with two axis numbers
        if key_prefix in ["CD", "PC"]:
            keys_to_swap = []
            for i in [1, 2]:
                for j in [1, 2]:
                    k = f"{key_prefix}{i}_{j}"
                    if k in header:
                        keys_to_swap.append((i, j, header[k]))
            # Swap (1,2) with (2,1)
            for i, j, val in keys_to_swap:
                header[f"{key_prefix}{j}_{i}"] = val
        else:
            k1 = f"{key_prefix}1"
            k2 = f"{key_prefix}2"
            if k1 in header and k2 in header:
                header[k1], header[k2] = header[k2], header[k1]

    # Save transposed CCDData
    return ccd

def load_frames(directory, file_list, extension=FITS_EXTENSION):
    """
    Load a list of FITS frames as CCDData objects.

    Parameters
    ----------
    directory : str
        Directory containing the FITS files.
    file_list : list of str
        FITS filenames without the extension.
    extension : str, optional
        FITS filename extension.

    Returns
    -------
    frames : list of CCDData
        Loaded CCDData objects.
    """

    return [
        CCDData.read(os.path.join(directory, filename + extension), unit="adu")
        for filename in file_list
    ]

def load_frames_from_config(directory, config, list_key, fname_func, level="raw"):
    """
    Load frames specified by a configuration list.

    Parameters
    ----------
    directory : str
        Processing directory.
    config : dict
        Pipeline configuration dictionary.
    list_key : str
        Configuration key containing the frame list.
    fname_func : callable
        Function that converts a configured filename to the filename
        corresponding to the requested processing level.
    level : str, optional
        Processing level ("raw", "merged", etc.).

    Returns
    -------
    frames : list of CCDData
        Loaded frames.
    """

    if not config.get(list_key):
        return []

    names = [fname_func(name, level=level) for name in config[list_key]]

    return load_frames(directory, names, extension=FITS_EXTENSION)

def process_frames(ccddata_frames, header=None, sigma_clip=5,
                   interpolate_bad_columns=False, interpolate_bad_columns_sigma=2, interpolate_bad_columns_iterations=10, transpose=False):
    """ interpolate bad columns and combine frames using avsigclip method.
        uses list generated by load_fits_data()
        runs interpol_bad_columns() - not well implemented yet
        then combine the images using AVSIGCLIP method.
    """
    
    # start list to store the reduced images
    reduced_frames = []
    
    # loop into the raw images
    for frame in ccddata_frames:
        
        if interpolate_bad_columns:
            frame = interpol_bad_columns(frame, threshold_sigma=interpolate_bad_columns_sigma,
                                         iterations=interpolate_bad_columns_iterations, ccddata=True)
        # store the processed frames into a list
        reduced_frames.append(frame)
    
    # combine with average and sigma clipping method
    master_frame = combine(reduced_frames, method='average', sigma_clip=True, scale=None,
                           sigma_clip_low_thresh=sigma_clip, sigma_clip_high_thresh=sigma_clip,
                           sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=3500e6) # mem_limit = 3.5Gb(RAM)
    
    # add header keywords
    master_frame.header.set('ST_IPCOL', value=interpolate_bad_columns, comment='Interpolated bad columns' )
    master_frame.header.set('ST_COMBI', value='avsigclip',             comment='Image combining method'   )
    master_frame.header.set('ST_CSIGC', value=sigma_clip,              comment='Sigma clipping threshold' )
    master_frame.header.set('ST_NCOMB', value=len(reduced_frames),     comment='Number of combined images')
    
    return master_frame         

def scale_saturated_thar_spectrum(infolder, thar_files, saturation_threshold=65000, output_file=None, return_hdu=False):
    """
    Replaces saturated pixels in a long-exposure FITS image with scaled values 
    from a short-exposure FITS image. Ideal for ThAr lamp spectrum.

    Parameters
    ----------
    infolder : str
        Path to the FITS files.
    thar_files : str
        list of FITS files, generally a short and a long exposure.
    saturation_threshold : float, default=65000
        Pixel value above which pixels are considered saturated.
    output_file : str, optional
        Name of the resulting FITS file. if not provided, will add "_scaled.fits" at the end of the fits file.
    """

    # start list to store data
    data    = []
    headers = []
    texp    = []
    
    # Read data
    for file in thar_files:
        with fits.open(infolder / file) as hdul:
            hdr = hdul[0].header
            texp_hdr = hdr.get("EXPTIME", hdr.get("AEXPTIME")) #change on header keywords after 2027-07
            
            
            data.append(hdul[0].data.astype(np.float64))
            headers.append(hdr)
                
            #texp.append(hdul[0].header["AEXPTIME"])
            texp.append(texp_hdr)
    
    texp = np.array(texp)
    data = np.array(data)
    
    # get index for long and short exposures
    file_long = np.argmax(texp)
    file_short = np.argmin(texp)
    
    # get exposure times
    texp_long  = texp[file_long]
    texp_short = texp[file_short]

    # saves output header as the long exposure
    header_out = headers[file_long]
    
    # retrieve long and short exposures
    short_data = data[file_short,:,:]
    long_data  = data[file_long,:,:]
    
    # Scale short exposure to match long exposure
    scale_factor = texp_long / texp_short
    scaled_short = short_data * scale_factor

    # Identify saturated pixels
    saturated_mask = long_data >= saturation_threshold

    # Replace saturated pixels with scaled short exposure
    corrected_data = np.copy(long_data)
    corrected_data[saturated_mask] = scaled_short[saturated_mask]

    # Save result
    if return_hdu:
        # Create primary HDU
        hdu = fits.PrimaryHDU(data=corrected_data, header=header_out)
        return hdu
    else:
        if output_file is None:
            output_file=infolder / thar_files[0].replace(FITS_EXTENSION,"_scaled"+FITS_EXTENSION)

        fits.writeto(output_file, corrected_data, header_out, overwrite=True)

        print(f"Output saved to {output_file}")
        
def load_trace_table(filename):
    """
    Load a saved spectral-order trace table.

    Parameters
    ----------
    filename : str or Path
        Trace table filename.

    Returns
    -------
    trace_y : ndarray
        Y pixel coordinates.
    trace_x : ndarray
        X pixel coordinates.
    """

    tbl = Table.read(filename, format="ascii")

    trace_x = np.asarray(tbl["trace_x"])
    trace_y = np.asarray(tbl["trace_y"])

    return trace_y, trace_x

def read_master_tracing(trace_data, trace_default_directory, n_orders, poly_order=4, sigma_clip_threshold=5, displ_max_noise=1.5, debug=False, show_plot=False):
    """ Read positions of the master tracing measurements and perform a polynomial fit using fit_trace_stellar_profile() """
    # start an empty list
    trace_poly_coeffs_init = []

    # loop through the orders to create a list of tracing coefficients
    for n in range(n_orders):

        trace_coeffs = fit_trace_stellar_profile(trace_data, trace_default_directory / f"trace_xy_{n+1:02d}{TABLE_EXTENSION}", 
                                                 order=n+1, poly_order=poly_order, sigma_clip_threshold=sigma_clip_threshold, 
                                                        show_plot=show_plot, displ_max_noise=displ_max_noise, debug=debug)
        # append the tracing coefficients to the list
        trace_poly_coeffs_init.append(trace_coeffs)
        
    return trace_poly_coeffs_init


def read_trace(filename=f"trace{TABLE_EXTENSION}"):
    tbl = Table.read(filename, format="ascii")
    trace_x = np.array(tbl["trace_x"])
    trace_y = np.array(tbl["trace_y"])
    return trace_x, trace_y


def fit_trace_stellar_profile(fits_file, trace_file, order, poly_order=4, sigma_clip_threshold=3, y_range=[500,3500],
                              show_plot=True, displ_max_noise=10, debug=True):
    """
    Trace a stellar profile along the y-axis and fit a polynomial x=f(y).

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    x_guess : float
        Initial guess of the x position at y_center.
    y_center : int
        Central y position (default=2048).
    half_width : int
        Half-width of the extraction box around the current x position.
    step : int
        Step size in y when tracing.
    poly_order : int
        Degree of the polynomial fit.

    Returns
    -------
    y_points : ndarray
        Y positions traced.
    x_points : ndarray
        Measured X positions along the trace.
    poly_coeffs : ndarray
        Polynomial coefficients for x=f(y).
    """

    # Load image
    if isinstance(fits_file, str):
        image = fits.getdata(fits_file)
    else:
        image = np.asarray(fits_file)

    # Load image
    #image = fits.getdata(fits_file)
    ny, nx = image.shape

    # Estimate background noise
    background = np.median(image)
    noise = 1.4826 * np.median(np.abs(image - background))
    
    # Define y-range
    if y_range is None:
        y_min, y_max = 0, ny - 1
    else:
        y_min, y_max = y_range

    x_points, y_points = read_trace(trace_file)

    # apply sigma-clip on datapoints
    clip_mask = ~sigma_clip(x_points, sigma=sigma_clip_threshold, maxiters=5).mask
    
    # Fit polynomial x=f(y)
    coeffs = np.polyfit(y_points[clip_mask], x_points[clip_mask], deg=poly_order)
    poly = np.poly1d(coeffs)

    res = abs(poly(y_points) - x_points)
    
    # Fit polynomial x=f(y)
    coeffs = np.polyfit(y_points[clip_mask], x_points[clip_mask], deg=poly_order)
    poly = np.poly1d(coeffs)

    # Plot trace
    if show_plot:
        plt.figure(figsize=(10,10))
        plt.imshow(image, origin="lower", cmap="gray", aspect='auto', vmin=0, vmax=displ_max_noise*noise)
        plt.plot(x_points[clip_mask], y_points[clip_mask], '.g', markersize=5, label=f'Measured [{np.sum(clip_mask)}/{len(x_points)}]')
        plt.plot(x_points[~clip_mask], y_points[~clip_mask], 'xr', markersize=5, label=f'Outliers [{np.sum(~clip_mask)}]')
        plt.plot(poly(y_points), y_points, '--', label=f'Trace [ndeg={poly_order}]')
        plt.legend()
        pix_border=10
        plt.xlim(min(x_points)-pix_border,max(x_points)+pix_border)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Order={order}")
        plt.show()

    return coeffs

import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, amp, x0, sigma, offset):
    return amp * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def fit_order_center(profile, xpix):
    """ Function to determine the center of the spatial profile of each spectral order using a gaussian model """

    # remove local background
    background = np.median(profile)
    profile = profile - background

    if np.max(profile) <= 0:
        return None

    amp0 = np.max(profile)
    x0 = xpix[np.argmax(profile)]
    sigma0 = 3.0

    try:

        popt, _ = curve_fit(
            gaussian,
            xpix,
            profile,
            p0=[amp0, x0, sigma0, 0],
            bounds=(
                [0, xpix.min(), 0.5, -np.inf],
                [np.inf, xpix.max(), 20.0, np.inf]
            )
        )

        amp, center, sigma, offset = popt

        fwhm = 2.355 * sigma

        return center, amp, fwhm

    except:
        return None
    
def estimate_center_fwhm(profile, xpix):
    """ Function to determine the half-peak width of theof the spatial profile """

    background = np.median(profile)
    profile = profile - background

    peak = np.max(profile)

    if peak <= 0:
        return None

    halfmax = peak / 2.0

    above = np.where(profile > halfmax)[0]

    if len(above) < 2:
        return None

    left = xpix[above[0]]
    right = xpix[above[-1]]

    return 0.5 * (left + right)

def trace_stellar_profile(
        fits_file,
        x_guess,
        y_center=2048,
        y_range=[500,3500],
        half_width=10,
        sigma_threshold=3,
        step=1,
        poly_order=4,
        number_of_iterations=3,
        collapse_rows=5,
        max_shift=2.0,
        max_failures=5,
        use_gaussian=True,
        use_fwhm_fallback=True,
        show_plot=True,
        displ_max_noise=10,
        debug=True):
    """
    Trace a stellar profile along the y-axis and fit a polynomial x=f(y).

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    x_guess : float
        Initial guess of the x position at y_center.
    y_center : int
        Central y position (default=2048).
    half_width : int
        Half-width of the extraction box around the current x position.
    step : int
        Step size in y when tracing.
    poly_order : int
        Degree of the polynomial fit.

    Returns
    -------
    y_points : ndarray
        Y positions traced.
    x_points : ndarray
        Measured X positions along the trace.
    poly_coeffs : ndarray
        Polynomial coefficients for x=f(y).
    """

    # Load image
    if isinstance(fits_file, str):
        image = fits.getdata(fits_file)
    else:
        image = np.asarray(fits_file)


    # Load image
    #image = fits.getdata(fits_file)
    ny, nx = image.shape

    # Estimate background noise
    background = np.median(image)
    noise = 1.4826 * np.median(np.abs(image - background))
    
    # Define y-range
    if y_range is None:
        y_min, y_max = 0, ny - 1
    else:
        y_min, y_max = y_range
    
    trace_half_height = collapse_rows // 2

    x_min = int(max(0, x_guess - half_width))
    x_max = int(min(nx, x_guess + half_width + 1))
    
    profile = np.sum(image[max(0, y_center-trace_half_height):min(ny, y_center+trace_half_height+1), x_min:x_max],axis=0)

    xpix = np.arange(x_min, x_max)

    fit_result = fit_order_center(profile, xpix)

    if fit_result is not None:
        x_center_fit, amp, fwhm = fit_result

    else:
        x_center_fit = x_guess
    
    fwhm_threshold = 10
    
    def follow_trace(direction, x_start, debug=debug):
        """Trace in one direction (+1 for up, -1 for down)"""

        y_positions = []
        x_positions = []

        #x_current = float(x_guess)
        x_current = float(x_start)

        n_fail = 0

        y = y_center
        y = y_center + direction * step

        while (direction > 0 and y <= y_max) or \
              (direction < 0 and y >= y_min):

            x_min = int(max(0, x_current - half_width))
            x_max = int(min(nx, x_current + half_width + 1))

            if x_max - x_min < 5:
                break

            # --------------------------------------------------
            # collapse several rows to improve S/N
            # --------------------------------------------------

            trace_half_height = 2

            y1 = max(0, y - trace_half_height)
            y2 = min(ny, y + trace_half_height + 1)

            profile = np.sum(
                image[y1:y2, x_min:x_max],
                axis=0
            )

            xpix = np.arange(x_min, x_max)

            peak = np.max(profile) - np.median(profile)
            
            if peak < sigma_threshold * noise:

                n_fail += 1

                if n_fail >= max_failures:
                    if debug:
                        print(f"Trace stopped at y={y}")
                    break

                y += direction * step
                continue
                
            

            # --------------------------------------------------
            # Gaussian fit
            # --------------------------------------------------

            fit_result = fit_order_center(profile, xpix)

            success = False

            if fit_result is not None:

                center, amp, fwhm = fit_result

                shift = np.abs(center - x_current)
                

                if shift < max_shift and fwhm < fwhm_threshold and fwhm > 1.5:

                    x_current = center
                    success = True
                
                
                if debug:
                    print(
                        f"y={y:4d} "
                        f"x_current={x_current:8.2f} "
                        f"fit={center:8.2f} "
                        f"shift={shift:5.2f}/{max_shift} "
                        f"fwhm={fwhm:5.2f}/{fwhm_threshold} "
                        f"success={success}"
                    )
            
            # --------------------------------------------------
            # fallback: FWHM midpoint
            # --------------------------------------------------

            success_fwhm = False
            
            if not success:

                center = estimate_center_fwhm(profile, xpix)

                if center is not None:

                    shift = np.abs(center - x_current)

                    if shift < max_shift:

                        x_current = center
                        success = True

                        # save for plotting
                        success_fwhm = success
                    
                        
            # --------------------------------------------------
            # bookkeeping
            # --------------------------------------------------

            if success:

                n_fail = 0

                y_positions.append(y)
                x_positions.append(x_current)

            else:

                n_fail += 1

                if n_fail >= max_failures:
                    break

                # keep previous solution
                y_positions.append(y)
                x_positions.append(x_current)

            y += direction * step

        return np.array(y_positions), np.array(x_positions)

    # Trace upward and downward
    y_up,   x_up   = follow_trace(+1, x_center_fit)
    y_down, x_down = follow_trace(-1, x_center_fit)

    # Combine both directions and include center
    #y_points = np.concatenate((y_down[::-1], [y_center], y_up))
    #x_points = np.concatenate((x_down[::-1], [x_guess], x_up))
    y_points = np.concatenate((y_down[::-1], [y_center], y_up))
    x_points = np.concatenate((x_down[::-1], [x_center_fit], x_up))
    
    ## new code here
    for _ in range(number_of_iterations):
        
        if len(y_points) < poly_order + 1:
            raise RuntimeError(f"Insufficient valid trace points "
                               f"({len(y_points)}) for polynomial degree {poly_order}")

        coeff = np.polyfit(y_points, x_points, poly_order)

        model = np.polyval(coeff, y_points)

        resid = x_points - model

        sigma = 1.4826 * np.median(np.abs(resid))

        good = np.abs(resid) < 3*sigma

        y_points = y_points[good]
        x_points = x_points[good]
    
    # Plot trace
    if show_plot:
        plt.figure(figsize=(10,10))
        plt.imshow(image, origin="lower", cmap="gray", aspect='auto', vmin=0, vmax=displ_max_noise*noise)
        plt.plot(x_points, y_points, '+r', markersize=5, label=f'Measured [{len(x_points)}]')
        plt.legend()
        pix_border=10
        plt.xlim(min(x_points)-pix_border,max(x_points)+pix_border)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()    
    
    return y_points, x_points


def save_trace_table(trace_y, trace_x, n_order, output_dir, overwrite=True):
    """ reads the (x,y) position of the trace and saves it as a txt file """
    tbl = Table([trace_x, trace_y], names=("trace_x", "trace_y"))
    tbl.write(output_dir / f"trace_xy_{n_order:02d}{TABLE_EXTENSION}", format="ascii", overwrite=overwrite)

##################
def refit_trace_prev(trace_data, trace_channel, trace_poly_coeffs, trace_directory, trace_window=20, trace_ndeg=4, sigma_clip_threshold=3., skip_orders=[None],
                plot_results=True, plot_transpose=True, plot_percentile=[1,99], quiet=True, output_file_prefix=None):
    
    # avoid warning messages
    if quiet:
        warnings.filterwarnings('ignore', message="fit may be poorly conditioned")
    

    fitter     = fitting.LinearLSQFitter()

    refined_trace_poly = []
    for n, trace_poly in enumerate(trace_poly_coeffs):

        poly_guess = np.poly1d(trace_poly)

        trace_db_file = trace_directory / f"trace_xy_{n+1:02d}{TABLE_EXTENSION}"

        x_vals, y_vals = read_trace(trace_db_file)

        
        if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):

            refit_trace = True
            
            # --- 2. Prepare arrays for refined trace ---
            x_refined  = np.zeros_like(y_vals, dtype=float)
            x_shift    = np.zeros_like(y_vals, dtype=float)
            
            
            # --- 3. Loop through y positions to refine the trace ---
            for i, y in enumerate(y_vals):

                x_guess = poly_guess(y)

                # Define window range safely to avoid adding positions outside the CCD
                x_min = max(int(x_guess - trace_window), 0)
                x_max = min(int(x_guess + trace_window), trace_data.shape[1]-1)

                # sum the trace over a few pixels in the dispersion direction
                n_pix_sum = 3

                if n_pix_sum > 0:
                    profile = np.sum(trace_data[int(y-n_pix_sum):int(y+n_pix_sum), int(x_min):int(x_max+1)], axis=0)
                else:
                    profile = trace_data[int(y), int(x_min):int(x_max+1)]

                x_window = np.arange(x_min, x_max+1)

                # Smooth the profile to reduce noise / spectral line impact
                profile_smooth = medfilt(profile, kernel_size=3)
                # position of the peak
                x_refined_peak = x_window[np.argmax(profile)]
                peak_max = np.argmax(profile)
                try:
                    peak_max2, _ = refine_peak_positions(profile_smooth, [peak_max], window=trace_window, 
                                                         plot_fits=False, n_col=1, output=None)
                    x_refined_peak2 = x_window[peak_max]
                except:
                    x_refined_peak2 = x_refined_peak

                x_refined[i] = x_refined_peak2

            # --- 4. Keep only valid positions ---
            y_valid = y_vals#[mask_valid]
            x_valid = x_refined#[mask_valid]

            # --- 5. Sigma-clip outliers relative to first guess ---
            poly_fit = fitter(models.Polynomial1D(degree=trace_ndeg), y_valid, x_valid)
            residuals = x_valid - poly_guess(y_valid)
            mask_clip = sigma_clip(residuals, sigma=sigma_clip_threshold).mask

            x_clean = x_valid[~mask_clip]
            y_clean = y_valid[~mask_clip]

            shift_trace = np.median(x_shift)

            # --- 6. Fit refined polynomial ---
            poly_refined = fitter(models.Polynomial1D(degree=trace_ndeg), y_clean, x_clean)
            poly_coeffs = poly_refined.parameters
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            poly_coeffs = np.flip(poly_coeffs)
            refined_trace_poly.append(poly_coeffs)


        else: # if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
            print(f"Skipping tracing refit for order n={n+1}")
            
            refit_trace = False
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            #poly_coeffs = np.flip(poly_guess)
            poly_coeffs = np.array(poly_guess)
            refined_trace_poly.append(poly_coeffs)
        

        dx = np.max(x_vals) - np.min(x_vals)

        # --- 7. Plot result for visual check ---

        if plot_results:

            xmin, xmax = np.min(x_vals) - dx*0.2, np.max(x_vals) + dx*0.2
            if xmin < 0:
                xmin = 0
            if xmax > trace_data.shape[1]:
                xmax = trace_data.shape[1]
                

            y_range=[ 500,1500] if trace_channel == 'blue' else [1000,3000]

            vmin, vmax = np.percentile(trace_data[y_range,int(xmin):int(xmax)], plot_percentile)
                
            if plot_transpose:
                plt.figure(figsize=(14,4))
                plt_img = plt.imshow(np.transpose(trace_data), origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(y_vals, poly_guess(y_vals),   ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(y_vals, poly_refined(y_vals), ls='--', color='red',  lw=1, label='Refined trace')
                plt.scatter(y_vals,              x_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace:
                    plt.scatter(y_valid[~mask_clip], x_valid[~mask_clip], s= 5, color='black', label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(y_valid[mask_clip],  x_valid[mask_clip],  s=10, marker='x', color='red', label=f'Excluded points ({np.sum(mask_clip)})')
                plt.ylim(xmin,xmax)
                
                #plt.xlim(min(y_vals)-50,max(y_vals)+50)
                ymin = max(min(y_vals)-50, 0)
                ymax = min(max(y_vals)+50, trace_data.shape[0])
                
                plt.xlim(ymin,ymax)
                plt.ylabel('X pixel')
                plt.xlabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(f'Order Tracing Refinement (n={n+1})')
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()

            else:
                plt.figure(figsize=(4,8))
                plt_img = plt.imshow(trace_data, origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(poly_guess(y_vals),   y_vals, ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(poly_refined(y_vals), y_vals, ls='--', color='red',  lw=1, label='Refined trace')
                plt.scatter(x_vals,              y_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace:
                    plt.scatter(x_valid[~mask_clip], y_valid[~mask_clip], s= 5, color='black',
                                label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(x_valid[mask_clip],  y_valid[mask_clip],  s=10, marker='x', color='red',
                                label=f'Excluded points ({np.sum(mask_clip)})')
                plt.xlim(xmin,xmax)
                plt.ylim(min(y_vals)-50,max(y_vals)+50)
                plt.xlabel('X pixel')
                plt.ylabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(f'Order Tracing Refinement (n={n+1})')
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()
        
        
    return refined_trace_poly


##################

def refit_trace(trace_data, trace_channel, trace_poly_coeffs, trace_directory, trace_window=20, trace_ndeg=4, sigma_clip_threshold=3., skip_orders=[None],
                plot_results=True, plot_transpose=True, plot_percentile=[1,99], quiet=True, output_file_prefix=None,
                use_reference_fallback=True, reference_poly_coeffs=None,
                divergence_threshold=3.0, fallback_y_center=None, fallback_half_width=15,
                fallback_collapse_rows=5, y_ranges_valid=None):
    """
    Refit spectral order traces on the observed flat/quartz frame.
 
    For orders where the refined polynomial diverges from the reference by more than
    ``divergence_threshold`` pixels (measured at the order centre), the function falls
    back to the reference polynomial shape and applies only a rigid pixel offset
    estimated from the data at that centre row.  When the signal is too faint to
    measure the offset reliably the reference polynomial is used unchanged (offset=0).
 
    Parameters
    ----------
    trace_data : ndarray
        2-D CCD frame used for tracing (flat / quartz).
    trace_channel : str
        ``'blue'`` or ``'red'`` – controls the display y-range in plots.
    trace_poly_coeffs : list of ndarray
        Initial polynomial coefficients x=f(y) for each order, as returned by
        ``read_master_tracing``.  These serve as the starting point for the refit
        and, when ``use_reference_fallback=True``, also as the reference shape if
        ``reference_poly_coeffs`` is *not* supplied.
    trace_directory : Path
        Directory containing the ``trace_xy_NN.txt`` reference tables.
    trace_window : int
        Half-width in x (pixels) of the search window for peak finding.
    trace_ndeg : int
        Polynomial degree for the refined fit.
    sigma_clip_threshold : float
        Sigma threshold for outlier rejection during refit.
    skip_orders : list of int or [None]
        Order numbers (1-based) to skip the refit for.  ``[None]`` means none.
    plot_results : bool
        Plot each order after refitting.
    plot_transpose : bool
        Transpose image in plot (x along y-axis).
    plot_percentile : list of two floats
        Percentile range for image display scaling.
    quiet : bool
        Suppress polynomial-fit warnings.
    output_file_prefix : Path or None
        If set, save per-order plots to ``<prefix>_nNN<PLOT_EXTENSION>``.
 
    use_reference_fallback : bool
        Enable the reference-table fallback for diverging orders (default True).
    reference_poly_coeffs : list of ndarray or None
        Reference polynomial coefficients to use as the fallback shape.  When
        ``None``, ``trace_poly_coeffs`` (the initial guess) is used as the
        reference.  Pass the coefficients loaded from an earlier epoch's DB here
        if you want a truly independent reference.
    divergence_threshold : float
        Maximum allowed shift in pixels between the refined polynomial and the
        reference at the order centre before the fallback is triggered
        (default 3.0 px).
    fallback_y_center : int or None
        Row at which the centre offset is measured.  ``None`` uses the midpoint
        of the order's y-range from the reference table.
    fallback_half_width : int
        Half-width in x (pixels) of the window used to measure the offset peak
        at ``fallback_y_center`` (default 15).
    fallback_collapse_rows : int
        Number of rows collapsed around ``fallback_y_center`` for the offset
        profile (default 5).
    y_ranges_valid : dict of {int: (int, int)} or None
        Per-order (1-based) tuple ``(y_min, y_max)`` giving the y-range where
        the order was actually detected in the observation (i.e. the span
        returned by ``trace_stellar_profile``).  When supplied:
 
        - The peak-finding loop (step 3) uses ``poly_ref`` as the x-guess
          outside the valid range instead of ``poly_guess``, so the search
          window stays centred correctly even where the observed polynomial
          extrapolates badly.
        - Points outside the valid range are excluded from the sigma-clip and
          polynomial refit (steps 5–6); inside the valid range the normal refit
          proceeds as usual.
        - The fallback offset measurement (step 6b) is clamped to within the
          valid y-range so it is always measured where the flux is reliable.
 
        ``None`` (default) disables all of the above and the function behaves
        as before.
 
    Returns
    -------
    refined_trace_poly : list of ndarray
        Refined polynomial coefficients for every order.
    fallback_orders : list of int
        1-based order numbers for which the fallback was applied.
    """
 
    # avoid warning messages
    if quiet:
        warnings.filterwarnings('ignore', message="fit may be poorly conditioned")
 
    # If no independent reference is supplied, use the initial guess coefficients
    if reference_poly_coeffs is None:
        reference_poly_coeffs = trace_poly_coeffs
 
    fitter = fitting.LinearLSQFitter()
 
    refined_trace_poly = []
    fallback_orders    = []          # track which orders used the fallback
 
    for n, trace_poly in enumerate(trace_poly_coeffs):
 
        poly_guess = np.poly1d(trace_poly)
        poly_ref   = np.poly1d(reference_poly_coeffs[n])

        trace_db_file = trace_directory / f"trace_xy_{n+1:02d}{TABLE_EXTENSION}"
 
        x_vals, y_vals = read_trace(trace_db_file)
 
        
        if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
 
            refit_trace = True
 
            # --- Resolve per-order valid y-range ---
            # y_obs_min/max: the rows where the order was actually detected.
            # Outside this range the observed polynomial may extrapolate badly,
            # so we use poly_ref as the search anchor and exclude those rows
            # from the refit.
            if y_ranges_valid is not None and (n + 1) in y_ranges_valid:
                y_obs_min, y_obs_max = y_ranges_valid[n + 1]
            else:
                y_obs_min, y_obs_max = int(y_vals.min()), int(y_vals.max())
 
            # Boolean mask: True where the row is inside the observed range
            mask_obs_range = (y_vals >= y_obs_min) & (y_vals <= y_obs_max)
 
            # --- 2. Prepare arrays for refined trace ---
            x_refined  = np.zeros_like(y_vals, dtype=float)
            x_shift    = np.zeros_like(y_vals, dtype=float)
            
            
            # --- 3. Loop through y positions to refine the trace ---
            for i, y in enumerate(y_vals):
 
                # Inside the observed range: trust poly_guess as the anchor.
                # Outside: use poly_ref so the search window stays centred even
                # where the observed polynomial extrapolates badly.
                if mask_obs_range[i]:
                    x_guess = poly_guess(y)
                else:
                    x_guess = poly_ref(y)
 
                # Define window range safely to avoid adding positions outside the CCD
                x_min = max(int(x_guess - trace_window), 0)
                x_max = min(int(x_guess + trace_window), trace_data.shape[1]-1)
 
                # sum the trace over a few pixels in the dispersion direction
                n_pix_sum = 3
 
                if n_pix_sum > 0:
                    profile = np.sum(trace_data[int(y-n_pix_sum):int(y+n_pix_sum), int(x_min):int(x_max+1)], axis=0)
                else:
                    profile = trace_data[int(y), int(x_min):int(x_max+1)]
 
                x_window = np.arange(x_min, x_max+1)
 
                # Smooth the profile to reduce noise / spectral line impact
                profile_smooth = medfilt(profile, kernel_size=3)
                # position of the peak
                x_refined_peak = x_window[np.argmax(profile)]
                peak_max = np.argmax(profile)
                try:
                    peak_max2, _ = refine_peak_positions(profile_smooth, [peak_max], window=trace_window, 
                                                         plot_fits=False, n_col=1, output=None)
                    x_refined_peak2 = x_window[peak_max]
                except:
                    x_refined_peak2 = x_refined_peak
 
                x_refined[i] = x_refined_peak2
 
            # --- 4. Keep only valid positions ---
            # Restrict refit to rows within the observed y-range so noisy
            # border rows do not pull the polynomial fit.
            y_valid = y_vals[mask_obs_range]
            x_valid = x_refined[mask_obs_range]
 
            # --- 5. Sigma-clip outliers relative to first guess ---
            poly_fit = fitter(models.Polynomial1D(degree=trace_ndeg), y_valid, x_valid)
            residuals = x_valid - poly_guess(y_valid)
            mask_clip = sigma_clip(residuals, sigma=sigma_clip_threshold).mask
 
            x_clean = x_valid[~mask_clip]
            y_clean = y_valid[~mask_clip]
 
            shift_trace = np.median(x_shift)
 
            # --- 6. Fit refined polynomial ---
            poly_refined = fitter(models.Polynomial1D(degree=trace_ndeg), y_clean, x_clean)
            poly_coeffs = poly_refined.parameters
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            poly_coeffs = np.flip(poly_coeffs)
 
            # --- 6b. Reference-table fallback for diverging orders ---
            used_fallback = False
            if use_reference_fallback:
 
                # Evaluate both polynomials at the order centre row.
                # Clamp y_fb to within the observed valid range so the offset
                # is always measured where the flux is reliable.
                if fallback_y_center is not None:
                    y_fb = int(np.clip(fallback_y_center, y_obs_min, y_obs_max))
                else:
                    y_fb = int(np.clip(np.median(y_vals), y_obs_min, y_obs_max))
 
                x_ref_center      = poly_ref(y_fb)
                x_refined_center  = np.poly1d(poly_coeffs)(y_fb)
                divergence        = abs(x_refined_center - x_ref_center)
 
                if divergence > divergence_threshold:
 
                    # --- measure a pixel offset at y_fb from the data ---
                    # Use poly_guess (the observed polynomial, which already tracks
                    # the order accurately) as the Gaussian search anchor rather
                    # than x_ref_center.  This prevents the search window from
                    # drifting onto an adjacent brighter order when the reference
                    # is offset from the current observation.
                    ny, nx = trace_data.shape
                    x_obs_center = poly_guess(y_fb)   # observed order position at y_fb
                    x_fb_min = max(int(x_obs_center) - fallback_half_width, 0)
                    x_fb_max = min(int(x_obs_center) + fallback_half_width + 1, nx)
 
                    r_half = fallback_collapse_rows // 2
                    y_fb_min = max(y_fb - r_half, 0)
                    y_fb_max = min(y_fb + r_half + 1, ny)
 
                    profile_fb = np.sum(
                        trace_data[y_fb_min:y_fb_max, x_fb_min:x_fb_max],
                        axis=0
                    )
                    xpix_fb = np.arange(x_fb_min, x_fb_max)
 
                    offset = 0.0   # default: use reference as-is
                    fit_fb = fit_order_center(profile_fb, xpix_fb)
                    if fit_fb is not None:
                        x_peak_measured, x_sigma, _ = fit_fb
                        # Validate fit: reject if sigma is degenerate (too small or
                        # too large) or if peak landed at the window edge, which
                        # indicates the Gaussian locked onto the wrong feature.
                        at_edge = (abs(x_peak_measured - x_fb_min) < 1.5 or
                                   abs(x_peak_measured - (x_fb_max - 1)) < 1.5)
                        bad_sigma = (x_sigma < 0.3 or x_sigma > fallback_half_width)
                        if not at_edge and not bad_sigma:
                            # offset = shift needed to move poly_ref onto the observed order
                            offset = x_peak_measured - x_ref_center
                        else:
                            # Gaussian unreliable — fall back to the median offset
                            # between poly_guess and poly_ref across the valid y-range
                            y_sample = np.linspace(y_obs_min, y_obs_max, 20)
                            offset = float(np.median(
                                np.polyval(np.array(poly_guess), y_sample) -
                                np.polyval(np.array(poly_ref),   y_sample)
                            ))
                            if not quiet:
                                print(f"  Order {n+1:02d}: Gaussian unreliable at y={y_fb} "
                                      f"(edge={at_edge}, bad_sigma={bad_sigma}) "
                                      f"→ using median offset={offset:+.2f} px")
 
                    # Shift the reference polynomial by the measured offset.
                    # Only the constant term (coeffs[-1] in numpy convention) changes;
                    # the shape of the curve is preserved from the reference epoch.
                    ref_coeffs = np.array(poly_ref)        # copy reference coefficients
                    ref_coeffs[-1] += offset               # numpy poly: highest degree first
 
                    poly_coeffs = ref_coeffs
                    poly_refined = np.poly1d(poly_coeffs)  # keep poly_refined consistent for plotting
 
                    used_fallback = True
                    fallback_orders.append(n + 1)
 
                    if not quiet:
                        print(
                            f"  Order {n+1:02d}: divergence={divergence:.2f} px > threshold={divergence_threshold:.2f} px "
                            f"→ fallback applied (offset={offset:+.2f} px at y={y_fb})"
                        )
 
            refined_trace_poly.append(poly_coeffs)  # always append, whether fallback or normal refit
 
 
        else: # if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
            print(f"Skipping tracing refit for order n={n+1}")
            
            refit_trace = False
            used_fallback = False
 
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            #poly_coeffs = np.flip(poly_guess)
            poly_coeffs = np.array(poly_guess)
            refined_trace_poly.append(poly_coeffs)
        
 
        dx = np.max(x_vals) - np.min(x_vals)
 
        # --- 7. Plot result for visual check ---
 
        if plot_results:
 
            xmin, xmax = np.min(x_vals) - dx*0.2, np.max(x_vals) + dx*0.2
            if xmin < 0:
                xmin = 0
            if xmax > trace_data.shape[1]:
                xmax = trace_data.shape[1]
                
 
            y_range=[ 500,1500] if trace_channel == 'blue' else [1000,3000]
 
            vmin, vmax = np.percentile(trace_data[y_range,int(xmin):int(xmax)], plot_percentile)
 
            # Build a title that flags fallback use
            order_title = f'Order Tracing Refinement (n={n+1})'
            if used_fallback:
                order_title += '  [FALLBACK]'
                
            if plot_transpose:
                plt.figure(figsize=(14,4))
                plt_img = plt.imshow(np.transpose(trace_data), origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(y_vals, poly_guess(y_vals),   ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(y_vals, poly_refined(y_vals), ls='--', color='red',  lw=1,
                             label='Fallback trace' if used_fallback else 'Refined trace')
                if used_fallback:
                    plt.plot(y_vals, poly_ref(y_vals), ls='-.', color='cyan', lw=1, label='Reference shape')
                plt.scatter(y_vals,              x_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace and not used_fallback:
                    plt.scatter(y_valid[~mask_clip], x_valid[~mask_clip], s= 5, color='black', label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(y_valid[mask_clip],  x_valid[mask_clip],  s=10, marker='x', color='red', label=f'Excluded points ({np.sum(mask_clip)})')
                    plt.scatter(y_vals[~mask_obs_range], x_refined[~mask_obs_range], s=5, marker='s', color='orange', label=f'Outside obs range ({np.sum(~mask_obs_range)})')
                plt.ylim(xmin,xmax)
                
                #plt.xlim(min(y_vals)-50,max(y_vals)+50)
                ymin = max(min(y_vals)-50, 0)
                ymax = min(max(y_vals)+50, trace_data.shape[0])
                
                plt.xlim(ymin,ymax)
                plt.ylabel('X pixel')
                plt.xlabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(order_title)
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()
 
            else:
                plt.figure(figsize=(4,8))
                plt_img = plt.imshow(trace_data, origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(poly_guess(y_vals),   y_vals, ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(poly_refined(y_vals), y_vals, ls='--', color='red',  lw=1,
                             label='Fallback trace' if used_fallback else 'Refined trace')
                if used_fallback:
                    plt.plot(poly_ref(y_vals), y_vals, ls='-.', color='cyan', lw=1, label='Reference shape')
                plt.scatter(x_vals,              y_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace and not used_fallback:
                    plt.scatter(x_valid[~mask_clip], y_valid[~mask_clip], s= 5, color='black',
                                label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(x_valid[mask_clip],  y_valid[mask_clip],  s=10, marker='x', color='red',
                                label=f'Excluded points ({np.sum(mask_clip)})')
                    plt.scatter(x_refined[~mask_obs_range], y_vals[~mask_obs_range], s=5, marker='s', color='orange',
                                label=f'Outside obs range ({np.sum(~mask_obs_range)})')
                plt.xlim(xmin,xmax)
                plt.ylim(min(y_vals)-50,max(y_vals)+50)
                plt.xlabel('X pixel')
                plt.ylabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(order_title)
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()
        
        
    return refined_trace_poly, fallback_orders

def refit_trace_last(trace_data, trace_channel, trace_poly_coeffs, trace_directory, trace_window=20, trace_ndeg=4, sigma_clip_threshold=3., skip_orders=[None],
                plot_results=True, plot_transpose=True, plot_percentile=[1,99], quiet=True, output_file_prefix=None,
                use_reference_fallback=True, reference_poly_coeffs=None,
                divergence_threshold=3.0, fallback_y_center=None, fallback_half_width=15,
                fallback_collapse_rows=5, y_ranges_valid=None):
    """
    Refit spectral order traces on the observed flat/quartz frame.
 
    For orders where the refined polynomial diverges from the reference by more than
    ``divergence_threshold`` pixels (measured at the order centre), the function falls
    back to the reference polynomial shape and applies only a rigid pixel offset
    estimated from the data at that centre row.  When the signal is too faint to
    measure the offset reliably the reference polynomial is used unchanged (offset=0).
 
    Parameters
    ----------
    trace_data : ndarray
        2-D CCD frame used for tracing (flat / quartz).
    trace_channel : str
        ``'blue'`` or ``'red'`` – controls the display y-range in plots.
    trace_poly_coeffs : list of ndarray
        Initial polynomial coefficients x=f(y) for each order, as returned by
        ``read_master_tracing``.  These serve as the starting point for the refit
        and, when ``use_reference_fallback=True``, also as the reference shape if
        ``reference_poly_coeffs`` is *not* supplied.
    trace_directory : Path
        Directory containing the ``trace_xy_NN.txt`` reference tables.
    trace_window : int
        Half-width in x (pixels) of the search window for peak finding.
    trace_ndeg : int
        Polynomial degree for the refined fit.
    sigma_clip_threshold : float
        Sigma threshold for outlier rejection during refit.
    skip_orders : list of int or [None]
        Order numbers (1-based) to skip the refit for.  ``[None]`` means none.
    plot_results : bool
        Plot each order after refitting.
    plot_transpose : bool
        Transpose image in plot (x along y-axis).
    plot_percentile : list of two floats
        Percentile range for image display scaling.
    quiet : bool
        Suppress polynomial-fit warnings.
    output_file_prefix : Path or None
        If set, save per-order plots to ``<prefix>_nNN<PLOT_EXTENSION>``.
 
    use_reference_fallback : bool
        Enable the reference-table fallback for diverging orders (default True).
    reference_poly_coeffs : list of ndarray or None
        Reference polynomial coefficients to use as the fallback shape.  When
        ``None``, ``trace_poly_coeffs`` (the initial guess) is used as the
        reference.  Pass the coefficients loaded from an earlier epoch's DB here
        if you want a truly independent reference.
    divergence_threshold : float
        Maximum allowed shift in pixels between the refined polynomial and the
        reference at the order centre before the fallback is triggered
        (default 3.0 px).
    fallback_y_center : int or None
        Row at which the centre offset is measured.  ``None`` uses the midpoint
        of the order's y-range from the reference table.
    fallback_half_width : int
        Half-width in x (pixels) of the window used to measure the offset peak
        at ``fallback_y_center`` (default 15).
    fallback_collapse_rows : int
        Number of rows collapsed around ``fallback_y_center`` for the offset
        profile (default 5).
    y_ranges_valid : dict of {int: (int, int)} or None
        Per-order (1-based) tuple ``(y_min, y_max)`` giving the y-range where
        the order was actually detected in the observation (i.e. the span
        returned by ``trace_stellar_profile``).  When supplied:
 
        - The peak-finding loop (step 3) uses ``poly_ref`` as the x-guess
          outside the valid range instead of ``poly_guess``, so the search
          window stays centred correctly even where the observed polynomial
          extrapolates badly.
        - Points outside the valid range are excluded from the sigma-clip and
          polynomial refit (steps 5–6); inside the valid range the normal refit
          proceeds as usual.
        - The fallback offset measurement (step 6b) is clamped to within the
          valid y-range so it is always measured where the flux is reliable.
 
        ``None`` (default) disables all of the above and the function behaves
        as before.
 
    Returns
    -------
    refined_trace_poly : list of ndarray
        Refined polynomial coefficients for every order.
    fallback_orders : list of int
        1-based order numbers for which the fallback was applied.
    """
 
    # avoid warning messages
    if quiet:
        warnings.filterwarnings('ignore', message="fit may be poorly conditioned")
 
    # If no independent reference is supplied, use the initial guess coefficients
    if reference_poly_coeffs is None:
        reference_poly_coeffs = trace_poly_coeffs
 
    fitter = fitting.LinearLSQFitter()
 
    refined_trace_poly = []
    fallback_orders    = []          # track which orders used the fallback
 
    for n, trace_poly in enumerate(trace_poly_coeffs):
 
        poly_guess = np.poly1d(trace_poly)
        poly_ref   = np.poly1d(reference_poly_coeffs[n])
        
        trace_db_file = trace_directory / f"trace_xy_{n+1:02d}{TABLE_EXTENSION}"
 
        x_vals, y_vals = read_trace(trace_db_file)
 
        
        if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
 
            refit_trace = True
 
            # --- Resolve per-order valid y-range ---
            # y_obs_min/max: the rows where the order was actually detected.
            # Outside this range the observed polynomial may extrapolate badly,
            # so we use poly_ref as the search anchor and exclude those rows
            # from the refit.
            if y_ranges_valid is not None and (n + 1) in y_ranges_valid:
                y_obs_min, y_obs_max = y_ranges_valid[n + 1]
            else:
                y_obs_min, y_obs_max = int(y_vals.min()), int(y_vals.max())
 
            # Boolean mask: True where the row is inside the observed range
            mask_obs_range = (y_vals >= y_obs_min) & (y_vals <= y_obs_max)
 
            # --- 2. Prepare arrays for refined trace ---
            x_refined  = np.zeros_like(y_vals, dtype=float)
            x_shift    = np.zeros_like(y_vals, dtype=float)
            
            
            # --- 3. Loop through y positions to refine the trace ---
            for i, y in enumerate(y_vals):
 
                # Inside the observed range: trust poly_guess as the anchor.
                # Outside: use poly_ref so the search window stays centred even
                # where the observed polynomial extrapolates badly.
                if mask_obs_range[i]:
                    x_guess = poly_guess(y)
                else:
                    x_guess = poly_ref(y)
 
                # Define window range safely to avoid adding positions outside the CCD
                x_min = max(int(x_guess - trace_window), 0)
                x_max = min(int(x_guess + trace_window), trace_data.shape[1]-1)
 
                # sum the trace over a few pixels in the dispersion direction
                n_pix_sum = 3
 
                if n_pix_sum > 0:
                    profile = np.sum(trace_data[int(y-n_pix_sum):int(y+n_pix_sum), int(x_min):int(x_max+1)], axis=0)
                else:
                    profile = trace_data[int(y), int(x_min):int(x_max+1)]
 
                x_window = np.arange(x_min, x_max+1)
 
                # Smooth the profile to reduce noise / spectral line impact
                profile_smooth = medfilt(profile, kernel_size=3)
                # position of the peak
                x_refined_peak = x_window[np.argmax(profile)]
                peak_max = np.argmax(profile)
                try:
                    peak_max2, _ = refine_peak_positions(profile_smooth, [peak_max], window=trace_window, 
                                                         plot_fits=False, n_col=1, output=None)
                    x_refined_peak2 = x_window[peak_max]
                except:
                    x_refined_peak2 = x_refined_peak
 
                x_refined[i] = x_refined_peak2
 
            # --- 4. Keep only valid positions ---
            # Restrict refit to rows within the observed y-range so noisy
            # border rows do not pull the polynomial fit.
            y_valid = y_vals[mask_obs_range]
            x_valid = x_refined[mask_obs_range]
 
            # --- 5. Sigma-clip outliers relative to first guess ---
            poly_fit = fitter(models.Polynomial1D(degree=trace_ndeg), y_valid, x_valid)
            residuals = x_valid - poly_guess(y_valid)
            mask_clip = sigma_clip(residuals, sigma=sigma_clip_threshold).mask
 
            x_clean = x_valid[~mask_clip]
            y_clean = y_valid[~mask_clip]
 
            shift_trace = np.median(x_shift)
 
            # --- 6. Fit refined polynomial ---
            poly_refined = fitter(models.Polynomial1D(degree=trace_ndeg), y_clean, x_clean)
            poly_coeffs = poly_refined.parameters
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            poly_coeffs = np.flip(poly_coeffs)
 
            # --- 6b. Reference-table fallback for diverging orders ---
            used_fallback = False
            if use_reference_fallback:
 
                # Evaluate both polynomials at the order centre row.
                # Clamp y_fb to within the observed valid range so the offset
                # is always measured where the flux is reliable.
                if fallback_y_center is not None:
                    y_fb = int(np.clip(fallback_y_center, y_obs_min, y_obs_max))
                else:
                    y_fb = int(np.clip(np.median(y_vals), y_obs_min, y_obs_max))
 
                x_ref_center      = poly_ref(y_fb)
                x_refined_center  = np.poly1d(poly_coeffs)(y_fb)
                divergence        = abs(x_refined_center - x_ref_center)

                if divergence > divergence_threshold:
 
                    # --- measure a pixel offset at y_fb from the data ---
                    ny, nx = trace_data.shape
                    x_fb_min = max(int(x_ref_center) - fallback_half_width, 0)
                    x_fb_max = min(int(x_ref_center) + fallback_half_width + 1, nx)
 
                    r_half = fallback_collapse_rows // 2
                    y_fb_min = max(y_fb - r_half, 0)
                    y_fb_max = min(y_fb + r_half + 1, ny)
 
                    profile_fb = np.sum(
                        trace_data[y_fb_min:y_fb_max, x_fb_min:x_fb_max],
                        axis=0
                    )
                    xpix_fb = np.arange(x_fb_min, x_fb_max)
 
                    offset = 0.0   # default: use reference as-is
                    fit_fb = fit_order_center(profile_fb, xpix_fb)
                    if fit_fb is not None:
                        x_peak_measured, _, _ = fit_fb
                        offset = x_peak_measured - x_ref_center

                    # Shift the reference polynomial by the measured offset.
                    # Only the constant term (coeffs[-1] in numpy convention) changes;
                    # the shape of the curve is preserved from the reference epoch.
                    ref_coeffs = np.array(poly_ref)        # copy reference coefficients
                    ref_coeffs[-1] += offset               # numpy poly: highest degree first
 
                    poly_coeffs = ref_coeffs
                    poly_refined = np.poly1d(poly_coeffs)  # keep poly_refined consistent for plotting
 
                    used_fallback = True
                    fallback_orders.append(n + 1)
 
                    if not quiet:
                        print(
                            f"  Order {n+1:02d}: divergence={divergence:.2f} px > threshold={divergence_threshold:.2f} px "
                            f"→ fallback applied (offset={offset:+.2f} px at y={y_fb})"
                        )
 
            refined_trace_poly.append(poly_coeffs)  # always append, whether fallback or normal refit
 
 
        else: # if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
            print(f"Skipping tracing refit for order n={n+1}")
            
            refit_trace = False
            used_fallback = False
 
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            #poly_coeffs = np.flip(poly_guess)
            poly_coeffs = np.array(poly_guess)
            refined_trace_poly.append(poly_coeffs)
        
 
        dx = np.max(x_vals) - np.min(x_vals)
 
        # --- 7. Plot result for visual check ---
 
        if plot_results:
 
            xmin, xmax = np.min(x_vals) - dx*0.2, np.max(x_vals) + dx*0.2
            if xmin < 0:
                xmin = 0
            if xmax > trace_data.shape[1]:
                xmax = trace_data.shape[1]
                
 
            y_range=[ 500,1500] if trace_channel == 'blue' else [1000,3000]
 
            vmin, vmax = np.percentile(trace_data[y_range,int(xmin):int(xmax)], plot_percentile)
 
            # Build a title that flags fallback use
            order_title = f'Order Tracing Refinement (n={n+1})'
            if used_fallback:
                order_title += '  [FALLBACK]'
                
            if plot_transpose:
                plt.figure(figsize=(14,4))
                plt_img = plt.imshow(np.transpose(trace_data), origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(y_vals, poly_guess(y_vals),   ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(y_vals, poly_refined(y_vals), ls='--', color='red',  lw=1,
                             label='Fallback trace' if used_fallback else 'Refined trace')
                if used_fallback:
                    plt.plot(y_vals, poly_ref(y_vals), ls='-.', color='cyan', lw=1, label='Reference shape')
                plt.scatter(y_vals,              x_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace and not used_fallback:
                    plt.scatter(y_valid[~mask_clip], x_valid[~mask_clip], s= 5, color='black', label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(y_valid[mask_clip],  x_valid[mask_clip],  s=10, marker='x', color='red', label=f'Excluded points ({np.sum(mask_clip)})')
                    plt.scatter(y_vals[~mask_obs_range], x_refined[~mask_obs_range], s=5, marker='s', color='orange', label=f'Outside obs range ({np.sum(~mask_obs_range)})')
                plt.ylim(xmin,xmax)
                
                #plt.xlim(min(y_vals)-50,max(y_vals)+50)
                ymin = max(min(y_vals)-50, 0)
                ymax = min(max(y_vals)+50, trace_data.shape[0])
                
                plt.xlim(ymin,ymax)
                plt.ylabel('X pixel')
                plt.xlabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(order_title)
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()
 
            else:
                plt.figure(figsize=(4,8))
                plt_img = plt.imshow(trace_data, origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(poly_guess(y_vals),   y_vals, ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(poly_refined(y_vals), y_vals, ls='--', color='red',  lw=1,
                             label='Fallback trace' if used_fallback else 'Refined trace')
                if used_fallback:
                    plt.plot(poly_ref(y_vals), y_vals, ls='-.', color='cyan', lw=1, label='Reference shape')
                plt.scatter(x_vals,              y_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace and not used_fallback:
                    plt.scatter(x_valid[~mask_clip], y_valid[~mask_clip], s= 5, color='black',
                                label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(x_valid[mask_clip],  y_valid[mask_clip],  s=10, marker='x', color='red',
                                label=f'Excluded points ({np.sum(mask_clip)})')
                    plt.scatter(x_refined[~mask_obs_range], y_vals[~mask_obs_range], s=5, marker='s', color='orange',
                                label=f'Outside obs range ({np.sum(~mask_obs_range)})')
                plt.xlim(xmin,xmax)
                plt.ylim(min(y_vals)-50,max(y_vals)+50)
                plt.xlabel('X pixel')
                plt.ylabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(order_title)
                plt.tight_layout()
                if output_file_prefix is not None:
                    #plt.savefig(output_file_prefix + f"_n{(n+1):02d}{PLOT_EXTENSION}", dpi=300)
                    output_file = output_file_prefix.parent / ( output_file_prefix.name + f"_n{n+1:02d}{PLOT_EXTENSION}" )
                    plt.savefig(output_file, dpi=300)
                plt.show()
        
        
    return refined_trace_poly, fallback_orders

def save_polyfits_tracing_csv(order, polyfits_x_along_y, output_csv, show_messages=False):
    """
        Read a list contaning the polynimal coefficients and save them as a csv file for future use. 
        The first row contains the order information, and the following rows will store the Polynomial coefficients.
    
    """

    # Convert to string representation of coefficients
    string_array = np.empty_like(polyfits_x_along_y)
    
    if polyfits_x_along_y.ndim > 1:
        
        pixel_array = order
        for i in range(polyfits_x_along_y.shape[0]):   # spatial
            for j in range(polyfits_x_along_y.shape[1]): # spectral lines
                coeffs = polyfits_x_along_y[i, j].parameters  # list of coefficients
                coeffs_txt = np.array2string(coeffs, separator=',')
                coeffs_txt = coeffs_txt#.replace(' ','').replace('[','').replace(']','')
                string_array[i, j] =  coeffs_txt #','.join(map(float, coeffs))
                
        string_array = np.vstack((pixel_array,string_array))
        
    else:
        pixel_array = np.array([1])
        for i in range(polyfits_x_along_y.shape[0]):   # spatial
            coeffs = polyfits_x_along_y[i]#.parameters  # list of coefficients
            coeffs_txt = np.array2string(coeffs, separator=',')
            coeffs_txt = coeffs_txt#.replace(' ','').replace('[','').replace(']','')
            string_array[i] =  coeffs_txt #','.join(map(float, coeffs))

        string_array = np.concatenate((pixel_array,string_array))
        
    # Save to CSV
    np.savetxt(output_csv, string_array, fmt='%s', delimiter=';')
    if show_messages:
        print(f'File {output_csv} was created.')


def read_polyfits_tracing_csv(directory, file_prefix, n_orders):
    """
        Read a CSV file created by save_polyfits_tracing_csv() and returns a numpy array of with tracing polynomial coefficients. 
    
    """
    polyfits = [] 
    
    trace_poly_coeffs = []
    
    for order in np.arange(1,n_orders+1): 
    
        file_csv = directory / f"{file_prefix}{order:02d}{CSV_EXTENSION}"

        # Load string array
        loaded_strings = np.genfromtxt(file_csv, dtype=str, delimiter=';')
        coeff_strings = loaded_strings[1:]
                
        # Convert back to Polynomial1D array
        coeffs = []
        for i in range(coeff_strings.shape[0]):
            coeffs_txt = coeff_strings[i]
            #print(i, coeffs_txt)
            #coeffs = list(map(float, coeffs_txt.split(',')))
            coeffs.append(float(coeffs_txt))
        polyfits.append(np.array(coeffs))
                
    return polyfits


def linearize_trace(fits_file, poly_coeffs, x_left=-8, x_right=12, 
                    y_range=None, flip_axis=False, displ_percent=[1,99], show_plot=True, output_file=None):
    """
    Fast linearization of a curved stellar trace using vectorized interpolation.

    Parameters
    ----------
    fits_file : str or ndarray
        Input FITS file or 2D numpy array.
    poly_coeffs : ndarray
        Polynomial coefficients for x = f(y).
    x_left : int
        Pixels to extract to the left of the trace.
    x_right : int
        Pixels to extract to the right of the trace.
    y_range : tuple (y_min, y_max), optional
        Vertical range of pixels to extract. If None, full range is used.
    output_file : str
        Path to save the linearized FITS file.
    show_plot : bool
        If True, plot the linearized image.

    Returns
    -------
    linearized : ndarray
        Rectified 2D array (width, height) where width = x_right - x_left + 1.
    """

    # Load image
    if isinstance(fits_file, str):
        image = fits.getdata(fits_file)
    else:
        image = np.asarray(fits_file)

    ny, nx = image.shape
    poly = np.poly1d(poly_coeffs)

    # Define y-range
    if y_range is None:
        y_min, y_max = 0, ny - 1
    else:
        y_min, y_max = y_range

    y_min = max(0, y_min)
    y_max = min(ny - 1, y_max)

    # Prepare arrays
    width = x_right - x_left + 1
    height = y_max - y_min + 1

    y_indices = np.arange(y_min, y_max + 1)
    x_offsets = np.arange(x_left, x_right + 1)

    # Compute x centers for all y
    x_centers = poly(y_indices)  # shape: (height,)

    # mask out tracing curves lying outside the image.
    mask = ( x_centers < 0 ) ^ (x_centers > nx )
    
    # Build coordinate grids
    y_grid = np.repeat(y_indices[np.newaxis, :], width, axis=0)  # (width, height)
    x_grid = (x_centers[np.newaxis, :] + x_offsets[:, np.newaxis])  # (width, height)

    # Interpolate in one call
    coords = np.array([y_grid, x_grid])
    linearized = map_coordinates(image, coords, order=1, mode='nearest')

    linearized[:,mask] = 0.0

    if flip_axis:
        linearized = np.flip(linearized,axis=1)
    
    # Save to FITS
    if output_file is not None:
        fits.writeto(output_file, linearized, overwrite=True)
        print(f"Linearized trace saved to {output_file}")
    # Optional plot
    if show_plot:
        plt.figure(figsize=(16, 4))
        
        vmin, vmax = np.percentile(linearized, displ_percent)
        
        img = plt.imshow(linearized, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        plt.axhline(width*0.5, ls='--')
        plt.xlabel("Extraction Width (pixels)")
        plt.ylabel("Y (linearized)")
        plt.title(f"Linearized Trace (y: {y_min}-{y_max})")
        plt.colorbar(img, label="Flux")
        plt.tight_layout()
        plt.show()

    return linearized
