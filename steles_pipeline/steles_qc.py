# steles_qc.py
from .steles_imports import *

def quality_control_master_frames(quartz, arc, sci,
    quartz_header=None, arc_header=None, sci_header=None,
    channel="blue", transpose=False,
    output_path=None, save_plot=True, show_plot=False):
    """
    Generate QC plot for the master flat, arc, and science frames.

    Parameters
    ----------
    quartz, arc, sci : ndarray
        Master frame data.

    quartz_header, arc_header, sci_header : FITS header, optional
        Headers associated with each master frame.

    channel : str
        STELES channel ("blue" or "red").

    transpose : bool
        Whether the images need to be transposed for plotting.

    output_path : str, optional
        Full path for the output PNG.

    save_plot : bool
        Save the plot if True.

    show_plot : bool
        Display the plot if True.
    """

    def transpose_array(array):
        return np.flip(array.T, axis=1)

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    quartz = quartz.copy()
    arc    = arc.copy()
    sci    = sci.copy()

    if transpose:
        quartz = transpose_array(quartz)
        arc = transpose_array(arc)
        sci = transpose_array(sci)

    # ---------------------------------------------------------
    # Axis labels
    # ---------------------------------------------------------

    x_title = ("columns <-> dispersion (pixel)" if channel == "blue" else "columns <-> orders (pixel)")

    y_title = ("lines <-> orders (pixel)" if channel == "blue" else "lines <-> dispersion (pixel)")

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # ---------------------------------------------------------
    # Flat
    # ---------------------------------------------------------

    img1 = ax1.imshow(quartz, vmin=0, vmax=np.percentile(quartz, 95), origin="lower", aspect="auto",)

    plt.colorbar(img1, ax=ax1, pad=0.01)

    ax1.set_title(quartz_header.get("OBJECT", "FLAT") if quartz_header is not None else "FLAT")

    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_title)

    # ---------------------------------------------------------
    # Arc
    # ---------------------------------------------------------

    img2 = ax2.imshow(arc, vmin=0, vmax=np.percentile(arc, 99), origin="lower", aspect="auto")

    plt.colorbar(img2, ax=ax2, pad=0.01)

    ax2.set_title(arc_header.get("OBJECT", "ARC") if arc_header is not None else "ARC")

    ax2.set_xlabel(x_title)
    ax2.set_ylabel(y_title)

    # ---------------------------------------------------------
    # Science
    # ---------------------------------------------------------

    img3 = ax3.imshow(sci, vmin=0, vmax=np.percentile(sci, 98), origin="lower",aspect="auto")

    plt.colorbar(img3, ax=ax3, pad=0.01)

    ax3.set_title(sci_header.get("OBJECT", "SCIENCE") if sci_header is not None else "SCIENCE")

    ax3.set_xlabel(x_title)
    ax3.set_ylabel(y_title)

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if save_plot and output_path is not None:
        plt.savefig(output_path, dpi=300)

    # ---------------------------------------------------------
    # Show / close
    # ---------------------------------------------------------

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

        
# =========================================================
# Stage 5b: Quality control tracing
# =========================================================
def quality_control_tracing(trace_data, trace_poly_coeffs, y_range, output_path=None, save_plot=True, show_plot=False):
    """
    Generate a QC plot of the traced spectral orders.

    Parameters
    ----------
    trace_data : ndarray
        Science image used for tracing.

    trace_poly_coeffs : array-like
        Polynomial coefficients for each traced spectral order.

    y_range : tuple or list
        Range of y pixels used for tracing.

    output_path : str, optional
        Full path for the output PNG.

    save_plot : bool
        Save the plot if True.

    show_plot : bool
        Display the plot if True.

    Returns
    -------
    intersections : list
        List of intersections found between adjacent orders.
        Each entry is a dictionary containing the order numbers
        and intersection coordinates.
    """

    # Prepare image
    plot_data = trace_data.copy()
    plot_data[plot_data <= 0] = 0.1

    y_axis = np.arange( y_range[0], y_range[1] + 1)

    # Start plot

    fig = plt.figure(figsize=(15, 5))

    img = plt.imshow( np.log10(plot_data), origin="lower", aspect="auto", vmin=0, vmax=3, cmap="viridis")

    # Trace orders
    intersections = []
    poly_last = None

    for n, coeffs in enumerate(trace_poly_coeffs):

        poly = np.poly1d(coeffs)

        # Check intersection with previous order
        if poly_last is not None:

            f_diff = poly - poly_last

            roots = np.roots(f_diff)
            roots = roots[np.isreal(roots)].real

            for y in roots:

                x = poly(y)

                if ( 0 < x < trace_data.shape[1] and 0 < y < trace_data.shape[0] ):

                    intersections.append({"order_1": n,
                                          "order_2": n + 1,
                                          "x": float(x),
                                          "y": float(y)})

        # Plot trace
        x_axis = poly(y_axis)
        mask = x_axis > 0

        plt.plot(x_axis[mask], y_axis[mask], alpha=0.5, lw=5)

        plt.plot( x_axis[mask], y_axis[mask], color="black", alpha=0.75, lw=0.5)

        poly_last = poly

    # Plot formatting
    plt.xlim(0, trace_data.shape[1])
    plt.ylim(0, trace_data.shape[0])

    plt.xlabel("x-axis (along orders)")
    plt.ylabel("y-axis (dispersion)")

    plt.colorbar(img, pad=0.01, label="log(adu)")

    plt.tight_layout()

    # Save
    if save_plot and output_path is not None:
        plt.savefig(output_path, dpi=600)

    # Show / close
    if show_plot:
        plt.show()
    else:
        plt.close()

    return intersections    

# =========================================================
# Stage 6: Extraction
# =========================================================
def quality_control_apertures(datasets, trace_poly_coeffs,
                              aperture_extract, y_range, plot_orders,
                              aperture_vmax=None, output_dir=None,
                              save_plot=True, show_plot=False):
    """
    Generate QC plots showing spectral-order traces and extraction apertures.

    Parameters
    ----------
    datasets : dict
        Dictionary containing the datasets to plot. Expected format:

            {
                "flat": (data, output_name),
                "arc":  (data, output_name),
                "sci":  (data, output_name),
            }

        where data is a 2D numpy array and output_name is the base
        filename used for the QC plot.

    trace_poly_coeffs : array-like
        Polynomial coefficients for each traced spectral order.

    aperture_extract : tuple/list
        Lower and upper aperture offsets relative to the trace.

    y_range : tuple/list
        Range of y pixels used for tracing.

    plot_orders : list
        Indices of the orders to include in the QC plots.

    aperture_vmax : dict, optional
        Maximum log(ADU) value for each dataset type.
        Example:
            {"flat": 4, "arc": 4, "sci": 4}

    output_dir : str, optional
        Directory where plots will be saved.

    save_plot : bool
        Save plots if True.

    show_plot : bool
        Display plots if True.
    """

    # Validate inputs
    if trace_poly_coeffs is None:
        raise RuntimeError("Trace coefficients not available")

    if len(plot_orders) == 0:
        raise ValueError("No aperture orders defined")

    if aperture_vmax is None:
        aperture_vmax = 3.0

    # Helper to prepare data
    def prepare(data):
        plot_data = data.copy()
        plot_data[plot_data <= 0] = 0.1
        return np.log10(plot_data)

    # Generate one plot for each dataset
    for kind, (data, name) in datasets.items():

        plot_data = prepare(data)

        vmax = aperture_vmax.get(kind, 4)

        n_plots = len(plot_orders)

        fig = plt.figure(figsize=(18, 6))

        plt.suptitle(f"{kind.upper()} frame")

        # Plot each selected order
        for i, order in enumerate(plot_orders):

            poly = np.poly1d(trace_poly_coeffs[order])

            y_axis = np.arange(y_range[0],y_range[1] + 1)

            x_axis = poly(y_axis)

            mask = x_axis > 0

            # Dynamic plotting window
            x_valid = x_axis[mask]

            if len(x_valid) == 0:
                continue

            x_min = x_valid.min()
            x_max = x_valid.max()

            dx = x_max - x_min

            x_min -= 0.25 * dx
            x_max += 0.25 * dx

            x_min = max(0, x_min)
            x_max = min(data.shape[1], x_max)

            # Plot image
            ax = plt.subplot(1, n_plots, i + 1)

            img = ax.imshow(plot_data, origin="lower", aspect="auto", vmin=0, vmax=vmax, cmap="viridis")

            # Trace
            ax.plot(x_axis[mask], y_axis[mask], color="red",  lw=1, label="Trace")

            # Extraction apertures
            ax.plot(x_axis[mask] + aperture_extract[0], y_axis[mask], color="black", lw=1, label="Aperture")

            ax.plot(x_axis[mask] + aperture_extract[1], y_axis[mask],color="black", lw=1)

            # Formatting
            ax.set_xlim(x_min, x_max)

            ax.set_ylim(0, data.shape[0])

            ax.set_title(f"Order={order + 1}")

            ax.set_xlabel("x-axis")

            if i == 0:
                ax.set_ylabel("y-axis")
                ax.legend()

        # Colorbar
        plt.colorbar(img, pad=0.01, label="log(adu)")

        plt.tight_layout()

        # Save
        if save_plot and output_dir is not None:
            #output_path = os.path.join(output_dir, name + f"_aperture{PLOT_EXTENSION}")
            output_path = output_dir / f"{name}_aperture{PLOT_EXTENSION}"
            plt.savefig(output_path, dpi=300)

        # Show / close
        if show_plot:
            plt.show()
        else:
            plt.close()
            
# PART II
def qc_wavelength_rms(wavesol_stats,
                      output_dir=None, output_name="wavelength",
                      save_plot=True, show_plot=False,
                      figsize=(10, 4), nbins=20):
    """
    Generate a histogram of wavelength-solution RMS per order.

    Parameters
    ----------
    wavesol_stats : list of dict
        Wavelength-solution statistics for each order. Each dictionary
        must contain an "rms" entry.

    output_dir : str, optional
        Directory where the QC plot will be saved.

    output_name : str
        Base name of the output file.

    save_plot : bool
        Save the plot if True.

    show_plot : bool
        Display the plot if True.

    figsize : tuple
        Figure size.

    dpi : int
        Output resolution.

    nbins : int
        Number of histogram bins.

    Returns
    -------
    stats : dict
        RMS statistics in mA.
    """

    # Retrieve valid RMS values
    rms = np.asarray([s["rms"] for s in wavesol_stats if np.isfinite(s["rms"])])

    if len(rms) == 0:
        raise RuntimeError("No valid wavelength RMS measurements available")

    # Convert Angstrom -> mA
    rms *= 1000

    # Statistics
    rms_stat = np.nanpercentile( rms,[15.9, 50, 84.1])

    stats = {"rms_16": float(rms_stat[0]),
             "rms_median": float(rms_stat[1]),
             "rms_84": float(rms_stat[2]),
             "rms_max": float(np.nanmax(rms)),
             "n_orders": int(len(rms)),}
    
    # Plot
    plt.figure(figsize=figsize)
    hist_bins = np.linspace(0, np.nanmax(rms), num=nbins)

    plt.hist(rms, bins=hist_bins)

    plt.axvline(rms_stat[0], ls=":", color="black", label=(r"RMS$_{16\%}$="
                                                           f"{rms_stat[0]:.2f}"
                                                           r"m$\AA$") )

    plt.axvline(rms_stat[1], ls="--", color="black", label=(r"RMS$_{50\%}$="
                                                            f"{rms_stat[1]:.2f}"
                                                            r"m$\AA$") )

    plt.axvline(rms_stat[2], ls="-.", color="black", label=(r"RMS$_{84\%}$="
                                                            f"{rms_stat[2]:.2f}"
                                                            r"m$\AA$") )

    plt.xlim(0,)
    plt.xlabel(r"RMS per order ($m\AA$)")
    plt.ylabel("Number of orders")
    plt.title("Wavelength Solution RMS")
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Save
    output_path = None
    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, output_name + f"_qc_rms_hist{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_rms_hist{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=600)

    # Show / close
    if show_plot:
        plt.show()
    else:
        plt.close()

    return stats, output_path

def qc_line_residuals(wavesol_stats, n_orders, output_dir=None, output_name="wavelength",
                      save_plot=True, show_plot=False, figsize=(10, 4)):
    """
    Generate a QC plot of wavelength-calibration line residuals.

    Returns
    -------
    stats : dict
        Global residual statistics in Angstroms.
    output_path : str or None
        Path of the saved plot.
    """

    # Collect residuals
    all_residuals = []

    for s in wavesol_stats:
        res = s["residuals"]

        if isinstance(res, str):
            res = res.strip("[]")
            res = [float(x) for x in res.split()] if res else []

        res = np.asarray(res, dtype=float)
        res = res[np.isfinite(res)]

        if len(res) > 0:
            all_residuals.extend(res)

    all_residuals = np.asarray(all_residuals, dtype=float)

    if len(all_residuals) == 0:
        raise RuntimeError("No valid wavelength residuals available")

    # Statistics
    stats = {"residual_median": float(np.nanmedian(all_residuals)),
             "residual_rms": float(np.sqrt(np.nanmean(all_residuals ** 2))),
             "residual_std": float(np.nanstd(all_residuals)),
             "residual_max_abs": float(np.nanmax(np.abs(all_residuals))),
             "n_lines": int(len(all_residuals)),
            }

    # Plot
    plt.figure(figsize=figsize)

    for s in wavesol_stats:
        res = s["residuals"]

        if isinstance(res, str):
            res = res.strip("[]")
            res = [float(x) for x in res.split()] if res else []

        res = np.asarray(res, dtype=float)
        res = res[np.isfinite(res)]

        if len(res) == 0:
            continue

        plt.scatter(np.full_like(res, s["order"]), res, s=10, alpha=0.6)

    plt.axhline(0, color="black", lw=1)
    plt.xlim(0, n_orders + 1)
    plt.xlabel("Order")
    plt.ylabel(r"Residual ($\AA$)")
    plt.title("Line Residuals")
    plt.tight_layout()

    # Save
    output_path = None

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, output_name + f"_qc_residuals{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_residuals{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=600)

    # Show / close
    if show_plot:
        plt.show()
    else:
        plt.close()

    return stats, output_path

def qc_dispersion_stability(wavesol_stats, n_orders, output_dir=None, output_name="wavelength",
                            save_plot=True, show_plot=False, figsize=(10, 6)):
    """
    Generate QC plots showing dispersion as a function of order and wavelength.

    Returns
    -------
    stats : dict
        Global dispersion statistics in Angstrom/pixel.
    output_path : str or None
        Path of the saved plot.
    """

    orders = np.asarray([s["order"]      for s in wavesol_stats])
    wave_c = np.asarray([s["mid_wave"]   for s in wavesol_stats])
    disp   = np.asarray([s["dispersion"] for s in wavesol_stats])

    valid = np.isfinite(orders) & np.isfinite(wave_c) & np.isfinite(disp)

    if not np.any(valid):
        raise RuntimeError("No valid dispersion measurements available")

    orders = orders[valid]
    wave_c = wave_c[valid]
    disp   = disp[valid]

    # Statistics
    stats = {"dispersion_median": float(np.nanmedian(disp)),
             "dispersion_std": float(np.nanstd(disp)),
             "dispersion_min": float(np.nanmin(disp)),
             "dispersion_max": float(np.nanmax(disp)),
            }

    # Plot
    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.plot(orders, disp, "o-")
    plt.xlim(0, n_orders + 1)
    plt.xlabel("Order")
    plt.ylabel(r"Dispersion ($\AA$/pix)")
    plt.suptitle("Dispersion Stability")

    plt.subplot(212)
    plt.plot(wave_c, disp, "o-")
    plt.xlim(np.nanmin(wave_c), np.nanmax(wave_c))
    plt.xlabel(r"Central Wavelength per order ($\AA$)")
    plt.ylabel(r"Dispersion ($\AA$/pix)")

    plt.tight_layout()
    
    # Save
    output_path = None

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, output_name + f"_qc_dispersion{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_dispersion{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=600)

    # Show / close
    if show_plot:
        plt.show()
    else:
        plt.close()

    return stats, output_path

def qc_rms_over_dispersion(wavesol_stats, n_orders, output_dir=None, output_name="wavelength",
                           velocity_scale=True, save_plot=True, show_plot=False,
                           figsize=(10, 4)):
    """
    Generate a QC plot of wavelength-solution RMS relative to dispersion.

    Parameters
    ----------
    wavesol_stats : list of dict
        Wavelength-solution statistics for each order.
    n_orders : int
        Total number of spectral orders.
    velocity_scale : bool
        If True, express RMS as an equivalent velocity in km/s.
        Otherwise, express RMS in pixels.
    Returns
    -------
    stats : dict
        Global RMS/dispersion quality metrics.
    output_path : str or None
        Path of the saved plot.
    """
    
    # Prepare data
    valid = [s for s in wavesol_stats if s["dispersion"] > 0 and np.isfinite(s["rms"]) and np.isfinite(s["mid_wave"])]

    if len(valid) == 0:
        raise RuntimeError("No valid wavelength-solution measurements available")

    orders = np.asarray([s["order"] for s in valid])
    wave_c = np.asarray([s["mid_wave"] for s in valid])

    if velocity_scale:
        vals = np.asarray([(s["rms"] / s["mid_wave"]) * 299792 for s in valid])
        ylabel = "RMS (km/s)"
    else:
        vals = np.asarray([s["rms"] / s["dispersion"] for s in valid])
        ylabel = "RMS / Dispersion (pix)"

    # Statistics and classification
    # gaussian definition
    #vals_stat = np.nanpercentile(vals, [16, 50, 84.1])
    #rms_med   = vals_stat[1]
    #rms_sigma = abs(vals_stat[2] - vals_stat[1])
    #idx_good   = ( vals <  (rms_med + 1 * rms_sigma) )
    #idx_1sigma = ( vals >= (rms_med + 1 * rms_sigma) ) & ( vals < ( rms_med + 3 * rms_sigma) )
    #idx_3sigma = ( vals >= (rms_med + 3 * rms_sigma) )
    
    # Robust statistics of the RMS distribution
    rms_med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - rms_med))
    rms_sigma = 1.4826 * mad
    
    idx_good = vals < rms_med + rms_sigma
    idx_1sigma = ( (vals >= rms_med + rms_sigma) & (vals < rms_med + 3 * rms_sigma) )
    idx_3sigma = vals >= rms_med + 3 * rms_sigma


    stats = {"rms_median": float(rms_med),
             "rms_sigma": float(rms_sigma),
             "n_valid": int(len(vals)),
             "n_good": int(np.sum(idx_good)),
             "n_1sigma": int(np.sum(idx_1sigma)),
             "n_3sigma": int(np.sum(idx_3sigma)),
             "frac_good": float(np.mean(idx_good)),
             "frac_1sigma": float(np.mean(idx_1sigma)),
             "frac_3sigma": float(np.mean(idx_3sigma)),
            }

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(figsize=figsize)

    plt.plot(orders[idx_good], vals[idx_good], "+", ms=8, zorder=1, color='green',
             label=fr"RMS < 1$\sigma$ ({np.sum(idx_good)}/{len(vals)})")

    plt.plot(orders[idx_1sigma], vals[idx_1sigma], "*", ms=5, zorder=1, color='orange',
             label=fr"1$\sigma$ < RMS < 3$\sigma$ ({np.sum(idx_1sigma)}/{len(vals)})")

    plt.plot(orders[idx_3sigma], vals[idx_3sigma], "x", ms=5, zorder=1, color='red',
             label=fr"RMS > 3$\sigma$ ({np.sum(idx_3sigma)}/{len(vals)})")

    plt.axhline(rms_med, ls="--", lw=1, zorder=0, alpha=0.5,
                label=fr"RMS$_{{50\%}}$={rms_med:.2f}")

    plt.axhline(rms_med + rms_sigma, ls="-.", lw=1, zorder=0, alpha=0.5,
                label=fr"RMS$_{{50\%}}$+1$\sigma$={rms_med+rms_sigma:.2f}")

    plt.axhline(rms_med + 3 * rms_sigma, ls=":", lw=1, zorder=0, alpha=0.5,
                label=fr"RMS$_{{50\%}}$+3$\sigma$={rms_med+3*rms_sigma:.2f}")

    # Label orders with moderately/badly poor fits
    dvals = np.nanmax(vals) - np.nanmin(vals)

    if dvals > 0:
        for i, idx in enumerate(idx_1sigma):
            if idx:
                plt.text(orders[i], vals[i] + 0.04 * dvals, orders[i],
                         ha="center", va="center", fontsize=10, zorder=2)

        for i, idx in enumerate(idx_3sigma):
            if idx:
                plt.text(orders[i], vals[i] - 0.06 * dvals, orders[i],
                         ha="center", va="center", fontsize=10, zorder=2)

    # Axes
    plt.xlabel("Order")
    plt.ylabel(ylabel)
    plt.ylim(0, )
    plt.xlim(0, n_orders + 1)
    plt.legend(fontsize=8, framealpha=0.35)
    plt.title("Wavelength Solution Quality")
    plt.tight_layout()

    # Save
    output_path = None

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, output_name + f"_qc_rms_over_disp{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_rms_over_disp{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=600)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return stats, output_path

def qc_extracted_products(sci_extracted, sci_header, channel, ext_flux, ext_wave, ext_noise,
                          ext_blaze, ext_blaze_flux, output_dir=None, output_name="spectrum",
                          save_plot=True, show_plot=False, figsize=(24, 6), fontsize=14, dpi=300):
    """
    Generate QC plots for the final STELES extracted 1D products.

    Displays flux, wavelength, blaze, blaze-corrected flux, and SNR
    for all spectral orders.
    """

    object_name = sci_header.get("OBJECT", None)

    # Extract data
    plot_flux  = sci_extracted[ext_flux].copy()
    plot_wave  = sci_extracted[ext_wave].copy()
    plot_noise = sci_extracted[ext_noise].copy()
    plot_blaze = sci_extracted[ext_blaze].copy()
    plot_bflux = sci_extracted[ext_blaze_flux].copy()

    # Red channel: reverse order direction consistently
    if channel == "red":
        plot_flux  = plot_flux[::-1, :]
        plot_wave  = plot_wave[::-1, :]
        plot_noise = plot_noise[::-1, :]
        plot_blaze = plot_blaze[::-1, :]
        plot_bflux = plot_bflux[::-1, :]

    # Calculate SNR
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_plot = plot_flux / plot_noise

    snr_valid = snr_plot[np.isfinite(snr_plot) & (snr_plot > 0)]

    if len(snr_valid) > 0:
        snr_stats = {"snr_median": float(np.nanmedian(snr_valid)),
                     "snr_90th": float(np.nanpercentile(snr_valid, 90)),
                     "snr_max": float(np.nanmax(snr_valid)),
                    }
    else:
        snr_stats = {"snr_median": np.nan,
                     "snr_90th": np.nan,
                     "snr_max": np.nan,
                    }

    # Prepare plot
    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(1, 5, figsize=figsize)

    # Flux
    flux_safe = np.where(plot_flux > 1, plot_flux, 1)
    flux_log = np.log10(flux_safe)

    img = axes[0].imshow(flux_log, aspect="auto", origin="lower",
                         vmin=np.nanpercentile(flux_log, 0.1), vmax=np.nanpercentile(flux_log, 99.9))
    axes[0].set_title("Flux")
    axes[0].set_xlabel("Dispersion")
    axes[0].set_ylabel("Orders")
    plt.colorbar(img, ax=axes[0], label="log10(flux/ADU)")

    # Wavelength
    valid = np.isfinite(plot_wave) & (plot_wave > 0)
    if np.any(valid):
        wave_min = np.nanmin(plot_wave[valid])
        wave_max = np.nanmax(plot_wave[valid])
    else:
        wave_min, wave_max = 0, 1

    img = axes[1].imshow(plot_wave, aspect="auto", origin="lower", vmin=wave_min, vmax=wave_max)
    axes[1].set_title(r"Wavelength ($\AA$)")
    axes[1].set_xlabel("Dispersion")
    axes[1].set_ylabel("Orders")
    plt.colorbar(img, ax=axes[1], label=r"Wavelength ($\AA$)")

    # Blaze
    img = axes[2].imshow(plot_blaze, aspect="auto", origin="lower", vmin=0, vmax=np.nanpercentile(plot_blaze, 99.99))
    axes[2].set_title("Blaze")
    axes[2].set_xlabel("Dispersion")
    axes[2].set_ylabel("Orders")
    plt.colorbar(img, ax=axes[2], label="Blaze")

    # Blaze-corrected flux
    bflux_safe = np.where(plot_bflux > 0, plot_bflux, 0.001)

    img = axes[3].imshow(bflux_safe, aspect="auto", origin="lower",
                         vmin=np.nanpercentile(bflux_safe, 0.1), vmax=np.nanpercentile(bflux_safe, 99.9))
    axes[3].set_title("Flux (Blaze Corrected)")
    axes[3].set_xlabel("Dispersion")
    axes[3].set_ylabel("Orders")
    plt.colorbar(img, ax=axes[3], label="Blaze corrected flux (ADU)")

    # SNR
    snr_max = np.nanpercentile(snr_valid, 99.9) if len(snr_valid) > 0 else 1
    img = axes[4].imshow(snr_plot, aspect="auto", origin="lower", vmin=0, vmax=snr_max)
    axes[4].set_title("SNR")
    axes[4].set_xlabel("Dispersion")
    axes[4].set_ylabel("Orders")
    plt.colorbar(img, ax=axes[4], label="SNR")

    # Title / layout
    if object_name is not None:
        fig.suptitle(object_name, y=0.95)
    plt.tight_layout()
    
    # Save / show
    output_path = None
    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, output_name + f"_final_1d{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_final_1d{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return snr_stats, output_path

def qc_blaze(sci_extracted, ext_flux, ext_wave, ext_blaze, n_panels=4,
             output_dir=None, output_name="spectrum", save_plot=True,
             show_plot=False, figsize=(10, 15), dpi=300):
    """
    QC: Compare extracted spectrum with the blaze function for each order.

    Returns
    -------
    stats : dict
        Basic statistics describing the blaze comparison.
    output_paths : list
        Paths of the generated QC plots.
    """

    n_orders = sci_extracted.shape[1]
    n_panels = min(n_panels, n_orders)
    n_per_panel = int(np.ceil(n_orders / n_panels))

    output_paths = []
    n_valid_orders = 0
    
    blaze_qc_stats = []

    for ipanel in range(n_panels):

        start = ipanel * n_per_panel
        end = min((ipanel + 1) * n_per_panel, n_orders)

        if start >= n_orders:
            break

        orders = np.arange(start, end)

        fig, axes = plt.subplots(len(orders), 1, figsize=figsize, gridspec_kw={"hspace": 0.35})

        if len(orders) == 1:
            axes = [axes]

        for i, n in enumerate(orders):

            wave = sci_extracted[ext_wave, n, :]
            flux = sci_extracted[ext_flux, n, :]
            blaze = sci_extracted[ext_blaze, n, :]

            valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(blaze) & (blaze > 0)

            if np.sum(valid) < 20:
                continue

            n_valid_orders += 1

            w = wave[valid]
            f = flux[valid]
            b = blaze[valid]

            # --- Continuum estimate ---
            try:
                win = min(101, len(f) - 1)

                if win % 2 == 0:
                    win -= 1

                if win > 5:
                    cont = savgol_filter(f, win, 1)
                else:
                    cont = f

            except Exception:
                cont = f

            # --- Scale blaze to continuum ---
            scale = np.nanmedian(cont / b)
            blaze_scaled = b * scale
            
            # statistics
            ratio = cont / blaze_scaled
            residual = cont - blaze_scaled
            blaze_qc_stats.append({"order": int(n),
                                   "scale": float(scale),
                                   "median_ratio": float(np.nanmedian(ratio)),
                                   "scatter": float(np.nanstd(ratio)),
                                   "rms": float(np.sqrt(np.nanmean(residual**2))),
                                   "correlation": float(np.corrcoef(cont, blaze_scaled)[0, 1]),
                                   "n_pixels": int(len(w))})

            ax = axes[i]

            ax.plot(w, f, color="black", lw=0.5, label=f"Order {n + 1}")
            ax.plot(w, blaze_scaled, color="red", lw=0.7, alpha=0.8, label="Blaze")
            ax.plot(w, cont, color="blue", lw=0.5, alpha=0.6, label="Continuum")

            ax.set_xlim(np.nanmin(w), np.nanmax(w))
            ax.set_ylabel("Flux")

            if i == 0:
                ax.legend(fontsize=8)

        axes[-1].set_xlabel(r"Wavelength ($\AA$)")
        plt.tight_layout()

        output_path = None

        if save_plot and output_dir is not None:
            #output_path = os.path.join(output_dir, f"{output_name}_blaze_qc_panel{ipanel + 1}{PLOT_EXTENSION}")
            output_path = output_dir / f"{output_name}_blaze_qc_panel{ipanel + 1}{PLOT_EXTENSION}"
            plt.savefig(output_path, dpi=dpi)
            output_paths.append(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

    stats = {"n_orders": int(n_orders),
             "n_valid_orders": int(n_valid_orders),
             "n_panels": int(n_panels),
             "blaze_stats": blaze_qc_stats,
            }

    return stats, output_paths

def qc_snr_old(sci_extracted, ext_flux, ext_wave, ext_noise, n_orders, n_panels=4,
           output_dir=None, output_name="spectrum", save_plot=True,
           show_plot=False, figsize=(10, 15), dpi=300):

    wave_all = sci_extracted[ext_wave, :, :]
    wave_min, wave_max = np.nanmin(wave_all), np.nanmax(wave_all)
    dwave = wave_max - wave_min

    n_panels = min(n_panels, n_orders)
    output_path = None
    snr_stats = []

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, gridspec_kw={"hspace": 0.175})

    if n_panels == 1:
        axes = [axes]

    for row in range(n_panels):
        ax = axes[row]
        max_snr = 0
        plotted = False
        
        wave_min_n = wave_min + (dwave / n_panels) *  row
        wave_max_n = wave_min + (dwave / n_panels) * (row + 1)

        for order in range(n_orders):

            wave = sci_extracted[ext_wave, order, :]
            flux = sci_extracted[ext_flux, order, :]
            noise = sci_extracted[ext_noise, order, :]

            valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(noise) & (noise > 0)

            if np.sum(valid) < 20:
                continue

            w, f, n = wave[valid], flux[valid], noise[valid]
            snr = f / n

            if wave_min_n <= np.median(w) <= wave_max_n:

                ax.plot(w, snr, lw=0.5)
                ax.text(np.median(w), np.nanpercentile(snr, 99.9), f"{order + 1}",
                        ha="center", va="bottom", fontsize=10)

                wave_min_n = min(wave_min_n, np.nanmin(w))
                wave_max_n = max(wave_max_n, np.nanmax(w))
                max_snr = max(max_snr, np.nanmax(snr))
                plotted = True

                snr_stats.append({"order": int(order),
                                  "snr_median": float(np.nanmedian(snr)),
                                  "snr_max": float(np.nanmax(snr)),
                                  "snr_p95": float(np.nanpercentile(snr, 95)),
                                  "n_pixels": int(len(snr))})

        if plotted:
            ax.set_xlim(wave_min_n, wave_max_n)
            ax.set_ylim(0, max_snr * 1.1)

        ax.set_ylabel("SNR")

        if row == n_panels - 1:
            ax.set_xlabel(r"Wavelength ($\AA$)")

    plt.tight_layout()

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, f"{output_name}_qc_snr{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_snr{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    stats = {"n_orders": int(n_orders),
             "n_valid_orders": int(len(snr_stats)),
             "snr_stats": snr_stats}

    return stats, [output_path] if output_path is not None else []

def qc_snr(sci_extracted, ext_flux, ext_wave, ext_noise, n_orders, n_panels=4,
           output_dir=None, output_name="spectrum", save_plot=True,
           show_plot=False, figsize=(10, 15), dpi=300):

    wave_all = sci_extracted[ext_wave, :, :]
    wave_min, wave_max = np.nanmin(wave_all), np.nanmax(wave_all)
    dwave = wave_max - wave_min

    n_panels = min(n_panels, n_orders)
    output_path = None
    snr_stats = []

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, gridspec_kw={"hspace": 0.175})

    if n_panels == 1:
        axes = [axes]

    for row in range(n_panels):
        ax = axes[row]
        max_snr = 0
        plotted = False

        # Fixed boundaries for this panel
        panel_min = wave_min + (dwave / n_panels) * row
        panel_max = wave_min + (dwave / n_panels) * (row + 1)

        # Actual wavelength range of plotted orders
        plot_min =  np.inf
        plot_max = -np.inf

        for order in range(n_orders):

            wave = sci_extracted[ext_wave, order, :]
            flux = sci_extracted[ext_flux, order, :]
            noise = sci_extracted[ext_noise, order, :]

            valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(noise) & (noise > 0)

            if np.sum(valid) < 20:
                continue

            w, f, n = wave[valid], flux[valid], noise[valid]
            snr = f / n

            if panel_min <= np.median(w) <= panel_max:

                ax.plot(w, snr, lw=0.5)

                ax.text(np.median(w), np.nanpercentile(snr, 99.9), f"{order + 1}", ha="center", va="bottom", fontsize=10)

                # Update ONLY the plotting range
                plot_min = min(plot_min, np.nanmin(w))
                plot_max = max(plot_max, np.nanmax(w))

                max_snr = max(max_snr, np.nanmax(snr))
                plotted = True

                snr_stats.append({
                    "order":      int(order),
                    "snr_median": float(np.nanmedian(snr)),
                    "snr_max":    float(np.nanmax(snr)),
                    "snr_p95":    float(np.nanpercentile(snr, 95)),
                    "n_pixels":   int(len(snr))
                })

        if plotted:
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(0, max_snr * 1.1)

        ax.set_ylabel("SNR")

        if row == n_panels - 1:
            ax.set_xlabel(r"Wavelength ($\AA$)")

    plt.tight_layout()

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, f"{output_name}_qc_snr{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_snr{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    stats = {"n_orders": int(n_orders),
             "n_valid_orders": int(len(snr_stats)),
             "snr_stats": snr_stats}

    return stats, [output_path] if output_path is not None else []





def qc_final_1d_spectra_old(sci_extracted, ext_wave, ext_blaze_flux, n_orders,
                        n_panels=4, output_dir=None, output_name="spectrum",
                        save_plot=True, show_plot=False, figsize=(20, 15), dpi=300):

    wave_all = sci_extracted[ext_wave, :, :]
    wave_min, wave_max = np.nanmin(wave_all), np.nanmax(wave_all)
    dwave = wave_max - wave_min

    n_panels = min(n_panels, n_orders)
    output_path = None
    flux_stats = []

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, gridspec_kw={"hspace": 0.175})

    if n_panels == 1:
        axes = [axes]

    for row in range(n_panels):

        wave_min_n = wave_min + (dwave / n_panels) * row
        wave_max_n = wave_min + (dwave / n_panels) * (row + 1)

        ax = axes[row]
        max_flux = 0
        min_flux = np.inf
        plotted = False

        for order in range(n_orders):

            wave = sci_extracted[ext_wave, order, :]
            flux = sci_extracted[ext_blaze_flux, order, :]

            valid = np.isfinite(wave) & np.isfinite(flux)

            if np.sum(valid) < 20:
                continue

            w, f = wave[valid], flux[valid]

            if wave_min_n <= np.median(w) <= wave_max_n:

                ax.plot(w, f, lw=0.5)
                ax.text(np.median(w), np.nanpercentile(f, 99.9), f"{order + 1}",
                        ha="center", va="bottom", fontsize=10)

                wave_min_n = min(wave_min_n, np.nanmin(w))
                wave_max_n = max(wave_max_n, np.nanmax(w))

                positive = f[f > 0]
                if len(positive) > 0:
                    min_flux = min(min_flux, np.nanmin(positive))

                max_flux = max(max_flux, np.nanmax(f))
                plotted = True

                flux_stats.append({
                    "order": int(order),
                    "flux_median": float(np.nanmedian(f)),
                    "flux_max": float(np.nanmax(f)),
                    "flux_min": float(np.nanmin(f)),
                    "n_pixels": int(len(f))
                })

        if plotted:
            ax.set_xlim(wave_min_n, wave_max_n)

            if np.isfinite(min_flux) and max_flux > 0:
                ax.set_ylim(min_flux * 0.9, max_flux * 1.1)

        ax.set_ylabel("Blaze-corrected flux")

        if row == n_panels - 1:
            ax.set_xlabel(r"Wavelength ($\AA$)")

    plt.tight_layout()

    if save_plot and output_dir is not None:
        #output_path = os.path.join(output_dir, f"{output_name}_qc_final1d{PLOT_EXTENSION}")
        output_path = output_dir / f"{output_name}_qc_final1d{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    stats = {"n_orders": int(n_orders),
             "n_valid_orders": int(len(flux_stats)),
             "n_panels": int(n_panels),
             "flux_stats": flux_stats}

    return stats, [output_path] if output_path is not None else []


def qc_final_1d_spectra(sci_extracted, ext_wave, ext_blaze_flux, n_orders,
                        n_panels=4, output_dir=None, output_name="spectrum",
                        save_plot=True, show_plot=False, figsize=(20, 15), dpi=300):

    wave_all = sci_extracted[ext_wave, :, :]
    wave_min, wave_max = np.nanmin(wave_all), np.nanmax(wave_all)
    dwave = wave_max - wave_min

    n_panels = min(n_panels, n_orders)
    output_path = None
    flux_stats = []

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, gridspec_kw={"hspace": 0.175})

    if n_panels == 1:
        axes = [axes]

    for row in range(n_panels):

        ax = axes[row]
        max_flux = 0
        min_flux = np.inf
        plotted = False

        # Fixed wavelength boundaries for selecting orders
        panel_min = wave_min + (dwave / n_panels) * row
        panel_max = wave_min + (dwave / n_panels) * (row + 1)

        # Actual wavelength range of plotted orders
        plot_min = np.inf
        plot_max = -np.inf

        for order in range(n_orders):

            wave = sci_extracted[ext_wave, order, :]
            flux = sci_extracted[ext_blaze_flux, order, :]

            valid = np.isfinite(wave) & np.isfinite(flux)

            if np.sum(valid) < 20:
                continue

            w, f = wave[valid], flux[valid]

            if panel_min <= np.median(w) <= panel_max:

                ax.plot(w, f, lw=0.5)

                ax.text( np.median(w), np.nanpercentile(f, 99.9), f"{order + 1}", ha="center",  va="bottom",fontsize=10)

                # Update only the plotting range
                plot_min = min(plot_min, np.nanmin(w))
                plot_max = max(plot_max, np.nanmax(w))

                positive = f[f > 0]

                if len(positive) > 0:
                    min_flux = min(min_flux, np.nanmin(positive))

                max_flux = max(max_flux, np.nanmax(f))
                plotted = True

                flux_stats.append({
                    "order":       int(order),
                    "flux_median": float(np.nanmedian(f)),
                    "flux_max":    float(np.nanmax(f)),
                    "flux_min":    float(np.nanmin(f)),
                    "n_pixels":    int(len(f))
                })

        if plotted:

            ax.set_xlim(plot_min, plot_max)

            if np.isfinite(min_flux) and max_flux > 0:
                ax.set_ylim(min_flux * 0.9, max_flux * 1.1)

        ax.set_ylabel("Blaze-corrected flux")

        if row == n_panels - 1:
            ax.set_xlabel(r"Wavelength ($\AA$)")

    plt.tight_layout()

    if save_plot and output_dir is not None:
        output_path = output_dir / f"{output_name}_qc_final1d{PLOT_EXTENSION}"
        plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()

    stats = {
        "n_orders":       int(n_orders),
        "n_valid_orders": int(len(flux_stats)),
        "n_panels":       int(n_panels),
        "flux_stats":     flux_stats
    }

    return stats, [output_path] if output_path is not None else []