"""
Utilities for building libraries and other utility functions.

Authors: David A. Sheen, Nathan A. Mahynski
"""
import matplotlib
import itertools
import os

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt

from typing import Union, Any
from numpy.typing import NDArray

def bin_spectrum(
    spec_to_bin: NDArray[np.floating], 
    window_size: int = 4, 
    window_size_y: Union[int, None] = None
) -> NDArray[np.floating]:
    """
    Coarsen HSQC NMR spectrum into discrete histograms.
    
    Parameters
    ----------
    spec_to_bin : ndarray(float, ndim=1)
        Raw HSQC NMR spectrum to bin.

    window_size : int, optional(default=4)
        How many neighboring bins to sum together during binning.  A `window_size > 1` will coarsen the spectra.

    window_size_y : int, optional(default=None)
        Window size to use in the `y` direction (axes 0) if different from `window_size`.  If `None`, uses `window_size`. 

    Returns
    -------
    spectrum : ndarray(float, ndim=2)
        Coarsened HSQC NMR spectrum.
    """
    
    if window_size_y is None:
        window_size_y = window_size
    
    cnv = np.zeros_like(spec_to_bin[::window_size_y,::window_size,])
    for n,m in itertools.product(
        np.arange(window_size_y),
        np.arange(window_size),
    ):
        this_array = spec_to_bin[n::window_size_y,m::window_size,]
        cnv += this_array

    return cnv

def plot_nmr(
    data,
    extent,
    norm: Union[str, None] = None,
    ax: Union[matplotlib.pyplot.Axes, None] = None,
    cmap='Reds'
) -> tuple["matplotlib.image.AxesImage", "matplotlib.pyplot.colorbar"]:
    """
    Plot a single HSQC NMR spectrum.

    Parameters
    ----------
    data : ndarray(float, ndim=2)
        HSQC NMR spectrum.

    extent : list
        The bounding box in data coordinates that the spectrum will fill: (left, right, bottom, top). Uses `origin=lower` convention.
    
    norm : str, optional(default=None)
        The normalization method used to scale data to the [0, 1] range before mapping to colors using `cmap`.

    ax : matplotlib.pyplot.Axes, optional(default=None)
        Axes to plot the image on.
    
    cmap : str, optional(default='Reds')
        The `matplotlib.colors.Colormap` instance or registered colormap name used to map scalar data to colors.

    Returns
    -------
    image : matplotlib.image.AxesImage
        HSQC NMR spectrum as an image.

    colorbar : matplotlib.pyplot.colorbar
        Colorbar to go with the image.
    """
    image = ax.imshow(
        data,
        cmap=cmap,
        aspect='auto',
        norm=norm,
        extent=extent,
        origin='lower',
    )
    
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label('Intensity')
    
    return image, colorbar  

class Bruker:
    """Methods to manipulate Bruker HSQC NMR spectra."""

    @staticmethod
    def read_target_spectra(spectra_directory: str, absolute_value: bool = False) -> tuple[list, list]:
        """
        Read Bruker HSQC NMR spectra from a directory.
        
        Parameters
        ----------
        spectra_directory : str
            Directory where HSQC NMR spectra are stored in default Bruker format.

        absolute_value : bool, optional(default=False) 
            Whether or not to take the absolute value of the HSQC NMR spectra read.

        Returns
        -------
        specdirs : list[str]
            List of directories each HSQC NMR spectra is extracted from.

        data_list : list[ndarray]
            List of all HSQC NMR spectra obtained.
        """
        specdirs = os.listdir(spectra_directory)
        
        data_list = []
        for sd in specdirs:
            data_dir = '/'.join(('spectra', sd, 'pdata', '1'))
            _, data = ng.bruker.read_pdata(dir=data_dir)
            if absolute_value:
                data_list += [np.abs(data)]
            else:
                data_list += [data]

        return specdirs, data_list

    def import_single_raw(sd: str) -> dict:
        """
        Import a single raw Bruker HSQC NMR spectra.
        
        Parameters
        ----------
        sd : str
            Spectrum directory tag to read in.

        Returns
        -------
        spectrum : dict
            Dictionary containing {'data': spectra, 'extent': extent, 'uc0_scale': ax0_scale, 'uc1_scale':ax1_scale}
        """
        this_spec = dict()
        data_dir = '/'.join(('spectra', sd, 'pdata', '1'))
        dic, data = ng.bruker.read_pdata(dir=data_dir)
        u = ng.bruker.guess_udic(dic,data)

        # Extract axis scale information from metadata
        # Axis 0 is the y axis, axis 1 is the x axis
        uc0 = ng.fileiobase.uc_from_udic(u, 0)
        ax0_scale = uc0.ppm_scale() # ppm read locations (should be uniformly spaced)
        ax0_lim = uc0.ppm_limits() # upper and lower bounds

        uc1 = ng.fileiobase.uc_from_udic(u, 1)
        ax1_scale = uc1.ppm_scale()
        ax1_lim = uc1.ppm_limits()

        extent = [ax1_lim[0], ax1_lim[1], ax0_lim[0], ax0_lim[1],] # limits are chosen so ppm goes in the correct direction

        this_spec['data'] = data
        this_spec['extent'] = extent
        this_spec['uc0_scale'] = ax0_scale
        this_spec['uc1_scale'] = ax1_scale
        
        return this_spec