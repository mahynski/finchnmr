"""
Functions for defining a substance measured with HSQC NMR.

Authors: David A. Sheen, Nathan A. Mahynski
"""
import copy
import itertools
import matplotlib
import os
import skimage

import nmrglue as ng
import numpy as np
import scipy.interpolate as spint

from numpy.typing import NDArray
from typing import Any, Union, ClassVar

class Substance:
    _data: ClassVar[NDArray[np.floating]]
    _extent: ClassVar[list]
    _uc0_scale: ClassVar[NDArray[np.floating]]
    _uc1_scale: ClassVar[NDArray[np.floating]]
    _interp_fcn: ClassVar[Any]
    
    def __init__(self, filename: Union[str, None] = None, style: str = 'bruker') -> None:
        """
        Instantiate the class.
        
        If `filename=None` no data is read.
        
        Parameters
        ----------
        filename : str, optional(default=None)
            If specified, read data from this file; otherwise create an empty class.
            
        style : str, optional(default="bruker")
            Manufacturer of NMR instrument which dictates how to extract this information. If `filename=None` this is ignored. 
            
        Example
        -------
        >>> s = Substance('test_data/my_substance/pdata/1', style='bruker')
        """
        if filename is not None:
            self.read(filename=filename, style=style)
    
    @property
    def data(self):
        """Return the 2D HSQC NMR spectrum."""
        return copy.deepcopy(self._data)
    
    @property
    def extent(self):
        """Return the bounds of the spectrum."""
        return copy.deepcopy(self._extent)
    
    @property
    def scale(self):
        """Return the grid points the spectrum is reported on."""
        return (self._uc0_scale, self._uc1_scale)
    
    @property
    def interp_fcn(self):
        """Return the interpolation function."""
        return self._interp_fcn
    
    @staticmethod
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

    def read(self, filename, style='bruker'):
        """
        Read HSQC NMR spectrum from a directory created by the instrument.
        
        Parameters
        ----------
        style : str, optional(default='bruker')
            Manufacturer of NMR instrument which dictates how to extract this information. At the moment only 'bruker' is supported.
            
        Example
        -------
        >>> s = Substance()
        >>> s.read('test_data/my_substance/pdata/1', style='bruker')
        """
        MAX_Y_SIZE = 256
        MAX_ASPECT_RATIO = 4
        BIN_SCALE = 16
            
        # Conver to absolute path
        filename = os.path.abspath(filename)
            
        try:
            # Read the pdata, extracting the nmr data and the metadata dictionary
            if style.lower() == 'bruker':
                if filename.split('/')[-2] != 'pdata':
                    raise ValueError('For style="bruker" the filename should include the path to the subfolder in "pdata".')
                dic, data_raw = ng.bruker.read_pdata(dir=filename)
                u = ng.bruker.guess_udic(dic, data_raw)
            else:
                raise ValueError(f'Unrecognized manufacturer: {style}')

            # This is very custom logic
            if data_raw.shape[1] > MAX_Y_SIZE:
                binning_size = 16
                if data_raw.shape[1] / data_raw.shape[0] > MAX_ASPECT_RATIO:
                    binning_size_y = binning_size // BIN_SCALE
                else:
                    binning_size_y = binning_size
            else:
                binning_size = 1
                binning_size_y = 1
                
            # Extract axis scale information from metadata; axis 0 is the y axis, axis 1 is the x axis
            uc0 = ng.fileiobase.uc_from_udic(u, 0)
            uc0_scale = uc0.ppm_scale()[binning_size_y // 2::binning_size_y] # ppm read locations (should be uniformly spaced)
    
            uc1 = ng.fileiobase.uc_from_udic(u, 1)
            uc1_scale = uc1.ppm_scale()[binning_size // 2::binning_size]
        except Exception as e:
            raise Exception(f'Unable to read substance in {filename} : {e}')
                
        self._data = self.bin_spectrum(data_raw, window_size=binning_size, window_size_y=binning_size_y)
        self._extent = [uc1_scale.max(), uc1_scale.min(), uc0_scale.max(), uc0_scale.min()] # Limits are chosen so ppm goes in the correct direction
        self._uc0_scale = uc0_scale
        self._uc1_scale = uc1_scale
        self._interp_fcn = spint.RegularGridInterpolator(
            (self._uc0_scale, self._uc1_scale),
            self._data,
            fill_value=0,
            bounds_error=False,
            method='cubic',
        )
            
        return
    
    def from_xml(self, filename):
        return
    
    def fit(self, reference: "Substance") -> "Substance":
        """
        Align this substance to another one which serves as a reference.
        
        Parameters
        ----------
        reference : Substance
            Substance to align this one with this (match extent, etc.).
            
        Returns
        -------
        aligned : Substance
            New Substance which is a version of this one, but is now aligned/matched with `reference`.
        """
        aligned = Substance()
        
        # 1. Crop and pad to match reference_substance
        aligned._data, aligned._uc0_scale, aligned._uc1_scale = self._crop_and_pad(reference)
        aligned._extent = [aligned._uc1_scale.max(), aligned._uc1_scale.min(), aligned._uc0_scale.max(), aligned._uc0_scale.min()]
        
        # 2. Resize and take absolute value
        aligned._data = np.abs(
            skimage.transform.resize(
                aligned._data,
                reference_substance._data.shape
            )
        )
        
        return aligned
    
    def _crop_and_pad(self, reference_substance: "Substance") -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute overlap masks.

        Parameters
        ----------
        reference_substance : Substance
            Another Substance we would like to "align" this NMR spectrum with.
            
        Returns
        -------
        data_padded : ndarray(float, ndim=2)
            Padded 2D spectrum.
            
        uc0_pad : ndarray(float, ndim=1)
            Padded uc0 axis.
        
        uc1_pad : ndarray(float, ndim=1)
            Padded uc1 axis.
        """
        def crop_overlap(scale_to_crop, target_scale):
            """Remove all values of `scale_to_crop` that are outside the bounds prescribed by `target_scale`. For best results, both `scale_to_crop` and `target_scale` are monotonic."""
            overlap_mask = np.logical_and(
                scale_to_crop >= target_scale.min(),
                scale_to_crop <= target_scale.max(),
            )

            return overlap_mask, scale_to_crop[overlap_mask]
        
        def pad_scale(scale_to_pad, target_scale, max_side='left'):
            """Pad `scale_to_pad` so that it has roughly the same extent as `target_scale`. The `max_side` keyword defines whether the scales are monotonically increasing or decreasing. Also, `scale_to_pad` must be monotonic and uniformly spaced."""
            scale_inc = scale_to_pad[0] - scale_to_pad[1] # Determine uniform increment size

            # Increments to add on the maximum side
            max_to_pad = target_scale.max() - scale_to_pad.max() # Absolute distance from end of scale to end of target scale
            max_incs = int(max(max_to_pad//scale_inc, 0)) # Number of increments to add

            # Increments to add on the minimum side
            min_to_pad = scale_to_pad.min() - target_scale.min()
            min_incs = int(max(min_to_pad//scale_inc,0))

            # Pad scale_to_pad with the appropriate values
            if max_side.lower() == 'left':
                # Scale has highest value at [0], values decrease with increasing [index]
                pad_left = np.linspace(
                    scale_to_pad.max() + max_incs*scale_inc,
                    scale_to_pad.max() + scale_inc,
                    max_incs
                )
                pad_right = np.linspace(
                    scale_to_pad.min() - scale_inc,
                    scale_to_pad.min() - min_incs*scale_inc,
                    min_incs
                )
            else:
                pad_left = np.linspace(
                    scale_to_pad.max() + scale_inc,
                    scale_to_pad.max() + max_incs*scale_inc,
                    max_incs
                )
                pad_right = np.linspace(
                    scale_to_pad.min() - min_incs*scale_inc,
                    scale_to_pad.min() - scale_inc,
                    min_incs
                )

            padded_scale = np.concatenate(
                (np.concatenate((pad_left, scale_to_pad)), pad_right)
            )

            return (max_incs, min_incs), (pad_left, pad_right), padded_scale
    
        overlap_mask0, uc0_overlap = crop_overlap(self._uc0_scale, reference_substance._uc0_scale)
        overlap_mask1, uc1_overlap = crop_overlap(self._uc1_scale, reference_substance._uc1_scale)
        
        data_overlap = self._data[overlap_mask0, :][:, overlap_mask1]

        pad0, _, uc0_pad = pad_scale(uc0_overlap[:], reference_substance._uc0_scale)
        pad1, _, uc1_pad = pad_scale(uc1_overlap[:], reference_substance._uc1_scale)

        data_padded = np.pad(data_overlap, (pad0, pad1))

        return data_padded, uc0_pad, uc1_pad
    
    def plot(
        self,
        norm: Union[str, None] = None,
        ax: Union[matplotlib.pyplot.Axes, None] = None,
        cmap='RdBu'
    ) -> tuple["matplotlib.image.AxesImage", "matplotlib.pyplot.colorbar"]:
        """
        Plot a single HSQC NMR spectrum.

        Parameters
        ----------
        norm : str or matplotlib.colors.Normalize, optional(default=None)
            The normalization method used to scale data to the [0, 1] range before mapping to colors using `cmap`.  If `None`, a `matplotlib.colors.SymLogNorm` is used.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot the image on.

        cmap : str, optional(default='RdBu')
            The `matplotlib.colors.Colormap` instance or registered colormap name used to map scalar data to colors.

        Returns
        -------
        image : matplotlib.image.AxesImage
            HSQC NMR spectrum as an image.

        colorbar : matplotlib.pyplot.colorbar
            Colorbar to go with the image.
        """
        if ax is None:
            _, ax = plt.subplots()
            
        if norm is None:
            norm = matplotlib.colors.SymLogNorm(linthresh=np.max(np.abs(self._data))/100)
            
        image = ax.imshow(
            self._data,
            cmap=cmap,
            aspect='auto',
            norm=norm,
            extent=self._extent,
            origin='lower',
        )

        colorbar = plt.colorbar(image, ax=ax)
        colorbar.set_label('Intensity')

        return image, colorbar  