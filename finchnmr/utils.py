"""
Utilities for building libraries and other utility functions.

Authors: David A. Sheen, Nathan A. Mahynski
"""
import itertools
import matplotlib
import os
import copy

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import scipy.interpolate as spint

from typing import Union, Any, ClassVar
from numpy.typing import NDArray

# def bin_spectrum(
#     spec_to_bin: NDArray[np.floating],
#     window_size: int = 4,
#     window_size_y: Union[int, None] = None
# ) -> NDArray[np.floating]:
#     """
#     Coarsen HSQC NMR spectrum into discrete histograms.

#     Parameters
#     ----------
#     spec_to_bin : ndarray(float, ndim=1)
#         Raw HSQC NMR spectrum to bin.

#     window_size : int, optional(default=4)
#         How many neighboring bins to sum together during binning.  A `window_size > 1` will coarsen the spectra.

#     window_size_y : int, optional(default=None)
#         Window size to use in the `y` direction (axes 0) if different from `window_size`.  If `None`, uses `window_size`.

#     Returns
#     -------
#     spectrum : ndarray(float, ndim=2)
#         Coarsened HSQC NMR spectrum.
#     """

#     if window_size_y is None:
#         window_size_y = window_size

#     cnv = np.zeros_like(spec_to_bin[::window_size_y,::window_size,])
#     for n,m in itertools.product(
#         np.arange(window_size_y),
#         np.arange(window_size),
#     ):
#         this_array = spec_to_bin[n::window_size_y,m::window_size,]
#         cnv += this_array

#     return cnv


# def plot_nmr(
#     data,
#     extent,
#     norm: Union[str, None] = None,
#     ax: Union[matplotlib.pyplot.Axes, None] = None,
#     cmap='Reds'
# ) -> tuple["matplotlib.image.AxesImage", "matplotlib.pyplot.colorbar"]:
#     """
#     Plot a single HSQC NMR spectrum.

#     Parameters
#     ----------
#     data : ndarray(float, ndim=2)
#         HSQC NMR spectrum.

#     extent : list
#         The bounding box in data coordinates that the spectrum will fill: (left, right, bottom, top). Uses `origin=lower` convention.

#     norm : str, optional(default=None)
#         The normalization method used to scale data to the [0, 1] range before mapping to colors using `cmap`.

#     ax : matplotlib.pyplot.Axes, optional(default=None)
#         Axes to plot the image on.

#     cmap : str, optional(default='Reds')
#         The `matplotlib.colors.Colormap` instance or registered colormap name used to map scalar data to colors.

#     Returns
#     -------
#     image : matplotlib.image.AxesImage
#         HSQC NMR spectrum as an image.

#     colorbar : matplotlib.pyplot.colorbar
#         Colorbar to go with the image.
#     """
#     image = ax.imshow(
#         data,
#         cmap=cmap,
#         aspect='auto',
#         norm=norm,
#         extent=extent,
#         origin='lower',
#     )

#     colorbar = plt.colorbar(image, ax=ax)
#     colorbar.set_label('Intensity')

#     return image, colorbar

# class Bruker:
#     """Methods to manipulate Bruker HSQC NMR spectra."""

#     @staticmethod
#     def read_target_spectra(spectra_directory: str, absolute_value: bool = False) -> tuple[list, list]:
#         """
#         Read Bruker HSQC NMR spectra from a directory.

#         Parameters
#         ----------
#         spectra_directory : str
#             Directory where HSQC NMR spectra are stored in default Bruker format.

#         absolute_value : bool, optional(default=False)
#             Whether or not to take the absolute value of the HSQC NMR spectra read.

#         Returns
#         -------
#         specdirs : list[str]
#             List of directories each HSQC NMR spectra is extracted from.

#         data_list : list[ndarray]
#             List of all HSQC NMR spectra obtained.
#         """
#         specdirs = os.listdir(spectra_directory)

#         data_list = []
#         for sd in specdirs:
#             data_dir = '/'.join(('spectra', sd, 'pdata', '1'))
#             _, data = ng.bruker.read_pdata(dir=data_dir)
#             if absolute_value:
#                 data_list += [np.abs(data)]
#             else:
#                 data_list += [data]

#         return specdirs, data_list

#     def import_single_raw(sd: str) -> dict:
#         """
#         Import a single raw Bruker HSQC NMR spectra.

#         Parameters
#         ----------
#         sd : str
#             Spectrum directory tag to read in.

#         Returns
#         -------
#         spectrum : dict
#             Dictionary containing {'data': spectra, 'extent': extent, 'uc0_scale': ax0_scale, 'uc1_scale':ax1_scale}
#         """
#         this_spec = dict()
#         data_dir = '/'.join(('spectra', sd, 'pdata', '1'))
#         dic, data = ng.bruker.read_pdata(dir=data_dir)
#         u = ng.bruker.guess_udic(dic,data)

#         # Extract axis scale information from metadata
#         # Axis 0 is the y axis, axis 1 is the x axis
#         uc0 = ng.fileiobase.uc_from_udic(u, 0)
#         ax0_scale = uc0.ppm_scale() # ppm read locations (should be uniformly spaced)
#         ax0_lim = uc0.ppm_limits() # upper and lower bounds

#         uc1 = ng.fileiobase.uc_from_udic(u, 1)
#         ax1_scale = uc1.ppm_scale()
#         ax1_lim = uc1.ppm_limits()

#         extent = [ax1_lim[0], ax1_lim[1], ax0_lim[0], ax0_lim[1],] # limits are chosen so ppm goes in the correct direction

#         this_spec['data'] = data
#         this_spec['extent'] = extent
#         this_spec['uc0_scale'] = ax0_scale
#         this_spec['uc1_scale'] = ax1_scale

#         return this_spec


# class Substance:
#     _data: NDArray[np.floating]
#     _extent: list
#     _uc0_scale:
#     _uc1_scale
#     _interp_fcn: Any

#     def __init__(self, filename=None, style='bruker'):
#         if filename is not None:
#             self.read(filename=filename, style=style)
#         return

#     @property
#     def data(self):
#         return copy.deepcopy(self._data)

#     @property
#     def extent(self):
#         return copy.deepcopy(self._extent)

#     @property
#     def scale(self):
#         return (self._uc0_scale, self.__uc1_scale)

#     @property
#     def fitted(self):
#         return copy.deepcopy(self._fitted)

#     def read(self, filename, style='bruker'):
#         return

#     def from_xml(self, filename):
#         return

#     def fit(self, reference_substance: "Substance") -> "Substance":
#         # crop_and_pad etc. and store results in fitted
#         # This can be stored in a library
#         fitted = Substance()

#         return fitted

#     @staticmethod
#     def _crop_overlap(scale_to_crop: NDArray[np.floating], target_scale: NDArray[np.floating]) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
#         """
#         Remove all values of `scale_to_crop` that are outside the bounds prescribed by `target_scale`.

#         For best results, both `scale_to_crop` and `target_scale` are monotonic.

#         Parameters
#         ----------
#         scale_to_crop : ndarray(float, ndim=1)
#             Array to mask if values are outside `target_scale`.

#         target_scale : ndarray(float, ndim=1)
#             Bounds to keep `scale_to_crop` inside.

#         Returns
#         -------
#         overlap_mask : ndarray(bool, ndim=1)
#             Logical mask that masks out values beyond the `target_scale` range.

#         scale_to_crop : ndarray(float, ndim=1)
#             Masked `scale_to_crop` array containing value in range.
#         """
#         overlap_mask = np.logical_and(
#             scale_to_crop > target_scale.min(),
#             scale_to_crop < target_scale.max(),
#         )

#         return overlap_mask, scale_to_crop[overlap_mask]

#     @staticmethod
#     def _pad_scale(scale_to_pad: NDArray[np.floating], target_scale: NDArray[np.floating], max_side: str = 'left') -> tuple[tuple, tuple, float]:
#         """
#         Pad `scale_to_pad` so that it has roughly the same extent as `target_scale`.

#         The `max_side` keyword defines whether the scales are monotonically increasing or decreasing.
#         Also, `scale_to_pad` must be monotonic and uniformly spaced.

#         Parameters
#         ----------
#         scale_to_pad : ndarray(float, ndim=1)

#         target_scale : ndarray(float, ndim=1)

#         max_side : str, optional(default="left")

#         Returns
#         -------

#         """
#         scale_inc = scale_to_pad[0] - scale_to_pad[1] # Determine uniform increment size

#         # Increments to add on the maximum side
#         max_to_pad = target_scale.max() - scale_to_pad.max() # Absolute distance from end of scale to end of target scale
#         max_incs = int(max(max_to_pad//scale_inc, 0)) # Number of increments to add

#         # Increments to add on the minimum side
#         min_to_pad = scale_to_pad.min() - target_scale.min()
#         min_incs = int(max(min_to_pad//scale_inc,0))

#         # Pad scale_to_pad with the appropriate values
#         # Scale has highest value at [0], values decrease with increasing [index]
#         if max_side.lower() == 'left':
#             pad_left = np.linspace(
#                 scale_to_pad.max() + max_incs*scale_inc,
#                 scale_to_pad.max() + scale_inc,
#                 max_incs
#             )
#             pad_right = np.linspace(
#                 scale_to_pad.min() - scale_inc,
#                 scale_to_pad.min() - min_incs*scale_inc,
#                 min_incs
#             )

#         padded_scale = np.concatenate(
#             (np.concatenate((pad_left, scale_to_pad)), pad_right)
#         )

#         return (max_incs, min_incs), (pad_left, pad_right), padded_scale

#     @staticmethod
#     def _crop_and_pad(substance_data: dict, spec_data: dict) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
#         """
#         Compute overlap masks.

#         Parameters
#         ----------

#         Returns
#         -------

#         """
#         overlap_mask1, uc1_overlap = crop_overlap(substance_data['uc1_scale'], spec_data['uc1_scale'])
#         overlap_mask0, uc0_overlap = crop_overlap(substance_data['uc0_scale'], spec_data['uc0_scale'])

#         substance_overlap = substance_data['data'][overlap_mask0, :][:, overlap_mask1]

#         pad0, padded0, uc0_pad = pad_scale(uc0_overlap[:], spec_data['uc0_scale'])
#         pad1, padded1, uc1_pad = pad_scale(uc1_overlap[:], spec_data['uc1_scale'])

#         substance_padded = np.pad(substance_overlap, (pad0, pad1))

#         return substance_padded, uc0_pad, uc1_pad


# class Library:
#     """
#     instead, make this contain a list of substances
#     when fit, all substances are fit in a loop to a fixed reference
#     a member called "data" or something flattens this into an X matrix that can be used for fitting
#     the library should use a substance list when instantiated but do not store these to keep size down
#     """


#     MAX_Y_SIZE: ClassVar[int]
#     MAX_ASPECT_RATIO: ClassVar[int]
#     BIN_SCALE: ClassVar[int]

#     def __init__(self, substance_path, binning_size_default = 16, style='bruker'):
#         self.MAX_Y_SIZE: 256
#         self.MAX_ASPECT_RATIO: 4
#         self.BIN_SCALE: 16

#         # 1. Load the data from disk
#         substance_library = dict()
#         for substance_name in [x for x in os.listdir(substance_path) if x.endswith('HSQC')]:
#             substance_data = dict()
#             full_path = '/'.join((substance_path, substance_name, '1/pdata/1/'))

#             try:
#                 # Read the pdata, extracting the nmr data and the metadata dictionary
#                 if style.lower() == 'bruker':
#                     dic, data_raw = ng.bruker.read_pdata(dir=full_path)
#                     u = ng.bruker.guess_udic(dic, data_raw)
#                 else:
#                     raise ValueError(f'Unrecognized manufacturer: {style}')

#                 # This is very custom logic
#                 if data_raw.shape[1] > self.MAX_Y_SIZE:
#                     binning_size = binning_size_default
#                     if data_raw.shape[1] / data_raw.shape[0] > self.MAX_ASPECT_RATIO:
#                         binning_size_y = binning_size // self.BIN_SCALE
#                     else:
#                         binning_size_y = binning_size
#                 else:
#                     binning_size = 1

#                 # Extract axis scale information from metadata
#                 # Axis 0 is the y axis, axis 1 is the x axis
#                 uc0 = ng.fileiobase.uc_from_udic(u, 0)
#                 ax0_scale = uc0.ppm_scale()[binning_size_y // 2::binning_size_y] # ppm read locations (should be uniformly spaced)
#                 ax0_lim = uc0.ppm_limits() # upper and lower bounds

#                 uc1 = ng.fileiobase.uc_from_udic(u, 1)
#                 ax1_scale = uc1.ppm_scale()[binning_size // 2::binning_size]
#                 ax1_lim = uc1.ppm_limits()

#                 extent = [ax1_lim[0], ax1_lim[1], ax0_lim[0], ax0_lim[1],] # limits are chosen so ppm goes in the correct direction

#                 data = bin_spectrum(data_raw, window_size=binning_size, window_size_y=binning_size_y)

#                 substance_data['data'] = data
#                 substance_data['extent'] = extent
#                 substance_data['uc0_scale'] = ax0_scale
#                 substance_data['uc1_scale'] = ax1_scale
#                 substance_data['interp_fcn'] = spint.RegularGridInterpolator(
#                     (ax0_scale,ax1_scale),
#                     data,
#                     fill_value=0,bounds_error=False,
#                     method='cubic',
#                 )

#                 substance_library[substance_name] = substance_data
#             except OSError:
#                 warnings.warn(f'No data found for substance {substance_name}')

#         self.library = substance_library

#     def fit(self, spec_data):

#         # 2. Crop and Pad
#         for this_sub_name, v in self.library.items():
#             this_sub = self.library[this_sub_name]
#             sub_pad, uc0_pad, uc1_pad = crop_and_pad(this_sub, spec_data)

#             ax0_lim = uc0_pad.max(), uc0_pad.min()
#             ax1_lim = uc1_pad.max(), uc1_pad.min()

#             extent = [ax1_lim[0], ax1_lim[1], ax0_lim[0], ax0_lim[1],]

#             this_sub['data_padded'] = sub_pad
#             this_sub['uc0_padded'] = uc0_pad
#             this_sub['uc1_padded'] = uc1_pad
#             this_sub['extent_padded'] = extent

#         # 3. Resize and take absolute value
#         self.reconstructed_spectra = dict()
#         for this_sub_name,v in self.library.items():
#             this_sub = self.library[this_sub_name]
#             sub_dat_resize = skimage.transform.resize(
#                 this_sub["data_padded"],
#                 spec_data['data'].shape
#             )
#             self.reconstructed_spectra[this_sub_name] = np.abs(sub_dat_resize)

#     def save(self, filename):
#         return
