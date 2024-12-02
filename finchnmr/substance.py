"""
Functions for defining a substance measured with HSQC NMR.

Authors: Nathan A. Mahynski, David A. Sheen
"""
import copy
import itertools
import matplotlib
import os
import skimage

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import plotly.express as px

from numpy.typing import NDArray
from typing import Any, Union, ClassVar, Literal

import warnings


class Substance:
    """Substance that was measured with HSQC NMR."""

    _name: ClassVar[str]
    _data: ClassVar[NDArray[np.floating]]
    _extent: ClassVar[tuple[float, float, float, float]]
    _uc0_scale: ClassVar[NDArray[np.floating]]
    _uc1_scale: ClassVar[NDArray[np.floating]]
    _window_size: ClassVar[int]
    _window_size_y: ClassVar[Union[int, None]]

    def __init__(
        self,
        pathname: Union[str, None] = None,
        name: str = "",
        style: str = "bruker",
        warning: Literal[
            "default", "error", "ignore", "always", "all", "module", "once"
        ] = "error",
    ) -> None:
        """
        Instantiate the substance.

        If `pathname=None` no data is read.

        Parameters
        ----------
        pathname : str, optional(default=None)
            If specified, read data from this folder; otherwise create an empty class.

        name : str, optional(default="")
            Name of the substance, e.g., "octanol".

        style : str, optional(default="bruker")
            Manufacturer of NMR instrument which dictates how to extract this information. If `pathname=None` this is ignored.

        warning : str, optional(default="error")
            How to handle warnings thrown when reading from disk; 'error' causes an Exception to be thrown stopping the code, however, if you are confident that the warnings are not relevant, you can set this to 'default' to simply report the warnings instead.

        Example
        -------
        >>> s = Substance('test_data/my_substance/pdata/1', style='bruker')
        """
        if pathname is not None:
            self.read(
                pathname=pathname, name=name, style=style, warning=warning
            )

    @property
    def name(self) -> str:
        """Return the name of the substance."""
        return self._name

    @property
    def data(self) -> NDArray[np.floating]:
        """Return the 2D HSQC NMR spectrum."""
        return copy.deepcopy(self._data)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return the bounds of the spectrum."""
        return copy.deepcopy(self._extent)

    @property
    def scale(self) -> tuple:
        """Return the grid points the spectrum is reported on."""
        return (self._uc0_scale, self._uc1_scale)

    def flatten(self) -> NDArray[np.floating]:
        """Return a flattened (1D) version of the data."""
        return self._data.flatten()

    def unflatten(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Unflatten or reshape data back to the original 2D shape."""
        if data.ndim > 1:
            raise Exception("Data to unflatten should be one dimensional.")
        return data.reshape(self._data.shape)

    def _set_data(self, data: NDArray[np.floating]) -> None:
        """Manually assign data; used only under very special circumstances."""
        setattr(self, "_data", copy.deepcopy(data))

    @staticmethod
    def bin_spectrum(
        spec_to_bin: NDArray[np.floating],
        window_size: int = 4,
        window_size_y: Union[int, None] = None,
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

        cnv = np.zeros_like(
            spec_to_bin[
                ::window_size_y,
                ::window_size,
            ]
        )
        for n, m in itertools.product(
            np.arange(window_size_y),
            np.arange(window_size),
        ):
            this_array = spec_to_bin[
                n::window_size_y,
                m::window_size,
            ]
            cnv += this_array

        return cnv

    def read(
        self,
        pathname: str,
        name: str = "",
        style: str = "bruker",
        warning: Literal[
            "default", "error", "ignore", "always", "all", "module", "once"
        ] = "error",
    ) -> None:
        """
        Read HSQC NMR spectrum from a directory created by the instrument.

        Parameters
        ----------
        pathname : str, optional(default=None)
            Read data from this folder.

        name : str, optional(default=None)
            Name of the substance, e.g., "octanol".

        style : str, optional(default='bruker')
            Manufacturer of NMR instrument which dictates how to extract this information. At the moment only 'bruker' is supported.

        warning : str, optional(default="error")
            How to handle warnings thrown when reading from disk; 'error' causes an Exception to be thrown stopping the code, however, if you are confident that the warnings are not relevant, you can set this to 'default' to simply report the warnings instead.

        Example
        -------
        >>> s = Substance()
        >>> s.read('test_data/my_substance/pdata/1', name='my_substance', style='bruker')
        """
        MAX_Y_SIZE = 256
        MAX_ASPECT_RATIO = 4
        BIN_SCALE = 16

        # Conver to absolute path
        pathname = os.path.abspath(pathname)

        try:
            with warnings.catch_warnings():
                # Adjust warning behavior during this stage
                warnings.simplefilter(warning)

                # Read the pdata, extracting the nmr data and the metadata dictionary
                if style.lower() == "bruker":
                    if pathname.split("/")[-2] != "pdata":
                        raise ValueError(
                            'For style="bruker" the pathname should include the path to the subfolder in "pdata".'
                        )
                    dic, data_raw = ng.bruker.read_pdata(dir=pathname)
                    u = ng.bruker.guess_udic(dic, data_raw)
                else:
                    raise ValueError(f"Unrecognized manufacturer: {style}")

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
                uc0_scale = uc0.ppm_scale()[
                    binning_size_y // 2 :: binning_size_y
                ]  # ppm read locations (should be uniformly spaced)

                uc1 = ng.fileiobase.uc_from_udic(u, 1)
                uc1_scale = uc1.ppm_scale()[binning_size // 2 :: binning_size]
        except Exception as e:
            raise Exception(f"Unable to read substance in {pathname} : {e}")

        setattr(
            self,
            "_data",
            self.bin_spectrum(
                data_raw, window_size=binning_size, window_size_y=binning_size_y
            ),
        )
        setattr(self, "_window_size", binning_size)
        setattr(self, "_window_size_y", binning_size_y)
        setattr(
            self,
            "_extent",
            (
                uc1_scale.max(),
                uc1_scale.min(),
                uc0_scale.max(),
                uc0_scale.min(),
            ),
        )  # Limits are chosen so ppm goes in the correct direction
        setattr(self, "_uc0_scale", uc0_scale)
        setattr(self, "_uc1_scale", uc1_scale)
        setattr(self, "_name", name)

        return

    def from_xml(self, filename: str) -> None:
        """Read substance from XML peak list."""
        raise NotImplementedError

    def fit(self, reference: "Substance") -> "Substance":
        """
        Align this substance to another one which serves as a reference.

        This also transforms the intensities to absolute values.

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
        _data, _uc0_scale, _uc1_scale = self._crop_and_pad(reference)
        setattr(aligned, "_data", _data)
        setattr(aligned, "_uc0_scale", _uc0_scale)
        setattr(aligned, "_uc1_scale", _uc1_scale)
        setattr(
            aligned,
            "_extent",
            (
                aligned._uc1_scale.max(),
                aligned._uc1_scale.min(),
                aligned._uc0_scale.max(),
                aligned._uc0_scale.min(),
            ),
        )

        # 2. Resize
        setattr(
            aligned,
            "_data",
            skimage.transform.resize(aligned._data, reference._data.shape),
        )

        return aligned

    def _crop_and_pad(
        self, reference: "Substance"
    ) -> tuple[
        NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
    ]:
        """
        Crop and/or pad the 2D HSQC NMR spectrum to be aligned with the reference HSQC NMR spectrum.

        Parameters
        ----------
        reference : Substance
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

        def pad_scale(scale_to_pad, target_scale, max_side="left"):
            """Pad `scale_to_pad` so that it has roughly the same extent as `target_scale`. The `max_side` keyword defines whether the scales are monotonically increasing or decreasing. Also, `scale_to_pad` must be monotonic and uniformly spaced."""
            scale_inc = (
                scale_to_pad[0] - scale_to_pad[1]
            )  # Determine uniform increment size

            # Increments to add on the maximum side
            max_to_pad = (
                target_scale.max() - scale_to_pad.max()
            )  # Absolute distance from end of scale to end of target scale
            max_incs = int(
                max(max_to_pad // scale_inc, 0)
            )  # Number of increments to add

            # Increments to add on the minimum side
            min_to_pad = scale_to_pad.min() - target_scale.min()
            min_incs = int(max(min_to_pad // scale_inc, 0))

            # Pad scale_to_pad with the appropriate values
            if max_side.lower() == "left":
                # Scale has highest value at [0], values decrease with increasing [index]
                pad_left = np.linspace(
                    scale_to_pad.max() + max_incs * scale_inc,
                    scale_to_pad.max() + scale_inc,
                    max_incs,
                )
                pad_right = np.linspace(
                    scale_to_pad.min() - scale_inc,
                    scale_to_pad.min() - min_incs * scale_inc,
                    min_incs,
                )
            else:
                pad_left = np.linspace(
                    scale_to_pad.max() + scale_inc,
                    scale_to_pad.max() + max_incs * scale_inc,
                    max_incs,
                )
                pad_right = np.linspace(
                    scale_to_pad.min() - min_incs * scale_inc,
                    scale_to_pad.min() - scale_inc,
                    min_incs,
                )

            padded_scale = np.concatenate(
                (np.concatenate((pad_left, scale_to_pad)), pad_right)
            )

            return (max_incs, min_incs), (pad_left, pad_right), padded_scale

        overlap_mask0, uc0_overlap = crop_overlap(
            self._uc0_scale, reference._uc0_scale
        )
        overlap_mask1, uc1_overlap = crop_overlap(
            self._uc1_scale, reference._uc1_scale
        )

        data_overlap = self._data[overlap_mask0, :][:, overlap_mask1]

        pad0, _, uc0_pad = pad_scale(uc0_overlap[:], reference._uc0_scale)
        pad1, _, uc1_pad = pad_scale(uc1_overlap[:], reference._uc1_scale)

        data_padded = np.pad(data_overlap, (pad0, pad1))

        return data_padded, uc0_pad, uc1_pad

    def plot(
        self,
        norm: Union["matplotlib.colors.Normalize", None] = None,
        ax: Union["matplotlib.pyplot.Axes", None] = None,
        cmap="RdBu",
        absolute_values=False,
        backend: str = "mpl",
        title: Union[str, None] = None
    ):
        """
        Plot a single HSQC NMR spectrum.

        Parameters
        ----------
        norm : str or matplotlib.colors.Normalize, optional(default=None)
            The normalization method used to scale data to the [0, 1] range before mapping to colors using `cmap`.  If `None`, a `matplotlib.colors.SymLogNorm` is used. This is currently only supported for the matplotlib backend.

        ax : matplotlib.pyplot.Axes, optional(default=None)
            Axes to plot the image on. This is currently only supported for the matplotlib backend.

        cmap : str, optional(default='RdBu')
            The `matplotlib.colors.Colormap` instance or registered colormap name used to map scalar data to colors.  String names are largely similar between the plotting backends and can usually be used interchangeably.

        absolute_values : bool, optional(default=False)
            Whether to plot the absolute values of the data (intensities).

        backend : str, optional(default='mpl')
            Plotting library to use; the default 'mpl' uses matplotlib and is not interactive, while 'plotly' will yield interactive plots.

        title : str, optional(default=None)
            Optional title to put on plot; otherwise this defaults to the substance's name.

        Returns
        -------
        if backend == 'mpl':

            image : matplotlib.image.AxesImage
                HSQC NMR spectrum as an image.

            colorbar : matplotlib.pyplot.colorbar
                Colorbar to go with the image.

        if backend == 'plotly':

            image : plotly.graph_objs._figure.Figure
                HSQC NMR spectrum as an image.
        """
        if backend == "mpl":
            if ax is None:
                _, ax = plt.subplots()

            if norm is None:
                norm = matplotlib.colors.SymLogNorm(
                    linthresh=np.max(np.abs(self._data)) / 100
                )

            image_mpl = ax.imshow(
                self._data if not absolute_values else np.abs(self._data),
                cmap=cmap,
                aspect="auto",
                norm=norm,
                extent=self._extent,
                origin="lower",
            )
            ax.set_xlabel(r"$\omega_2-^1$H (ppm)")
            ax.set_ylabel(r"$\omega_1-^{13}$C (ppm)")

            colorbar = plt.colorbar(image_mpl, ax=ax)
            colorbar.set_label("Intensity")

            plt.gca().set_title(self._name if title is None else title)

            return image_mpl, colorbar
        elif backend == "plotly":
            image_plt = px.imshow(
                self._data if not absolute_values else np.abs(self._data),
                x=self._uc1_scale,
                y=self._uc0_scale,
                text_auto=False,
                aspect="auto",
                origin="upper",
                color_continuous_scale=cmap,
                title=self._name if title is None else title,
            )
            image_plt.update_xaxes(autorange="reversed")
            image_plt.update_layout(xaxis_title=r"$\omega_2-^1{\rm H~(ppm)}$")
            image_plt.update_layout(yaxis_title=r"$\omega_1-^{13}{\rm C~(ppm)}$")
            image_plt.update_layout(coloraxis_colorbar=dict(title="Intensity"))
            image_plt.update_traces(
                hovertemplate="x: %{x}<br>y: %{y}<br>Intensity: %{z}"
            )

            return image_plt
        else:
            raise ValueError(f"Unrecognized backend {backend}")
