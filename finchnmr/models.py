"""
Tools to build linear models.

Authors: Nathan A. Mahynski
"""
import pickle
import utils

import sklearn.linear_model as sklm
import numpy as np

from typing import Union, Any, Sequence
from numpy.typing import NDArray

class LASSO:
    """Fit LASSO model(s) from the scikit-learn library."""

    @staticmethod
    def calculate_models(
        target_spectra: list[NDArray[np.floating]], 
        library_spectra: NDArray[np.floating], 
        lasso_kw: Union[dict, None] = None, 
        binning_kw: Union[dict, None] = None
    ) -> tuple[list, list]:
        """
        Compute a LASSO model for a list of target spectra.

        Parameters
        ----------
        target_spectra : list(ndarray(float, ndim=1))
            List of flattened HSQC NMR spectra to fit.

        library_spectra : ndarray(float, ndim=2)
            Flattened HSQC NMR spectra of a library of known compounds to fit against.  Coefficients will follow the order of spectra provided here.

        lasso_kw : dict, optional(default=None)
            Keyword arguments to an `sklearn.linear_model.LASSO` model.

        binning_kw : dict, optional(default=None)
            Keyword arguments to `utils.bin_spectrum`.

        Returns
        -------
        models_list : list
            List of `sklearn.linear_model.LASSO` models trained.

        spectra_list : list(ndarray(float, ndim=1))
            List of HSQC NMR spectra fit during training (coarsened versions of `target_spectra`).
        """
        models_list = []
        spectra_list = []
        
        if binning_kw is None:
            binning_kw = {}
        
        if lasso_kw is None:
            lasso_kw = {}

        for target_spectrum in target_spectra:
            binned_spectrum = utils.bin_spectrum(target_spectrum, **binning_kw)
            data_to_fit = binned_spectrum.reshape(1,-1)
            
            lasso_model = sklm.Lasso(**lasso_kw)
            lasso_model.fit(library_spectra.T, data_to_fit.T)
            
            models_list += [lasso_model]
            spectra_list += [binned_spectrum]

        return models_list, spectra_list

    @staticmethod
    def plot_coefs(
        models_list: list[Any],
        norm: Union[str, None] = None,
        **kwargs : Any,
    ) -> None:
        """
        Plot the coefficients in list of LASSO models.

        Parameters
        ----------
        models_list : list
            List of fitted LASSO models (see `LASSO.calculate_models`).
        
        norm : str, optional(default=None)
            The normalization method used to scale data to the [0, 1] range before mapping to colors.
        """
        coefs_list = [m.coef_ for m in models_list]

        coefs_array = np.stack(coefs_list)

        plt.imshow(coefs_array.T, norm=norm, **kwargs)
        plt.colorbar()