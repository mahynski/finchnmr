"""
Tools to build linear models.

Authors: Nathan A. Mahynski
"""
import pickle
import utils

import sklearn.linear_model as sklm
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from typing import Union, Any, Sequence
from numpy.typing import NDArray

class _Model(RegressorMixin, BaseEstimator);
    """
    Model base class wrapper for linear models.
    
    The main reason this is needed is to define a consistent interface to a method to access the model coefficients after fitting.
    """
    def __init__(self):
        """
        Note that the sklearn API requires all estimators (subclasses of this) to specify all the parameters that can be set at the class level in their __init__ as explicit keyword arguments (no *args or **kwargs).
        """
        self.model_ = None
        
    def fit(self, X, y):
        if self.model_ is None:
            raise Exception("model has not been set yet.")
        else:
            self.__model = self.model_(**self.get_params())
        _ = self.__model.fit(X, y)
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        return self.__model.predict(X)
        
    def model(self):
        return self.__model

    def coeff(self):
        raise NotImplementedError
        
class LASSO(_Model):
    def __init__(self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        self.set_params()
        self.model_ = sklm.Lasso
        
    def coeff(self):
        return self.model().coef_
    
class Utils: # not needed anymore
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