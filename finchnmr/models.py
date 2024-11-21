"""
Tools to build linear models.

Authors: Nathan A. Mahynski
"""
import pickle
import utils

import sklearn.linear_model as sklm
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from itertools import product

from typing import Union, Any, Sequence
from numpy.typing import NDArray

def optimize_models(
    target_spectra: list[NDArray[np.floating]], 
    library_spectra: NDArray[np.floating], 
    model: "_Model",
    param_grid: dict(str, list),
    model_kw: Union[dict[str, Any], None] = None, 
    binning_kw: Union[dict, None] = None
) -> tuple[list, list]:
    """
    Optimize a model to fit each wild spectra in a list.

    Parameters
    ----------
    target_spectra : list(ndarray(float, ndim=1))
        List of flattened HSQC NMR spectra to fit.

    library_spectra : ndarray(float, ndim=2)
        Flattened HSQC NMR spectra of a library of known compounds to fit against.  Coefficients will follow the order of spectra provided here.

    model : _Model
        Uninstantiated model class to fit the spectra with.

    param_grid : dict(str, list)
        Dictionary of parameter grid to search over; this follows the same convention as `sklearn.model_selection.GridSearchCV`.

    model_kw : dict(str, Any), optional(default=None)
        Default keyword arguments to your model.

    binning_kw : dict, optional(default=None)
        Keyword arguments to `utils.bin_spectrum`.

    Returns
    -------
    models_list : list(_Model)
        List of optimized models fit to each target HSQC NMR spectrum.

    spectra_list : list(ndarray(float, ndim=1))
        List of HSQC NMR spectra fit during training (coarsened versions of `target_spectra`).

    Example
    -------
    >>> models_list, spectra_list = finchnmr.models.optimize_models(
    ...     target_spectra=target_spectra,
    ...     library_spectra=library_spectra,
    ...     model=finchnmr.models.LASSO,
    ...     param_grid={'alpha':np.logspace(-5, 1, 100)},
    ...     model_kw={'fit_intercept':False, 'positive':True},
    ...     binning_kw=None
    ... )
    """
    models_list = []
    spectra_list = []
        
    if binning_kw is None:
        binning_kw = {}

    def build_fitted_model_(model_kw, param_set, library_spectra, data_to_fit):
        if model_kw is None:
            estimator = model() # Use model default parameters
        else:
            estimator = model(**model_kw) # Set basic parameters manually

        estimator.set_params(param_set) # Set specific parameters (alpha, etc.)
        _ = estimator.fit(library_spectra.T, data_to_fit.T)

        return estimator

    def unroll_(param_grid):
        param_sets = []
        for values in product(*param_grid.values()):
            combination = dict(zip(param_grid.keys(), values))
            param_sets.append(combination)

        return param_sets

    for target_spectrum in target_spectra:
        binned_spectrum = utils.bin_spectrum(target_spectrum, **binning_kw)
        data_to_fit = binned_spectrum.reshape(1,-1)

        param_sets = unroll_(param_grid)
        scores = []
        for param_set_ in param_sets:
            estimator_ = build_fitted_model_(model_kw, param_set_, library_spectra, data_to_fit)
            scores.append(estimator_.score(library_spectra.T, data_to_fit.T))

        # Fit final estimator with the "best" parameters
        estimator = build_fitted_model_(model_kw, param_sets[np.argmax(scores)], library_spectra, data_to_fit)

        models_list += [estimator]
        spectra_list += [binned_spectrum]

    return models_list, spectra_list

def plot_coeffs(
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
    coefs_list = [m.coeff() for m in models_list]
    coefs_array = np.stack(coefs_list)
    plt.imshow(coefs_array.T, norm=norm, **kwargs)
    plt.colorbar()

class _Model(RegressorMixin, BaseEstimator):
    """
    Model base class wrapper for linear models.
    
    The main reason this is needed is to define a consistent interface to a method to access the model coefficients after fitting.
    """
    def __init__(self):
        """
        Instantiate the model.

        Note that the sklearn API requires all estimators (subclasses of this) to specify all the parameters that can be set at the class level in their __init__ as explicit keyword arguments (no *args or **kwargs).
        """
        self.model_ = None
        
    def fit(self, X, y):
        """
        Fit the model.
        
        Parameters
        ----------
        X : ndarray(float, ndim=1)
            Flattened library HSQC NMR spectra.

        y : ndarray(float, ndim=1)
            Flattened target HSQC NMR spectra to fit.

        Returns
        -------
        self : _Model
            Fitted model.
        """
        if self.model_ is None:
            raise Exception("model has not been set yet.")
        else:
            self.__model = self.model_(**self.get_params())

        _ = self.__model.fit(X, y)
        self.is_fitted_ = True

        return self
        
    def predict(self, X):
        """
        Predict the (flattened) target HSQC spectra.
        
        Parameters
        ----------
        X : ndarray(float, ndim=1)
            Flattened library HSQC NMR spectra.

        Returns
        -------
        spectrum : ndarray(float, ndim=1)
            Predicted spectrum fit to the given library.
        """
        return self.__model.predict(X)
        
    def model(self):
        """Return the fitted model."""
        return self.__model

    def coeff(self):
        """Return the coefficients in the model."""
        raise NotImplementedError
        
class LASSO(_Model):
    def __init__(
        self, 
        alpha=1.0, 
        *, 
        fit_intercept=True, 
        precompute=False, 
        copy_X=True, 
        max_iter=1000, 
        tol=0.0001, 
        warm_start=False, 
        positive=False, 
        random_state=None, 
        selection='cyclic'
    ):
        """
        Instantiate the class.

        Inputs are identical to sklearn.linear_model.Lasso.
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        """
        self.set_params(**{
            'alpha': alpha, 
            'fit_intercept': fit_intercept, 
            'precompute': precompute, 
            'copy_X': copy_X, 
            'max_iter': max_iter, 
            'tol': tol, 
            'warm_start': warm_start, 
            'positive': positive, 
            'random_state': random_state, 
            'selection': selection
        })
        self.model_ = sklm.Lasso
        
    def coeff(self):
        """Return the LASSO model coefficients."""
        return self.model().coef_