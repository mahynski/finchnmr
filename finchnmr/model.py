"""
Tools to build models.

Authors: Nathan A. Mahynski
"""
import pickle

import sklearn.linear_model as sklm
import numpy as np

from . import utils
from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
from typing import Union, Any, Sequence, ClassVar
from numpy.typing import NDArray

def optimize_models(
    target_spectra: list[NDArray[np.floating]], 
    library_spectra: NDArray[np.floating], 
    model: "_Model",
    param_grid: dict[str, list],
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
    ...     param_grid={'alpha': np.logspace(-5, 1, 100)},
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
    Plot the coefficients in list of models.

    Parameters
    ----------
    models_list : list
        List of fitted models (see `optimize_models`).
        
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
    model: ClassVar[Any]
        
    def __init__(self) -> None:
        """
        Instantiate the model.

        Note that the sklearn API requires all estimators (subclasses of this) to specify all the parameters that can be set at the class level in their __init__ as explicit keyword arguments (no *args or **kwargs).
        """
        self.model_ = None
        
    def set_params(self, **parameters: Any) -> "_Model":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_Model":
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
            self.model = self.model_(**self.get_params())

        _ = self.model.fit(X, y)
        self.is_fitted_ = True

        return self
        
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
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
        return self.model.predict(X)

    def coeff(self) -> NDArray[np.floating]:
        """Return the coefficients in the model."""
        raise NotImplementedError
        
class LASSO(_Model):
    alpha: ClassVar[float]
    precompute: ClassVar[bool]
    copy_X: ClassVar[bool]
    max_iter: ClassVar[int]
    tol: ClassVar[float]
    warm_start: ClassVar[bool]
    random_state: ClassVar[Union[int, None]]
    selection: ClassVar[str]
        
    def __init__(
        self, 
        alpha: float = 1.0, 
        precompute: bool = False, 
        copy_X: bool = True, 
        max_iter: int = 10000, 
        tol: float = 0.0001, 
        warm_start: bool = False, 
        random_state: Union[int, None] = None, 
        selection: str = 'cyclic'
    ) -> None:
        """
        Instantiate the class.

        Inputs are identical to `sklearn.linear_model.Lasso` except for `fit_intercept` and `positive` which are forced to be `False` and `True`, respectively. Also, `max_iter` is increased from 1,000 to 10,000 by default.
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        """
        self.set_params(**{
            'alpha': alpha, 
            'fit_intercept': False, # Always assume no offset
            'precompute': precompute, 
            'copy_X': copy_X, 
            'max_iter': max_iter, 
            'tol': tol, 
            'warm_start': warm_start, 
            'positive': True, # Force coefficients to be positive
            'random_state': random_state, 
            'selection': selection
        })
        self.model_ = sklm.Lasso
        
    def coeff(self) -> NDArray[np.floating]:
        """Return the LASSO model coefficients."""
        return self.model.coef_