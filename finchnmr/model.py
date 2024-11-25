"""
Tools to build models.

Authors: Nathan A. Mahynski
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as sklm

from . import library
from . import substance
from . import utils

from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
from typing import Union, Any, Sequence, ClassVar
from numpy.typing import NDArray

def optimize_models(
    targets: list[substance.Substance], 
    library: library.Library, 
    model: "_Model",
    param_grid: dict[str, list],
    model_kw: Union[dict[str, Any], None] = None, 
) -> list:
    """
    Optimize a model to fit each wild spectra in a list.

    Parameters
    ----------
    targets : list[Substance]
        Unknown/wild HSQC NMR spectrum to fit with the library.
        
    library : Library
        Library of HSQC NMR spectra to use for fitting `targets`.
        
    model : _Model
        Uninstantiated model class to fit the spectra with.
        
    param_grid : dict(str, list)
        Dictionary of parameter grid to search over; this follows the same convention as `sklearn.model_selection.GridSearchCV`.

    model_kw : dict(str, Any), optional(default=None)
        Default keyword arguments to your model. If `None` then the `model` defaults are used.

    Returns
    -------
    models_list : list(_Model)
        List of optimized models fit to each target HSQC NMR spectrum.

    Example
    -------
    >>> models_list, spectra_list = finchnmr.models.optimize_models(
    ...     targets=[target],
    ...     library=library,
    ...     model=finchnmr.models.LASSO,
    ...     param_grid={'alpha': np.logspace(-5, 1, 100)},
    ... )
    """
    optimized_models = []

    def build_fitted_model_(model_kw, param_set, library, target):
        """Create and train the model."""
        if model_kw is None:
            estimator = model() # Use model default parameters
        else:
            estimator = model(**model_kw) # Set basic parameters manually

        estimator.set_params(param_set) # Set specific parameters (alpha, etc.)
        _ = estimator.fit(library.X, target.flatten())

        return estimator

    def unroll_(param_grid):
        """Create every possible combination of parameters in the grid."""
        param_sets = []
        for values in product(*param_grid.values()):
            combination = dict(zip(param_grid.keys(), values))
            param_sets.append(combination)

        return param_sets

    param_sets = unroll_(param_grid)
    for target in targets:
        library.fit(target) # Align library with target

        scores = []
        for param_set in param_sets:
            estimator_ = build_fitted_model_(model_kw, param_set, library, target)
            scores.append(estimator_.score(library.X, target.flatten()))

        # Fit final estimator with the "best" parameters
        estimator = build_fitted_model_(model_kw, param_sets[np.argmax(scores)], library, target)

        optimized_models += [estimator]

    return optimized_models

def plot_coeffs(
    optimized_models: list[Any],
    norm: Union[str, None] = None,
    **kwargs : Any,
) -> None:
    """
    Plot the coefficients in list of models.

    Parameters
    ----------
    optimized_models : list
        List of fitted models (see `optimize_models`).
        
    norm : str, optional(default=None)
        The normalization method used to scale data to the [0, 1] range before mapping to colors.
    """
    coefs_list = [m.coeff() for m in optimized_models]
    coefs_array = np.stack(coefs_list)
    plt.imshow(coefs_array.T, norm=norm, **kwargs)
    plt.colorbar()

class _Model(RegressorMixin, BaseEstimator):
    """
    Model base class wrapper for linear models.
    
    The main reason this is needed is to define a consistent interface to a method to access the model coefficients after fitting.
    """
    model: ClassVar[Any]
    model_: ClassVar[Any]
        
    def __init__(self) -> None:
        """
        Instantiate the model.

        Note that the sklearn API requires all estimators (subclasses of this) to specify all the parameters that can be set at the class level in their __init__ as explicit keyword arguments (no *args or **kwargs).
        """
        setattr(self, "model_", None)
        
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
            setattr(self, "model", self.model_(**self.get_params()))

        _ = self.model.fit(X, y)
        setattr(self, "is_fitted_ ", True)

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
        setattr(self, "model_", sklm.Lasso)
        
    def coeff(self) -> NDArray[np.floating]:
        """Return the LASSO model coefficients."""
        return self.model.coef_