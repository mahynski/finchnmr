"""
Tools to build models.

Authors: Nathan A. Mahynski
"""
import pickle
import tqdm
import copy

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as sklm

from . import library
from . import substance

from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
from typing import Union, Any, Sequence, ClassVar
from numpy.typing import NDArray


def optimize_models(
    targets: list["substance.Substance"],
    nmr_library: "library.Library",
    nmr_model: "_Model",
    param_grid: dict[str, list],
    model_kw: Union[dict[str, Any], None] = None,
) -> list:
    """
    Optimize a model to fit each wild spectra in a list.

    Parameters
    ----------
    targets : list[Substance]
        Unknown/wild HSQC NMR spectrum to fit with the `nmr_library`.

    nmr_library : Library
        Library of HSQC NMR spectra to use for fitting `targets`.

    nmr_model : _Model
        Uninstantiated model class to fit the spectra with.

    param_grid : dict(str, list)
        Dictionary of parameter grid to search over; this follows the same convention as `sklearn.model_selection.GridSearchCV`.

    model_kw : dict(str, Any), optional(default=None)
        Default keyword arguments to your model. If `None` then the `nmr_model` defaults are used.

    Returns
    -------
    optimized_models : list(_Model)
        List of optimized models fit to each target HSQC NMR spectrum.

    Example
    -------
    >>> optimized_models = finchnmr.model.optimize_models(
    ...     targets=[target],
    ...     nmr_library=nmr_library,
    ...     nmr_model=finchnmr.model.LASSO,
    ...     param_grid={'alpha': np.logspace(-5, 1, 100)},
    ... )
    """
    optimized_models = []

    def build_fitted_model_(model_kw, param_set, nmr_library, target):
        """Create and train the model."""
        if model_kw is None:
            estimator = nmr_model()  # Use model default parameters
        else:
            estimator = nmr_model(**model_kw)  # Set basic parameters manually

        estimator.set_params(
            **param_set
        )  # Set specific parameters (alpha, etc.)
        _ = estimator.fit(nmr_library, target)

        return estimator

    def unroll_(param_grid):
        """Create every possible combination of parameters in the grid."""
        param_sets = []
        for values in product(*param_grid.values()):
            combination = dict(zip(param_grid.keys(), values))
            param_sets.append(combination)

        return param_sets

    param_sets = unroll_(param_grid)
    for i, target in tqdm.tqdm(
        enumerate(targets), desc="Iterating through targets"
    ):

        scores = []
        for param_set in tqdm.tqdm(
            param_sets, desc="Iterating through parameter sets"
        ):
            try:
                estimator_ = build_fitted_model_(
                    model_kw, param_set, nmr_library, target
                )
            except:
                pass  # Do not score this model
            else:
                scores.append(estimator_.score())

        if len(scores) == 0:
            raise Exception(f"Unable to fit any models for target index {i}")

        # Fit final estimator with the "best" parameters
        estimator = build_fitted_model_(
            model_kw, param_sets[np.argmax(scores)], nmr_library, target
        )

        optimized_models += [estimator]

    return optimized_models


def plot_coeffs(
    optimized_models: list[Any],
    norm: Union[str, None] = None,
    **kwargs: Any,
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
    """Model base class wrapper for linear models."""

    model: ClassVar[Any]
    model_: ClassVar[Any]
    _nmr_library: ClassVar["library.Library"]

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

    def fit(
        self, nmr_library: "library.Library", target: "substance.Substance"
    ) -> "_Model":
        """
        Fit the model.

        The library is "fit"/aligned to the target first, then the linear model is fit.

        Parameters
        ----------
        nmr_library : Library
            Library of HSQC NMR spectra to use for fitting `unknown`.

        target : substance.Substance
            Unknown/wild HSQC NMR spectrum to fit with the `nmr_library`.

        Returns
        -------
        self : _Model
            Fitted model.
        """
        if self.model_ is None:
            raise Exception("model has not been set yet.")
        else:
            setattr(self, "model", self.model_(**self.get_params()))
            setattr(self, "_nmr_library", copy.deepcopy(nmr_library))
            
        # Align library with target
        self._nmr_library.fit(target)
        
        # Target needs to take absolute value and optionally be scaled in [0, 1]; same as when library is aligned.
        
        # ...
        
        
        # When predicting / reconstructing, scale the model output back to "real" units
        
        

        _ = self.model.fit(
            self._nmr_library.X, self._nmr_library._fit_to.flatten()
        )
        setattr(self, "is_fitted_ ", True)

        return self

    def predict(self) -> NDArray[np.floating]:
        """
        Predict the (flattened) target HSQC spectra.

        Returns
        -------
        spectrum : ndarray(float, ndim=1)
            Predicted (flattened) spectrum fit to the given `nmr_library`.
        """
        return self.model.predict(self._nmr_library.X)

    def score(self) -> float:
        """
        Score the model's performance (fit).

        Returns
        -------
        score : float
            Coefficient of determination of the model that uses `nmr_library` to predict `target`.
        """
        return self.model.score(
            self._nmr_library.X, self._nmr_library._fit_to.flatten()
        )

    def reconstruct(self) -> "substance.Substance":
        """Reconstruct a 2D HSQC NMR spectrum using the fitted model."""
        reconstructed = copy.deepcopy(self._nmr_library._fit_to)
        reconstructed._set_data(reconstructed.unflatten(self.predict()))

        return reconstructed

    def coeff(self) -> NDArray[np.floating]:
        """Return the coefficients in the model."""
        raise NotImplementedError


class LASSO(_Model):
    """LASSO model from sklearn."""

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
        selection: str = "cyclic",
    ) -> None:
        """
        Instantiate the class.

        Inputs are identical to `sklearn.linear_model.Lasso` except for `fit_intercept` and `positive` which are forced to be `False` and `True`, respectively. Also, `max_iter` is increased from 1,000 to 10,000 by default.
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        """
        self.set_params(
            **{
                "alpha": alpha,
                "fit_intercept": False,  # Always assume no offset
                "precompute": precompute,
                "copy_X": copy_X,
                "max_iter": max_iter,
                "tol": tol,
                "warm_start": warm_start,
                "positive": True,  # Force coefficients to be positive
                "random_state": random_state,
                "selection": selection,
            }
        )
        setattr(self, "model_", sklm.Lasso)

    def coeff(self) -> NDArray[np.floating]:
        """Return the LASSO model coefficients."""
        return self.model.coef_
