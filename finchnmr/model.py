"""
Tools to build models.

Authors: Nathan A. Mahynski
"""
import copy
import tqdm

import numpy as np
import sklearn.linear_model as sklm

from . import library
from . import substance
from . import analysis

from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
from typing import Union, Any, ClassVar
from numpy.typing import NDArray


def optimize_models(
    targets: list["substance.Substance"],
    nmr_library: "library.Library",
    nmr_model: "_Model",
    param_grid: dict[str, list],
    model_kw: Union[dict[str, Any], None] = None,
) -> tuple[list["_Model"], list["analysis.Analysis"]]:
    """
    Optimize a model to fit each wild spectra in a list.

    All combinations of parameters in `param_grid` are tested and the best performer is retained.

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

    analyses : list(Analysis)
        List of analysis objects to help visualize and understand each fitted model.

    Example
    -------
    >>> target = finchnmr.substance.Substance(...) # Load target(s)
    >>> nmr_library = finchnmr.library.Library(...) # Create library
    >>> optimized_models, analyses = finchnmr.model.optimize_models(
    ...     targets=[target],
    ...     nmr_library=nmr_library,
    ...     nmr_model=finchnmr.model.LASSO,
    ...     param_grid={'alpha': np.logspace(-5, 1, 100)},
    ... )
    >>> analyses[0].plot_top_spectra(k=5)
    """
    optimized_models = []
    analyses = []

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
        print(i)
        scores = []
        for param_set in tqdm.tqdm(
            param_sets, desc="Iterating through parameter sets"
        ):
            print('param set : ', param_set)
            try:
                estimator_ = build_fitted_model_(
                    model_kw, param_set, nmr_library, target
                )
            except Exception as e:
                print(e)
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
        analyses += [analysis.Analysis(model=estimator)]

    return optimized_models, analyses


class _Model(RegressorMixin, BaseEstimator):
    """Model base class wrapper for linear models."""

    model: ClassVar[Any]
    model_: ClassVar[Any]
    _nmr_library: ClassVar["library.Library"]
    _score: ClassVar[float]
    _scale_y: ClassVar[NDArray[np.floating]]
    is_fitted_: ClassVar[bool]

    def __init__(self) -> None:
        """
        Instantiate the model.

        Note that the sklearn API requires all estimators (subclasses of this) to specify all the parameters that can be set at the class level in their __init__ as explicit keyword arguments (no *args or **kwargs).
        """
        setattr(self, "is_fitted_", False)
        setattr(self, "model_", None)

    def set_params(self, **parameters: Any) -> "_Model":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def target(self):
        """Return the target this model is meant to reproduce."""
        if self.is_fitted_:
            return copy.deepcopy(self._nmr_library._fit_to)
        else:
            raise Exception("Model has not been fit yet.")

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
            setattr(self, "model", self.model_(**self.get_model_params()))
            setattr(self, "_nmr_library", copy.deepcopy(nmr_library))

        # Align library with target - this also saves target internally
        self._nmr_library.fit(target)

        # Transform library to normalize
        X, _ = self.transform(self._nmr_library.X)

        # Tansform target in a similar way
        y, scale_y = self.transform(target.flatten().reshape(-1, 1))
        setattr(self, "_scale_y", scale_y)
        #         y = target.flatten().reshape(-1, 1)
        #         setattr(self, "_scale_y", 1.0)

        # Fit the model
        _ = self.model.fit(X, y)

        # Store the score of this fit
        setattr(self, "_score", self.model.score(X, y))

        setattr(self, "is_fitted_", True)
        return self

    @staticmethod
    def transform(X):
        X_t = np.abs(
            X
        )  # Convert library intensities to absolute values, [0, inf)
        scale = np.max(X_t, axis=0)  # Scale library to [0, 1]
        return X_t / scale, scale

    def predict(self) -> NDArray[np.floating]:
        """
        Predict the (flattened) target HSQC spectra.

        Returns
        -------
        spectrum : ndarray(float, ndim=1)
            Predicted (flattened) spectrum fit to the given `nmr_library`.
        """
        if not self.is_fitted_:
            raise Exception("Model has not been fit yet.")

        y_pred = self.model.predict(self.transform(self._nmr_library.X)[0])

        # When predicting / reconstructing, scale the model output back to
        # "real" units.  This is still absolute value / intensity space but
        # has the proper magnitude for comparison with the target spectrum.
        return y_pred * self._scale_y

    def score(self) -> float:
        """
        Score the model's performance (fit).

        Returns
        -------
        score : float
            Coefficient of determination of the model that uses `nmr_library` to predict `target`.
        """
        if not self.is_fitted_:
            raise Exception("Model has not been fit yet.")
        return self._score

    def reconstruct(self) -> "substance.Substance":
        """Reconstruct a 2D HSQC NMR spectrum using the fitted model."""
        if not self.is_fitted_:
            raise Exception("Model has not been fit yet.")
        reconstructed = self.target()
        reconstructed._set_data(reconstructed.unflatten(self.predict()))

        return reconstructed

    def importances(self) -> NDArray[np.floating]:
        """Return the importances of each feature in the model."""
        raise NotImplementedError

    def get_model_params(self) -> dict[str, Any]:
        """Get the parameters needed to instantiate the model."""
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

    fit_intercept: ClassVar[bool]
    positive: ClassVar[bool]

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
        super().__init__()

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

    def get_model_params(self) -> dict[str, Any]:
        """Return the parameters for an sklearn.linear_model.Lasso model."""
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "precompute": self.precompute,
            "copy_X": self.copy_X,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "warm_start": self.warm_start,
            "positive": self.positive,
            "random_state": self.random_state,
            "selection": self.selection,
        }

    def importances(self) -> NDArray[np.floating]:
        """Return the Lasso model coefficients as importances."""
        return self.model.coef_
