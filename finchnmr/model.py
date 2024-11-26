"""
Tools to build models.

Authors: Nathan A. Mahynski
"""
import copy
import matplotlib
import pickle
import tqdm

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
) -> tuple[list["_Model"], list["Analysis"]]:
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
        analyses += [Analysis(model=estimator)]

    return optimized_models, analyses


def plot_stacked_importances(
    optimized_models: list[Any],
    norm: Union[str, None] = None,
    figsize: Union[tuple[int, int], None] = None,
    **kwargs: Any,
) -> tuple["matplotlib.image.AxesImage", "matplotlib.pyplot.colorbar"]:
    """
    Plot the importance values in list of models.

    Parameters
    ----------
    optimized_models : list
        List of fitted models (see `optimize_models`).

    norm : str, optional(default=None)
        The normalization method used to scale data to the [0, 1] range before mapping to colors.

    kwargs : dict, optional(default=None)
        Additional keyword arguments for pyplot.imshow() function.

    Returns
    -------
    image : matplotlib.image.AxesImage
        Feature importances as an image of a grid where each column corresponds to a different model and each row to a different feature (in the unrolled HSQC NMR spectrum).

    colorbar : matplotlib.pyplot.colorbar
        Colorbar to go with the image.

    Example
    -------
    >>> optimized_models, analyses = finchnmr.model.optimize_models(...)
    >>> plot_stacked_importances(optimized_models)
    """
    imp_list = [m.importances() for m in optimized_models]
    imps_array = np.stack(imp_list)
    _, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(imps_array.T, norm=norm, **kwargs)
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label("Importance")

    return image, colorbar


class Analysis:
    """Set of analysis methods for analyzing fitted models."""

    _model: ClassVar["_Model"]

    def __init__(self, model: "_Model") -> None:
        """
        Instantiate the class.

        Parameters
        ----------
        model : _Model
            Fitted model to some target, see `optimize_models`.
        """
        setattr(self, "_model", model)

    def plot_top_spectra(
        self,
        k: int = 5,
        plot_width: int = 3,
        figsize: Union[tuple[int, int], None] = (10, 5),
    ) -> NDArray["matplotlib.pyplot.Axes"]:
        """
        Plot the HSQC NMR spectra that are the most importance to the model.

        Parameters
        ----------
        k : int, optional(default=5)
            Number of most important spectra to plot.  If -1 then plot them all.

        plot_width : int, optional(default=3)
            Number of subplots the grid will have along its width.

        figsize : tuple(int, int), optional(default=(10,5))
            Size of final figure.

        Returns
        -------
        axes : ndarray(matplotlib.pyplot.Axes, ndim=1)
            Flattened array of axes on which the spectra are plotted.
        """
        if k == -1:
            k = len(self._model.importances())

        plot_depth = int(np.ceil(k / plot_width))
        fig, axes_ = plt.subplots(
            nrows=plot_depth, ncols=plot_width, figsize=figsize
        )
        axes = axes_.flatten()

        # Plot the NMR spectra
        for i, (idx_, importances_) in enumerate(
            sorted(
                list(enumerate(self._model.importances())),
                key=lambda x: np.abs(x[1]),
                reverse=True,
            )[:k]
        ):
            s_ = self._model._nmr_library.substance_by_index(idx_)
            s_.plot(ax=axes[i])
            axes[i].set_title(
                s_.name + "\nI = {}".format("%.4f" % importances_)
            )

        # Trim of extra subplots
        for i in range(k, plot_depth * plot_width):
            axes[i].remove()

        return axes

    def plot_top_importances(
        self,
        k: int = 5,
        by_name: bool = False,
        figsize: Union[tuple[int, int], None] = None,
    ) -> "matplotlib.pyplot.Axes":
        """
        Plot the importances of the top substances in the model.

        Parameters
        ----------
        k : int, optional(default=5)
            Number of top importances to plot.  If -1 then plot them all.

        by_name : bool, optional(default=False)
            HSQC NMR spectra will given by integer index in the library by default; if True, the use the associated substance name instead.

        figsize : tuple(int, int), optional(default=None))
            Size of final figure.

        Returns
        -------
        axes : matplotlib.pyplot.Axes
            Horizontal bar chart the importances are plotted on in descending order.
        """
        if k == -1:
            k = len(self._model.importances())

        sorted_importances = sorted(
            list(enumerate(self._model.importances())),
            key=lambda x: np.abs(x[1]),
            reverse=True,
        )[:k]

        if by_name:
            labels = [
                self._model._nmr_library.substance_by_index(x[0]).name
                for x in sorted_importances
            ]
        else:
            labels = [str(x[0]) for x in sorted_importances]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        axes.barh(
            y=np.arange(k)[::-1],
            width=[h[1] for h in sorted_importances],
            align="center",
            tick_label=labels,
        )
        axes.set_xlabel("Importance")

        return axes

    def build_residual(self) -> "substance.Substance":
        """
        Create a substance whose spectrum is comprised of the residual (true spectrum - model).

        Returns
        -------
        residual : substance.Substance
            Artificial substance whose spectrum is the residual.
        """
        target = self._model.target()
        reconstructed = self._model.reconstruct()

        residual = copy.deepcopy(
            self._model.target()
        )  # Create a new copy of the target as a baseline
        residual._set_data(target.data - reconstructed.data)

        return residual

    def plot_residual(
        self, **kwargs
    ) -> tuple["matplotlib.image.AxesImage", "matplotlib.pyplot.colorbar"]:
        """
        Plot the residual (target - reconstructed) spectrum.

        An artificial substance is created representing the residual (see `build_residual`).  This is what is plotted, so it may be manipulated accordingly.

        Parameters
        ----------
        kwargs : dict, optional(default=None)
            Keyword arguments for `substance.Substance.plot`.

        Returns
        -------
        image : matplotlib.image.AxesImage
            HSQC NMR resdual spectrum as an image.

        colorbar : matplotlib.pyplot.colorbar
            Colorbar to go with the image.

        Example
        -------
        >>> a = Analysis(...)
        >>> a.plot_residual(absolute_values=True)
        """
        residual = self.build_residual()

        image, colorbar = residual.plot(**kwargs)
        plt.gca().set_title("Target - Reconstructed")

        return image, colorbar


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
