"""
Tools to analyze models.

Authors: Nathan A. Mahynski
"""
import copy
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from . import substance
from . import model # Only needed for type checking

from typing import Union, Any, ClassVar
from numpy.typing import NDArray

def plot_stacked_importances(
    optimized_models: list[Any],
    figsize: Union[tuple[int, int], None] = None,
    backend: str = 'mpl',
    **imshow_kwargs: Any,
):
    """
    Plot the importance values in list of models.

    Parameters
    ----------
    optimized_models : list
        List of fitted models (see `optimize_models`).
        
    figsize : tuple(int, int), optional(default=None)
        Figure size; this is currently only supported for the matplotlib backend.

    backend : str, optional(default='mpl')
        Plotting library to use; the default 'mpl' uses matplotlib and is not interactive, while 'plotly' will yield interactive plots.

    imshow_kwargs : dict, optional(default=None)
        Additional keyword arguments for {backend}.imshow() function; e.g., "cmap" or "color_continuous_scale".

    Returns
    -------
    if backend == 'mpl':
    
        image : matplotlib.image.AxesImage
            Feature importances as an image of a grid where each column corresponds to a different model and each row to a different feature (in the unrolled HSQC NMR spectrum).

        colorbar : matplotlib.pyplot.colorbar
            Colorbar to go with the image.

    if backend == 'plotly':
        
        image : plotly.graph_objs._figure.Figure
            Feature importances as an image of a grid where each column corresponds to a different model and each row to a different feature (in the unrolled HSQC NMR spectrum).
                
    Example
    -------
    >>> optimized_models, analyses = finchnmr.model.optimize_models(...)
    >>> plot_stacked_importances(optimized_models, backend='mpl', cmap='RdBu')
    >>> plot_stacked_importances(optimized_models, backend='plotly', color_continuous_scale='RdBu')
    """
    imp_list = [m.importances() for m in optimized_models]
    imps_array = np.stack(imp_list)
    
    if backend == 'mpl':
        _, ax = plt.subplots(figsize=figsize)

        image_mpl = ax.imshow(imps_array.T, **imshow_kwargs)
        ax.set_xlabel('Model Index')
        ax.set_ylabel('Library Substance Index')
        
        colorbar = plt.colorbar(image_mpl, ax=ax)
        colorbar.set_label("Importance")

        return image_mpl, colorbar
    elif backend == 'plotly':
        image_plt = px.imshow(
            imps_array.T, 
            text_auto=False,
            aspect="auto",
            **imshow_kwargs
        )
            
        image_plt.update_layout(xaxis_title="Model Index")
        image_plt.update_layout(yaxis_title="Library Substance Index")
        image_plt.update_layout(
            coloraxis_colorbar=dict(
                title="Importance" 
            )
        )

        # https://stackoverflow.com/questions/73649907/plotly-express-imshow-hover-text/73658192#73658192
        names = np.array([[s.name for s in m._nmr_library._substances] for m in optimized_models]).T
        z1, z2 = np.dstack([np.indices(imps_array.T.shape)])
        customdata = np.dstack((z1, z2, names))
        
        image_plt.update(data=[
            {
                'customdata': customdata,
                'hovertemplate': "Substance Index: %{customdata[0]}<br>Model Index: %{customdata[1]}<br>Substance: %{customdata[2]}"
            }
        ])
        
        return image_plt
    else:
        raise ValueError(f"Unrecognized backend {backend}")
        

class Analysis:
    """Set of analysis methods for analyzing fitted models."""

    _model: ClassVar["model._Model"]

    def __init__(self, model: "model._Model") -> None:
        """
        Instantiate the class.

        Parameters
        ----------
        model : _Model
            Fitted model to some target, see `optimize_models`.
        """
        setattr(self, "_model", model)

    def get_top_substances(
            self,
            k: int = 5
    ) -> tuple[list["substance.Substance"], list[float]]:
        """
        Retrieve the most important substances to the model.
        
        Parameters
        ----------
        k : int, optional(default=5)
            Number of most important spectra to retrieve.
        
        Returns
        -------
        top_substances : list(Substance)
            The most important substances, sorted from highest to lowest by the absolute value of their importance.

        top_importances : list(float)
            Importance of each substance, sorted from highest to lowest by the absolute value of their importance.
        """
        top_substances = []
        top_importances = []
        for i, (idx_, importances_) in enumerate(
            sorted(
                list(enumerate(self._model.importances())),
                key=lambda x: np.abs(x[1]),
                reverse=True,
            )[:k]
        ):
            top_substances.append(self._model._nmr_library.substance_by_index(idx_))
            top_importances.append(importances_)
        
        return top_substances, top_importances

    def plot_top_spectra(
        self,
        k: int = 5,
        plot_width: int = 3,
        figsize: Union[tuple[int, int], None] = (10, 5),
    ):
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
            Flattened array of axes on which the top HSQC NMR spectra are plotted.
        """
        if k == -1:
            k = len(self._model.importances())

        plot_depth = int(np.ceil(k / plot_width))

        fig, axes_ = plt.subplots(
            nrows=plot_depth, ncols=plot_width, figsize=figsize
        )
        axes = axes_.flatten()

        # Plot the NMR spectra
        top_substances, top_importances = self.get_top_substances(k=k)
        for s_, importance_ in zip(top_substances, top_importances):
            s_.plot(ax=axes[i])
            axes[i].set_title(
                s_.name + "\nI = {}".format("%.4f" % importance_)
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
    ):
        """
        Plot the residual (target - reconstructed) spectrum.

        An artificial substance is created representing the residual (see `build_residual`).  This is what is plotted, so it may be manipulated accordingly. Refer to the kwargs in `substance.Substance.plot`.

        Parameters
        ----------
        kwargs : dict, optional(default=None)
            Keyword arguments for `substance.Substance.plot`.

        Returns
        -------
        By default, or if kwargs['backend'] == 'mpl' in kwargs:
        
            image : matplotlib.image.AxesImage
                HSQC NMR resdual spectrum as an image.

            colorbar : matplotlib.colorbar.Colorbar
                Colorbar to go with the image.

        if kwargs['backend'] == 'plotly':
        
            image : plotly.graph_objs._figure.Figure
                HSQC NMR spectrum as an image.
                
        Example
        -------
        >>> a = Analysis(...)
        >>> a.plot_residual(absolute_values=True, backend='mpl')
        >>> a.plot_residual(absolute_values=True, backend='plotly', cmap='viridis')
        """
        residual = self.build_residual()

        result = residual.plot(**kwargs)
        if 'backend' in kwargs:
            if kwargs['backend'] == 'mpl':
                plt.gca().set_title("Target - Reconstructed")
            elif kwargs['backend'] == 'plotly':
                result.update_layout(title="Target - Reconstructed")
            else:
                raise ValueError(f"Unrecognized backend {backend}")
                
        return result