"""
Functions for defining a library of substances measured with HSQC NMR.

Authors: Nathan A. Mahynski, David A. Sheen
"""
import copy
import pickle

from . import substance

import numpy as np

from numpy.typing import NDArray
from typing import ClassVar


class Library:
    """Library of substances for fitting new unknowns."""

    is_fitted_: ClassVar[bool]
    _substances: ClassVar[list["substance.Substance"]]
    _fit_to: "substance.Substance"
    _X: ClassVar[NDArray[np.floating]]

    def __init__(self, substances: list["substance.Substance"]) -> None:
        """
        Instantiate the library.

        Parameters
        ----------
        substances : list(Substance)
            List of substances in the library.

        Example
        -------
        >>> substances = []
        >>> head = '../../../spectra_directory/'
        >>> for sample_ in os.listdir(head):
        ...     pathname_ = os.path.join(
        ...         os.path.abspath(
        ...             os.path.join(
        ...                 head,
        ...                 sample_
        ...             )
        ...         ), 'pdata/1'
        ...     )
        >>>     substances.append(finchnmr.substance.Substance(pathname_))
        >>> L = finchnmr.library.Library(substances=substances)
        """
        setattr(self, "_substances", substances)
        setattr(self, "is_fitted_", False)

    def fit(self, reference: "substance.Substance") -> "Library":
        """
        Align all substances to another one which serves as a reference.

        Parameters
        ----------
        reference : Substance
            Substance to align all substances in the library with (match extent, etc.).

        Returns
        -------
        self
        """
        aligned = []
        for sub in self._substances:
            aligned.append(sub.fit(reference).flatten())
        setattr(self, "_X", np.array(aligned, dtype=np.float64).T)
        setattr(self, "_fit_to", reference)
        setattr(self, "is_fitted_", True)

        return self

    @property
    def X(self) -> NDArray[np.floating]:
        """
        Return a copy of the data in the library.

        Returns
        -------
        X : ndarray(float, ndim=2)
            This data is arranged in a 2D array, where each column is the flattened HSQC NMR spectrum of a different substance (row). The ordering follows that with which the library was instantiated.

        Example
        -------
        >>> L = finchnmr.library.Library(substances=substances)
        >>> L.fit(substance=new_compound)
        >>> L.X
        """
        if self.is_fitted_:
            return self._X.copy()
        else:
            raise Exception(
                "Library has not been fit to a reference substance yet."
            )

    def substance_by_index(self, idx: int) -> "substance.Substance":
        """
        Retrieve a substance from the library by index.

        Parameters
        ----------
        idx : int
            Index of the substance in the library.

        Returns
        -------
        substance : Substance
            Desired substance.
        """
        return copy.copy(self._substances[idx])

    def substance_by_name(self, name: str) -> "substance.Substance":
        """
        Retrieve a substance from the library by name.

        Parameters
        ----------
        name : str
            Name of the substance in the library.

        Returns
        -------
        substance : Substance
            Desired substance.
        """
        for s in self._substances:
            if s.name == name:
                return copy.copy(s)
        raise ValueError(f"No substance with name {name} in library.")

    def save(self, filename: str) -> None:
        """
        Pickle library to a file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=4)
