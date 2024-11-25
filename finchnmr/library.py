"""
Functions for defining a library of substances measured with HSQC NMR.

Authors: Nathan A. Mahynski, David A. Sheen
"""
import pickle

from . import substance

import numpy as np

from numpy.typing import NDArray
from typing import ClassVar

class Library:
    substances: ClassVar[list[substance.Substance]]
    is_fitted_: ClassVar[bool]
    _fit_to: substance.Substance
    _X: ClassVar[NDArray[np.floating]]
        
    def __init__(self, substances: list[substance.Substance]) -> None:
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
        setattr(self, "substances", substances)
        setattr(self, "is_fitted_", False)
        
    def fit(self, reference: substance.Substance) -> Library:
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
        for sub in self.substances:
            aligned.append(sub.fit(reference).flatten())
        setattr(self, "_X", np.array(aligned, dtype=np.float64))
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
            This data is arranged in a 2D array, where each row is the flattened HSQC NMR spectrum of a different substance. The ordering follows that with which the library was instantiated.
        
        Example
        -------
        >>> L = finchnmr.library.Library(substances=substances)
        >>> L.fit(substance=new_compound)
        >>> L.X
        """
        if self.is_fitted_:
            return self._X.copy()
        else:
            raise Exception("Library has not been fit to a reference substance yet.")
            
    def save(self, filename: str) -> None:
        """
        Pickle library to a file.
        
        Parameters
        ----------
        filename : str
            Filename to write to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=4)