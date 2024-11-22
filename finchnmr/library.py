"""
Functions for defining a library of substances measured with HSQC NMR.

Authors: Nathan A. Mahynski, David A. Sheen
"""
import pickle

import numpy as np

from numpy.typing import NDArray
from typing import ClassVar

class Library:
    substances: ClassVar[list["Substance"]]
    is_fitted_: ClassVar[bool]
    _X: ClassVar[NDArray[np.floating]]
        
    def __init__(self, substances: list["Substance"]) -> None:
        """
        Instantiate the library.
        
        Parameters
        ----------
        substances : list(Substance)
            List of substances in the library.
        """
        self.substances = substances
        self.is_fitted_ = False
        
    def fit(self, reference: "Substance") -> None:
        """
        Align all substances to another one which serves as a reference.
        
        Parameters
        ----------
        reference : Substance
            Substance to align all substances in the library with (match extent, etc.).
        """
        aligned = []
        for sub in self.sustances:
            aligned.append(sub.fit(reference).flatten())
        self._X = np.array(aligned, dtype=np.float64)
        self.is_fitted_ = True
        
    @property
    def X(self) -> NDArray[np.floating]:
        """
        Return a copy of the data in the library.
        
        This data is arranged in a 2D array, where each row is the flattened HSQC NMR spectrum of a different substance. The ordering follows that with which the library was instantiated.
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