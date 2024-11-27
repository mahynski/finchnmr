.. finchnmr documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FINCHnmr documentation
========================

.. image:: https://github.com/mahynski/finchnmr/actions/workflows/python-app.yml/badge.svg?branch=main
   :target: https://github.com/mahynski/finchnmr/actions
.. image:: https://readthedocs.org/projects/finchnmr/badge/?version=latest
   :target: https://finchnmr.readthedocs.io/en/latest/?badge=latest
.. image:: https://codecov.io/github/mahynski/finchnmr/graph/badge.svg?token=DsrQIbklpB
   :target: https://codecov.io/gh/mahynski/finchnmr
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. image:: https://zenodo.org/badge/331207062.svg
   :target: https://zenodo.org/badge/latestdoi/331207062

----

FINCHnmr is lightweight toolkit for fitting 2D `heteronuclear single-quantum coherence (HSQC) nuclear magnetic resonance (NMR)<https://en.wikipedia.org/wiki/Heteronuclear_single_quantum_coherence_spectroscopy>`_ data to a known library of substances.  This predicts the presence and relative concentration of these compounds, and the residual (error) can be interpreted as the sum of the remaining unknown compounds present.  For a live demonstration, visit `https://finchnmr-demo.streamlit.app/ <https://finchnmr-demo.streamlit.app/>`_. Although originally designed to work with (1H-13C) data, FINCHnmr will work with any 2D NMR as long as the library used matches the sample being predicted / analyzed.

There are two approaches to generating the spectral libraries.  Both methods are demonstrated in the documentation.

1. Library spectra are taken directly from the `Biological Magnetic Resonance Bank (BMRB) <https://bmrb.io/>`_. These spectra are automatically padded and resized so that they match the extent (2D grid) and resolution of the wild spectra being fit.
2. Library spectra are reconstructed from a feature list of peak locations by placing bivariate Gaussians at these locations; assumptions must be made about the spread of these distributions in both dimentions of frequency shift space.

License Information
###################
* See `LICENSE.md <https://github.com/mahynski/finchnmr/blob/main/LICENSE.md>`_ for more information.

* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by `NIST <https://www.nist.gov/>`_.
  
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   install
   examples

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
