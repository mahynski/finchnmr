![Workflow](https://github.com/mahynski/finchnmr/actions/workflows/python-app.yml/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/finchnmr/badge/?version=latest)](https://finchnmr.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/mahynski/finchnmr/graph/badge.svg?token=DsrQIbklpB)](https://codecov.io/github/mahynski/finchnmr)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<!--[![DOI](https://zenodo.org/badge/331207062.svg)](https://zenodo.org/badge/latestdoi/331207062)-->

<!--
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
-->

FINCHnmr : [FI]tti[N]g 13[C] 1[H] HSQC [NMR]
===

<img src="https://raw.githubusercontent.com/mahynski/finchnmr/main/docs/_static/logo_small.png" height="100" align="left" />

FINCHnmr is lightweight toolkit for fitting 2D heteronuclear single-quantum coherence nuclear magnetic resonance [(HSQC NMR)](https://en.wikipedia.org/wiki/Heteronuclear_single_quantum_coherence_spectroscopy) data to a known library of substances.  This predicts the presence and relative concentration of these compounds, and the residual (error) can be interpreted as the sum of the remaining unknown compounds present.  For a live demonstration, visit our [HuggingFace space](https://huggingface.co/spaces/mahynski/finchnmr-demo) or try it out on [streamlit's community cloud](https://finchnmr-demo.streamlit.app/). Although originally designed to work with (1H-13C) data, FINCHnmr will work with any 2D NMR as long as the library used matches the sample being predicted / analyzed.
<br/>

There are two approaches to generating the spectral libraries.  Both methods are demonstrated in the documentation.

1. Library spectra are taken directly from the [Biological Magnetic Resonance Bank (BMRB)](https://bmrb.io/). These spectra are automatically padded and resized so that they match the extent (2D grid) and resolution of the wild spectra being fit.
2. Library spectra are reconstructed from a feature list of peak locations by placing bivariate Gaussians at these locations; assumptions must be made about the spread of these distributions in both dimentions of frequency shift space.

Installation
===

We recommend creating a [virtual environment](https://docs.python.org/3/library/venv.html) or, e.g., a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) then installing finchnmr with [pip](https://pip.pypa.io/en/stable/):

~~~bash
$ pip install finchnmr
~~~

You can also install from this GitHub repo source:

~~~bash
$ git clone git@github.com:mahynski/finchnmr.git
$ cd finchnmr
$ pip install .
$ python -m pytest # Optional unittests
~~~

Documentation
===

Documentation is hosted at [https://finchnmr.readthedocs.io/](https://finchnmr.readthedocs.io/) via [readthedocs](https://about.readthedocs.com/).

The logo was generated using Google Gemini (Imagen 3) with the prompt "Design a logo involving a finch and NMR" on Nov. 9, 2024.

License Information
===

* See LICENSE.md for more information.
* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by [NIST](https://www.nist.gov/).
