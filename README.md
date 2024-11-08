FINCHnmr : [FI]tti[N]g [C]13 [H]1 HSQC [NMR]
===

<img src="docs/_static/logo_small.png" height="100" align="left" />

FINCHnmr is lightweight toolkit for fitting 2D HSQC (H1, C13) NMR data to a known library of substances.  This predicts the presence and relative concentration of these compounds, and the residual (error) can be interpreted as the sum of the remaining unknown compounds present.  For a live demonstration, visit [https://finchnmr-demo.streamlit.app/](https://finchnmr-demo.streamlit.app/).

<br/>

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

Notes
===

Look at [LASSO CV](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LassoCV.html)
See [sparse-lm](https://cedergrouphub.github.io/sparse-lm/) as an alternative to LASSO

~~~bash
$ mypy --ignore-missing-imports my_new_file.py
~~~
