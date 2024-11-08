FINCHnmr : [FI]tti[N]g [C]13 [H]1 HSQC [NMR]
===

<img src="docs/_static/logo_small.png" height="100" align="left" />

FINCHnmr is lightweight toolkit for fitting 2D HSQC (H1, C13) NMR data to a known library of substances.  This predicts the presence and relative concentration of these compounds, and the residual (error) can be interpreted as the sum of the remaining unknown compounds present.

<br/>
<br/>
<br/>
<br/>

Documentation
===

Documentation is stored in the `docs/` folder and is currently set up to use [sphinx](https://www.sphinx-doc.org/en/master/).

First build the `requirements.txt` needed to build the documentation.

~~~bash
$ cd docs
$ pip install Sphinx
$ pip install pip-tools
$ pip-compile requirements.in
~~~

Adjust the `docs/conf.py` as desired. Then run `docs/make_docs.sh` to setup the documentation initially.  You can manually add and adjust later.

~~~bash
$ bash make_docs.sh
~~~

Go to [https://about.readthedocs.com/](https://about.readthedocs.com/) to link your repo to build and host the documentation automatically!  The `.readthedocs.yml` file contains the configuration for this which you can adjust as needed.

Unittests
===

Build [unittests](https://docs.python.org/3/library/unittest.html) in the `tests/` directory.  The `pyproject.toml` automatically configures pytest to look in `tests/`.  The following will run all unittests in this directory.

~~~bash
$ python -m pytest
~~~

The GitHub workflow in `.github/workflows/python-app.yml` will also run these tests and perform coverage checks using this command.  This workflow is triggered automatically on the main branch, but you can adjust this file so this is automatically triggered on others as well.

Linting
===

Automatic code linting is provided via [pre-commit](https://pre-commit.com/); refer to the `.pre-commit-config.yaml` file for the specific configuration which you can adjust as needed.

Run pre-commit to lint new code, then commit the changes.

~~~bash
$ pre-commit run --all-files
~~~

Best Practices
===

Other best practices include [typing](https://docs.python.org/3/library/typing.html) - see [mypy](https://mypy-lang.org/).

Notes
===

Look at [LASSO CV](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LassoCV.html)
See [sparse-lm](https://cedergrouphub.github.io/sparse-lm/) as an alternative to LASSO

~~~bash
$ mypy --ignore-missing-imports my_new_file.py
~~~
