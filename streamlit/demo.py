"""
Interactive demonstration of FINCHnmr.

Author: Nathan A. Mahynski
"""
import finchnmr
import os
import shutil
import zipfile

import numpy as np
import streamlit as st

from datasets import load_dataset
from finchnmr import analysis, library, model, substance
from streamlit_extras.add_vertical_space import add_vertical_space

UPLOAD_FOLDER = "uploaded_nmr"

# ----------------------------- CACHED FUNCTIONS -----------------------------
@st.cache_data
def build_library():
    """Build NMR library from HF."""
    nmr_dataset = load_dataset(
        "mahynski/bmrb-hsqc-nmr-1H13C",
        split="train",
        token=st.secrets["HF_TOKEN"],
        trust_remote_code=True,
    )
    substances = [
        finchnmr.substance.Substance(
            pathname=d['pathname'],
            name=d['name'],
            warning='ignore'
        ) for d in nmr_dataset
    ]
    lib = finchnmr.library.Library(substances)
    return lib

# @st.cache_data
def build_model(_target, _lib, _param_grid, _nmr_model, _model_kw):
    """Build model for target."""
    optimized_models, analyses = finchnmr.model.optimize_models(
        targets=[_target],
        nmr_library=_lib,
        nmr_model=_nmr_model,
        param_grid=_param_grid, 
        model_kw=_model_kw, 
    )
    return optimized_models, analyses

# --------------------------------- SIDEBAR ----------------------------------
st.set_page_config(layout="wide")
st.header("Analyze an HSQC NMR Spectra with FINCHnmr")
st.logo("docs/_static/logo_small.png", size='large', link="https://github.com/mahynski/finchnmr")

with st.sidebar:
    st.image("docs/_static/logo_small.png")
    st.markdown('''
    ## About this application    
    :heavy_check_mark: This tool is intended to demonstrate the use of [finchnmr](https://github.com/mahynski/finchnmr) to characterize the composition of mixture of compounds.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in the [finchnmr documentation](https://finchnmr.readthedocs.io/en/latest/index.html) for reproducible, high-quality analysis.

    This tool is provided "as-is" without warranty.  See our [License](https://github.com/mahynski/finchnmr/blob/a9c3504ea012fbd2452218fb2cd6924972bb88dc/LICENSE.md) for more details.
    ''')
    
    add_vertical_space(1)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')
    
with st.popover("Example Upload Directory"):
    st.text("example/\n├── acqu\n├── acqu2\n├── acqu2s\n├── acqus\n├── audita.txt\n├── cpdprg2\n├── format.temp\n├── fq1list\n├── pdata\n│       └── 1\n│                  ├── 2ii\n│                  ├── 2ir\n│                  ├── 2ri\n│                  ├── 2rr\n│                  ├── assocs\n│                  ├── auditp.txt\n│                  ├── clevels\n│                  ├── curdat2\n│                  ├── outd\n│                  ├── proc\n│                  ├── proc2\n│                  ├── proc2s\n│                  ├── procs\n│                  ├── thumb.png\n│                  └── title\n├── prosol_History\n├── pulseprogram\n├── scon2\n├── ser\n├── specpar\n├── spnam14\n├── spnam3\n├── spnam31\n├── spnam7\n├── uxnmr.info\n└── uxnmr.par\n")

# ----------------------------------- MAIN -----------------------------------
uploaded_file = st.file_uploader(
    label="Upload a directory output by a Bruker HSQC NMR instrument to start. This should be provided as .zip file. Refer to the dropdown above for an example of the directory structure which should be provided, e.g., as example.zip.",
    type=['zip'], 
    accept_multiple_files=False, 
    key=None, 
    help="", 
    on_change=None, 
    label_visibility="visible"
)

if uploaded_file is not None:
    if os.path.isdir(f'./{UPLOAD_FOLDER}/'):
        shutil.rmtree(f'./{UPLOAD_FOLDER}/')
        
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        z.extractall(f'./{UPLOAD_FOLDER}/')
            
    head = os.listdir(f'./{UPLOAD_FOLDER}/')
    if len(head) != 1:
        raise Exception("Uploaded zip file should contain exactly 1 folder.")
    else:
        head = head[0]
        
    # Create substance
    target = finchnmr.substance.Substance(
        pathname=os.path.abspath(f'./{UPLOAD_FOLDER}/{head}/pdata/1'),
        name=head,
        warning='ignore'
    )
    
    col1_, col2_ = st.columns(2)
    
    with col1_:
        # Plot the substance with plotly
        cmap_option = st.selectbox(
            "Colormap",
            ("Reds", "Blues", "Viridis", "Plasma", "RdBu"),
            index=0,
        )
        st.plotly_chart(target.plot(absolute_values=True, backend='plotly', cmap=cmap_option))

    with col2_:
        # Load reference library from HF
        with st.spinner(text="Building HSQC Library (this can take a minute the first time)..."):
            lib = build_library()
        st.success("Library has been built and cached!")

        # Select model type
        model_ = st.selectbox(label="Choose a model", options=["Lasso"], index=0)

        if model_:
            nmr_model = None
            param_grid = {}
            model_kw = {}

            # Set parameters and model kwargs
            with st.form(key='model_settings'):
                if model_.lower() == "lasso":
                    nmr_model = finchnmr.model.LASSO
                    start_alpha_ = st.number_input(label="Smallest alpha (log base)", min_value=-16, max_value=16, value="min", step=1)
                    stop_alpha_ = st.number_input(label="Largest alpha (log base)", min_value=-16, max_value=16, value=0, step=1)
                    n_ = st.slider(label="Number of alpha values in logscale", min_value=1, max_value=100, value=1, step=1)
                    
                    # Set of alphas to check
                    st.write("Hyperparameters")
                    param_grid = {'alpha': np.logspace(start_alpha_, stop_alpha_, int(n_))} # Select a range of alpha values to examine sparsity

                    # Lasso configuration
                    st.write("Model Configuration")
                    max_iter_ = st.number_input(label="Max number of iterations to converge, see [Lasso documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)", min_value=1, max_value=100000, value=1000, step=1)
                    selection_ = st.selectbox(label='Selection scheme, see [Lasso documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)', options=['selection', 'random'], index=0)
                    tol_ = st.number_input(label="Convergence tolerance, see [Lasso documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)", min_value=None, max_value=None, value=0.0001, format="%0.4f", step=0.0001)
                    model_kw = {'max_iter':int(max_iter_), 'selection':selection_, 'random_state':42, 'tol':tol_} 

                submit_button = st.form_submit_button("Start Building Model", icon=":material/start:")

            # Build the model
            if submit_button:
                stop_btn = st.button("Stop Building Model", type="primary", icon=":material/block:")
                with st.spinner(text="Building..."):
                    optimized_models, analyses = build_model(_target=target, _lib=lib, _param_grid=param_grid, _nmr_model=nmr_model, _model_kw=model_kw)
                st.success("Model has been built and cached!")

    # Now present the analysis / results
