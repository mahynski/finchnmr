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
    
    optimized_models = []

    tab1_, tab2_ = st.tabs(["Configure Model", "Analyze Results"]) 
    with tab1_:
        st.subheader('Configure Model')

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
            with st.spinner(text="Building HSQC Library from [HuggingFace](https://huggingface.co/datasets/mahynski/bmrb-hsqc-nmr-1H13C) (this can take a minute the first time)..."):
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

                        # Set of alphas to check
                        st.write("Hyperparameters")
                        start_alpha_ = st.number_input(label="Smallest alpha (log base)", min_value=-16, max_value=16, value="min", step=1)
                        stop_alpha_ = st.number_input(label="Largest alpha (log base)", min_value=-16, max_value=16, value=0, step=1)
                        n_ = st.slider(label="Number of alpha values in logscale", min_value=1, max_value=100, value=1, step=1)
                        param_grid = {'alpha': np.logspace(start_alpha_, stop_alpha_, int(n_))} # Select a range of alpha values to examine sparsity

                        # Lasso configuration
                        st.divider()
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
                    st.success("Model has been built and cached!", icon="✅")

    # Now present the analysis / results
    with tab2_:
        import pickle
        optimized_models = [pickle.load(open("streamlit/example_model.pkl", 'rb'))] # TEMP
        analyses = [pickle.load(open("streamlit/example_analysis.pkl", 'rb'))] # TEMP

        if len(optimized_models) > 0:
            st.subheader('Observe how well the model fits the original spectrum.')

            model_ = optimized_models[0] # We only fit one model
            analysis_ = analyses[0]

            # Plot original vs. reconstructed and residual
            col3_, col4_ = st.columns(2)
            with col3_:
                # Plot the substance with plotly
                cmap_option3 = st.selectbox(
                    "Colormap",
                    ("Reds", "Blues", "Viridis", "Plasma", "RdBu"),
                    index=0,
                    key='compare_orig',
                )
                st.plotly_chart(
                    target.plot(absolute_values=True, backend='plotly', cmap=cmap_option3, title='Original Spectrum'),
                    key='compare_orig_plot',
                )

            with col4_:
                cmap_option4 = st.selectbox(
                    "Colormap",
                    ("Reds", "Blues", "Viridis", "Plasma", "RdBu"),
                    index=0,
                    key='compare_recon'
                )
                st.plotly_chart(
                    model_.reconstruct().plot(absolute_values=True, backend='plotly', cmap=cmap_option4, title='Model Reconstruction'),
                    key='compare_recon_plot'
                )

            col5_, col6_ = st.columns(2)
            with col5_:
                cmap_option5 = st.selectbox(
                    "Colormap",
                    ("Reds", "Blues", "Viridis", "Plasma", "RdBu"),
                    index=0,
                    key='compare_resid'
                )
                st.plotly_chart(
                    analysis_.build_residual().plot(absolute_values=True, backend='plotly', cmap=cmap_option5, title='Residual'),
                    key='compare_resid_plot'
                )

            # Plot most important spectra
            max_n_ = len(analysis_._model.importances())
            default_n_ = np.min([10, max_n_])
            with col6_:
                n_imp_ = st.slider(
                    label='Visualize the most important N spectra in the library',
                    value=default_n_, 
                    min_value=1, 
                    max_value=max_n_,
                    step=1
                )
                st.plotly_chart(
                    analysis_.plot_top_importances(k=n_imp_, by_name=True, backend='plotly'),
                    use_container_width=True
                )

            st.divider()
            st.subheader('Visualize the most important substances from the library used')

            # Now plot the important spectra themselves
            top_substances, top_importances = analysis_.get_top_substances(k=n_imp_)
            n_cols_ = st.slider(
                label='Number of columns',
                value=3, 
                min_value=1, 
                max_value=n_imp_,
                step=1
            )
            n_rows_ = np.max([1, int(np.ceil(n_imp_ / n_cols_))])
            ctr = 0
            for row_idx in range(n_rows_):
                for col_, col_idx in zip(st.columns(n_cols_), range(n_cols_)):
                    with col_:
                        if ctr < n_imp_:
                            cmap_option_ = st.selectbox(
                                "Colormap",
                                ("Reds", "Blues", "Viridis", "Plasma", "RdBu"),
                                index=0,
                                key=f"cmap_option_{ctr}_"
                            )
                            st.plotly_chart(
                                top_substances[ctr].plot(absolute_values=True, backend='plotly', cmap=cmap_option_)
                            )  
                        ctr += 1

