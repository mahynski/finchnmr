"""
Interactive demonstration of FINCHnmr.

Author: Nathan A. Mahynski
"""
import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('FINCHnmr: [FI]tti[N]g 13[C] 1[H] HSQC NMR')
    st.markdown('''
    ## About this application
    This tool uses the [finchnmr](https://github.com/mahynski/finchnmr) python 
    package for analysis.
    
    :heavy_check_mark: It is intended as a tool to demonstrate the use of [finchnmr](https://github.com/mahynski/finchnmr) to characterize the composition of mixture of compounds.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in 
    [finchnmr documentation](https://finchnmr.readthedocs.io/en/latest/index.html) for reproducible, high-quality
    analysis.

    This tool is provided "as-is" without warranty.  See our [License](https://github.com/mahynski/finchnmr/blob/a9c3504ea012fbd2452218fb2cd6924972bb88dc/LICENSE.md) for more details.
    ''')
    add_vertical_space(2)
    
    uploaded_file = st.file_uploader(
        label="Upload a directory output by a Bruker HSQC NMR instrument to analyze. This should be provided as .zip file.",
        type=['zip'], 
        accept_multiple_files=False, 
        key=None, 
        help="", 
        on_change=None, 
        label_visibility="visible"
    )
    
    add_vertical_space(2)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')