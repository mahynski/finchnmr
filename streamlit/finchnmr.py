"""
Interactive demonstration of FINCHnmr.

Author: Nathan A. Mahynski
"""
import zipfile

import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('FINCHnmr: [FI]tti[N]g 13[C] 1[H] HSQC NMR')
    st.markdown('''
    ## About this application    
    :heavy_check_mark: This tool is intended to demonstrate the use of [finchnmr](https://github.com/mahynski/finchnmr) to characterize the composition of mixture of compounds.

    :x: It is not intended to be used in production.  Instead, use the Jupyter notebooks provided in the [finchnmr documentation](https://finchnmr.readthedocs.io/en/latest/index.html) for reproducible, high-quality analysis.

    This tool is provided "as-is" without warranty.  See our [License](https://github.com/mahynski/finchnmr/blob/a9c3504ea012fbd2452218fb2cd6924972bb88dc/LICENSE.md) for more details.
    ''')
    
    add_vertical_space(1)
    st.write('Made by ***Nate Mahynski***')
    st.write('nathan.mahynski@nist.gov')
    
st.header("Analyze an HSQC NMR Spectra with FINCHnmr")

# st.text("The directory structure should look something like this:\n\nexperiment-42.zip\n|\t\tacqu\n|    acqu2\n|    acqu2s\n|    acqus\n|    audita.txt\n|    cpdprg2\n|    format.temp\n|    fq1list\n|\n----pdata\n|   |\n|   ----1\n|       |    2ii\n|       |    2ir\n|       |    2ri\n|       |    2rr\n|       |    assocs\n  |       |    auditp.txt\n  |       |    clevels\n|       |    curdat2\n|       |    outd\n|       |    proc\n|       |    proc2\n |       |    proc2s\n |       |    procs\n|       |    thumb.png\n|       |    title\n|    prosol_History\n |    pulseprogram\n|    scon2\n|    ser\n|    specpar\n|    spnam14\n|    spnam3\n|    spnam31\n |    spnam7\n|    uxnmr.info\n|    uxnmr.par\n")

uploaded_file = st.file_uploader(
    label="Upload a directory output by a Bruker HSQC NMR instrument to start. This should be provided as .zip file.",
    type=['zip'], 
    accept_multiple_files=False, 
    key=None, 
    help="", 
    on_change=None, 
    label_visibility="visible"
)

if uploaded_file is not None:
    # If zip file, extract contents
    if file.type == "application/zip":
        with zipfile.ZipFile(file, 'r') as z:
            z.extractall('.')