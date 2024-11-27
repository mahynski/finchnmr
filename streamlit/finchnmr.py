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

st.markdown('''
The directory structure should look something like this:

experiment-42.zip
|    acqu  
|    acqu2  
|    acqu2s  
|    acqus  
|    audita.txt  
|    cpdprg2  
|    format.temp  
|    fq1list  
|
----pdata  
|   |
|   ----1
|       |    2ii  
|       |    2ir  
|       |    2ri  
|       |    2rr  
|       |    assocs  
|       |    auditp.txt  
|       |    clevels  
|       |    curdat2  
|       |    outd  
|       |    proc  
|       |    proc2  
|       |    proc2s  
|       |    procs  
|       |    thumb.png 
|       |    title
|    prosol_History  
|    pulseprogram  
|    scon2  
|    ser  
|    specpar  
|    spnam14  
|    spnam3  
|    spnam31  
|    spnam7  
|    uxnmr.info  
|    uxnmr.par
''')

uploaded_file = st.file_uploader(
    label="Upload a directory output by a Bruker HSQC NMR instrument to start. This should be provided as .zip file.",
    type=['zip'], 
    accept_multiple_files=False, 
    key=None, 
    help="", 
    on_change=None, 
    label_visibility="visible"
)

if len(file_uploaded) > 0:
    for file in file_uploaded:
        # If zip file, extract contents
        if file.type == "application/zip":
            with zipfile.ZipFile(file, 'r') as z:
                z.extractall('.')