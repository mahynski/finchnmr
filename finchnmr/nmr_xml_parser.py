"""
Parse XML files.

Authors: David A. Sheen, Nathan A. Mahynski
"""
import xml
import xml.etree.ElementTree as ET
import pandas as pd

from typing import Any

types_dict = dict(
    F1='float64',
    F2='float64',
    annotation='str',
    intensity='float64',
    type='int',
)

def parse_peak_file(xml_file: str) -> "pd.DataFrame":
    """
    Parse the XML file in a Pandas DataFrame.
    
    Parameters
    ----------
    xml_file : str
        Name of .xml file to parse.

    Returns
    -------
    dataframe : pd.DataFrame
        DataFrame of NMR features
    """
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Build a Pandas Dataframe from a dictionary
    index = 0 # Index by integer
    peak_dict: dict[int, Any] = dict()
    for child in root:
        for sub in child:
            # NMR features are labeled with the tag 'Peak2D'; there is also a metadata header, which we ignore here.
            if 'Peak2D' in sub.tag:
                peak_dict[index] = sub.attrib
                index += 1
    
    peak_df = pd.DataFrame.from_dict(peak_dict, orient='index').astype(types_dict)
    
    return peak_df