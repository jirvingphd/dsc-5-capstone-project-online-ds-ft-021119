import matplotlib.pyplot as plt
import matplotlib as mpl
from bsds.functions_combined_BEST import *
import bsds.functions_sklearn as sk
import bsds.data as data
import bsds.ts_functions as ts
import bsds.app_functions as app

def display_side_by_side(*args):
    """Display all input dataframes side by side. Also accept captioned styler df object (df_in = df.style.set_caption('caption')
    Modified from Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side"""
    from IPython.display import display_html
    import pandas
    html_str=''
    for df in args:
        if type(df) == pandas.io.formats.style.Styler:
            html_str+= '&nbsp;'
            html_str+=df.render()
        else:
            html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
