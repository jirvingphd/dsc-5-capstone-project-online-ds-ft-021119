import dash
import dash_table
import pandas as pd
from IPython.display import display

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


## IMPORT STANDARD PACKAGES
from bs_ds.imports import *
import bs_ds as bs

## Import custom capstone functions
# from functions_combined_BEST import ihelp, ihelp_menu, reload
from pprint import pprint
import pandas as pd


# Import plotly and cufflinks for iplots
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo 
import cufflinks as cf
# cf.go_offline()
import functions_combined_BEST as ji


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
dash.Dash(assets_ignore='z_external_stylesheet')
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


dash.Dash(assets_ignore=['z_external_stylesheet','header.css']) #'typography.css'

twitter_df = pd.read_csv('data/_twitter_df_with_stock_price.csv',index_col=0, parse_dates=True)
# stock_df = pd.read_csv('data/_stock_df_with_technical_indicators.csv', index_col=0, parse_dates=True)

## functions to use:
# ji.search_for_tweets_with_word(twitter_df, word = ___, display_n =5, from_column='content')

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')


app = dash.Dash(__name__)

app.layout = html.Div(id='main-div',children=[
    dcc.Input(id='search-word',type='text',value='russia'),
    html.Div(id='search-for-tweets',children=[
        dash_table.DataTable(id='output-tweets', )
    ]) #,children=[        html.Iframe(id='iframe')]
])


@app.callback(Output(component_id='search-for-tweets', component_property = 'children'), [Input(component_id='search-word',component_property='value')])
def search_tweets(word):
    from IPython.display import Markdown
    from temp_code import search_for_tweets_with_word
    res = search_for_tweets_with_word(twitter_df,word=word,display_n=10,as_df=True,from_column='content')
    # display(res)
    import json
    
    # out_df =res.reset_index().to_json()
    return  res.__repr__()


if __name__ == '__main__':
    app.run_server(debug=True)