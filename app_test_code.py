import dash
import dash_table
import pandas as pd
from IPython.display import display

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


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

app.layout = html.Div(id='main-div',
    children=[
        
    ## TWITTER SEARCH APP - Start
        html.Div(id='app-twitter-search', className='app',
                 style={'border':'2px solid slategray'}, children=[                        
            html.H2(id='app-title',children="SEARCH TRUMP'S TWEETS" ,style={'text-align':'center'},className='app'),
        
            html.Div(id='full-search-menu', children= [
            
        
                html.Div(id='menu-input', className='interface',
                         style={'flex':'30%'}, children=[
                            
                        html.Label('Word to Find', className='menu-label',
                                style={'margin-right':2}),
                        dcc.Input(id='search-word',
                                  type='text',
                                  value='Russia',
                                  style={'margin-right':'5%'}),

                        html.Label('# of Tweets to Show',className='menu-label'),
                        dcc.Input(id='display-n', 
                                value=10,
                                type='number',
                                style={'width':'10%','margin-left':'2%'})
                        ]),
                
                html.Button(id='submit-button',
                        n_clicks=0,
                        children='Submit'
                        )
                ]),
        dcc.Markdown(id='display_results',
                    )
        ]),
    ## TWITTER SEARCH APP - End

])


@app.callback(Output(component_id='display_results', component_property = 'children'),
            [Input(component_id='submit-button',component_property='n_clicks'),
             Input(component_id='display-n',component_property='value')],
            [State(component_id='search-word',component_property='value')])
def search_tweets(n_clicks, display_n, word):
    from IPython.display import Markdown, display
    from temp_code import search_for_tweets_with_word
        
    res = search_for_tweets_with_word(twitter_df,word=word,display_n=display_n,as_md=True,from_column='content')
    return  res


if __name__ == '__main__':
    app.run_server(debug=True)