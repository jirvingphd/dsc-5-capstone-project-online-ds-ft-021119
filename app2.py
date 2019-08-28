# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

## IMPORT STANDARD PACKAGES
from bs_ds.imports import *
import bs_ds as bs

## Import custom capstone functions
import functions_combined_BEST as ji
from functions_combined_BEST import ihelp, ihelp_menu, reload
from pprint import pprint
import pandas as pd


# Import plotly and cufflinks for iplots
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo 
import cufflinks as cf
# cf.go_offline()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
dash.Dash(assets_ignore='z_external_stylesheet')


## Open file firectory 
with open('data/filename_dictionary.json', 'r') as f:
    import json
    file_dict_json = f.read()
    file_dict = json.loads(file_dict_json)

## Load in data
stock_df_filename = file_dict['stock_df']['stock_df_with_indicators']
stock_df = ji.load_processed_stock_data(stock_df_filename)

df_models_dict = {}
for model in ['model_0A','model_0B','model_1','model_2','model_3']:
    
    for res in ['df_model','df_results']:
        filename = file_dict[model][res]
        model_dict[model][res] = pd.read_csv(filename,index_col=0,parse_dates=True)
        # model_dict[model]['df_results'] = pd.read_excel()


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

fig = ji.plotly_time_series(stock_df,y_col='price', as_figure=True)
fig_indicators = ji.plotly_technical_indicators(stock_df)



app.layout = html.Div(id='main_container',children=[


    html.Div(id='header',children=[ #main child1

        html.H1(children="Predicting Stock Market Fluctuations With Donald Trump's Tweets"),
        html.P(id='my_name', children=dcc.Markdown('''
        James M. Irving, Ph.D.

        https://github.com/jirvingphd

        [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)

        '''))
        ]
        ),

    html.Div(id='intro',children=[ #main child2
        dcc.Markdown('''
        ___
        
        ## PROJECT GOAL:
        * **Use President Trump's tweets (NLP and other features) to predict fluctuations in the stock market (using S&P 500 as index).**

            
        ### DATA USED:

        * **All Donald Trump tweets from inaugaration day 2017 to today (for now) - 06/20/19**
            * Extracted from http://www.trumptwitterarchive.com/
        * **Minute-resolution data for the S&P500 covering the same time period.**
            * IVE S&P500 Index from - http://www.kibot.com/free_historical_data.aspx
            
        ### MAJOR REFERENCES / INSPIRATION / PRIOR WORK IN FIELD:
        1. **Stanford Scientific Poster Using NLP ALONE to predict if stock prices increase or decrease 5 mins after Trump tweets.**  
            - [Poster PDF LINK](http://cs229.stanford.edu/proj2017/final-posters/5140843.pdf)
            - [Evernote Summary Notes Link](https://www.evernote.com/l/AAoL1CyhPV1GoIzSgq59GO10x6xfEeVDo5s/)
        2. **TowardsDataScience Blog Plost on "Using the latest advancements in deep learning to predict stock price movements."** 
            - [Blog Post link](https://towardsdatascience.com/aifortrading-2edd6fac689d)
            - [Evernote Summary](https://www.evernote.com/l/AApvQ8Xh8b9GBLhrD0m8w4H1ih1oVM8wkEw/)


        ## DATA AND MODEL OUTLINE

        ### Natural Language Processing for Twitter Data:

        - **FEATURES ENGINEERED/USED:**
            - Extract features from Trump's tweets: perform the NLP analysis to generate the features about his tweets to use in final model
                * [x] Tweet sentiment score
                * [x] Tweet frequency per timebin
                * [x] upper-to-lowercase-ratio
                * [x] retweet-count
                * [x] favorite-count
            
        * **PREDICTIVE MODEL** 
            - *Binary classification using Word2Vec Embeddings with an LSTM to predict +/- price change per tweet.*
                * [x] Fit word2vec model on tweets, use vectors to create an embedding layer for model.
                * [x] Feed in  X data = content of tweet, y = stock_price change 60 mins after tweet.

            ''')
            ]
        ), # end div

        html.Div(id='tweets',children= [ # main child 3
            
            html.H2(children="EXPLORING TRUMP'S TWEETS"),
            html.Div([
                dcc.Input(id='search',value='russia',type='text'),
                html.Div(id)
            ]),
        ]
    )
)


if __name__ == '__main__':
    app.run_server(debug=True)