# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
dash.Dash(assets_ignore='z_external_stylesheet')

import pandas as pd
from dash.dependencies import Input, Output
import functions_combined_BEST as ji



# Import some functions directly
from functions_combined_BEST import ihelp, ihelp_menu, reload

## IMPORT STANDARD PACKAGES
from bs_ds.imports import *
from pprint import pprint

# Import my custom functions 
import functions_combined_BEST as ji
import my_keras_functions as jik
import bs_ds as bs

# Import plotly and cufflinks for iplots
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo 
import cufflinks as cf
# cf.go_offline()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

stock_df = ji.load_processed_stock_data()


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
fig = ji.plotly_time_series(stock_df,y_col='price', as_figure=True)
fig_indicators = ji.plotly_technical_indicators(stock_df)



app.layout = html.Div(children=[

    html.H1(children="Predicting Stock Market Fluctuations With Donald Trump's Tweets"),
    html.P(id='my_name', children=dcc.Markdown('''
    James M. Irving, Ph.D.

    https://github.com/jirvingphd

    [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)

    ''')),

    html.Div(className='main',children=[dcc.Markdown('''
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
            '''),

            html.H2('PLACEHOLDER - TWEET NLP /MODELING'),
            html.H3('Visuals from Twitter NLP anaylsis'),
            html.H3('Model Details and Results'),
            html.H2('PLACEHOLDER - Modeling Stock Price'),
            dcc.Graph(figure=fig),

            dcc.Markdown('''
    ### STOCK MARKET PRICE DATA (S&P 500):

    - **ENGINEER FEATURES:** 
        - *Extract features about the stock data -calculate the technical indices for the S&P 500 discussed in his article.*  [ see reference #2 - blog post ]
            * [x] 7 days moving average 
            * [x] 21 days moving average
            * [x] exponential moving average
            * [x] momentum
            * [x] Bollinger bands
            * [x] MACD
            
    * **PREDICTIVE MODELING:**
        - *Generate stock price predictions based only historical data using....*
            * [x] a binary classification using Tweet NLP to predict stock price up/down
            * [x] a SARIMAX model using stock price alone
            * [x] a LSTM neural network using price alone
            * [x] a LSTM using price and technical indicators 
            * [x] a LSTM using price, technical indicators, NLP-Model #1's predicitons, tweet-statistics
            
    ### FINAL MODEL - FEED ALL ABOVE FEATURES INTO:

    - **NEURAL NETWORK *REGRESSION* MODEL TO PREDICT *ACTUAL S&P 500 PRICE* AT 1 HOUR-1 DAY FOLLOWING TWEETS**
        - Final Model Target is based more on blog post's construction (ref#2), but takes output of model like ref#1

    ''',
    ),
    # html.Img(src="https://raw.githubusercontent.com/jirvingphd/dsc-5-capstone-project-online-ds-ft-021119/master/figures/annotated_GAN_for_stock_market.jpeg", width=650), #"/assets/annotated_GAN_for_stock_market.jpeg",width=700)
    dcc.Markdown("___")
    ]
    ),



    html.Div(
        children = dcc.Markdown(
            """
            ___

            ## DATA ANALYSIS DETAILS AND EQUATIONS 
            ### Technical Indicators - Explanation & Equations
            1. **7 and 21 day moving averages**
            ```python
            df['ma7'] df['price'].rolling(window = 7 ).mean() #window of 7 if daily data
            df['ma21'] df['price'].rolling(window = 21).mean() #window of 21 if daily data
            ```    
            2. **MACD(Moving Average Convergence Divergence)**

            > Moving Average Convergence Divergence (MACD) is a trend-following momentumindicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.

            >The result of that calculation is the MACD line. A nine-day EMA of the MACD, called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals. 

            > Traders may buy the security when the MACD crosses above its signal line and sell - or short - the security when the MACD crosses below the signal line. Moving Average Convergence Divergence (MACD) indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.  - _[from Investopedia](https://www.investopedia.com/terms/m/macd.asp)_

            ```python
            df['ewma26'] = pd.ewma(df['price'], span=26)
            df['ewma12'] = pd.ewma(df['price'], span=12)
            df['MACD'] = (df['12ema']-df['26ema'])
            ```
            3. **Exponentially weighted moving average**
            ```python
            dataset['ema'] = dataset['price'].ewm(com=0.5).mean()
            ```
            4. **Bollinger bands**
                > Bollinger Bands® are a popular technical indicators used by traders in all markets, including stocks, futures and currencies. There are a number of uses for Bollinger Bands®, including determining overbought and oversold levels, as a trend following tool, and monitoring for breakouts. There are also some pitfalls of the indicators. In this article, we will address all these areas.
                > Bollinger bands are composed of three lines. One of the more common calculations of Bollinger Bands uses a 20-day simple moving average (SMA) for the middle band. The upper band is calculated by taking the middle band and adding twice the daily standard deviation, the lower band is the same but subtracts twice the daily std. - _[from Investopedia](https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends/)_

                - Boilinger Upper Band:<br>
                $BOLU = MA(TP, n) + m * \sigma[TP, n ]$<br><br>
                - Boilinger Lower Band<br>
                $ BOLD = MA(TP,n) - m * \sigma[TP, n ]$
                - Where:
                    - $MA$  = moving average
                    - $TP$ (typical price) = $(High + Low+Close)/ 3$
                    - $n$ is number of days in smoothing period
                    - $m$ is the number of standard deviations
                    - $\sigma[TP, n]$ = Standard Deviations over last $n$ periods of $TP$

            ```python
            # Create Bollinger Bands
            dataset['20sd'] = pd.stats.moments.rolling_std(dataset['price'],20)
            dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
            dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
            ```

            5. **Momentum**
            > "Momentum is the rate of acceleration of a security's price or volume – that is, the speed at which the price is changing. Simply put, it refers to the rate of change on price movements for a particular asset and is usually defined as a rate. In technical analysis, momentum is considered an oscillator and is used to help identify trend lines." - _[from Investopedia](https://www.investopedia.com/articles/technical/081501.asp)_

                - $ Momentum = V - V_x$
                - Where:
                    - $V$ = Latest Price
                    - $V_x$ = Closing Price
                    - $x$ = number of days ago

            ```python
            # Create Momentum
            dataset['momentum'] = dataset['price']-1
            ```
            """
         ), 
        #  style = body_style
    ),

    html.Div(
    dcc.Graph(id='stock-graph',
    figure = fig_indicators
    )
)])

if __name__ == '__main__':
    app.run_server(debug=True)

####
# @app.callback(Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')])