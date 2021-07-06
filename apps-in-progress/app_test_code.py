import dash
import dash_table
import pandas as pd
from IPython.display import display

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

## Importing bsds functions
import sys
sys.path.append("../")
from bsds.app_functions import search_for_tweets_prior_hour
# from bsds.imports import *
import bsds as bs

## Import custom capstone functions
# from functions_combined_BEST import ihelp, ihelp_menu, reload
from pprint import pprint
import pandas as pd


# Import plotly and cufflinks for iplots
import plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"
import plotly.graph_objs as go
import plotly.offline as pyo 
import cufflinks as cf
cf.go_offline()
import bsds.functions_combined_BEST as ji


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
dash.Dash(assets_ignore='z_external_stylesheet')
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


dash.Dash(assets_ignore=['z_external_stylesheet','header.css']) #'typography.css'

twitter_df = pd.read_csv('../data/_twitter_df_with_stock_price.csv',index_col=0, parse_dates=True)

## Load in data
stock_df_filename = '../data/_stock_df_with_technical_indicators.csv'
stock_df = ji.load_processed_stock_data_plotly(stock_df_filename)


## APP- SEARCH FOR TWEETS BY STOCKHOUR
fig_price = ji.plotly_time_series(stock_df,figsize=(800,400),y_col='price', as_figure=True)#, show_fig=False)
fig_price.update({'layout':{
    'clickmode':'event+select'}}
)
# fig = go.Figure(data=fig_price['data'],layout=temp_layout)

## functions to use:
# ji.search_for_tweets_with_word(twitter_df, word = ___, display_n =5, from_column='content')

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

## Load in README
with open('../README.md','r') as f:
    README = f.read()
    
app = dash.Dash(__name__)

app.layout = html.Div(id='main-div',
    children=[
        html.Div(id='app-tweets-from-date', style={'border':'2px solid green','border-padding':'2%'} ,children=[
            
            dcc.Graph(id='stock_price',figure=fig_price,clear_on_unhover=True),
            dcc.Markdown(id='display-stock-tweets',
                         style={'min-height':'300px','border':'1px solid darkgreen'}),
            ]),
    
    #     # dcc.Markdown(README),
    # ## TWITTER SEARCH APP - Start
    #     html.Div(id='app-twitter-search-date', className='app',
    #              style={'border':'2px solid slategray'}, children=[                        
    #         html.H2(id='app-title-date',children="TRUMP'S TWEETS BY DATE" ,style={'text-align':'center'},className='app'),
        
    #         html.Div(id='full-search-menu-date', children= [
            
        
    #             html.Div(id='menu-input-date', className='interface',
    #                      style={'flex':'30%'}, children=[
                            
    #                     html.Label('Date', className='menu-label',
    #                             style={'margin-right':2}),
    #                     dcc.Input(id='search-date',
    #                               type='text',
    #                               value='08/23/2019',
    #                               style={'margin-right':'5%'}),

    #                     html.Label('# of Tweets to Show',className='menu-label'),
    #                     dcc.Input(id='display-n-date', 
    #                             value=10,
    #                             type='number',
    #                             style={'width':'10%','margin-left':'2%'})
    #                     ]),
                
    #             html.Button(id='submit-button-date',
    #                     n_clicks=0,
    #                     children='Submit'
    #                     )
    #             ]),
    #     dcc.Markdown(id='display_results_date',
    #                 )
    #     ]),
    ## TWITTER SEARCH APP - End
    
        
    # ## TWITTER SEARCH APP - Start
    #     html.Div(id='app-twitter-search', className='app',
    #              style={'border':'2px solid slategray'}, children=[                        
    #         html.H2(id='app-title',children="SEARCH TRUMP'S TWEETS" ,style={'text-align':'center'},className='app'),
        
    #         html.Div(id='full-search-menu', children= [
            
        
    #             html.Div(id='menu-input', className='interface',
    #                      style={'flex':'30%'}, children=[
                            
    #                     html.Label('Word to Find', className='menu-label',
    #                             style={'margin-right':2}),
    #                     dcc.Input(id='search-word',
    #                               type='text',
    #                               value='Russia',
    #                               style={'margin-right':'5%'}),

    #                     html.Label('# of Tweets to Show',className='menu-label'),
    #                     dcc.Input(id='display-n', 
    #                             value=10,
    #                             type='number',
    #                             style={'width':'10%','margin-left':'2%'})
    #                     ]),
                
    #             html.Button(id='submit-button',
    #                     n_clicks=0,
    #                     children='Submit'
    #                     )
    #             ]),
    #     dcc.Markdown(id='display_results',
    #                 )
    #     ]),
    # ## TWITTER SEARCH APP - End

])

### STOCK HOUR -> TWEETS
@app.callback(
    Output('display-stock-tweets', 'children'),
    [Input('stock_price', 'hoverData'),
     Input('stock_price','clickData')])#'clickData')])
def display_tweets_from_stocks(hoverData,clickData,twitter_df=twitter_df):
    # from temp_code import search_for_tweets_prior_hour
    import pandas as pd
    # res=[]
    
    if hoverData is not None:
        stock_hour = hoverData['points'][0]['x']
        stock_hour = pd.to_datetime(stock_hour)
        res_hover = search_for_tweets_prior_hour(twitter_df=twitter_df, stock_hour=stock_hour)
        if len(res_hover) == 0:
            res_hover= "### No Tweets found"# for {stock_hour}"
                
    if clickData is not None:
        stock_hour = clickData['points'][0]['x']
        stock_hour = pd.to_datetime(stock_hour)
        res_click = search_for_tweets_prior_hour(twitter_df=twitter_df, stock_hour=stock_hour)
        if len(res_click) == 0:
            res_click='### No Tweets found'# for {stock_hour}'
        
    if (hoverData is None):
        if clickData is not None:
            return res_click
    elif hoverData is not None:
        return res_hover

        # if len(stock_hour)==0:
        #     res = f'No Tweets Found for {stock_hour}'
            
    # elif hoverData is not None:
    #     stock_hour = hoverData['points'][0]['x']
    #     stock_hour = pd.to_datetime(stock_hour)
    #     res = search_for_tweets_prior_hour(twitter_df=twitter_df, stock_hour=stock_hour)
    #     if len(res)==0:
    #         res = f'No Tweets Found for {stock_hour}'
    else:
        return 'Hover or click to show Tweets.'    



# ### TWITTER WORD SEARCH
# @app.callback(Output(component_id='display_results', component_property = 'children'),
#             [Input(component_id='submit-button',component_property='n_clicks'),
#              Input(component_id='display-n',component_property='value')],
#             [State(component_id='search-word',component_property='value')])
# def search_tweets(n_clicks, display_n, word):
#     from IPython.display import Markdown, display
#     from temp_code import search_for_tweets_with_word
        
#     res = search_for_tweets_with_word(twitter_df,word=word,display_n=display_n,as_md=True,from_column='content')
#     return  res


# ### TWIITTER DATE SEARCH
# @app.callback(Output(component_id='display_results_date', component_property = 'children'),
#             [Input(component_id='submit-button-date',component_property='n_clicks'),
#              Input(component_id='display-n-date',component_property='value')],
#             [State(component_id='search-date',component_property='value')])
# def search_tweets_for_date(n_clicks, display_n, date):
#     from IPython.display import Markdown, display
#     from temp_code import search_for_tweets_by_date
        
#     res = search_for_tweets_by_date(twitter_df,date=date,display_n=display_n,as_md=True,from_column='content')
#     return  res


if __name__ == '__main__':
    app.run_server(debug=False,host='127.0.0.1')