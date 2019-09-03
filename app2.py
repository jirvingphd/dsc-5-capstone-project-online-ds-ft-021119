# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

## IMPORT STANDARD PACKAGES
from bs_ds.imports import *
import bs_ds as bs

## Import custom capstone functions
from functions_combined_BEST import ihelp, ihelp_menu, reload
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
# warnings.filterwarnings('ignore')
dash.Dash(assets_ignore='z_external_stylesheet')
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


## Load in text to display
with open ('assets/text/intro.txt','r') as f:
    md_intro = f.read() 

with open ('assets/text/model_details_stocks.txt','r') as f:
    md_model_details_stocks=f.read()

with open ('assets/text/model_details_nlp.txt','r') as f:
    md_model_details_nlp=f.read()

with open('assets/text/technical_indicators.txt','r') as f:
    md_tech_indicators = f.read()

with open('assets/text/nlp_intro.txt','r') as f:
    md_nlp_intro = f.read()

with open('assets/text/nlp_data_intro.txt','r') as f:
    md_nlp_data_intro = f.read()

## NLP Figure Image locations
wordclouds_top_words = "assets/images/wordcloud_top_words_by_delta_price.png"
wordclouds_unique_words = "assets/images/wordcloud_unique_words_by_delta_price.png"

## NLP Model Training Img Locations
img_conf_mat_NLP_fig = 'assets/model0A/model0A_conf_matrix.png'
img_keras_history_NLP_fig = 'assets/model0A/model0A_keras_history.png'
NLP_model_summary = 'assets/model0A/model0A_summary.txt'

## Load in data
stock_df_filename = 'data/_stock_df_with_technical_indicators.csv'
stock_df = ji.load_processed_stock_data_plotly(stock_df_filename)

twitter_df = pd.read_csv('data/_twitter_df_with_stock_price.csv',
index_col=0,parse_dates=True)


## Specify all assets for model 1
df_model1 = pd.read_csv('results/model1/best/model1_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results1 = pd.read_excel('results/model1/best/model1_df_results.xlsx',index_col=0)
img_model1_history = 'results/model1/best/model1_keras_history.png'

df_train_test1=df_model1[['true_train_price','true_test_price']]
model1_train_test_fig = ji.plotly_time_series(df_train_test1,as_figure=True,show_fig=False)

with open('results/model1/best/model1_summary.txt','r') as f:
    txt_model1_summary = "```"
    txt_model1_summary+=f.read()
    txt_model1_summary+= "\n```"

fig_model1 = ji.plotly_true_vs_preds_subplots(df_model1, show_fig=False)




## Specify all assets for model 2
df_model2 = pd.read_csv('results/model2/best/model2_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results2 = pd.read_excel('results/model2/best/model2_df_results.xlsx',index_col=0)
img_model2_history = 'results/model2/best/model2_keras_history.png'

df_train_test1=df_model2[['true_train_price','true_test_price']]
model2_train_test_fig = ji.plotly_time_series(df_train_test1,as_figure=True,show_fig=False)

with open('results/model2/best/model2_summary.txt','r') as f:
    txt_model2_summary = "```"
    txt_model2_summary+=f.read()
    txt_model2_summary+= "\n```"

fig_model2 = ji.plotly_true_vs_preds_subplots(df_model2, show_fig=False)




## Specify all assets for model 1
df_model3 = pd.read_csv('results/model3/best/model3_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results3 = pd.read_excel('results/model3/best/model3_df_results.xlsx',index_col=0)
img_model3_history = 'results/model3/best/model3_keras_history.png'

df_train_test1=df_model3[['true_train_price','true_test_price']]
model3_train_test_fig = ji.plotly_time_series(df_train_test1,as_figure=True,show_fig=False)

with open('results/model3/best/model2_summary.txt','r') as f:
    txt_model3_summary = "```"
    txt_model3_summary+=f.read()
    txt_model3_summary+= "\n```"

fig_model3 = ji.plotly_true_vs_preds_subplots(df_model3, show_fig=False)



# df_model2 = pd.read_csv('results/model_2/df_model.csv',index_col=0,parse_dates=True)

# df_model3 = pd.read_csv('')

# fig_model2 = ji.plotly_true_vs_preds_subplots(df_model2,show_fig=False)



fig_price = ji.plotly_time_series(stock_df,y_col='price', as_figure=True)#, show_fig=False)
fig_indicators = ji.plotly_technical_indicators(stock_df, as_figure=True, show_fig=False)
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

md_example_tweet_forms = ji.display_same_tweet_diff_cols(twitter_df, columns=['content','content_min_clean',
'cleaned_stopped_content'], for_dash=True)#,'cleaned_stopped_tokens'


app = dash.Dash(__name__)# external_stylesheets=external_stylesheets)

app.layout = html.Div(id='main-div',children=[

    html.H1(id='main_header',className='main_header', children="Predicting Stock Market Fluctuations With Donald Trump's Tweets"),

    html.P(id='my_name', children=[
        dcc.Markdown('''
        James M. Irving, Ph.D.

        https://github.com/jirvingphd

        [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)

        ''')]),
    
        html.Div(id='1_intro',
        children=[ #main child2
        dcc.Markdown(md_intro)
        ]),

        html.Div(id="2_NLP", children=[
            html.H1('NATURAL LANGUAGE PROCESSING'),
            dcc.Markdown(md_nlp_intro),
            html.Div(id='show_tweet_forms',
            children=[
                dcc.Markdown(ji.display_same_tweet_diff_cols(twitter_df,for_dash=True))
                ]),
            html.Div(id='nlp_data',children=[
                html.H3('When Trump Tweets, does the market react?'),
                dcc.Graph(figure=fig_price),
                dcc.Markdown(md_nlp_data_intro),
                dcc.Graph(figure=ji.plotly_price_histogram(twitter_df)),
                dcc.Graph(figure=ji.plotly_pie_chart(twitter_df,show_fig=False),style={'height':'50%'}),
                html.Div(id='wordcloud-figures',className='image',children=[ 
                    html.H3("Does Trump's word choice differ between classes?"),
                    html.H4('Most Frequent Words In Tweets - by Stock +/- Class'),
                    html.Img(id='wordcloud-top-words',className='image',src=wordclouds_top_words, 
                width=500,style={'padding':'2%'}),
                html.H4('Most Frequent Words Unique to Each Class'),
                html.Img(id='wordcloud-unque-words',className='image',src=wordclouds_unique_words,
                width=500,style={'padding':'2%'})
                ])
            ]),
        ]),

        html.Div(id='3_stock_market_data', children= [ 
            html.H1('MODELING THE S&P 500'),
            html.H3('Model 1: Predicting Price Using Price Alone'),
            html.Div(id='model_1_results', children=[
                dcc.Graph(figure=model1_train_test_fig),#fig_price),
                dcc.Markdown(className='quoted_tweet',children=txt_model1_summary,style={'width':'30%',
                'ha':'center'}),#md_model_details_stocks),
                html.Img(src=img_model1_history),
                dcc.Graph( id='fig_model1_results',figure=fig_model1)
            ])
        ]), # stock_market_data children,
                
        # html.Div(id='tech_indicators',children=[
        # html.H2("Modeling Stock Price + Technical Indicators"),
        # dcc.Graph(id='indicators_figure', figure = fig_indicators),
        # dcc.Markdown(md_tech_indicators),
        # html.H3('Model Results'),
        # dcc.Graph(figure=fig_model2)
        # ])
    

]
)

if __name__ == '__main__':
    app.run_server(debug=True)