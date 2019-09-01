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


## Load in data
stock_df_filename = 'data/_stock_df_with_technical_indicators.csv'
stock_df = ji.load_processed_stock_data_plotly(stock_df_filename)

twitter_df = pd.read_csv('data/_twitter_df_with_stock_price.csv',
index_col=0,parse_dates=True)

df_model1 = pd.read_csv('assets/model_1/df_model.csv',index_col=0,parse_dates=True)
df_model2 = pd.read_csv('assets/model_2/df_model.csv',index_col=0,parse_dates=True)

fig_model1 = ji.plotly_true_vs_preds_subplots(df_model1, show_fig=False)
fig_model2 = ji.plotly_true_vs_preds_subplots(df_model1,show_fig=False)


fig_price = ji.plotly_time_series(stock_df,y_col='price', as_figure=True,show_fig=False)
fig_indicators = ji.plotly_technical_indicators(stock_df, as_figure=True, show_fig=False)
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

md_example_tweet_forms = ji.display_same_tweet_diff_cols(twitter_df, columns=['content','content_min_clean',
'cleaned_stopped_content'], for_dash=True)#,'cleaned_stopped_tokens'


##
nlp_df = pd.read_csv('data/_twitter_df_only_delta_price_pos_neg.csv',
index_col=0,parse_dates=True)

get_floats =nlp_df['cleaned_stopped_lemmas'].apply(lambda x: isinstance(x,float))
nlp_df =nlp_df[~get_floats]


twitter_df_groups,twitter_group_text = ji.get_group_texts_for_word_cloud(nlp_df, 
                                                                      text_column='cleaned_stopped_tokens', 
                                                                      groupby_column='delta_price_class')


fig_wordcloud =ji.compare_word_clouds(text1=twitter_df_groups['pos']['joined'],
                       label1='Stock Market Increased',
                       text2= twitter_df_groups['neg']['joined'],
                       label2='Stock Market Decreased',
                       twitter_shaped = True, verbose=1,
                       suptitle_y_loc=0.75,
                       suptitle_text='Most Frequent Words by Stock Price +/- Change',
                       save_file=False,
                       wordcloud_cfg_dict={'collocations':True},
                       filepath_folder='',**{'subplot_titles_fontdict':{'fontsize':26,'fontweight':'bold'},
                        'suptitle_fontdict':{'fontsize':40,'fontweight':'bold'},
                         'group_colors':{'group1':'green','group2':'red'},
                        })
app = dash.Dash(__name__)# external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(id='main_header',className='main_header', children="Predicting Stock Market Fluctuations With Donald Trump's Tweets"),

    html.P(id='my_name', children=dcc.Markdown('''
        James M. Irving, Ph.D.

        https://github.com/jirvingphd

        [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)

        ''')),
    
        html.Div(id='1_intro',
        children=[ #main child2
        dcc.Markdown(md_intro)
        ]),

        html.Div(id="2_NLP", children=[
            dcc.Markdown(md_nlp_intro),
            html.Div(id='show_tweet_forms',
            children=[
                dcc.Markdown(ji.display_same_tweet_diff_cols(twitter_df,for_dash=True))
                ]),
            html.Div(id='nlp_data',children=[
                dcc.Markdown(md_nlp_data_intro),
                dcc.Graph(figure=ji.plotly_price_histogram(twitter_df)),
                dcc.Graph(figure=ji.plotly_pie_chart(twitter_df,show_fig=False))
            ]),
        ]),

        html.Div(id='3_stock_market_data', children= [ 
            html.H2('MODELING THE STOCK MARKET'),
            html.H3('Model 1: Predicting Price Using Price Alone'),
            html.Div(id='model_1_results', children=[
                dcc.Graph(figure=fig_price),
                dcc.Markdown(className='quoted_tweet',children=md_model_details_stocks),
                dcc.Graph( id='fig_model1_results',figure=fig_model1)
            ])
        ]), # stock_market_data children,
                
        html.Div(id='tech_indicators',children=[
        html.H2("Modeling Stock Price + Technical Indicators"),
        dcc.Graph(id='indicators_figure', figure = fig_indicators),
        dcc.Markdown(md_tech_indicators),
        html.H3('Model Results'),
        dcc.Graph(figure=fig_model2)
        ])
    

]
)

if __name__ == '__main__':
    app.run_server(debug=True)