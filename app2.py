# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

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
cf.go_offline()
import functions_combined_BEST as ji


# Suppress warnings
import warnings
# warnings.filterwarnings('ignore')
dash.Dash(assets_ignore='z_external_stylesheet')
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


## Load in text to display
with open ('assets/text/intro.txt','r') as f:
    md_intro = f.read() 
with open ('assets/text/model_overview.txt','r') as f:
    md_data_overview = f.read()
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
img_conf_mat_NLP_fig = 'assets/images/model0A_conf_matrix.png'
img_keras_history_NLP_fig = 'assets/images/model0A_keras_history.png'
NLP_model_summary = 'assets/model0A/model0A_summary.txt'

## Load in data
stock_df_filename = 'data/_stock_df_with_technical_indicators.csv'
stock_df = ji.load_processed_stock_data_plotly(stock_df_filename)

twitter_df = pd.read_csv('data/_twitter_df_with_stock_price.csv',
index_col=0,parse_dates=True)


## Specify all assets for model 1
df_model1 = pd.read_csv('results/model1/best/model1_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results1 = pd.read_excel('results/model1/best/model1_df_results.xlsx',index_col=0)
img_model1_history = 'assets/images/model1_keras_history.png'

df_train_test1=df_model1[['true_train_price','true_test_price']]
# model1_train_test_fig = ji.plotly_time_series(df_train_test1,as_figure=True,show_fig=False)

with open('results/model1/best/model1_summary.txt','r') as f:
    txt_model1_summary = "```"
    txt_model1_summary+=f.read()
    txt_model1_summary+= "\n```"

fig_model1 = ji.plotly_true_vs_preds_subplots(df_model1, show_fig=False)




## Specify all assets for model 2
df_model2 = pd.read_csv('results/model2/best/model2_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results2 = pd.read_excel('results/model2/best/model2_df_results.xlsx',index_col=0)
img_model2_history = 'assets/images/model2_keras_history.png'

df_train_test2=df_model2[['true_train_price','true_test_price']]
# model2_train_test_fig = ji.plotly_time_series(df_train_test2,as_figure=True,show_fig=False)

with open('results/model2/best/model2_summary.txt','r') as f:
    txt_model2_summary = "```"
    txt_model2_summary+=f.read()
    txt_model2_summary+= "\n```"

fig_model2 = ji.plotly_true_vs_preds_subplots(df_model2, show_fig=False)




## Specify all assets for model 1
df_model3 = pd.read_csv('results/model3/best/model3_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_results3 = pd.read_excel('results/model3/best/model3_df_results.xlsx',index_col=0)
img_model3_history = 'assets/images/model3_keras_history.png'

df_train_test1=df_model3[['true_train_price','true_test_price']]
# model3_train_test_fig = ji.plotly_time_series(df_train_test1,as_figure=True,show_fig=False)

with open('results/model3/best/model3_summary.txt','r') as f:
    txt_model3_summary = "```"
    txt_model3_summary+=f.read()
    txt_model3_summary+= "\n```"

fig_model3 = ji.plotly_true_vs_preds_subplots(df_model3, show_fig=False)

## Specify all assets for model 2
df_modelxgb = pd.read_csv('results/modelxgb/modelxgb_df_model_true_vs_preds.csv',index_col=0,parse_dates=True)
df_resultsxgb = pd.read_excel('results/modelxgb/modelxgb_df_results.xlsx',index_col=0)
df_importance = pd.read_csv('results/modelxgb/df_importance.csv')
importance_fig = df_importance.sort_values(by='weight', ascending=True).iplot(kind='barh',theme='solar',
                                                                    title='Feature Importance',
                                                                    xTitle='Relative Importance<br>(sum=1.0)',
                                                                    asFigure=True)


fig_modelxgb = ji.plotly_true_vs_preds_subplots(df_modelxgb, true_train_col='true_train_price',
                                true_test_col='true_test_price', pred_test_columns='pred_test_price',
                                show_fig=False)

fig_feature_importance = df_importance.sort_values(by='weight', ascending=True).iplot(kind='barh',theme='solar',
                                                                    title='Feature Importance',
                                                                    xTitle='Relative Importance<br>(sum=1.0)',
                                                                    asFigure=True)


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
        dcc.Markdown(md_intro),
        dcc.Markdown(md_data_overview)
        ]),

        html.Div(id="2_NLP", children=[
            html.H1('NATURAL LANGUAGE PROCESSING'),
            dcc.Markdown(md_nlp_intro),
            html.Div(id='example-tweets', className='interactive_output',
            children=[
                dcc.Markdown(id='show-tweet-forms',children=ji.display_same_tweet_diff_cols(twitter_df,for_dash=True)),
                html.Button(id='fetch-tweets',n_clicks=0,children='Fetch New Tweet'),#,style={'fontSize':28}),
                ],                #tyle={'border':'5px solid cornflowerblue'}
                ),
            html.Div(id='nlp_data',children=[
                html.H2('When Trump Tweets, does the market react?'),
                dcc.Graph(figure=fig_price,className='figure'),
                dcc.Markdown(md_nlp_data_intro),
                dcc.Graph(figure=ji.plotly_price_histogram(twitter_df),className='figure'),
                dcc.Graph(figure=ji.plotly_pie_chart(twitter_df,show_fig=False),className='figure',style={'height':'50%'}),
                html.Div(id='wordcloud-figures',className='image',children=[ 
                    html.H3("Do words matter (to the S&P 500)?"),
                    html.H4('Most Frequent Words In Tweets - by Stock +/- Class'),
                    html.Img(id='wordcloud-top-words',className='image',src=wordclouds_top_words, 
                width=500,style={'padding':'2%'}),
                html.H4('Most Frequent Words Unique to Each Class'),
                html.Img(id='wordcloud-unque-words',className='image',src=wordclouds_unique_words,
                width=500,style={'padding':'2%'})
                ])
            ]),
        ]),

        html.Div(id='3_stock_market_price_data', children= [ 
            html.H1('MODELING THE S&P 500'),
            html.H2('Model 1: Predicting Price Using Price Alone'),
            html.Div(id='model_1_results', children=[
                # dcc.Graph(figure=model1_train_test_fig),#fig_price),
                dcc.Markdown(children=txt_model1_summary,style={'width':'30%',
                'ha':'center'}),#md_model_details_stocks),
                html.Img(src=img_model1_history),
                dcc.Graph( id='fig_model1_results',figure=fig_model1,className='figure'),
                dash_table.DataTable(id='df_results_model1',
                columns = [{"name":i, "id":i,'editable':False} for i in df_results1.columns],
                data = df_results1.to_dict('records'),
                style_header={'backgroundColor':'black'},
                style_cell={'backgroundColor':'black','color':'white','textAlign':'center'}
                ),
            ])
        ]), # stock_market_data children,
                
        html.Div(id='4_stock_market_price_indicators_data', children= [ 
            html.H2('Model 2: Predicting Price with Technical Indicators'),
            html.Div(id='model_2_results', children=[
                # dcc.Graph(figure=model2_train_test_fig),#fig_price),
                dcc.Markdown(children=txt_model2_summary,style={'width':'30%',
                'ha':'center'}),#md_model_details_stocks),
                html.Img(src=img_model2_history),
                dcc.Graph( id='fig_model2_results',figure=fig_model2,className='figure'),
                dash_table.DataTable(id='df_results_model2',
                columns = [{"name":i, "id":i,'editable':False} for i in df_results2.columns],
                data = df_results2.to_dict('records'),
                style_header={'backgroundColor':'black'},
                style_cell={'backgroundColor':'black','color':'white','textAlign':'center'}
                )
                ])
        ]), # stock_market_data children,

        html.Div(id='5_stock_market_and_tweets_combined', children= [ 
            html.H2('Model 3: Predicting Price with Technical Indicators'),
            html.Div(id='model_3_results', children=[
                # dcc.Graph(figure=model3_train_test_fig),#fig_price),
                dcc.Markdown(children=txt_model3_summary,style={'width':'30%',
                'ha':'center'}),#md_model_details_stocks),
                html.Img(src=img_model3_history),
                dcc.Graph( id='fig_model3_results',className='figure',figure=fig_model3),
                dash_table.DataTable(id='df_results_model3',
                columns = [{"name":i, "id":i,'editable':False} for i in df_results3.columns],
                data = df_results3.to_dict('records'),
                style_header={'backgroundColor':'black'},
                style_cell={'backgroundColor':'black','color':'white','textAlign':'center'}
                )
                ])
        ]), # stock_market_data children,
                                        
        html.Div(id='6_xgb', children= [ 
            html.H2('Model X: Predicting Price Using Price, Indicators & Tweets'),
            html.Div(id='model_x_results', children=[
                # dcc.Graph(figure=modelxgb_train_test_fig),#fig_price),
                # dcc.Markdown(children=txt_modelxgb_summary,style={'width':'30%',
                # 'ha':'center'}),#md_model_details_stocks),
                # html.Img(src=img_model1_history.__repr__()),
                dcc.Graph( id='fig_modelx_results',figure=fig_modelxgb,className='figure'),
                dcc.Graph(id='feature_importance',figure=fig_feature_importance,className='figure'),
                dash_table.DataTable(id='df_results_modelX',
                columns = [{"name":i, "id":i,'editable':False} for i in df_resultsxgb.columns],
                data = df_resultsxgb.to_dict('records'),
                style_header={'backgroundColor':'black'},
                style_cell={'backgroundColor':'black','color':'white','textAlign':'center'}
                )
            ])
        ]), 

        html.Div(id='summary',children=[
            html.H1('Summary'),
            dcc.Markdown('Final Conclusions')
        ])
]
)

@app.callback(Output(component_id='show-tweet-forms',component_property='children'),
[Input(component_id='fetch-tweets',component_property='n_clicks')])
def get_new_tweets_to_show(n_clicks, twitter_df=twitter_df):

    md_tweets = ji.display_same_tweet_diff_cols(twitter_df,for_dash=True)
    return md_tweets
if __name__ == '__main__':
    app.run_server(debug=True)