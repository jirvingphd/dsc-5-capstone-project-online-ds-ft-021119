# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



import functions_combined_BEST as ji
from functions_combined_BEST import plotly_true_vs_preds_subplots,plotly_time_series


## Load in text to display
with open ('assets/text/intro.txt','r') as f:
    md_intro = f.read() 

with open ('assets/text/model_details.txt','r') as f:
    md_model_details=f.read()

with open('assets/text/technical_indicators.txt','r') as f:
    md_tech_indicators = f.read()



import pandas as pd
df_model1 = pd.read_csv('assets/model_1/df_model.csv',index_col=0,parse_dates=True)
df_model2 = pd.read_csv('assets/model_2/df_model.csv',index_col=0,parse_dates=True)

fig_model1 = plotly_true_vs_preds_subplots(df_model1, show_fig=False)
fig_model2 = plotly_true_vs_preds_subplots(df_model1,show_fig=False)
# df_dict[3] = pd.read_csv('assets/model_3/df_model.csv',index_col=0,parse_dates=True)


## Open file firectory 
with open('assets/filename_directory.json', 'r+') as f:
    import json
    file_dict_json = f.read()
    file_dict = json.loads(file_dict_json)
    
# import pandas as pd
# # df_models_dict = {}
# fig_dict={}
# for k,model_dir in file_dict.items():

# # for model in ['model_0A','model_0B','model_1','model_2','model_3']:
#     if 'model' in k:
#         fig_dict[k]={}
#         print(f'[i] found "model" in {k}')
#     else:
#         print(f'\n - skipping {k}')
#         continue

#     # for res in ['df_model','df_results']:
#     if 'df_model' in model_dir.keys():
#         print(f'\n{k}')
#         try:
#             filename = model_dir['df_model']
#             temp_df = pd.read_csv(filename,index_col=0,parse_dates=True)
#             # df_models_dict[model][res] = temp_df
#             fig_dict[model]['true_vs_preds'] = ji.plotly_true_vs_preds_subplots(temp_df)
#         except:
#             print(f'\n[!]{k}: "model_dir" not found.')
#     else:
#         print(f'\n[!] df_model not found in {k}')
        # model_dict[model]['df_results'] = pd.read_excel()

# # fig_model1 = ji.plotly_true_vs_preds_subplots(df_models_dict['model_1']['df_model'])
# # fig_model2 = ji.plotly_true_vs_preds_subplots(df_models_dict['model_2']['df_model'])
# # fig_model3 = ji.plotly_true_vs_preds_subplots(df_models_dict['model_3']['df_model'])
# ## Load in data
# stock_df_filename = '_stock_df_with_technical_indicators.csv'
# stock_df = ji.load_processed_stock_data_plotly(stock_df_filename)

fig_price = ji.plotly_time_series(stock_df,y_col='price', as_figure=True,show_fig=False)
# fig_indicators = ji.plotly_technical_indicators(stock_df, as_figure=True, show_fig=False)
# # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)





app.layout = html.Div(id='main_container',children=[

    # html.Div(id='header',children=[ #main child1

    #     html.H1(children="Predicting Stock Market Fluctuations With Donald Trump's Tweets"),
    #     html.P(id='my_name', children=dcc.Markdown('''
    #     James M. Irving, Ph.D.

    #     https://github.com/jirvingphd

    #     [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)

    #     '''))
    #     ]
    #     ),
    
    #     html.Div(
    #     id='intro',children=[ #main child2
    #     dcc.Markdown(md_intro)]
    #     ),
        
        html.Div(id='tweets',children= [   # main child 3
            html.H2(children="EXPLORING TRUMP'S TWEETS"),
            html.Div(children=[ dcc.Input(id='search',value='russia',type='text') ]
            ),
        html.H3('Visuals from Twitter NLP anaylsis'),
        ]),

        html.Div(id='model_details',children=[

            html.H2('Modeling Stock Price Alone'),
            dcc.Graph(figure=fig_price),
            dcc.Markdown(md_model_details),
            html.H3('Model Details and Results'),
            dcc.Graph(fig_model1),
            dcc.Markdown('> Summary Goes Here')
            ]),


        html.Div(id='tech_indicators',children=[
            html.H2("Modeling Stock Price + Technical Indicators"),
            dcc.Graph(id='indicators_figure', figure = fig_indicators),
            dcc.Markdown(md_tech_indicators),
            html.H3('Model Results'),
            dcc.Graph(figure=fig_model2)
            ]
        )

            
    ]
)

        
    

if __name__ == '__main__':
    app.run_server(debug=False)

####
# @app.callback(Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')])