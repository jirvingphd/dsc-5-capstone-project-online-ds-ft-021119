## For each model need
#1) df_model for true_vs_preds_subplots
#2) df_results for tables (if cant use html export) (includes time_Shifted_metrics)
#3) df_shifted with traces for all shifts (maybe plot)
#4) save normal true_vs_pred_subplots 
# def check_if_file_exists(filename,delim='/'):
#     import os
#     filename_parts = filename.split(delim)
# #     num_folders = len(filenameparts)-1
    
# #     check_path =filename_parts[0]
# #     for i in range(1,len(filename_parts)+1):
#     check_path = '/'.join(filename_parts[:-1])
#     current_files = os.listdir(check_path)
#     if filename_parts[-1] in current_files:
#         return True
#     else:
# #         return False
# def auto_filename_time(prefix='model',sep='_',timeformat='%m-%d-%Y_%I%M%p'):
#     """Generates a filename with a  base string + sep+ the current datetime formatted as timeformat."""
#     if prefix is None:
#         prefix=''
#     timesuffix=get_time(timeformat=timeformat, filename_friendly=True)
#     filename = f"{prefix}{sep}{timesuffix}"
#     return filename

def get_now(timeformat='%m-%d-%Y %T',raw=False,filename_friendly= False,replacement_seperator='-'):
    """Gets current time in local time zone. 
    if raw: True then raw datetime object returned without formatting.
    if filename_friendly: replace ':' with replacement_separator """
    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone

    now_utc = datetime.now(timezone('UTC'))
    now_local = now_utc.astimezone(get_localzone())

    if raw == True:
        return now_local
    
    else:
        now = now_local.strftime(timeformat)

    if filename_friendly==True:
        return now.replace(':',replacement_seperator).lower()
    else:
        return now

def save_word2vec(word2vec_model, file_dict=None, filename=None,parms_dict=None):
    if filename is None and file_dict is None:
        raise Exception('Must provide either filename or file_dict.')

    elif filename is None:
        filename = file_dict['word2vec']['base_filename']
    else:
        filename = filename
    
    if parms_dict is None:
        params = {}
        params['text_column'] ='?'
        params['window'] = word2vec_model.window
        params['min_count'] = word2vec_model.min_count
        params['epochs']=word2vec_model.epochs
        params['sg'] = word2vec_model.sg
        params['hs'] = word2vec_model.hs
        params['negative'] = word2vec_model.negative
        params['ns_exponent'] = word2vec_model.ns_exponent

    import pickle         
    with open(filename, 'wb') as f:
        pickle.dump(word2vec_model,f)

    print(f"[i] word2vec model saved as {filename}")


def load_word2vec(filename = None, file_dict=None):
    if filename is None and file_dict is None:
        raise Exception('Must provide either filename or file_dict.')

    elif filename is None:
        filename = file_dict['word2vec']['base_filename']
    else:
        filename = filename
    print(f'[i] Loading Word2Vec model from {filename}')
    import pickle 
    
    with open(filename,'rb') as f:
        word2vec_model = pickle.load(f)
    return word2vec_model


def load_filename_directory(json_filename= 'data/test_filename_dictionary.json',show_menu=True):
    import functions_combined_BEST as ji       
    import os
    import qgrid
    from IPython.display import display
    if os.path.isfile(json_filename):# and overwrite==False
        with open(json_filename,'r') as f:
            import json
            file_dict = json.loads(f.read())

        print(f"[i] filename_directory loaded from {json_filename}.")

        if show_menu:
            # qgrid.show_grid()
            display(ji.display_dict_dropdown(file_dict))
        return file_dict

    else:
        msg = f'[!] json_filename not found.'
        print(msg)
        raise Exception('File not found')

def def_filename_dictionary(json_filename = 'data/filename_dictionary.json',show_dict=True,
                              create_folders=True,load_prior=True,save_directory=False):#,overwrite=False):
    """ Creates a file_dict with either the model_# or twitter_df/stock_df as kwds.
    Value is the filename (with extension) that should be used for saving/loading the file.
    filename_dictionary is reutrned and also saved to disk.
    
    Ex: For using with save_model_dfs():
        >>save_model_dfs(file_dict,'model1',
                       df_model=df_model1,
                      df_results=dfs_results1,
                      df_shifted=df_shifted1)
                      
    Ex: Using on its own:
        file_dict = define_filename_dictionary()
        >> twitter_df.read_csv(file_dict['twitter_df']['twitter_df_pre_stock_price'])
    """           
    import functions_combined_BEST as ji       
    import os
    if load_prior ==True and os.path.isfile(json_filename):# and overwrite==False
        with open(json_filename,'r') as f:
            import json
            file_dict = json.loads(f.read())
            print(f"[i] filename_directory loaded from {json_filename}.")
    else:
        file_dict = {}
        
    if 'file_directory' not in file_dict.keys():
        file_dict['file_directory']={}
        file_dict['file_directory']['history'] = ''
        file_dict['file_directory']['filename'] = json_filename
                  
    ## Create entries for #'d models
    for i in ['0A','0B']:
        
        if f"model_{i}" not in file_dict.keys():
            file_dict[f"model_{i}"] ={}
                    
            file_dict[f'model_{i}'] = {'base_filename':f'models/NLP/nlp_model{i}',
                                   'output_filenames':{'model':'',
                                                       'weights':'',
                                                       'excel':'',
                                                       'params':''}}
            file_dict[f'model_{i}']['model_summary'] =f"results/model{i}/model{i}_summary" 
            file_dict[f'model_{i}']['fig_conf_mat'] = f'results/model{i}/model{i}_conf_matrix'
            file_dict[f'model_{i}']['fig_conf_mat.ext'] = f'results/model{i}/model{i}_conf_matrix.png'
            file_dict[f'model_{i}']['fig_keras_history'] = f'results/model{i}/model{i}_keras_history'
            file_dict[f'model_{i}']['fig_keras_history.ext'] = f'results/model{i}/model{i}_keras_history.png'
        
        
    for i in ['xgb','1','2','3']:
        
        if f"model_{i}" not in file_dict.keys():

            file_dict[f"model_{i}"] ={}
            file_dict[f'model_{i}'] = {'base_filename':f'models/stocks/model{i}_',
                               'output_filenames':{'model':'',
                                                   'weights':'',
                                                   'excel':'',
                                                   'params':''}}

            file_dict[f'model_{i}']['df_model'] = f'results/model{i}/model{i}_df_model_true_vs_preds.csv'
            file_dict[f'model_{i}']['df_results'] = f'results/model{i}/model{i}_df_results.xlsx' 
            file_dict[f'model_{i}']['df_shifted'] = f'results/model{i}/model{i}_df_shifted.csv'

            file_dict[f'model_{i}']['fig'] = f'results/model{i}/model{i}_true_vs_preds'
            file_dict[f'model_{i}']['fig.ext'] = f'results/model{i}/model{i}_true_vs_preds.png'
            file_dict[f'model_{i}']['fig_shifted'] = f'results/model{i}/model{i}_true_vs_preds_shifted'
            file_dict[f'model_{i}']['fig_shifted.ext'] = f'results/model{i}/model{i}_true_vs_preds_shifted.png'
            
            file_dict[f'model_{i}']['fig_keras_history.ext'] = f'results/model{i}/model{i}_keras_history.png'
            file_dict[f'model_{i}']['fig_keras_history'] = f'results/model{i}/model{i}_keras_history'
            file_dict[f'model_{i}']['model_summary'] =f"results/model{i}/model{i}_summary" 



    ## ADD NAMES OF OTHER FILES
    if 'twitter_df' not in file_dict.keys():
        file_dict['twitter_df']={
            'raw_tweet_file':'data/trumptwitterarchive_export_iphone_only__08_23_2019.csv',
            'twitter_df_pre_stock_price':'data/_twitter_df_before_stock_price.csv',
            'twitter_df_post_stock_price':'data/_twitter_df_with_stock_price.csv'}
        
    if 'stock_df' not in file_dict.keys():

        file_dict['stock_df'] ={'raw_text_file':'data/IVE_bidask1min_08_23_2019.txt',
                                'raw_csv_file':'data/IVE_bidask1min_08_23_2019.csv',
                                'stock_df_with_indicators':'data/_stock_df_with_technical_indicators.csv',
                               'stock_df_input_model1':'data/stock_df_for_model1.csv',
                               'stock_price_for_twitter_df':"data/IVE_bidask1min_08_23_2019.csv"}

    if 'nlp_figures' not in file_dict.keys():
        file_dict['nlp_figures']={'word_clouds_compare':'figures/wordcloud_top_words_by_delta_price.png',
                                 'word_clouds_compare_unique':'figures/wordcloud_unique_words_by_delta_price.png',
                                  'freq_dist_plots':'figures/freq_dist_plots_by_delta_price.png'
                                 }
        
    if 'nlp_model_for_predictions' not in file_dict.keys():

        file_dict['nlp_model_for_predictions'] = {'base_filename':'models/best_final/nlp_classifier_model',
                                                 'output_filenames':{'model':'','weights':'',
                                                                     'excel':'','params':''}}
        
    if 'word2vec' not in file_dict.keys():
        file_dict['word2vec'] = {'base_filename':'models/word2vec/word2vec_model.pkl'}
    
    if 'df_combined' not in file_dict.keys():
        file_dict['df_combined'] = {}
        file_dict['df_combined']['pre_nlp'] = 'data/_combined_stock_data_raw_tweets.csv'
        file_dict['df_combined']['post_nlp'] = 'data/_combined_stock_data_plus_nlp.csv'
        file_dict['df_combined']['with_preds'] = 'data/_combined_stock_data_with_tweet_preds.csv'
    
    if show_dict:
        ji.display_dict_dropdown(file_dict)
        
    if save_directory:
        # check if file already exists and raise errror if no auto_increment_name
#         if full_filename in current_files and auto_increment_name==False:
        ## save the directory json file
        with open(json_filename,'w') as f:
            import json
            json_file_dict = json.dumps(file_dict)
            f.write(json_file_dict)
            print(f"[i] filename_directory saved to {json_filename}.")
            print('\t - use `update_file_directory(file_dict)` to update file.')

    if create_folders:
        print(f"[i] creating all required folders...")
        for k,v in file_dict.items():
            if isinstance(v,dict):
                 for k2,v2 in file_dict[k].items():
                    if '/' in v2:
                        ji.create_required_folders(v2,verbose=1)

            elif '/' in v:
                ji.create_required_folders(v,verbose=1)



    return file_dict

# def load_file_directory(file_dict_filename='')

def update_file_directory(file_dict):
    import functions_combined_BEST as ji
    file_dir = file_dict['file_directory']['filename']
    
    with open(file_dir,'w+') as f:
        import json
        json_file_dict = json.dumps(file_dict)
        f.write(json_file_dict)
        f.seek(0)
    print(f"[i]filename_directory updated, filename='{file_dir}'")
    

    # export = [df_model1,dfs_results,df_shifted]
# file_dict = define_filename_dictionary(show_dict=True,create_folders=True)