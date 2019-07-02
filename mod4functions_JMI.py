def quick_table(tuples, col_names=None, caption =None,display_df=True):
    """Accepts a bigram output tuple of tuples and makes captioned table."""
    import pandas as pd
    from IPython.display import display
    if col_names == None:
    
        df = pd.DataFrame.from_records(tuples)
        
    else:
        
        df = pd.DataFrame.from_records(tuples,columns=col_names)
        dfs = df.style.set_caption(caption)
        
        if display_df == True:
            display(dfs)
            
    return df

def compare_word_cloud(text1,label1,text2,label2):
    """Compares the wordclouds from 2 sets of texts"""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud1 = WordCloud(max_font_size=80, max_words=200, background_color='white').generate(' '.join(text1))
    wordcloud2 = WordCloud(max_font_size=80, max_words=200, background_color='white').generate(' '.join(text2))


    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,15))
    ax[0].imshow(wordcloud1, interpolation='bilinear')
    ax[0].set_aspect(1.5)
    ax[0].axis("off")
    ax[0].set_title(label1, fontsize=20)

    ax[1].imshow(wordcloud2, interpolation='bilinear')
    ax[1].set_aspect(1.5)
    ax[1].axis("off")
    ax[1].set_title(label2, fontsize=20)

    fig.tight_layout()
    return fig,ax

def transform_image_mask_white(val):
    """Will convert any pixel value of 0 (white) to 255 for wordcloud mask."""
    if val==0:
        return 255
    else:  
        return val

def open_image_mask(filename):
    import numpy as np
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    mask = np.array(Image.open(filename))
    







# Define get_tags_ats to accept a list of text entries and return all found tags and ats as 2 series/lists
def get_tags_ats(text_to_search,exp_tag = r'(#\w*)',exp_at = r'(@\w*)', output='series',show_counts=False):
    """Accepts a list of text entries to search, and a regex for tags, and a regex for @'s.
    Joins all entries in the list of text and then re.findsall() for both expressions.
    Returns a series of found_tags and a series of found_ats.'"""
    import re
    import pandas as pd
    # Create a single long joined-list of strings
    text_to_search_combined = ' '.join(text_to_search)
        
    # print(len(text_to_search_combined), len(text_to_search_list))
    found_tags = re.findall(exp_tag, text_to_search_combined)
    found_ats = re.findall(exp_at, text_to_search_combined)
    
    if output.lower() == 'series':
        found_tags = pd.Series(found_tags, name='tags')
        found_ats = pd.Series(found_ats, name='ats')
        
        if show_counts==True:
            print(f'\t{found_tags.name}:\n{found_tags.value_counts()} \n\n\t{found_ats.name}:\n{found_ats.value_counts()}')
                
    if (output.lower() != 'series') & (show_counts==True):
        raise Exception('output must be set to "series" in order to show_counts')
                       
    return found_tags, found_ats


def clean_text(series,is_tokens=False,return_tokens=False, urls=True, hashtags=True, mentions=True, remove_stopwords=True, verbose=False):
    """Accepts a series/df['column'] and tokenizes, removes urls, hasthtags, and @s using regex before tokenizing and removing stopwrods"""
    import pandas as pd
    import re, nltk
    from nltk.corpus import stopwords
    
    series_cleaned=series.copy()
    
    # Remove URLS
    if urls==True:
        urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")
        series_cleaned = series_cleaned.apply(lambda x: urls.sub(' ', x))
            
        if verbose==True:
            print('URLs removed...')
            
    # Remove hashtags
    if hashtags==True:
        hashtags = re.compile(r'(\#\w*)')
        series_cleaned = series_cleaned.apply(lambda x: hashtags.sub(' ', x))
        
        if verbose==True:
            print('Hashtags removed...')
    
    # Remove mentions
    if mentions==True:
        mentions = re.compile(r'(\@\w*)')
        series_cleaned = series_cleaned.apply(lambda x: mentions.sub(' ',x))

        if verbose==True:
            print('Mentions removed...')
    
    
    # Regexp_tokenize stopped words (to keep contractions)
    if is_tokens==False:
        pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
        series_cleaned = series_cleaned.apply(lambda x: nltk.regexp_tokenize(x,pattern))
        if verbose==True:
            print('Text regexp_tokenized...\n')
    
    
    # Filter Out Stopwords
    stopwords_list = []
    from nltk.corpus import stopwords
    import string
    
    # Generate Stopwords List
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['http','https','...','``','co','“','’','‘','”',
                       'rt',"n't","''","RT",'u','s',"'s",'?']#,'@','#']
    stopwords_list += [0,1,2,3,4,5,6,7,8,9]
    stopwords_list +=['RT','rt',';']
     
    if remove_stopwords==True:
        series_cleaned = series_cleaned.apply(lambda x: [w.lower() for w in x if w.lower() not in stopwords_list])
        # for s in range(len(series_cleaned)):
        #     text =[]
        #     text_stopped = []
        #     text = series_cleaned[s]
        #     text_stopped = [x.lower() for x in text if x.lower() not in stopwords_list]
        #     series_cleaned[s]= text_stopped
        
        if verbose==True:
            print('Stopwords removed...')
       
    if return_tokens==False:
        series_cleaned = series_cleaned.apply(lambda x: ' '.join(x))
    
    print('\n')
    return series_cleaned

def train_test_val_split(X,y,test_size=0.20,val_size=0.1):
    """Performs 2 successive train_test_splits to produce a training, testing, and validation dataset"""
    from sklearn.model_selection import train_test_split

    first_split_size = test_size + val_size
    second_split_size = val_size/(test_size + val_size)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=first_split_size)

    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=second_split_size)

    return X_train, X_test, X_val, y_train, y_test, y_val

def plot_keras_history(history):
    """Plots the history['acc','val','val_acc','val_loss']"""
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    x = range(1,len(acc)+1)
    
    fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
    ax[0].plot(x, acc,'b',label='Training Acc')
    ax[0].plot(x, val_acc,'r',label='Validation Acc')
    ax[0].legend()
    ax[1].plot(x, loss,'b',label='Training Loss')
    ax[1].plot(x, val_loss, 'r', label='Validation Loss')
    ax[1].legend()
    plt.show()
    return fig, ax


def plot_keras_history_custom(history,metrics=[('acc','loss'),('val_acc','val_loss')], figsize=(8,6)):
    """Plots the history['acc','val','val_acc','val_loss']"""
    plot_dict = {}
    
    import matplotlib.pyplot as plt
    for i,metric_tuple in enumerate(metrics):
         
        plot_dict[i] = {}
        
        for metric in metric_tuple:
            plot_dict[i][metric]= history.history[metric]
                       

    x_len = len(history.history[metrics[0][0]])
    x = range(1,x_len)
    
    fig,ax = plt.subplots(nrows=metrics.shape[0], ncols=1, figsize=figsize)
    
    for p in plot_dict.keys():
        
        for k,v in plot_dict[p]:
            ax[p].plot(x, plot_dict[p][v], label=k)
            ax[p].legend()
                    
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_auc_roc_curve(y_test, y_test_pred):
    """ Takes y_test and y_test_pred from a ML model and plots the AUC-ROC curve."""
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    auc = roc_auc_score(y_test, y_test_pred[:,1])

    FPr, TPr, _  = roc_curve(y_test, y_test_pred[:,1])
    plt.plot(FPr, TPr,label=f"AUC for CatboostClassifier:\n{round(auc,2)}" )

    plt.plot([0, 1], [0, 1],  lw=2,linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()



def display_random_tweets(df_tokenize,n=5 ,display_cols=['content','text_for_vectors','tokens'], group_labels=[],verbose=True):
    """Takes df_tokenize['text_for_vectors']"""
    import numpy as np
    import pandas as pd 
    from IPython.display import display
    if len(group_labels)==0:

        group_labels = display_cols

    
    random_tweets={}
    # Randomly pick n indices to display from specified col
    idx = np.random.choice(range(len(df_tokenize)), n)
    
    for i in range(len(display_cols)):
        
        group_name = str(group_labels[i])
        random_tweets[group_name] ={}

        # Select column data
        df_col = df_tokenize[display_cols[i]]
        

        tweet_group = {}
        tweet_group['index'] = idx
        
        chosen_tweets = df_col[idx]
        tweet_group['text'] = chosen_tweets

        # print(chosen_tweets)
        if verbose>0:
            with pd.option_context('max_colwidth',300):
                df_display = pd.DataFrame.from_dict(tweet_group)
                display(df_display.style.set_caption(f'Group: {group_name}'))


        random_tweets[group_name] = tweet_group
        
        # if verbose>0:
              
        #     for group,data in random_tweets.items():
        #         print(f'\n\nRandom Tweet for {group:>.{300}}:\n{"---"*20}')

        #         df = random_tweets[group]
        #         display(df)
    if verbose==0:
        return random_tweets
    else:
        return


def reload(mod):
    """Reloads the module from file."""
    from importlib import reload
    import sys
    print(f'Reloading...')
    return  reload(mod)


def process_df_full(df_full, raw_col='content_raw', fill_content_col='content',force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column."""
    import re
    import pandas as pd
    
    if force==False:
        if fill_content_col in df_full.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')


    # # create 'content_raw' column from 'content'
    # df_full[fill_content_col] = df_full['content'].copy()


    # Add has_RT and starts_RT columns
    # Creating columns for tweets that `has_RT` or `starts_RT`
    df_full['has_RT']=df_full[raw_col].str.contains('RT')
    df_full['starts_RT']=df_full[raw_col].str.contains('^RT')


    ## FIRST REMOVE THE RT HEADERS

    # Remove `RT @Mentions` FIRST:
    re_RT = re.compile(r'RT [@]?\w*:')

    # raw_col =  'content_raw'
    check_content_col =raw_col
    fill_content_col = fill_content_col

    df_full['content_starts_RT'] = df_full[check_content_col].apply(lambda x: re_RT.findall(x))
    df_full[fill_content_col] =  df_full[check_content_col].apply(lambda x: re_RT.sub(' ',x))


    ## SECOND REMOVE URLS
    # Remove urls with regex
    urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")

    check_content_col = fill_content_col
    fill_content_col = fill_content_col

    # df_full['content_urls'] = df_full[check_content_col].apply(lambda x: urls.findall(x))
    df_full[fill_content_col] =  df_full[check_content_col].apply(lambda x: urls.sub(' ',x))

    ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
    df_full['content_min_clean'] =  df_full[fill_content_col]




    ## REMOVE AND SAVE HASHTAGS, MENTIONS
    # Remove and save Hashtags
    hashtags = re.compile(r'\#\w*')

    check_content_col = fill_content_col
    fill_content_col = fill_content_col

    df_full['content_hashtags'] =  df_full[check_content_col].apply(lambda x: hashtags.findall(x))
    df_full[fill_content_col] =  df_full[check_content_col].apply(lambda x: hashtags.sub(' ',x))


    # Remove and save mentions (@)'s
    mentions = re.compile(r'\@\w*')

    check_content_col = fill_content_col
    fill_content_col = fill_content_col

    df_full['content_mentions'] =  df_full[check_content_col].apply(lambda x: mentions.findall(x))
    df_full[fill_content_col] =  df_full[check_content_col].apply(lambda x: mentions.sub(' ',x))

    return df_full



def load_orig_dataset(root_dir = 'russian-troll-tweets/', ext='.csv'):
    """Accepts a root_dir, finds all files that end with ext and loads into a dataframe."""
    import os
    import pandas as pd
    # root_dir = 'russian-troll-tweets/'
    # os.listdir('russian-troll-tweets/')
    filelist = [os.path.join(root_dir,file) for file in os.listdir(root_dir) if file.endswith(ext)]
    print(f'Loading {len(filelist)} files into dataframe...')
        # Vertically concatenate 
    df = pd.DataFrame()
    for file in filelist:
        df_new = pd.read_csv(file)
        df = pd.concat([df,df_new], axis=0)
    # df.info()

    return df

def ask_user_to_save(df, filename=None,encoding=None, ask=True, skip_saving=False, overwrite=False):
    """Asks user to save df as filename. If no filename specified: filename ='saved_df.csv'
    Set ask=False to save without asking."""
    import os, warnings

    if type(df)=='string':
        raise Exception('First input must be the dataframe to be saved.')
    if skip_saving==True:
        return print(f'Since skip_loading=True, no file was save.')

    if filename==None:
        filename='saved_df.csv'
    
    if ask==True:
        ans = input('Would you like to save the df to a .csv?(y/n):')
    else:
        ans = 'y'
    
    # If ans to save =='y'
    if ans.lower()=='y':

        # Check if the file already exists
        if filename in os.listdir():
        
            if overwrite==False:
                # raise Exception(f"{filename} already exists.")
                return warnings.warn(f"{filename} already exists.")

            if overwrite==True:
                warnings.warn(f"Overwriting {filename}.")

        df.to_csv(filename)
        print(f'{filename} successfully saved.')
    else:
        print('Ok. No file was saved. ')
        

def ask_user_to_load(filename, load_as_global = True ,ask=True, skip_loading=False, index_col=0, encoding=None):
    """Asks user to save df as filename. If no filename specified: filename ='saved_df.csv'
    Set ask=False to save without asking."""
    import os
    import pandas as pd

    
    if skip_loading==True:
        return print(f'Since skip_loading=True, no file was loaded.')
        

    if ask==True:
        ans = input('Would you like to load {filename} to a datafrane?(y/n):')

    else:
        ans = 'y'
        

    # If ans to load =='y'
    if ans.lower()=='y':
        
        if load_as_global == True:
            global df_
            df_ = pd.read_csv(filename, encoding=encoding,index_col=index_col) 
            print(f'{filename} loaded as global variable: "df_"')
            pass
        else:
            df_ = pd.read_csv(filename, encoding=encoding,index_col=index_col) 
            return df_
    else:
        return print('Ok. No file was loaded.')
        

def run_all_checkpoint(skip=False):
    ans = input('Continue running all?(y/n):')
    if ans.lower()=='y':
        return print('OK. Continuing to run...')
    else:
        raise Exception('User requested to stop running.')


## TO CHECK FOR STRINGS IN BOTH DATASETS:
def check_dfs_for_exp_list(df_controls, df_trolls, list_of_exp_to_check):
    df_resample = df_trolls
    for exp in list_of_exp_to_check:
    #     exp = '[Pp]eggy'
        print(f'For {exp}:')
        print(f"\tControl tweets: {len(df_controls.loc[df_controls['content_min_clean'].str.contains(exp)])}")
        print(f"\tTroll tweets: {len(df_resample.loc[df_resample['content_min_clean'].str.contains(exp)])}\n")
              
# list_of_exp_to_check = ['[Pp]eggy','[Mm]exico','nasty','impeachment','[mM]ueller']
# check_dfs_for_exp_list(df_controls, df_resample, list_of_exp_to_check=list_of_exp_to_check)


def get_group_texts_tokens(df_small, groupby_col='troll_tweet', group_dict={0:'controls',1:'trolls'}, column='content_stopped'):
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    text_dict = {}
    for k,v in group_dict.items():
        group_text_temp = df_small.groupby(groupby_col).get_group(k)[column]
        group_text_temp = ' '.join(group_text_temp)
        group_tokens = regexp_tokenize(group_text_temp, pattern)
        text_dict[v] = {}
        text_dict[v]['tokens'] = group_tokens
        text_dict[v]['text'] =  ' '.join(group_tokens)
            
    print(f"{text_dict.keys()}:['tokens']|['text']")
    return text_dict



def check_df_groups_for_exp(df_full, list_of_exp_to_check, check_col='content_min_clean', groupby_col='troll_tweet', group_dict={0:'Control',1:'Troll'}):      
    """Checks `check_col` column of input dataframe for expressions in list_of_exp_to_check and 
    counts the # present for each group, defined by the groupby_col and groupdict. 
    Returns a dataframe of counts."""
    
    list_of_results = []      

    header_list= ['Term']
    [header_list.append(x) for x in group_dict.values()]
    list_of_results.append(header_list)
    
    for exp in list_of_exp_to_check:
        curr_exp_list = [exp]
        
        for k,v in group_dict.items():
            df_group = df_full.groupby(groupby_col).get_group(k)
            curr_group_count = len(df_group.loc[df_group[check_col].str.contains(exp)])
            curr_exp_list.append(curr_group_count)
        
        list_of_results.append(curr_exp_list)
        
    df_results = bs.list2df(list_of_results, index_col='Term')
    return df_results


###########################################################################

def plot_fit_cloud(troll_cloud,contr_cloud,label1='Troll',label2='Control'):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(18,18))

    ax[0].imshow(troll_cloud, interpolation='gaussian')
    # ax[0].set_aspect(1.5)
    ax[0].axis("off")
    ax[0].set_title(label1, fontsize=40)

    ax[1].imshow(contr_cloud, interpolation='bilinear',)
    # ax[1].set_aspect(1.5)
    ax[1].axis("off")
    ax[1].set_title(label2, fontsize=40)
    plt.tight_layout()
    return fig, ax