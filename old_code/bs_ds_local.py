# from functions_combined_
def display_side_by_side(*args):
    """Display all input dataframes side by side. Also accept captioned styler df object (df_in = df.style.set_caption('caption')
    Modified from Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side"""
    from IPython.display import display_html
    import pandas
    html_str=''
    for df in args:
        if type(df) == pandas.io.formats.style.Styler:
            html_str+= '&nbsp;'
            html_str+=df.render()
        else:
            html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# -*- coding: utf-8 -*-


def reload(mod):
    """Reloads the module from file.
    Example:
    import my_functions_from_file as mf
    # after editing the source file:
    # mf.reload(mf)"""
    from importlib import reload
    import sys
    print(f'Reloading...\n')
    return  reload(mod)


def ihelp(function_or_mod, show_help=True, show_code=True,return_code=False,markdown=True,file_location=False):
    """Call on any module or functon to display the object's
    help command printout AND/OR soruce code displayed as Markdown
    using Python-syntax"""

    import inspect
    from IPython.display import display, Markdown
    page_header = '---'*28
    footer = '---'*28+'\n'
    if show_help:
        print(page_header)
        banner = ''.join(["---"*2,' HELP ',"---"*24,'\n'])
        print(banner)
        help(function_or_mod)
        # print(footer)
        
    import sys
    if "google.colab" in sys.modules:
        markdown=False

    if show_code:
        print(page_header)

        banner = ''.join(["---"*2,' SOURCE -',"---"*23])
        print(banner)
        try:
            import inspect
            source_DF = inspect.getsource(function_or_mod)

            if markdown == True:
                
                output = "```python" +'\n'+source_DF+'\n'+"```"
                display(Markdown(output))
            else:
                print(source_DF)

        except TypeError:
            pass
            # display(Markdown)


    if file_location:
        file_loc = inspect.getfile(function_or_mod)
        banner = ''.join(["---"*2,' FILE LOCATION ',"---"*21])
        print(page_header)
        print(banner)
        print(file_loc)

    # print(footer)

    if return_code:
        return source_DF


def ihelp_menu(function_names,show_help=False,show_source=True):
    """Accepts a list of functions or function_names as strings.
    if show_help: display `help(function`.
    if show_source: retreive source code and display as proper markdown syntax"""
    from ipywidgets import interact, interactive, interactive_output
    import ipywidgets as widgets
    from IPython.display import display
    # from functions_combined_BEST import ihelp

    import inspect
    import pandas as pd

    if isinstance(function_names,list)==False:
        function_names = [function_names]
    functions_dict = dict()
    for fun in function_names:
        if isinstance(fun, str):
            # module =
            functions_dict[fun] = eval(fun)

        elif inspect.isfunction(fun):

            members= inspect.getmembers(fun)
            member_df = pd.DataFrame(members,columns=['param','values']).set_index('param')

            fun_name = member_df.loc['__name__'].values[0]
            functions_dict[fun_name] = fun



    ## Check boxes
    check_help = widgets.Checkbox(description='show help(function)',value=True)
    check_source = widgets.Checkbox(description='show source code)',value=True)
    check_boxes = widgets.HBox(children=[check_help,check_source])

    ## dropdown menu (dropdown, label, button)
    dropdown = widgets.Dropdown(options=list(functions_dict.keys()))
    label = widgets.Label('Function Menu')
    button = widgets.ToggleButton(description='Show/hide',value=False)
    menu = widgets.HBox(children=[label,dropdown,button])
    full_layout = widgets.GridBox(children=[menu,check_boxes],box_style='warning')

    # out=widgets.Output(layout={'border':'1 px solid black'})
    def dropdown_event(change):
        show_ihelp(function=change.new)
    dropdown.observe(dropdown_event,names='values')

    def button_event(change):
        button_state = change.new
        if button_state:
            button.description
    #     show_ihelp(display_help=button_state)

    button.observe(button_event)
    show_output = widgets.Output()

    def show_ihelp(display_help=button,function=dropdown.value,show_help=check_help.value,show_code=check_source.value):
        from IPython.display import display
        show_output.clear_output()
        if display_help:
            if isinstance(function, str):
    #             with show_output:
    #                 ihelp(eval(function),show_help=show_help,show_code=show_code)
                display(ihelp(eval(function),show_help=show_help,show_code=show_code))
            else:
                display(ihelp(function,show_help=show_help,show_code=show_code))
        else:
            display('Press show to display ')
    #         show_output.clear_output()

    output = widgets.interactive_output(show_ihelp,{'display_help':button,
                                                   'function':dropdown,
                                                   'show_help':check_help,
                                                   'show_code':check_source})
    # with out:
    # with show_output:
    display(full_layout, output)#,show_output)






################################################### ADDITIONAL NLP #####################################################
## Adding in stopword removal to the actual dataframe
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += [0,1,2,3,4,5,6,7,8,9]

    return  stopwords_list


def apply_stopwords(stopwords_list,  text, tokenize=True,return_tokens=False, pattern = "([a-zA-Z]+(?:'[a-z]+)?)"):
    """EX: df['text_stopped'] = df['content'].apply(lambda x: apply_stopwords(stopwords_list,x))"""
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    if tokenize==True:
        from nltk import regexp_tokenize

        text = regexp_tokenize(text,pattern)

    stopped = [x.lower() for x in text if x.lower() not in stopwords_list]

    if return_tokens==True:
        return regexp_tokenize(' '.join(stopped),pattern)
    else:
        return ' '.join(stopped)

def empty_lists_to_strings(x):
    """Takes a series and replaces any empty lists with an empty string instead."""
    if len(x)==0:
        return ' '
    else:
        return ' '.join(x) #' '.join(tokens)

def load_raw_twitter_file(filename = 'data/trump_tweets_01202017_06202019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}):
    import pandas as pd

    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df




## NEW 07/11/19 - function for all sentiment analysis

def full_sentiment_analysis(twitter_df, source_column='content_min_clean',separate_cols=True):#, plot_results=True):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    source_column='content_min_clean'
    twitter_df['sentiment_scores'] = twitter_df[source_column].apply(lambda x: sid.polarity_scores(x))
    twitter_df['compound_score'] = twitter_df['sentiment_scores'].apply(lambda dict: dict['compound'])
    twitter_df['sentiment_class'] = twitter_df['compound_score'].apply(lambda score: 'pos' if score >=0 else 'neg')


    # Separate result dictionary into columns  (optional)
    if separate_cols==True:
        # Separate Scores into separate columns in df
        twitter_df_out = get_group_sentiment_scores(twitter_df)
    else:
        twitter_df_out = twitter_df


#     # plot results (optional)
#     if plot_results==True:

#         print("RESULTS OF SENTIMENT ANALYSIS BINARY CLASSIFICATION:\n",'-'*60)
#         # Normalized % of troll sentiment classes
#         plot_sent_class = twitter_df_out['sentiment_class'].value_counts()
#         plot_sent_class_norm = plot_sent_class/(sum(plot_sent_class))
#         print('\tNormalized Troll Classes:\n',plot_sent_class_norm)


#         with plt.style.context('seaborn-notebook'):
#             boxplot = df_sents.boxplot(column=['neg','neu','pos'],notch=True,figsize=(6,4))
#             boxplot.set_xticklabels(['Negative','Neutral','Positive']);
#             boxplot.set_title('Sentiment Scores By Word Type')
#             boxplot.set_ylabel('Sentiment Score')

    return twitter_df_out



# Write a function to extract the group scores from the dataframe
def get_group_sentiment_scores(df, score_col='sentiment_scores'):
    import pandas as pd
    series_df = df[score_col]
    series_neg = series_df.apply(lambda x: x['neg'])
    series_pos = series_df.apply(lambda x: x['pos'])
    series_neu = series_df.apply(lambda x: x['neu'])

    series_neg.name='neg'
    series_pos.name='pos'
    series_neu.name='neu'

    df = pd.concat([df,series_neg,series_neu,series_pos],axis=1)
    return df






def full_twitter_df_processing(df,raw_tweet_col='content', cleaned_tweet_col='content', case_ratio_col='content_min_clean',
sentiment_analysis_col='content_min_clean', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column."""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0

    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

    # if raw_tweet_col == cleaned_tweet_col:
    #     raw_tweets = 'content_raw'
    #     df[raw_tweets] = df[tweet_col].copy()


    if RT ==True:

        # Creating columns for tweets that `has_RT` or `starts_RT`
        df['has_RT']=df[raw_tweet_col].str.contains('RT')
        df['starts_RT']=df[raw_tweet_col].str.contains('^RT')

        ## FIRST REMOVE THE RT HEADERS

        # Remove `RT @Mentions` FIRST:
        re_RT = re.compile('RT [@]?\w*:')

        check_content_col = raw_tweet_col
        fill_content_col = cleaned_tweet_col

        df['content_starts_RT'] = df[check_content_col].apply(lambda x: re_RT.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: re_RT.sub(' ',x))
        track_fill_content_col+=1


    if urls==True:
        ## SECOND REMOVE URLS
        # Remove urls with regex
        urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")

        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        # df_full['content_urls'] = df_full[check_content_col].apply(lambda x: urls.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub(' ',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1

    if hashtags==True:

        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        ## REMOVE AND SAVE HASHTAGS, MENTIONS
        # Remove and save Hashtags
        hashtags = re.compile(r'\#\w*')

        df['content_hashtags'] =  df[check_content_col].apply(lambda x: hashtags.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: hashtags.sub(' ',x))
        track_fill_content_col+=1

        if str_tags_mentions==True:
            df['hashtag_strings'] = df['content_hashtags'].apply(lambda x: empty_lists_to_strings(x))


    if mentions==True:

        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        # Remove and save mentions (@)'s
        mentions = re.compile(r'\@\w*')


        df['content_mentions'] =  df[check_content_col].apply(lambda x: mentions.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: mentions.sub(' ',x))
        track_fill_content_col+=1

        if str_tags_mentions==True:
            df['mention_strings'] = df['content_mentions'].apply(lambda x: empty_lists_to_strings(x))


    # Creating content_stopped columns and then tokens_stopped column
    stop_col_name = fill_content_col+'_stop'
    stop_tok_col_name =  fill_content_col+'_stop_tokens'

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stop_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))


    ## Case Ratio Calculation (optional)
    if case_ratio_col is not None:
        df['case_ratio'] = df[case_ratio_col].apply(lambda x: case_ratio(x))

    ## Sentiment Analysis (optional)
    if sentiment_analysis_col is not None:
        df = full_sentiment_analysis(df,source_column=sentiment_analysis_col,separate_cols=True)

    df.sort_index(inplace=True)
    return df



def case_ratio(msg):
    """Accepts a twitter message (or used with .apply(lambda x:)).
    Returns the ratio of capitalized characters out of the total number of characters.

    EX:
    df['case_ratio'] = df['text'].apply(lambda x: case_ratio(x))"""
    import numpy as np
    msg_length = len(msg)
    test_upper = [1 for x in msg if x.isupper()]
    test_lower = [1 for x in msg if x.islower()]
    test_ratio = np.round(sum(test_upper)/msg_length,5)
    return test_ratio


#################################################### STOCK ##############################################################
def twitter_column_report(twitter_df, decision_map=None, sort_column=None, ascending=True, interactive=True):
    from ipywidgets import interact
    import pandas as pd
    df_dtypes=pd.DataFrame()
    df_dtypes = pd.DataFrame({'Column #': range(len(twitter_df.columns)),'Column Name':twitter_df.columns,
                              'Data Types':twitter_df.dtypes.astype('str')}).set_index('Column Name') #.set_index('Column Name')

    decision_map = {'object':'join','int64':'sum','bool':'to_list()?','float64':'drop and recalculate'}

    df_dtypes['Action'] = df_dtypes['Data Types'].map(decision_map)#column_list
#     df_dtypes.style.set_caption('DF Columns, Dtypes, and Course of Action')

    if sort_column is not None:
        df_dtypes.sort_values(by =sort_column,ascending=ascending, axis=0, inplace=True)
    if interactive==False:
        return df_dtypes
    else:

        @interact(column= df_dtypes.columns,direction={'ascending':True,'descending':False})
        def sort_df(column, direction):
            return df_dtypes.sort_values(by=column,axis=0,ascending=direction)



# def make_half_hour_range(twitter_df):
#     """Takes a df, rounds first timestamp down to nearest hour, last timestamp rounded up to hour.
#     Creates 30 minute intervals based that encompass all data."""
#     import pandas as pd
#     # Get timebin before the first timestamp that starts at 30m into the hour
#     ofst_30m_early=pd.offsets.Minute(-30)
#     start_idx = ofst_30m_early(twitter_df['date'].iloc[0].floor('H'))

#     # Get timbin after last timestamp that starts 30m into the hour.
#     ofst_30m_late =pd.offsets.Minute(30)
#     end_idx= ofst_30m_late(twitter_df['date'].iloc[-1].ceil('H'))


#     # Make time bins using the above start and end points
#     half_hour_range = pd.date_range(start =start_idx, end = end_idx, freq='30T')#.to_period()
#     half_hour_intervals = pd.interval_range(start=start_idx, end=end_idx,freq='30T',name='half_hour_bins',closed='left')

#     return half_hour_intervals
def make_time_index_intervals(twitter_df,col ='date', start=None,end=None, freq='CBH',num_offset=1):
    """Takes a df, rounds first timestamp down to nearest hour, last timestamp rounded up to hour.
    Creates 30 minute intervals based that encompass all data."""
    import pandas as pd

    if freq=='CBH':
        freq=pd.offsets.CustomBusinessHour(n=num_offset,start='09:30',end='16:30')
        ofst = pd.offsets.CustomBusinessHour(n=num_offset,start='09:30',end='16:30') #freq=custom_BH_freq()
        ofst_early = pd.offsets.CustomBusinessHour(n=-num_offset,start='09:30',end='16:30') #freq=custom_BH_freq()
    if freq=='T':
        ofst = pd.offsets.Minute(n=num_offset)
        ofst_early = pd.offsets.Minute(n=-num_offset)

    if freq=='H':
        ofst = pd.offsets.Hour(n=num_offset)
        ofst_early=pd.offsets.Hour(n=-num_offset)


    if start is None:
        # Get timebin before the first timestamp that starts
        start_idx = ofst.rollback(twitter_df[col].iloc[0])#.floor('H'))
    else:
        start_idx = pd.to_datetime(start)

    if end is None:
        # Get timbin after last timestamp that starts 30m into the hour.
        end_idx= ofst.rollforward(twitter_df[col].iloc[-1])#.ceil('H'))
    else:
        end_idx = pd.to_datetime(end)


    # Make time bins using the above start and end points
    time_range = pd.date_range(start =start_idx, end = end_idx, freq=freq)#.to_period()
    time_intervals = pd.interval_range(start=start_idx, end=end_idx,freq=freq,name='interval_index',closed='left')

    return time_intervals


#***########### FUNCTIONS FOR RESAMPLING AND BINNING TWITTER DATA
def int_to_ts(int_list, as_datetime=False, as_str=True):
    """Accepts one Panda's interval and returns the left and right ends as either strings or Timestamps."""
    import pandas as pd
    if as_datetime & as_str:
        raise Exception('Only one of `as_datetime`, or `as_str` can be True.')

    left_edges =[]
    right_edges= []

    for interval in int_list:
        int_str = interval.__str__()[1:-1]
        left,right = int_str.split(',')
        left_edges.append(left)
        right_edges.append(right)


    if as_str:
        return left_edges, right_edges

    elif as_datetime:
        left = pd.to_datetime(left)
        right = pd.to_datetime(right)
        return left,right


# Step 1:
def bin_df_by_date_intervals(test_df,time_intervals,column='date'):
    """Uses pd.cut with half_hour_intervals on specified column.
    Creates a dictionary/map of integer bin codes.
    Adds column"int_bins" with int codes.
    Adds column "left_edge" as datetime object representing the beginning of the time interval.
    Returns the updated test_df and a list of bin_codes."""
    import pandas as pd
    # Cut The Date column into interval bins,
    cut_date = pd.cut(test_df[column], bins=time_intervals)#,labels=list(range(len(half_hour_intervals))), retbins=True)
    test_df['int_times'] = cut_date

    # convert to str to be used as group names/codes
    unique_bins = cut_date.astype('str').unique()
    num_code = list(range(len(unique_bins)))

    # Dictioanry of number codes to be used for interval groups
    bin_codes = dict(zip(num_code,unique_bins))#.astype('str')


    # Mapper dictionary to convert intervals into number codes
    bin_codes_mapper = {v:k for k,v in bin_codes.items()}


    # Add column to the dataframe, then map integer code onto it
    test_df['int_bins'] = test_df['int_times'].astype('str').map(bin_codes_mapper)


    # Get the left edge of the bins to use later as index (after grouped)
    left_out, _ =int_to_ts(test_df['int_times'])#.apply(lambda x: int_to_ts(x))
    test_df['left_edge'] = pd.to_datetime(left_out)

    # bin codes to labels
    bin_codes = [(k,v) for k,v in bin_codes.items()]

    return test_df, bin_codes


def concatenate_group_data(group_df_or_series):
    """Accepts a series or dataframe from a groupby.get_group() loop.
    Adds TweetFreq column for # of rows concatenate. If input is series,
    TweetFreq=1 and series is returned."""

    import pandas as pd
    from pandas.api import types as tp

    if isinstance(group_df_or_series, pd.Series):

        group_data = group_df_or_series

#         group_data.index = group_df_or_series.index
        group_data['TweetFreq'] = 1

        return group_data

    # if the group is a dataframe:
    elif isinstance(group_df_or_series, pd.DataFrame):

        df = group_df_or_series

        # create an output series to collect combined data
        group_data = pd.Series(index=df.columns)
        group_data['TweetFreq'] = df.shape[0]


        for col in df.columns:

            combined=[]
            col_data = []

            col_data = df[col]
            combined=col_data.values

            group_data[col] = combined

    return group_data


#***#
# def collapse_df_by_group_indices(twitter_df,group_indices, new_col_order=None):
#     """Loops through the group_indices provided to concatenate each group into
#     a single row and combine into one dataframe with the ______ as the index"""

#     import pandas as pd
#     # Create a Panel to temporarily hold the group series and dataframes
#     # group_dict_to_df = {}
#     # create a dataframe with same columns as twitter_df, and index=group ids from twitter_groups
#     group_df_index = [x[0] for x in group_indices]


#     twitter_grouped = pd.DataFrame(columns=twitter_df.columns, index=group_df_index)
#     twitter_grouped['TweetFreq'] =0

#     for (idx,group_members) in group_indices:

#         group_df = twitter_df.loc[group_members]

#         combined_series = concatenate_group_data(group_df)

# #         twitter_grouped.loc[idx,:] = combined_series
#         twitter_grouped.loc[idx] = combined_series#.values

#     if new_col_order==None:
#         return twitter_grouped

#     else:
#         df_out = twitter_grouped[new_col_order].copy()
#         df_out.index = group_df_index#twitter_grouped.index
#         return df_out
def collapse_df_by_group_index_col(twitter_df,group_index_col='int_bins', new_col_order=None):
    """Loops through the group_indices provided to concatenate each group into
    a single row and combine into one dataframe with the ______ as the index"""

    import pandas as pd


    # Create a Panel to temporarily hold the group series and dataframes
    # group_dict_to_df = {}
    # create a dataframe with same columns as twitter_df, and index=group ids from twitter_groups

    group_indices = twitter_df.groupby(group_index_col).groups
    group_indices = [(k,v) for k,v in group_indices.items()]
    group_df_index = [x[0] for x in group_indices]


    # Create empty shell of twitter_grouped dataframe
    twitter_grouped = pd.DataFrame(columns=twitter_df.columns, index=group_df_index)
    twitter_grouped['TweetFreq'] =0


    # Loop through each group_indices
    for (idx,group_members) in group_indices:

        group_df = twitter_df.loc[group_members]

        # Call on concatenate_group_data to handle the merging of rows
        combined_series = concatenate_group_data(group_df)

#         twitter_grouped.loc[idx,:] = combined_series
        twitter_grouped.loc[idx] = combined_series#.values

    # Update Column order, if requested, otherwise return twitter_grouped
    if new_col_order==None:
        return twitter_grouped
    else:
        df_out = twitter_grouped[new_col_order].copy()
        df_out.index = group_df_index#twitter_grouped.index
        return df_out



def load_stock_price_series(filename='IVE_bidask1min.txt',
                               folderpath='data/',
                               start_index = '2017-01-23', freq='T'):
    import pandas as pd
    import numpy as np
    from IPython import display

    # Load in the text file and set headers
    fullfilename= folderpath+filename
    headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
    stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True,usecols=['Date','Time','BidClose'])

    # Create datetime index
    date_time_index = stock_df['Date']+' '+stock_df['Time']
    date_time_index = pd.to_datetime(date_time_index)
    stock_df.index=date_time_index

    # Select only the days after start_index
    stock_df = stock_df[start_index:]

    stock_price = stock_df['BidClose'].rename('stock_price')
    stock_price[stock_price==0] = np.nan

    return stock_price

def load_twitter_df(overwrite=True,set_index='time_index',verbose=2,replace_na=''):
    import pandas as pd
    from IPython.display import display
    # try: twitter_df
    # except NameError: twitter_df = None
    # if twitter_df is not None:
    #     print('twitter_df already exists.')
    #     if overwrite==True:
    #         print('Overwrite=True. deleting original...')
    #         del(twitter_df)

    # if twitter_df is None:
    print('loading twitter_df')

    twitter_df = pd.read_csv('data/trump_twitter_archive_df.csv', encoding='utf-8', parse_dates=True)
    twitter_df.drop('Unnamed: 0',axis=1,inplace=True)

    twitter_df['date']  = pd.to_datetime(twitter_df['date'])
    twitter_df['time_index'] = twitter_df['date'].copy()
    twitter_df.set_index(set_index,inplace=True,drop=True)


    # Fill in missing values before merging with stock data
    twitter_df.fillna(replace_na, inplace=True)
    twitter_df.sort_index(ascending=True, inplace=True)

    # RECASTING A COUPLE COLUMNS
    twitter_df['is_retweet'] = twitter_df['is_retweet'].astype('bool')
    twitter_df['id_str'] = twitter_df['id_str'].astype('str')
    twitter_df['sentiment_class'] = twitter_df['sentiment_class'].astype('category')

#         twitter_df.reset_index(inplace=True)
    # Check header and daterange of index
    if verbose>0:
        display(twitter_df.head(2))
        print(twitter_df.index[[0,-1]])
    return twitter_df


#################### GENERAL HELPER FUNCTIONS #####################
def is_var(name):
    x=[]
    try: eval(name)
    except NameError: x = None

    if x is None:
        return False
    else:
        return True

#################### TIMEINDEX FUNCTIONS #####################

def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

def get_day_window_size_from_freq(dataset, CBH=custom_BH_freq()):#, freq='CBH'):

    if dataset.index.freq == CBH: #custom_BH_freq():
        day_window_size =  7

    elif dataset.index.freq=='T':
        day_window_size = 60*24
    elif dataset.index.freq=='BH':
        day_window_size = 8
    elif dataset.index.freq=='H':
        day_window_size =24

    elif dataset.index.freq=='B':
        day_window_size=1
    elif dataset.index.freq=='D':
        day_window_size=1

    else:
        raise Exception(f'dataset freq={dataset.index.freq}')

    return day_window_size




def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_method='ffill',
                        verbose=3): #set_tz=True,

    import pandas as pd
    import numpy as np
    from IPython.display import display

    if verbose>1:
        # print(f"{'Index When:':>{10}}\t{'Freq:':>{20}}\t{'Index Start:':>{40}}\t{'Index End:':>{40}}")
        print(f"{'Index When:'}\t{'Freq:'}\t{'Index Start'}\t\t{'Index End:'}")
        print(f"Pre-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


    if freq=='CBH':
        freq=custom_BH_freq()
#         start_idx =

    # Change frequency to freq
    ive_df = ive_df.asfreq(freq,)#'min')

    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')

    # Report Success / Details
    if verbose>1:
        print(f"Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


    ## FILL AND TRACK TIMEPOINTS WITH MISSING DATA

    # Helper Function for adding column to track the datapoints that were filled
    def check_null_times(x):
        import numpy as np
        if np.isnan(x):
            return True
        else:
            return False

    ## CREATE A COLUMN TO TRACK ROWS TO BE FILLED
    # If col_to_fill provided, use that column to create/judge ive_df['filled_timebin']
    if col_to_fill!=None:
        ive_df['filled_timebin'] = ive_df[col_to_fill].apply(lambda x: check_null_times(x))#True if ive_df.isna().any()

    # if not provided, use all columns and sum results
    elif col_to_fill == None:
        # Prefill fol with 0's
        ive_df['filled_timebin']=0

        # loop through all columns and add results of check_null_times from each loop
    for col in ive_df.columns:
        if ive_df[col].dtypes=='float64':
            #ive_df['filled_timebin'] = ive_df[target_col].apply(lambda x: check_null_times(x))#True if ive_df.isna().any()
            curr_filled_timebin_col = ive_df[col].apply(lambda x: check_null_times(x))#True if ive_df.isna().any()

            # add results
            ive_df['filled_timebin'] +=  curr_filled_timebin_col

    ive_df['filled_timebin'] = ive_df['filled_timebin'] >0

    ## FILL IN NULL VALUES
    ive_df.fillna(method=fill_method, inplace=True)

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\nFilled {len(check_fill==True)}# of rows using method {fill_method}')

    # Report any remaning null values
    if verbose>0:
        res = ive_df.isna().sum()
        if res.any():
            print(f'Cols with Nulls:')
            print(res[res>0])
        else:
            print('No Remaining Null Values')

    # display header
    if verbose>2:
        from IPython.display import display
        display(ive_df.head())

    return ive_df


# Helper Function for adding column to track the datapoints that were filled
def check_null_times(x):
    import numpy as np
    if np.isnan(x):
        return True
    else:
        return False

##################### DATASET LOADING FUNCTIONS #####################
def load_raw_stock_data_from_txt(filename='IVE_bidask1min.txt',
                               folderpath='data/',
                               start_index = '2016-12-31',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    import pandas as pd
    import numpy as np
    from IPython.display import display

    # Load in the text file and set headers
    fullfilename= folderpath+filename
    headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
    stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

    # Create datetime index
    date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
    date_time_index = pd.to_datetime(date_time_index)
    stock_df.set_index(date_time_index, inplace=True)

    # Select only the days after start_index
    stock_df = stock_df[start_index:]
    print(f'\nRestricting stock_df to index {start_index}-forward')

    # Remove 0's from BidClose
    if clean==True:
        print(f"There are {len(stock_df.loc[stock_df['BidClose']==0])} '0' values for 'BidClose'")
        stock_df.loc[stock_df['BidClose']==0] = np.nan
        num_null = stock_df['BidClose'].isna().sum()
        print(f'\tReplaced 0 with np.nan. There are {num_null} null values to address.')

        if fill_or_drop_null=='drop':
            print("Since fill_or_drop_null=drop, dropping null values from BidClose.")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)
        elif fill_or_drop_null=='fill':
            print(f"Since fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.")

            stock_df['BidClose'].fillna(method=fill_method, inplace=True)

        if verbose>0:
            print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            print(f"Filling 0 values using method = {fill_method}")




    # call set_timeindex_freq to specify proper frequency
    if freq!=None:
        # Set the time index .
        print(f'Setting the timeindex to freq{freq}')
        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=verbose)

    # Display feedback
    if verbose>0:
        display(stock_df.head())
    if verbose>1:
        print(stock_df.index[[0,-1]],stock_df.index.freq)

    return stock_df





def load_stock_df_from_csv(filename='ive_sp500_min_data_match_twitter_ts.csv',
                           folderpath='/content/drive/My Drive/Colab Notebooks/Mod 5 Project/data/',
                          clean=True,freq='T',fill_method='ffill',verbose=2):
    import os
    import pandas as pd
    import numpy as np
    from IPython.display import display
    #         check_for_google_drive()

    # Check if user provided folderpath to append to filename
    if len(folderpath)>0:
        fullfilename = folderpath+filename
    else:
        fullfilename=filename

    # load in csv by fullfilename
    stock_df = pd.read_csv(fullfilename,index_col=0, parse_dates=True)
#     stock_df = set_timeindex_freq(stock_df,['BidClose'],freq=freq, fill_method=fill_method)

    if clean==True:

        if verbose>0:
            print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            print(f"Filling 0 values using method = {fill_method}")

        stock_df.loc[stock_df['BidClose']==0] = np.nan
        stock_df['BidClose'].fillna(method=fill_method, inplace=True)


    # Set the time index
    stock_df = set_timeindex_freq(stock_df,'BidClose',freq=freq, fill_method = fill_method, verbose=verbose)


    # Display info depending on verbose level
    if verbose>0:
        display(stock_df.head())

    if verbose>1:
        print(stock_df.index)

    return stock_df


def plot_time_series(stocks_df, freq=None, fill_method='ffill',figsize=(12,4)):

    df = stocks_df.copy()
    df.fillna(method=fill_method, inplace=True)
    df.dropna(inplace=True)

    if (df.index.freq==None) & (freq == None):
        xlabels=f'Time'

    elif (df.index.freq==None) & (freq != None):
        df = df.asfreq(freq)
        df.fillna(method=fill_method, inplace=True)
        df.dropna(inplace=True)
        xlabels=f'Time - Frequency = {freq}'

    else:
        xlabels=f'Time - Frequency = {df.index.freq}'

    ylabels="Price"

    raw_plot = df.plot(figsize=figsize)
    raw_plot.set_title('Stock Bid Closing Price ')
    raw_plot.set_ylabel(ylabels)
    raw_plot.set_xlabel(xlabels)


def stationarity_check(df, col='BidClose', window=80, freq='BH'):
    """From learn.co lesson: use ADFuller Test for Stationary and Plot"""
    import matplotlib.pyplot as plt
    TS = df[col].copy()
    TS = TS.asfreq(freq)
    TS.fillna(method='ffill',inplace=True)
    TS.dropna(inplace=True)
    # Import adfuller
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    import numpy as np

    # Calculate rolling statistics
    rolmean = TS.rolling(window = window, center = False).mean()
    rolstd = TS.rolling(window = window, center = False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller(TS) # change the passengers column as required

    #Plot rolling statistics:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
    ax[0].set_title('Rolling Mean & Standard Deviation')

    ax[0].plot(TS, color='blue',label='Original')
    ax[0].plot(rolmean, color='red', label='Rolling Mean',alpha =0.6)
    ax[1].plot(rolstd, color='black', label = 'Rolling Std')
    ax[0].legend()
    ax[1].legend()
#     plt.show(block=False)
    plt.tight_layout()

    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')
    print('\tIf p<.05 then timeseries IS stationary.')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    return None



def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    # UDEMY COURSE ALTERNATIVE TO STATIONARITY CHECK
    """
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

######## SEASONAL DECOMPOSITION
def plot_decomposition(TS, decomposition, figsize=(12,8),window_used=None):
    """ Plot the original data and output decomposed components"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    # Gather the trend, seasonality and noise of decomposed object
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fontdict_axlabels = {'fontsize':12}#,'fontweight':'bold'}

    # Plot gathered statistics
    fig, ax = plt.subplots(nrows=4, ncols=1,figsize=figsize)

    ylabel = 'Original'
    ax[0].plot(np.log(TS), color="blue")
    ax[0].set_ylabel(ylabel, fontdict=fontdict_axlabels)

    ylabel = label='Trend'
    ax[1].plot(trend, color="blue")
    ax[1].set_ylabel(ylabel, fontdict=fontdict_axlabels)

    ylabel='Seasonality'
    ax[2].plot(seasonal, color="blue")
    ax[2].set_ylabel(ylabel, fontdict=fontdict_axlabels)

    ylabel='Residuals'
    ax[3].plot(residual, color="blue")
    ax[3].set_ylabel(ylabel, fontdict=fontdict_axlabels)
    ax[3].set_xlabel('Time', fontdict=fontdict_axlabels)

    # Add title with window
    if window_used == None:
        plt.suptitle('Seasonal Decomposition', y=1.02)
    else:
        plt.suptitle(f'Seasonal Decomposition - Window={window_used}', y=1.02)

    # Adjust aesthetics
    plt.tight_layout()

    return ax


def seasonal_decompose_and_plot(ive_df,col='BidClose',freq='H',
                          fill_method='ffill',window=144,
                         model='multiplicative', two_sided=False,
                               plot_components=True):##WIP:
    """Perform seasonal_decompose from statsmodels.tsa.seasonal.
    Plot Output Decomposed Components"""
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose


    # TS = ive_df['BidClose'].asfreq('BH')
    TS = pd.DataFrame(ive_df[col])
    TS = TS.asfreq(freq)
    TS[TS==0]=np.nan
    TS.fillna(method='ffill',inplace=True)

    # Perform decomposition
    decomposition = seasonal_decompose(np.log(TS),freq=window, model=model, two_sided=two_sided)

    if plot_components==True:
        ax = plot_decomposition(TS, decomposition, window_used=window)

    return decomposition


### WIP FUNCTIONS
def make_date_range_slider(start_date,end_date,freq='D'):

    from ipywidgets import interact, interactive, Label, Box, Layout
    import ipywidgets as iw
    from datetime import datetime
    import pandas as pd
    # specify the date range from user input
    dates = pd.date_range(start_date, end_date,freq=freq)

    # specify formatting based on frequency code
    date_format_lib={'D':'%m/%d/%Y','H':'%m/%d/%Y: %T'}
    freq_format = date_format_lib[freq]


    # creat options list and index for SelectionRangeSlider
    options = [(date.strftime(date_format_lib[freq]),date) for date in dates]
    index = (0, len(options)-1)

    #     # Create out function to display outputs (not needed?)
    #     out = iw.Output(layout={'border': '1px solid black'})
    #     #     @out.capture()

    # Instantiate the date_range_slider
    date_range_slider = iw.SelectionRangeSlider(
        options=options, index=index, description = 'Date Range',
        orientation = 'horizontal',layout={'width':'500px','grid_area':'main'},#layout=Layout(grid_area='main'),
        readout=True)

    # Save the labels for the date_range_slider as separate items
    date_list = [date_range_slider.label[0], date_range_slider.label[-1]]
    date_label = iw.Label(f'{date_list[0]} -- {date_list[1]}',
                            layout=Layout(grid_area='header'))




#### TWITTER_STOCK MATCHING
def get_B_day_time_index_shift(test_df, verbose=1):
    import pandas as pd
    import numpy as np
    fmtYMD= '%Y-%m-%d'

    test_df['day']= test_df['date'].dt.strftime('%Y-%m-%d')
    test_df['time'] = test_df['date'].dt.strftime('%T')
    test_df['dayofweek'] = test_df['date'].dt.day_name()

    test_df_to_period = test_df[['date','content']]
    test_df_to_period = test_df_to_period.to_period('B')
    test_df_to_period['B_periods'] = test_df_to_period.index.values
    test_df_to_period['B_day'] = test_df_to_period['B_periods'].apply(lambda x: x.strftime(fmtYMD))



    test_df['B_day'] = test_df_to_period['B_day'].values
    test_df['B_shifted']=np.where(test_df['day']== test_df['B_day'],False,True)
    test_df['B_time'] = np.where(test_df['B_shifted'] == True,'09:30:00', test_df['time'])

    test_df['B_dt_index'] = pd.to_datetime(test_df['B_day'] + ' ' + test_df['B_time'])

    test_df['time_shift'] = test_df['B_dt_index']-test_df['date']

    if verbose > 0:
        test_df.head(20)

    return test_df

def reorder_twitter_df_columns(twitter_df, order=[]):
    if len(order)==0:
        order=['date','dayofweek','B_dt_index','source','content','content_raw','retweet_count','favorite_count','sentiment_scores','time_shift']
    twitter_df_out = twitter_df[order]
    twitter_df_out.index = twitter_df.index
    return twitter_df_out


def match_stock_price_to_tweets(tweet_timestamp,time_after_tweet= 30,time_freq ='T',stock_price=[]):#stock_price_index=stock_date_data):

    import pandas as pd
    import numpy as np
    from datetime import datetime as dt
    # output={'pre_tweet_price': price_at_tweet,'post_tweet_price':price_after_tweet,'delta_price':delta_price, 'delta_time':delta_time}
    output={}
    # convert tweet timestamp to minute accuracy
    ts=[]
    ts = pd.to_datetime(tweet_timestamp).round(time_freq)

    BH = pd.tseries.offsets.BusinessHour(start='09:30',end='16:30')
    BD = pd.tseries.offsets.BusinessDay()


    # checking if time is within stock_date_data
#     def roll_B_day_forward(ts):

    if ts not in stock_price.index:
        ts = BH.rollforward(ts)

        if ts not in stock_price.index:
            return np.nan#"ts2_not_in_index"

    # Get price at tweet time
    price_at_tweet = stock_price.loc[ts]

    if np.isnan(price_at_tweet):
        output['pre_tweet_price'] = np.nan
    else:
        output['pre_tweet_price'] = price_at_tweet


    # Use timedelta to get desired timepoint following tweet
    hour_freqs = 'BH','H','CBH'
    day_freqs = 'B','D'

    if time_freq=='T':
        ofst=pd.offsets.Minute(time_after_tweet)

    elif time_freq in hour_freqs:
        ofst=pd.offsets.Hour(time_after_tweet)

    elif time_freq in day_freqs:
        ofst=pd.offsets.Day(time_after_tweet)


    # get timestamp to check post-tweet price
    post_tweet_ts = ofst(ts)


    if post_tweet_ts not in stock_price.index:
#         post_tweet_ts =BD.rollforward(post_tweet_ts)
        post_tweet_ts = BH.rollforward(post_tweet_ts)

        if post_tweet_ts not in stock_price.index:
            return np.nan


    # Get next available stock price
    price_after_tweet = stock_price.loc[post_tweet_ts]
    if np.isnan(price_after_tweet):
        output['post_tweet_price'] = 'NaN in stock_price'
    else:
        # calculate change in price
        delta_price = price_after_tweet - price_at_tweet
        delta_time = post_tweet_ts - ts
        output['post_tweet_price'] = price_after_tweet
        output['delta_time'] = delta_time
        output['delta_price'] = delta_price

#         output={'pre_tweet_price': price_at_tweet,'post_tweet_price':price_after_tweet,'delta_price':delta_price, 'delta_time':delta_time}

    return output

def unpack_match_stocks(stock_dict):
    import pandas as pd
    stock_series = pd.Series(stock_dict)
    return stock_series



### KERAS
def my_rmse(y_true,y_pred):
    """RMSE calculation using keras.backend"""
    from keras import backend as kb
    sq_err = kb.square(y_pred - y_true)
    mse = kb.mean(sq_err,axis=-1)
    rmse =kb.sqrt(mse)
    return rmse



##### FROM CAPSTONE PROJECT OUTLINE AND ANALYSIS

def get_technical_indicators(dataset,make_price_from='BidClose'):

    import pandas as pd
    import numpy as np
    dataset['price'] = dataset[make_price_from].copy()
    if dataset.index.freq == custom_BH_freq():
        days = get_day_window_size_from_freq(dataset)#,freq='CBH')
    else:
        days = get_day_window_size_from_freq(dataset)

    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['price'].rolling(window=7*days).mean()
    dataset['ma21'] = dataset['price'].rolling(window=21*days).mean()

    # Create MACD
    dataset['26ema'] = dataset['price'].ewm(span=26*days).mean()
#     dataset['12ema'] = pd.ewma(dataset['price'], span=12)
    dataset['12ema'] = dataset['price'].ewm(span=12*days).mean()

    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
#     dataset['20sd'] = pd.stats.moments.rolling_std(dataset['price'],20)
    dataset['20sd'] = dataset['price'].rolling(20*days).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    # Create Exponential moving average
    dataset['ema'] = dataset['price'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['price']-days*1

    return dataset


def plot_technical_indicators(dataset, last_days=90):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    days = get_day_window_size_from_freq(dataset)

    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(10, 6), dpi=100)
#     shape_0 = dataset.shape[0]
#     xmacd_ = shape_0-(days*last_days)

    dataset = dataset.iloc[-(days*last_days):, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)

    # Plot first subplot
    ax[0].plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    ax[0].plot(dataset['price'],label='Closing Price', color='b')
    ax[0].plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    ax[0].plot(dataset['upper_band'],label='Upper Band', color='c')
    ax[0].plot(dataset['lower_band'],label='Lower Band', color='c')
    ax[0].fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    ax[0].set_title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    ax[0].set_ylabel('USD')
    ax[0].legend()

#     shape_0 = dataset.shape[0]
#     xmacd_ = shape_0-(days*last_days)
#     # Plot second subplot
#     ax[1].set_title('MACD')
#     ax[1].plot(dataset['MACD'],label='MACD', linestyle='-.')
#     ax[1].hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
#     ax[1].hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
#     ax[1].plot(dataset['momentum'],label='Momentum', color='b',linestyle='-')

#     ax[1].legend()
    plt.delaxes(ax[1])
    plt.show()

def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=True):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')

    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST
    day_freq = periods_per_day
    start_train_day =  stock_df.index[-1] - (num_train_days+num_test_days )*day_freq
    last_train_day = stock_df.index[-1] - num_test_days*day_freq
    # start_train_day = stock_df.index[-1] - num_train_days*day_freq
    # last_day = stock_df.index[-1] - num_test_days*day_freq

    train_data = stock_df.loc[start_train_day:last_train_day]#,'price']
    test_data = stock_df.loc[last_train_day:]#,'price']

    # train_data = stock_df.loc[start_train_day:last_day]#,'price']
    # test_data = stock_df.loc[last_day:]#,'price']

    if verbose>0:
        print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]}.')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]}.')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')

    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))


    if plot==True:
        if 'price' in stock_df.columns:
            plot_col ='price'
        elif 'price_labels' in stock_df.columns:
            plot_col = 'price_labels'

        fig = plt.figure(figsize=(8,4))
        train_data[plot_col].plot(label='Training')
        test_data[plot_col].plot(label='Test')
        plt.title('Training and Test Data for S&P500')
        plt.ylabel('Price')
        plt.xlabel('Trading Date/Hour')
        plt.legend()
        plt.show()

    return train_data, test_data




def make_scaler_library(df,transform=False,columns=[]):
    """Takes a df and fits a MinMax scaler to the columns specified (default is to use all columns).
    Returns a dictionary (scaler_library) with keys = columns, and values = its corresponding fit's MinMax Scaler

    Example Usage:
    scale_lib, df_scaled = make_scaler_library(df, transform=True)

    # to get the inverse_transform of a column with a different name:
    # use `inverse_transform_series`
    scaler = scale_lib['price'] # get scaler fit to original column  of interest
    price_column =  inverse_transform_series(df['price_labels'], scaler) #get the inverse_transformed series back
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler_dict = {}
    scaler_dict['index'] = df.index
    if len(columns)==0:
        user_cols = []
        columns = df.columns
    for col in columns:
        user_cols=columns
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler

    if transform==False:
        return scaler_dict

    elif transform==True:
        df_out = transform_cols_from_library(df, scaler_dict,columns=user_cols)
        return scaler_dict, df_out


def transform_cols_from_library(df,scaler_library,inverse=False,columns=[]):
    """Accepts a df and a scaler_library that was transformed using make_scaler_library.
    Inverse tansforms listed columns (if columns =[] then all columns)
    Returns a dataframe with all columns of original df."""
    df_out = df.copy()

    if len(columns)==0:
        columns = df.columns

    for col in columns:

        scaler = scaler_library[col]
        if hasattr(scaler, 'data_range_')==False:
            raise Exception(f'The scaler for {col} is not fitted.')


        if inverse==False:
            scaled_col = scaler.transform(df[col].values.reshape(-1,1))
        elif inverse==True:
            scaled_col = scaler.inverse_transform(df[col].values.reshape(-1,1))
        df_out[col] = scaled_col.ravel()
    return df_out

def inverse_transform_series(series, scaler):
    """Takes a series of df column and a fit scaler. Intended for use with make_scaler_library's dictionary
    Example Usage:
    scaler_lib, df_scaled = make_scaler_library(df, transform = True)
    series_inverse_transformed = inverse_transform_series(df['price_data'],scaler_lib['price'])
    """
    import pandas as pd

    series_tf = scaler.inverse_transform(series.values.reshape(-1,1))
    series_tf = pd.Series(series_tf.ravel(), index = series.index, name=series.name)
    return series_tf


def make_X_y_timeseries_data(data,x_window = 35, verbose=2,as_array=True):
    """Creates an X and Y time sequence trianing set from a pandas Series.
    - X_train is a an array with x_window # of samples for each row in X_train
    - y_train is one value per X_train window: the next time point after the X_window.
    Verbose determines details printed about the contents and shapes of the data.

    # Example Usage:
    X_train, y_train = make_X_y_timeseries(df['price'], x_window= 35)
    print( X_train[0]]):
    # returns: arr[X1,X2...X35]
    print(y_train[0])
    # returns  X36
    """

    import numpy as np
    import pandas as pd

    # Raise warning if null valoues
    if any(data.isna()):
        raise Exception('Function does not accept null values')

    # Optional display of input data shape and range
    if verbose>0:
        print(f'Input Range: {np.min(data)} - {np.max(data)}')
        print(f'Input Shape: {np.shape(data)}\n')


    # Save the index from the input data
    time_index_in = data.index
    time_index = data.index[x_window:]


    # Create Empty lists to receive binned X_train and y_train data
    X_train, y_train = [], []
    check_time_index = []

    # For every possible bin of x_window # of samples
    # create an X_train row with the X_window # of previous samples
    # create a y-train row with just one values - the next sample after the X_train window
    for i in range(x_window, data.shape[0]):
        check_time_index.append([data.index[i-x_window], data.index[i]])
        # Append a list of the past x_window # of timepoints
        X_train.append(data.iloc[i-x_window:i])#.values)

        # Append the next single timepoint's data
        y_train.append(data.iloc[i])#.values)

    if as_array == True:
        # Make X_train, y_train into arrays
        X_train, y_train = np.array(X_train), np.array(y_train)


    if verbose>0:
        print(f'\nOutput Shape - X: {X_train.shape}')
        print(f'Output Shape - y: {y_train.shape}')
        print(f'\nTimeindex Shape: {np.shape(time_index)}\n\tRange: {time_index[0]}-{time_index[-1]}')
        print(f'\tFrequency:',time_index.freq)
#     print(time_index)
#     print(check_time_index)
    return X_train, y_train, time_index


def make_df_timeseries_bins_by_column(df, x_window = 35, verbose=2,one_or_two_dfs = 1): #target_col='price',
    """ Function will take each column from the dataframe and create a train_data dataset  (with X and Y data), with
    each row in X containing x_window number of observations and y containing the next following observation"""
    import pandas as pd
    import numpy as np
    col_data  = {}
    time_index_for_df = []
    for col in df.columns:

        col_data[col] = {}
        col_bins, col_labels, col_idx =  make_X_y_timeseries_data(df[col], verbose=0, as_array=True)#,axis=0)
#         print(f'col_bins dtype={type(col_bins)}')
#         print(f'col_labels dtype={type(col_labels)}')

        ## ALTERNATIVE IS TO PLACE DF COLUMNS CREATION ABOVE HERE
        col_data[col]['bins']=col_bins
        col_data[col]['labels'] = col_labels
#         col_data[col]['index'] = col_idx
        time_index_for_df = col_idx

    # Convert the dictionaries into a dataframe
    df_timeseries_bins = pd.DataFrame(index=time_index_for_df)
#     df_timeseries_bins.index=time_index_for_df
#     print(time_index_for_df)
    # for each original column
    for colname,data_dict in col_data.items():

        #for column's new data bins,labels
        for data_col, X in col_data[colname].items():

            # new column title
            new_colname = colname+'_'+data_col
#             print(new_colname)
            make_col = []
            if data_col=='labels':
                df_timeseries_bins[new_colname] = col_data[colname][data_col]
            else:
                # turn array of lists into list of arrays
                for x in range(X.shape[0]):
                    x_data = np.array(X[x])
#                     x_data = X[x]
                    make_col.append(x_data)
                # fill in column's data
                df_timeseries_bins[new_colname] = make_col

#     print(df_timeseries_bins.index)
#     print(time_index_for_df)


    if one_or_two_dfs==1:
        return df_timeseries_bins

    elif one_or_two_dfs==2:
        df_bins = df_timeseries_bins.filter(regex=('bins'))
        df_labels = df_timeseries_bins.filter(regex=('labels'))

    return df_bins, df_labels



def predict_model_make_results_dict(model,scaler, X_test_in, y_test,test_index,
                                    X_train_in, y_train,train_index,
                                   return_as_dfs = False):# Get predictions and combine with true price

    """Accepts a fit keras model, X_test, y_test, and y_train data. Uses provided fit-scaler that transformed
    original data.
    By default (return_as_dfs=False): returns the results as a panel (dictioanry of dataframes), with panel['train'],panl['test']
    Setting return_as_dfs=True will return df_train, df_test"""
    import pandas as pd
    # Get predictions from model
    predictions = model.predict(X_test_in)

    # Get predicted price series (scaled and inverse_transformed)
    pred_price_scaled = pd.Series(predictions.ravel(),name='scaled_pred_price',index=test_index)
    pred_price = inverse_transform_series(pred_price_scaled, scaler).rename('pred_price')

    # Get true price series (scaled and inverse_transformed)
    true_price_scaled =  pd.Series(y_test,name='scaled_test_price',index=test_index)
    true_price = inverse_transform_series(true_price_scaled,scaler).rename('test_price')

    # combine all test data series into 1 dataframe
    df_test_data = pd.concat([true_price, pred_price,  true_price_scaled, pred_price_scaled],axis=1)#, columns=['predicted_price','true_price'], index=index_test)



    # Get predictions from model
    train_predictions = model.predict(X_train_in)

    # Get predicted price series (scaled and inverse_transformed)
    train_pred_price_scaled = pd.Series(train_predictions.ravel(),name='scaled_pred_train_price',index=train_index)
    train_pred_price = inverse_transform_series(train_pred_price_scaled, scaler).rename('pred_train_price')

    # Get training data scaled and inverse transformed into its own dataframe
    train_price_scaled = pd.Series(y_train,name='scaled_train_price',index= train_index)
    train_price =inverse_transform_series(train_price_scaled,scaler).rename('train_price')

    df_train_data = pd.concat([train_price, train_pred_price, train_price_scaled, train_pred_price_scaled],axis=1)


    # Return results as Panel or 2 dataframes
    if return_as_dfs==False:
        results = {'train':df_train_data,'test':df_test_data}
        return results

    else:

        return df_train_data, df_test_data



def plot_true_vs_preds_subplots(train_price, test_price, pred_price, subplots=False, verbose=0,figsize=(12,5)):

    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    # Check for null values
    train_null = train_price.isna().sum()
    test_null = test_price.isna().sum()
    pred_null = pred_price.isna().sum()
    null_test = train_null + test_null+pred_null
    if null_test>0:

        train_price.dropna(inplace=True)
        test_price.dropna(inplace=True)
        pred_price.dropna(inplace=True)

        if verbose>0:
            print(f'Dropping {null_test} null values.')


    ## CREATE FIGURE AND AX(ES)
    if subplots==True:
        # fig = plt.figure(figsize=figsize)#, constrained_layout=True)
        # ax1 = plt.subplot2grid((2, 9), (0, 0), rowspan=2, colspan=4)
        # ax2 = plt.subplot2grid((2, 9),(0,4), rowspan=2, colspan=5)
        fig, (ax1,ax2) = plt.subplots(figsize=figsize, nrows=1, ncols=2, sharey=False)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)


    ## Define plot styles by train/test/pred data type
    style_dict = {'train':{},'test':{},'pred':{}}
    style_dict['train']={'lw':2,'color':'blue','ls':'-', 'alpha':1}
    style_dict['test']={'lw':1,'color':'orange','ls':'-', 'alpha':1}
    style_dict['pred']={'lw':2,'color':'green','ls':'--', 'alpha':0.7}


    # Plot train_price if it is not empty.
    if len(train_price)>0:
        ax1.plot(train_price, label='price-training',**style_dict['train'])


    # Plot test and predicted price
    ax1.plot(test_price, label='true test price',**style_dict['test'])
    ax1.plot(pred_price, label='predicted price', **style_dict['pred'])#, label=['true_price','predicted_price'])#, label='price-predictions')
    ax1.legend()
    ax1.set_title('S&P500 Price: Forecast by LSTM-Neural-Network')
    ax1.set_xlabel('Business Day-Hour')
    ax1.set_ylabel('Stock Price')

    # Plot a subplot with JUST the test and predicted prices
    if subplots==True:

        ax2.plot(test_price, label='true test price',**style_dict['test'])
        ax2.plot(pred_price, label='predicted price', **style_dict['pred'])#, label=['true_price','predicted_price'])#, label='price-predictions')
        ax2.legend()
        plt.title('Predicted vs. Actual Price - Test Data')
        ax2.set_xlabel('Business Day-Hour')
        ax2.set_ylabel('Stock Price')
        plt.subplots_adjust(wspace=1)#, hspace=None)[source]¶


    # # ANNOTATING RMSE
    # RMSE = np.sqrt(mean_squared_error(test_price,pred_price))
    # bbox_props = dict(boxstyle="square,pad=0.5", fc="white", ec="k", lw=0.5)

    # plt.annotate(f"RMSE: {RMSE.round(3)}",xycoords='figure fraction', xy=(0.085,0.85),bbox=bbox_props)
    plt.tight_layout()
    if subplots==True:
        return fig, ax1,ax2
    else:
        return fig, ax1

# fig, ax = plot_price_vs_preds(df_train_price['train_price'],df_test_price['test_price'],df_test_price['pred_price'])

def print_array_info(X, name='Array'):
    """Test function for verifying shapes and data ranges of input arrays"""
    Xt=X
    print('X type:',type(Xt))
    print(f'X.shape = {Xt.shape}')
    print(f'\nX[0].shape = {Xt[0].shape}')
    print(f'X[0] contains:\n\t',Xt[0])


def arr2series(array,series_index=[],series_name='predictions'):
    """Accepts an array, an index, and a name. If series_index is longer than array:
    the series_index[-len(array):] """
    import pandas as pd
    if len(series_index)==0:
        series_index=list(range(len(array)))

    if len(series_index)>len(array):
        new_index= series_index[-len(array):]
        series_index=new_index

    preds_series = pd.Series(array.ravel(), index=series_index, name=series_name)
    return preds_series


def get_true_vs_model_pred_df(model, n_input, test_generator, test_data_index, df_test, train_generator, train_data_index, df_train, scaler=None,
                              inverse_tf=True, plot=True, verbose=2):
    """Accepts a model, the training and testing data TimeseriesGenerators, the test_index and train_index.
    Returns a dataframe with True and Predicted Values for Both the Training and Test Datasets."""
    import pandas as pd
    ## GET PREDICTIONS FROM MODEL
    test_predictions = pd.Series(model.predict_generator(test_generator).ravel(),
                                 index=test_data_index[n_input:], name='Predicted Test Price')

    train_predictions = pd.Series(model.predict_generator(train_generator).ravel(),
                                  index=train_data_index[n_input:], name='Predicted Training Price')

    # Make a series for true price to plot
    test_true_price = pd.Series( df_test['price'].rename('True Test Price').iloc[n_input:],
                                index= test_data_index[n_input:], name='True Test Price')

    train_true_price = pd.Series(df_train['price'].rename('True Training Price').iloc[n_input:],
                                 index = train_data_index[n_input:], name='True Train Price')

    # Combine all 4 into one dataframe
    df_show = pd.concat([train_true_price,train_predictions,test_true_price,test_predictions], axis=1)


    # CONVERT BACK TO ORIGINAL UNIT SCALE
    if inverse_tf==True:

        if scaler:
            for col in df_show.columns:
                df_show[col] = inverse_transform_series(df_show[col],scaler)

        else:
            raise Exception('Must pass a fit scaler to inverse_tf the units.')


    # PREVIEW DATA
    if verbose>1:
        df_show.head()

    if plot==True:
        plot_true_vs_preds_subplots(df_show['True Train Price'],df_show['True Test Price'],
                                    df_show['Predicted Test Price'], subplots=True)

    return df_show







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
    # from bs_ds import list2df
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

    df_results = list2df(list_of_results, index_col='Term')
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





###################### TWITTER AND STOCK PRICE DATA ######################
## twitter_df, stock_price = load_twitter_df_stock_price()
## twitter_df = get_stock_prices_for_twitter_data(twitter_df, stock_prices)
#
def load_twitter_df_stock_price(twitter_df,stock_price=None):# del stock_price
    from IPython.display import display
    if stock_price is None:
        stock_price = load_stock_price_series()


    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    # try: twitter_df
    # except NameError: twitter_df=None
    # if twitter_df is None:
    #     print('Loading twitter_df')
    #     twitter_df= load_raw_twitter_file()
    #     twitter_df = full_twitter_df_processing(twitter_df,cleaned_tweet_col='clean_content')

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)

    display(twitter_df.head())
    print(stock_price.index[0],stock_price.index[-1])
    print(twitter_df.index[0],twitter_df.index[-1])

    return twitter_df, stock_price

def get_stock_prices_for_twitter_data(twitter_df, stock_prices):
    # Get get the business day index to account for tweets during off-hours
    import pandas as pd
    import numpy as np

    twitter_df = get_B_day_time_index_shift(twitter_df,verbose=1)

    # Make temporary B_dt_index var in order to round that column to minute-resolution
    B_dt_index = twitter_df[['B_dt_index','B_day']]#.asfreq('T')
    B_dt_index['B_dt_index']= pd.to_datetime(B_dt_index['B_dt_index'])
    B_dt_index['B_dt_index']= B_dt_index['B_dt_index'].dt.round('T')

    # Get stock_prices for each twitter timestamp
    twitter_df['B_dt_minutes'] = B_dt_index['B_dt_index'].copy()
    twitter_df['stock_price_results'] = twitter_df['B_dt_minutes'].apply(lambda x: match_stock_price_to_tweets(x,time_after_tweet=60,stock_price=stock_prices))

    df_to_add = twitter_df['stock_price_results'].apply(lambda x: unpack_match_stocks(x))

    new_twitter_df = pd.concat([twitter_df,df_to_add], axis=1)


    twitter_df = new_twitter_df.loc[~new_twitter_df['post_tweet_price'].isna()]
    # twitter_df.drop(['0'],axis=1,inplace=True)
    twitter_df['delta_price_class'] = np.where(twitter_df['delta_price'] > 0,'pos','neg')

    twitter_df.drop([0],axis=1, inplace=True)
    # display(twitter_df.head())
    print(twitter_df.isna().sum())

    return twitter_df




def train_test_val_split(X,y,test_size=0.20,val_size=0.1):
    """Performs 2 successive train_test_splits to produce a training, testing, and validation dataset"""
    from sklearn.model_selection import train_test_split

    if val_size==0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
    else:

        first_split_size = test_size + val_size
        second_split_size = val_size/(test_size + val_size)

        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=first_split_size)

        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=second_split_size)

        return X_train, X_test, X_val, y_train, y_test, y_val



def plot_keras_history(history, title_text='',fig_size=(6,6),save_fig=False,no_val_data=False, filename_base='results/keras_history'):
    """Plots the history['acc','val','val_acc','val_loss']"""


    metrics = ['acc','loss','val_acc','val_loss']

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plot_metrics={}
    for metric in metrics:
        if metric in history.history.keys():
            plot_metrics[metric] = history.history[metric]

    # Set font styles:
    fontDict = {
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'title':{
            'fontsize':14,
            'fontweight':'normal',
            'ha':'center',
            }
        }
    # x = range(1,len(acc)+1)
    if no_val_data == True:
        fig_size = (fig_size[0],fig_size[1]//2)
        fig, ax = plt.subplots(figsize=fig_size)

        for k,v in plot_metrics.items():
            if 'acc' in k:
                color='b'
                label = 'Accuracy'
            if 'loss' in k:
                color='r'
                label = 'Loss'
            ax.plot(range(len(v)),v, label=label,color=color)

        plt.title('Model Training History')
        fig.suptitle(title_text,y=1.01,**fontDict['title'])
        ax.set_xlabel('Training Epoch',**fontDict['xlabel'])
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        plt.legend()
        plt.show()

    else:
        ## CREATE SUBPLOTS
        fig,ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, sharex=True)

        # Set color scheme for data type
        color_dict = {'val':'red','default':'b'}



        # Title Subplots
        fig.suptitle(title_text,y=1.01,**fontDict['title'])
        ax[1].set_xlabel('Training Epoch',**fontDict['xlabel'])

        ## Set plot params by metric and data type
        for metric, data in plot_metrics.items():
            x = range(1,len(data)+1)
            ## SET AXIS AND LABEL BY METRIC TYPE
            if 'acc' in metric.lower():
                ax_i = 0
                metric_title = 'Accuracy'

            elif 'loss' in metric.lower():
                ax_i=1
                metric_title = 'Loss'

            ## SET COLOR AND LABEL PREFIX BY DATA TYPE
            if 'val' in metric.lower():
                color = color_dict['val']
                data_label = 'Validation '+metric_title

            else:
                color = color_dict['default']
                data_label='Training ' + metric_title

            ## PLOT THE CURRENT METRIC AND LABEL
            ax[ax_i].plot(x, data, color=color,label=data_label)
            ax[ax_i].set_ylabel(metric_title,**fontDict['ylabel'])
            ax[ax_i].legend()

        plt.tight_layout()
        plt.show()

    if save_fig:
        if '.' not in filename_base:
            filename = filename_base+'.png'
        else:
            filename = filename_base
        fig.savefig(filename,facecolor='white', format='png', frameon=True)

        print(f'[io] Figure saved as {filename}')
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
    mask=[]
    mask = np.array(Image.open(filename))
    return mask


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


def get_time(timeformat='%m-%d-%y_%T%p',raw=False,filename_friendly= False,replacement_seperator='-'):
    """
    Gets current time in local time zone.
    if raw: True then raw datetime object returned without formatting.
    if filename_friendly: replace ':' with replacement_separator
    """
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


def auto_filename_time(prefix='',sep=' ',suffix='',ext='',fname_friendly=True,timeformat='%m-%d-%Y %T'):
    '''Generates a filename with a  base string + sep+ the current datetime formatted as timeformat.
     filename = f"{prefix}{sep}{suffix}{sep}{timesuffix}{ext}
    '''
    if prefix is None:
        prefix=''
    timesuffix=get_time(timeformat=timeformat, filename_friendly=fname_friendly)

    filename = f"{prefix}{sep}{suffix}{sep}{timesuffix}{ext}"
    return filename




def save_model_weights_params(model,model_params=None, filename_prefix = 'models/model', filename_suffix='', check_if_exists = True,
 auto_increment_name=True, auto_filename_suffix=True, save_model_layer_config_xlsx=True, sep='_', suffix_time_format = '%m-%d-%Y_%I%M%p'):
    """Saves a fit Keras model and its weights as a .json file and a .h5 file, respectively.
    auto_filename_suffix will use the date and time to give the model a unique name (avoiding overwrites).
    Returns the model_filename and weight_filename"""
    import json
    import pickle
    # from functions_combined_BEST import auto_filename_time
    # from bs_ds_local import auto_filename_time

    # create base model filename
    if auto_filename_suffix:
        filename = auto_filename_time(prefix=filename_prefix, sep=sep,timeformat=suffix_time_format )
    else:
        filename=filename_prefix


    ## Add suffix to filename
    full_filename = filename + filename_suffix
    full_filename = full_filename+'.json'


    ## check if file exists
    if check_if_exists:
        import os
        import pandas as pd
        current_files = os.listdir()

        # check if file already exists
        if full_filename in current_files and auto_increment_name==False:
            raise Exception('Filename already exists')

        elif full_filename in current_files and auto_increment_name==True:

            # check if filename ends in version #
            import re
            num_ending = re.compile(r'[vV].?(\d+).json')

            curr_file_num = num_ending.findall(full_filename)
            if len(curr_file_num)==0:
                v_num = '_v01'
            else:
                v_num = f"_{int(curr_file_num)+1}"

            full_filename = filename + v_num + '.json'

            print(f'{filename} already exists... incrementing filename to {full_filename}.')

    ## SAVE MODEL AS JSON FILE
    # convert model to json
    model_json = model.to_json()

    create_required_folders(full_filename)
    # save json model to json file
    with open(full_filename, "w") as json_file:
        json.dump(model_json,json_file)
    print(f'Model saved as {full_filename}')


    ## GET BASE FILENAME WITHOUT EXTENSION
    file_ext=full_filename.split('.')[-1]
    filename = full_filename.replace(f'.{file_ext}','')

    ## SAVE MODEL WEIGHTS AS HDF5 FILE
    weight_filename = filename+'_weights.h5'
    model.save_weights(weight_filename)
    print(f'Weights saved as {weight_filename}')


    ## SAVE MODEL LAYER CONFIG TO EXCEL FILE
    if save_model_layer_config_xlsx == True:

        excel_filename=filename+'_model_layers.xlsx'
        # Get modelo config df
        df_model_config = get_model_config_df(model)
        df_model_config.to_excel(excel_filename, sheet_name='Keras Model Config')
        print(f"Model configuration table saved as {excel_filename }")



    ## SAVE MODEL PARAMS TO PICKLE
    if model_params is not None:
        # import json
        import inspect
        import pickle# as pickle

        def replace_function(function):
            import inspect
            return inspect.getsource(function)

        ## Select good model params to save
        model_params_to_save = {}
        model_params_to_save['data_params'] = model_params['data_params']
        model_params_to_save['input_params'] = model_params['input_params']

        model_params_to_save['compile_params'] = {}
        model_params_to_save['compile_params']['loss'] = model_params['compile_params']['loss']

        ## Check for and replace functins in metrics
        metric_list =  model_params['compile_params']['metrics']

        # replace functions in metric list with source code
        for i,metric in enumerate(metric_list):
            if inspect.isfunction(metric):
                metric_list[i] = replace_function(metric)
        metric_list =  model_params['compile_params']['metrics']


        # model_params_to_save['compile_params']['metrics'] = model_params['compile_params']['metrics']
        model_params_to_save['compile_params']['optimizer_name'] = model_params['compile_params']['optimizer_name']
        model_params_to_save['fit_params'] = model_params['fit_params']

        ## save model_params_to_save to pickle
        model_params_filename=filename+'_model_params.pkl'
        try:
            with open(model_params_filename,'wb') as param_file:
                pickle.dump(model_params_to_save, param_file) #sort_keys=True,indent=4)
        except:
            print('Pickling failed')
    else:
        model_params_filename=''

    filename_dict = {'model':filename,'weights':weight_filename,'excel':excel_filename,'params':model_params_filename}
    return filename_dict#[filename, weight_filename, excel_filename, model_params_filename]


def load_model_weights_params(base_filename = 'models/model_',load_model_params=True, load_model_layers_excel=True, trainable=False,
model_filename=None,weight_filename=None, model_params_filename = None, excel_filename=None, verbose=1):
    """Loads in Keras model from json file and loads weights from .h5 file.
    optional set model layer trainability to False"""
    from IPython.display import display
    from keras.models import model_from_json
    import json

    ## Set model and weight filenames from base_filename if None:
    if model_filename is None:
        model_filename = base_filename+'.json'

    if weight_filename is None:
        weight_filename = base_filename+'_weights.h5'

    if model_params_filename is None:
        model_params_filename = base_filename + '_model_params.pkl'

    if excel_filename is None:
        excel_filename = base_filename + '_model_layers.xlsx'


    ## LOAD JSON MODEL
    with open(model_filename, 'r') as json_file:
        loaded_model_json = json.loads(json_file.read())
    loaded_model = model_from_json(loaded_model_json)

    ## LOAD MODEL WEIGHTS
    loaded_model.load_weights(weight_filename)
    print(f"Loaded {model_filename} and loaded weights from {weight_filename}.")

    # SET LAYER TRAINABILITY
    if trainable is False:
        for i, model_layer in enumerate(loaded_model.layers):
            loaded_model.get_layer(index=i).trainable=False
        if verbose>0:
            print('All model.layers.trainable set to False.')
        if verbose>1:
            print(model_layer,loaded_model.get_layer(index=i).trainable)

    # IF VERBOSE, DISPLAY SUMMARY
    if verbose>0:
        display(loaded_model.summary())
        print("Note: Model must be compiled again to be used.")


    ## START RETURN LIST WITH MODEL
    return_list = [loaded_model]

    ## LOAD MODEL_PARAMS PICKLE
    if load_model_params:
        import pickle
        model_params = pickle.load(model_params_filename)
        return_list.append(model_params)

    ## LOAD EXCEL OF MODEL LAYERS CONFIG
    if load_model_layers_excel:
        import pandas as pd
        df_model_layers = pd.read_excel(excel_filename)
        return_list.append(df_model_layers)

    return return_list[:]
    #     return loaded_model, model_params
    # else:
    #     return loaded_model



def display_dict_dropdown(dict_to_display ):
    """Display the model_params dictionary as a dropdown menu."""
    from ipywidgets import interact
    from IPython.display import display
    from pprint import pprint

    dash='---'
    print(f'{dash*4} Dictionary Contents {dash*4}')

    @interact(dict_to_display=dict_to_display)
    def display_params(dict_to_display):
        # print(dash)
        pprint(dict_to_display)
        return #params.values();



def show_random_img(image_array, n=1):
    """Display n rendomly-selected images from image_array"""
    from keras.preprocessing.image import array_to_img, img_to_array, load_img
    import numpy as np
    from IPython.display import display
    i=1
    while i <= n:
        choice = np.random.choice(range(0,len(image_array)))
        print(f'Image #:{choice}')
        display(array_to_img(image_array[choice]))
        i+=1
    return


def check_class_balance(df,col ='delta_price_class_int',note='',
                        as_percent=True, as_raw=True):
    import numpy as np
    dashes = '---'*20
    print(dashes)
    print(f'CLASS VALUE COUNTS FOR COL "{col}":')
    print(dashes)
    # print(f'Class Value Counts (col: {col}) {note}\n')

    ## Check for class value counts to see if resampling/balancing is needed
    class_counts = df[col].value_counts()

    if as_percent:
        print('- Classes (%):')
        print(np.round(class_counts/len(df)*100,2))
    # if as_percent and as_raw:
    #     # print('\n')
    if as_raw:
        print('- Class Counts:')
        print(class_counts)
    print('---\n')


def index_report(df, label='',time_fmt = '%Y-%m-%d %T', return_index_dict=False):
    """Sorts dataframe index, prints index's start and end points and its datetime frequency.
    if return_index_dict=True then it returns these values in a dictionary as well as printing them."""
    import pandas as pd
    df.sort_index(inplace=True)

    index_info = {'index_start': df.index[0].strftime(time_fmt), 'index_end':df.index[-1].strftime(time_fmt),
                'index_freq':df.index.freq}

    if df.index.freq is None:
        try:
            index_info['inferred_index_freq'] = pd.infer_freq(df.index)
        except:
            index_info['inferred_index_freq'] = 'error'
    dashes = '---'*20
    # print('\n')
    print(dashes)
    print(f"\tINDEX REPORT:\t{label}")
    print(dashes)
    print(f"* Index Endpoints:\n\t{df.index[0].strftime(time_fmt)} -- to -- {df.index[-1].strftime(time_fmt)}")
    print(f'* Index Freq:\n\t{df.index.freq}')
    # print('\n')
    # print(dashes)

    if return_index_dict == True:
        return index_info
    else:
        return



def undersample_df_to_match_classes(df,class_column='delta_price_class', class_values_to_keep=None,verbose=1):
    """Resamples (undersamples) input df so that the classes in class_column have equal number of occruances.
    If class_values_to_keep is None: uses all classes. """
    import pandas as pd
    import numpy as np

    ##  Get value counts and classes
    class_counts = df[class_column].value_counts()
    classes = list(class_counts.index)

    if verbose>0:
        print('Initial Class Value Counts:')
        print('%: ',class_counts/len(df))

    ## use all classes if None
    if class_values_to_keep is None:
        class_values_to_keep = classes


    ## save each group's indices in dict
    class_dict = {}
    for curr_class in classes:

        if curr_class in class_values_to_keep:
            class_dict[curr_class] = {}

            idx = df.loc[df[class_column]==curr_class].index

            class_dict[curr_class]['idx'] = idx
            class_dict[curr_class]['count'] = len(idx)
        else:
            continue


    ## determine which class count to match
    counts = [class_dict[k]['count'] for k in class_dict.keys()]
    # get number of samples to match
    count_to_match = np.min(counts)

    if len(np.unique(counts))==1:
        raise Exception('Classes are already balanced')

    # dict_resample = {}
    df_sampled = pd.DataFrame()
    for k,v in class_dict.items():
        temp_df = df.loc[class_dict[k]['idx']]
        temp_df =  temp_df.sample(n=count_to_match)
        # dict_resample[k] = temp_df
        df_sampled =pd.concat([df_sampled,temp_df],axis=0)

    ## sort index of final
    df_sampled.sort_index(ascending=False, inplace=True)

    # print(df_sampled[class_column].value_counts())

    if verbose>0:
        check_class_balance(df_sampled, col=class_column)
        # class_counts = [class_column].value_counts()

        # print('Final Class Value Counts:')
        # print('%: ',class_counts/len(df))

    return df_sampled

def show_del_me_code(called_by_inspect_vars=False):
    """Prints code to copy and paste into a cell to delete vars using a list of their names.
    Companion function inspect_variables(locals(),print_names=True) will provide var names tocopy/paste """
    from pprint import pprint
    if called_by_inspect_vars==False:
        print("#[i]Call: `inspect_variables(locals(), print_names=True)` for list of var names")

    del_me = """
    del_me= []#list of variable names
    for me in del_me:
        try:
            exec(f'del {me}')
            print(f'del {me} succeeded')
        except:
            print(f'del {me} failed')
            continue
        """
    print(del_me)


def check_null_small(df,null_index_column=None):# return_idx=False):
    import pandas as pd
    import numpy as np

    res = df.isna().sum()
    idx = res.loc[res>0].index
    print('\n')
    print('---'*10)
    print('Columns with Null Values')
    print('---'*10)
    print(res[idx])
    print('\n')
    if null_index_column is not None:
        idx_null = df.loc[ df[null_index_column].isna()==True].index
        # return_index = idx_null[idx_null==True]
        return idx_null



def find_null_idx(df,column=None):
    """returns the indices of null values found in the series/column.
    if df is a dataframe and column is none, it returns a dictionary
    with the column names as a value and  null_idx for each column as the values.
    Example Usage:
    1)
    >> null_idx = get_null_idx(series)
    >> series_null_removed = series[null_idx]
    2)
    >> null_dict = get_null_idx()
    """
    import pandas as pd
    import numpy as np
    idx_null = []
    # Raise an error if df is a series and a column name is given
    if isinstance(df, pd.Series) and column is not None:
        raise Exception('If passing a series, column must be None')
    # else if its a series, get its idx_null
    elif isinstance(df, pd.Series):
        series = df
        idx_null = series.loc[series.isna()==True].index

    # else if its a dataframe and column is a string:
    elif isinstance(df,pd.DataFrame) and isinstance(column,str):
            series=df[column]
            idx_null = series.loc[series.isna()==True].index

    # else if its a dataframe
    elif isinstance(df, pd.DataFrame):
        idx_null = {}

        # if no column name given, use all columns as col_list
        if column is None:
            col_list =  df.columns
        # else use input column as col_list
        else:
            col_list = column

        ## for each column, get its null idx and add to dictioanry
        for col in col_list:
            series = df[col]
            idx_null[col] = series.loc[series.isna()==True].index
    else:
        raise Exception('Input df must be a pandas DataFrame or Series.')
    ## return the index or dictionary idx_null
    return idx_null



def dict_dropdown(dict_to_display,title='Dictionary Contents'):
    """Display the model_params dictionary as a dropdown menu."""
    from ipywidgets import interact
    from IPython.display import display
    from pprint import pprint

    dash='---'
    print(f'{dash*4} {title} {dash*4}')

    @interact(dict_to_display=dict_to_display)
    def display_params(dict_to_display=dict_to_display):

        # # if the contents of the first level of keys is dicts:, display another dropdown
        # if dict_to_display.values()
        display(pprint(dict_to_display))
        return #params.values();


def display_df_dict_dropdown(dict_to_display, selected_key=None):
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import interact, interactive
    import pandas as pd

    key_list = list(dict_to_display.keys())
    key_list.append('_All_')

    if selected_key is not None:
        selected_key = selected_key

    def view(eval_dict=dict_to_display,selected_key=''):

        from IPython.display import display
        from pprint import pprint

        if selected_key=='_All_':

            key_list = list(eval_dict.keys())
            outputs=[]

            for k in key_list:

                if type(eval_dict[k]) == pd.DataFrame:
                    outputs.append(eval_dict[k])
                    display(eval_dict[k].style.set_caption(k).hide_index())
                else:
                    outputs.append(f"{k}:\n{eval_dict[k]}\n\n")
                    pprint('\n',eval_dict[k])

            return outputs#pprint(outputs)

        else:
                k = selected_key
#                 if type(eval_dict(k)) == pd.DataFrame:
                if type(eval_dict[k]) == pd.DataFrame:
                     display(eval_dict[k].style.set_caption(k))
                else:
                    pprint(eval_dict[k])
                return [eval_dict[k]]

    w= widgets.Dropdown(options=key_list,value='_All_', description='Key Word')

    # old, simple
    out = widgets.interactive_output(view, {'selected_key':w})


    # new, flashier
    output = widgets.Output(layout={'border': '1px solid black'})
    if type(out)==list:
        output.append_display_data(out)
#         out =widgets.HBox([x for x in out])
    else:
        output = out
#     widgets.HBox([])
    final_out =  widgets.VBox([widgets.HBox([w]),output])
    display(final_out)
    return final_out#widgets.VBox([widgets.HBox([w]),output])#out])


def def_plotly_date_range_widgets(my_rangeselector=None,as_layout=True,as_dict=False):
    """old name; def_my_plotly_stock_layout,
    REPLACES DEF_RANGE_SELECTOR"""
    if as_dict:
        as_layout=False

    from plotly import graph_objs as go
    if my_rangeselector is None:
        my_rangeselector={'bgcolor': 'lightgray', #rgba(150, 200, 250, 1)',
                            'buttons': [{'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                                        {'count':3,'label':'3m','step':'month','stepmode':'backward'},
                                        {'count':6,'label':'6m','step':'month','stepmode':'backward'},
                                        {'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward'},
                                        {'step':'all'}, {'count':1,'step':'year', 'stepmode':'todate'}
                                        ],
                        'visible': True}

    my_layout = {'xaxis':{
        'rangeselector': my_rangeselector,
        'rangeslider':{'visible':True},
                         'type':'date'}}

    if as_layout:
        return go.Layout(my_layout)
    else:
        return my_layout

def def_cufflinks_solar_theme(as_layout=True, as_dict=False):
    from plotly import graph_objs as go
    if as_dict:
        as_layout=False
    # if as_layout and as_dict:
        # raise Exception('only 1 of as_layout, as_dict can be True')

    theme_dict = {'annotations': {'arrowcolor': 'grey11', 'fontcolor': 'beige'},
     'bargap': 0.01,
     'colorscale': 'original',
     'layout': {'legend': {'bgcolor': 'black', 'font': {'color': 'beige'}},
                'paper_bgcolor': 'black',
                'plot_bgcolor': 'black',
                'titlefont': {'color': 'beige'},
                'xaxis': {'gridcolor': 'lightgray',
                          'showgrid': True,
                          'tickfont': {'color': 'darkgray'},
                          'titlefont': {'color': 'beige'},
                          'zerolinecolor': 'gray'},
                'yaxis': {'gridcolor': 'lightgrey',
                          'showgrid': True,
                          'tickfont': {'color': 'darkgray'},
                          'titlefont': {'color': 'beige'},
                          'zerolinecolor': 'grey'}},
     'linewidth': 1.3}

    theme = go.Layout(theme_dict['layout'])
    if as_layout:
        return theme
    if as_dict:
        return theme.to_plotly_json()



def def_plotly_solar_theme_with_date_selector_slider(as_layout=True, as_dict=False):
    ## using code above
    if as_dict:
        as_layout=False
    solar_theme = def_cufflinks_solar_theme(as_layout=True)#['layout']
    stock_range_widget_layout = def_plotly_date_range_widgets()
    new_layout = solar_theme.update(stock_range_widget_layout)
    # new_layout = merge_dicts_by_keys(solar_theme['layout'],my_layout)
    if as_layout:
        return new_layout
    if as_dict:
        return new_layout.to_plotly_json()




def match_data_colors(fig1,fig2):
    color_dict = {}
    for data in fig1['data']:
        name = data['name']
        color_dict[name] = {'color':data['line']['color']}

    data_list =  fig2['data']
    for i,trace in enumerate(data_list):
        if trace['name'] in color_dict.keys():
            data_list[i]['line']['color'] = color_dict[trace['name']]['color']
    fig2['data'] = data_list
    return fig1,fig2





def plotly_true_vs_preds_subplots(df_model_preds,
                                true_train_col='true_train_price',
                                true_test_col='true_test_price',
                                pred_test_columns='pred_from_gen',
                                subplot_mode='lines+markers',marker_size=5,
                                title='S&P 500 True Price Vs Predictions ($)',
                                theme='solar',
                                verbose=0,figsize=(1000,500),
                                       debug=False,
                                show_fig=True):
    """y_col_kws={'col_name':line_color}"""

    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    import cufflinks as cf
    cf.go_offline()

    from plotly.offline import iplot#download_plotlyjs, init_notebook_mode, plot, iplot
#     init_notebook_mode(connected=True)


    ### MAKE THE LIST OF COLUMNS TO CREATE SEPARATE DATAFRAMES TO PLOT
    if isinstance(pred_test_columns,str):
        pred_test_columns = [pred_test_columns]
    if pred_test_columns is None:
        exclude_list = [true_train_col,true_test_col]
        pred_test_columns = [col for col in df_model_preds.columns if col not in exclude_list]

    fig1cols = [true_train_col,true_test_col]
    fig2cols = [true_test_col]

    [fig1cols.append(x) for x in pred_test_columns]
    [fig2cols.append(x) for x in pred_test_columns]

    ## CREATE FIGURE DATAFRAMES
    fig1_df = df_model_preds[fig1cols]
    fig2_df = df_model_preds[fig2cols].dropna()




    ## Get my_layout
    fig_1 = plotly_time_series(fig1_df,theme=theme,show_fig=False, as_figure=True,
                                  iplot_kwargs={'mode':'lines'})

    fig_2 = plotly_time_series(fig2_df,theme=theme,show_fig=False,as_figure=True,
                                  iplot_kwargs={'mode':subplot_mode,
                                               'size':marker_size})

    fig_1,fig_2 = match_data_colors(fig_1,fig_2)

    ## Create base layout and add figsize
    base_layout = def_plotly_solar_theme_with_date_selector_slider()
    update_dict={'height':figsize[1],
                 'width':figsize[0],
                 'title': title,
                'xaxis':{'autorange':True, 'rangeselector':{'y':-0.3}},
                 'yaxis':{'autorange':True},
                 'legend':{'orientation':'h',
                 'y':1.0,
                 'bgcolor':None}
                }

    base_layout.update(update_dict)
    base_layout=base_layout.to_plotly_json()

    # Create combined figure with uneven-sized plots
    specs= [[{'colspan':3},None,None,{'colspan':2},None]]#specs= [[{'colspan':2},None,{'colspan':1}]]
    big_fig = cf.subplots(theme=theme,
                          base_layout=base_layout,
                          figures=[fig_1,fig_2],
                          horizontal_spacing=0.1,
                          shape=[1,5],specs=specs)#,
    # big_fig['layout']['legend']['bgcolor']=None
    big_fig['layout']['legend']['y'] = 1.0
    big_fig['layout']['xaxis']['rangeselector']['y']=-0.3
    big_fig['layout']['xaxis2']['rangeselector'] = {'bgcolor': 'lightgray',
                                                    'buttons': [
                                                        {'count': 1,
                                                         'label': '1d',
                                                         'step': 'day',
                                                         'stepmode': 'backward'},
                                                        {'step':'all'}
                                                    ],'visible': True,
                                                    'y':-0.5}
    update_layout_dict={
                        'yaxis':{
                            'title':{'text': 'True Train/Test Price vs Predictions',
                                     'font':{'color':'white'}}},
                        'yaxis2':{'title':{'text':'Test Price vs Pred Price',
                                           'font':{'color':'white'}}},
                        'title':{'text':'S&P 500 True Price Vs Predictions ($)',
                        'font':{'color':'white'},
                        'y':0.95, 'pad':{'b':0.1,'t':0.1}
                        }
                       }


    layout = go.Layout(big_fig['layout'])
    # title_layout = go.layout.Title(text='S&P 500 True Price Vs Predictions ($)',font={'color':'white'},pad={'b':0.1,'t':0.1}, y=0.95)#                                'font':{'color':'white'}
    layout = layout.update(update_layout_dict)
    # big_fig['layout'] = layout.to_plotly_json()
    big_fig = go.Figure(data=big_fig['data'],layout=layout)

    fig_dict={}
    fig_dict['fig_1']=fig_1
    fig_dict['fig_2'] =fig_2
    fig_dict['big_fig']=big_fig


    if show_fig:
        iplot(big_fig)
    if debug == True:
        return fig_dict
    else:
        return big_fig


def plotly_time_series(stock_df,x_col=None, y_col=None,layout_dict=None,title='S&P500 Hourly Price',theme='solar',
as_figure = True,show_fig=True,fig_dim=(900,400),iplot_kwargs=None): #,name='S&P500 Price'):
    import plotly
    from IPython.display import display

    # else:
    import plotly.offline as py
    from plotly.offline import plot, iplot, init_notebook_mode

    import plotly.tools as tls
    import plotly.graph_objs as go
    import cufflinks as cf
    cf.go_offline()
    init_notebook_mode(connected=False)

    # py.init_notebook_mode(connected=True)
    # Set title
    if title is None:
        title = "Time series with range slider and selector"

    # %matplotlib inline
    if plotly.__version__<'4.0':
        if theme=='solar':
            solar_layout = def_cufflinks_solar_theme(as_layout=True)
            range_widgets = def_plotly_date_range_widgets(as_layout=True)
            my_layout = solar_layout.update(range_widgets)
        else:
            my_layout = def_plotly_date_range_widgets()

        ## Define properties to update layout
        update_dict = {'title':
                    {'text': title},
                    'xaxis':{'title':{'text':'Market Trading Day-Hour'}},
                    'yaxis':{'title':{'text':'Closing Price (USD)'}},
                    'height':fig_dim[1],
                    'width':fig_dim[0]}
        my_layout.update(update_dict)


        ## UPDATE LAYOUT WITH ANY OTHER USER PARAMS
        if layout_dict is not None:
            my_layout = my_layout.update(layout_dict)

        if iplot_kwargs is None:

            # if no columns specified, use the whole df
            if (y_col is None) and (x_col is None):
                fig = stock_df.iplot( layout=my_layout,world_readable=True,asFigure=True)#asDates=True,

            # else plot y_col
            elif (y_col is not None) and (x_col is None):
                fig = stock_df[y_col].iplot(layout=my_layout,world_readable=True,asFigure=True)#asDates=True,

            #  else plot x_col vs y_col
            else:
                fig = stock_df.iplot(x=x_col,y=y_col,  layout=my_layout,world_readable=True,asFigure=True)#asDates=True,

        else:

            # if no columns specified, use the whole df
            if (y_col is None) and (x_col is None):
                fig = stock_df.iplot( layout=my_layout,world_readable=True,asFigure=True,**iplot_kwargs)#asDates=True,

            # else plot y_col
            elif (y_col is not None) and (x_col is None):
                fig = stock_df[y_col].iplot(asDates=True, layout=my_layout,world_readable=True,asFigure=True,**iplot_kwargs)

            #  else plot x_col vs y_col
            else:
                fig = stock_df.iplot(x=x_col,y=y_col,  layout=my_layout,world_readable=True,asFigure=True,**iplot_kwargs)#asDates=True,


    ## IF using verson v4.0 of plotly
    else:
        # LEARNING HOW TO CUSTOMIZE SLIDER
        # ** https://plot.ly/python/range-slider/
        fig = go.Figure()

        fig.update_layout(
            title_text=title
        )

        fig.add_trace(go.Scatter(x=stock_df[x_col], y=stock_df[y_col]))#, name=name)) #df.Date, y=df['AAPL.Low'], name="AAPL Low",
        #                          line_color='dimgray'))
        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(

                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            ),

            yaxis = go.layout.YAxis(
                        autorange=True,
                        title=go.layout.yaxis.Title(
                            text = 'S&P500 Price',
                            font=dict(
                                # family="Courier New, monospace",
                                size=18,
                                color="#7f7f7f")
                        )
                )
            )

    if show_fig:
        iplot(fig)
    if as_figure:
        return fig



def preview_dict(d, n=5,print_or_menu='print',return_list=False):
    """Previews the first n keys and values from the dict"""

    from pprint import pprint
    list_keys = list(d.keys())
    prev_d = {}
    for key in list_keys[:n]:
        prev_d[key]=d[key]

    if 'print' in print_or_menu:
        pprint(prev_d)
    elif 'menu' in print_or_menu:
        display_dict_dropdown(prev_d)
    else:
        raise Exception("print_or_menu must be 'print' or 'menu'")

    if return_list:
        out = [(k,v) for k,v in prev_d.items()]
        return out
    else:
        pass


def disp_df_head_tail(df,n_head=3, n_tail=3,head_capt='df.head',tail_capt='df.tail'):
    """Displays the df.head(n_head) and df.tail(n_tail) and sets captions using df.style"""
    from IPython.display import display
    import pandas as pd
    df_h = df.head(n_head).style.set_caption(head_capt)
    df_t = df.tail(n_tail).style.set_caption(tail_capt)
    display(df_h, df_t)


def create_required_folders(full_filenamepath,folder_delim='/',verbose=1):
    """Accepts a full file name path include folders with '/' as default delimiter.
    Recursively checks for all sub-folders in filepath and creates those that are missing."""
    import os
    ## Creating folders needed
    check_for_folders = full_filenamepath.split(folder_delim)#'/')

    # if the splits creates more than 1 filepath:
    if len(check_for_folders)==1:
        return print('[!] No folders detected in provided full_filenamepath')
    else:# len(check_for_folders) >1:

        # set first foler to check
        check_path = check_for_folders[0]

        if check_path not in os.listdir():
            if verbose>0:
                print(f'\t- creating folder "{check_path}"')
            os.mkdir(check_path)

        ## handle multiple subfolders
        if len(check_for_folders)>2:

            ## for each subfolder:
            for folder in check_for_folders[1:-1]:
                base_folder_contents = os.listdir(check_path)

                # add the subfolder to prior path
                check_path = check_path + '/' + folder

                if folder not in base_folder_contents:#os.listdir():
                    if verbose>0:
                        print(f'\t- creating folder "{check_path}"')
                    os.mkdir(check_path)
        if verbose>1:
            print('Finished. All required folders have been created.')
        else:
            return


def inspect_variables(local_vars = None,sort_col='size',exclude_funcs_mods=True, top_n=10,return_df=False,always_display=True,
show_how_to_delete=True,print_names=False):
    """Displays a dataframe of all variables and their size in memory, with the
    largest variables at the top."""
    import sys
    import inspect
    import pandas as pd
    from IPython.display import display
    if local_vars is None:
        raise Exception('Must pass "locals()" in function call. i.e. inspect_variables(locals())')


    glob_vars= [k for k in globals().keys()]
    loc_vars = [k for k in local_vars.keys()]

    var_list = glob_vars+loc_vars

    var_df = pd.DataFrame(columns=['variable','size','type'])

    exclude = ['In','Out']
    var_list = [x for x in var_list if (x.startswith('_') == False) and (x not in exclude)]

    i=0
    for var in var_list:#globals().items():#locals().items():

        if var in loc_vars:
            real_var = local_vars[var]
        elif var in glob_vars:
            real_var = globals()[var]
        else:
            print(f"{var} not found.")

        var_size = sys.getsizeof(real_var)

        var_type = []
        if inspect.isfunction(real_var):
            var_type = 'function'
            if exclude_funcs_mods:
                continue
        elif inspect.ismodule(real_var):
            var_type = 'module'
            if exclude_funcs_mods:
                continue
        elif inspect.isbuiltin(real_var):
            var_type = 'builtin'
        elif inspect.isclass(real_var):
            var_type = 'class'
        else:

            var_type = real_var.__class__.__name__


        var_row = pd.Series({'variable':var,'size':var_size,'type':var_type})
        var_df.loc[i] = var_row#pd.concat([var_df,var_row],axis=0)#.join(var_row,)
        i+=1

    # if exclude_funcs_mods:
    #     var_df = var_df.loc[var_df['type'] not in ['function', 'module'] ]

    var_df.sort_values(sort_col,ascending=False,inplace=True)
    var_df.reset_index(inplace=True,drop=True)
    var_df.set_index('variable',inplace=True)
    var_df = var_df[['type','size']]

    if top_n is not None:
        var_df = var_df.iloc[:top_n]



    if always_display:
        display(var_df.style.set_caption('Current Variables by Size in Memory'))

    if show_how_to_delete:
        print('---'*15)
        print('## CODE TO DELETE MANY VARS AT ONCE:')
        show_del_me_code(called_by_inspect_vars=True)


    if print_names ==False:
        print('#[i] set `print_names=True` for var names to copy/paste.')
        print('---'*15)
    else:
        print('---'*15)
        print('Variable Names:\n')
        print_me = [f"{str(x)}" for x in var_df.index]
        print(print_me)

    if return_df:
        return var_df



def replace_bad_filename_chars(filename,replace_spaces=False, replace_with='_'):
    """removes any characters not allowed in Windows filenames"""
    bad_chars= ['<','>','*','/',':','\\','|','?']
    if replace_spaces:
        bad_chars.append(' ')

    for char in bad_chars:
        filename=filename.replace(char,replace_with)

    # verify name is not too long for windows
    if len(filename)>255:
        filename = filename[:256]
    return filename




def evaluate_classification_model(model,  X_train,X_test,y_train,y_test, history=None,binary_classes=True,
                            conf_matrix_classes= ['Decrease','Increase'],
                            normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
                            history_filename ='results/keras_history.png', save_conf_matrix_png=False,
                            conf_mat_filename= 'results/confusion_matrix.png',save_summary=False,
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix.
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix

    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
        ## update filenames
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        conf_mat_filename = filename_dict['conf_mat']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    if history is not None:
        plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)

    print('\n- Evaluating Training Data:')
    loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
    print(f'    - Accuracy:{accuracy_train:{numFmt}}')
    print(f'    - Loss:{loss_train:{numFmt}}')

    print('\n- Evaluating Test Data:')
    loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
    print(f'    - Accuracy:{accuracy_test:{numFmt}}')
    print(f'    - Loss:{loss_test:{numFmt}}\n')


    ## Get model predictions
    y_hat_train = model.predict_classes(X_train)
    y_hat_test = model.predict_classes(X_test)

    if y_test.ndim>1 or binary_classes==False:
        if binary_classes==False: 
            pass
        else:
            binary_classes = False
            print(f"[!] y_test was >1 dim, setting binary_classes to False")
        
        ## reduce dimensions of y_train and y_test
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)


    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)

    ## Get sklearn classification report 
    report_str = classification_report(y_test,y_hat_test)
    report_dict = classification_report(y_test,y_hat_test,output_dict=True)
    
    
    try:
        ## Create and display classification report
        # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
        df_report_temp = pd.DataFrame(report_dict)
        df_report_temp = df_report_temp.T#reset_index(inplace=True)

        df_report = df_report_temp[['precision','recall','f1-score','support']]
        display(df_report.round(4).style.set_caption('Classification Report'))
        print('\n')
    
    except:
        print(report_str)
        # print(report_dict)
        df_report = pd.DataFrame()

    ## if saving the model.summary() printout 
    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(report_str)

    ## Create and plot confusion_matrix
    import matplotlib.pyplot as plt
    conf_mat = confusion_matrix(y_test, y_hat_test)
    with plt.rc_context(rc={'figure.figsize':conf_matrix_figsize}): # rcParams['figure.figsize']
        fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes,
                                    normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
    if save_conf_matrix_png:
        fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)
        
        


    return df_report, fig





def evaluate_regression_model(model, history, train_generator, test_generator,true_train_series,
true_test_series,include_train_data=True,return_preds_df = False, save_history=False, history_filename ='results/keras_history.png', save_summary=False,
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix.
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix

    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'summary':summary_filename}
        ## update filenames
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    plot_keras_history( history,filename_base=history_filename,no_val_data=True, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)

        # # EVALUATE MODEL PREDICTIONS FROM GENERATOR
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_train[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_train[0]:{numFmt}}')

    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_test[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_test[0]:{numFmt}}')

    x_window = test_generator.length
    n_features = test_generator.data[0].shape[0]
    gen_df = get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
        n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

    regr_results = evaluate_regression(y_true=gen_df['true_from_gen'], y_pred=gen_df['pred_from_gen'],show_results=True,
                                metrics=['r2', 'RMSE', 'U'])


    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(regr_results.__repr__())


    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,gen_df],axis=1)
    else:
        df_all_preds = gen_df

    if return_preds_df:
        return df_all_preds



def evaluate_regression(y_true, y_pred, metrics=None, show_results=False, display_thiels_u_info=False):
    """Calculates and displays any of the following evaluation metrics: (passed as strings in metrics param)
    r2, MAE,MSE,RMSE,U
    if metrics=None:
        metrics=['r2','RMSE','U']
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    from bs_ds import list2df
    import inspect

    idx_true_null = find_null_idx(y_true)
    idx_pred_null = find_null_idx(y_pred)
    if all(idx_true_null == idx_pred_null):
        y_true.dropna(inplace=True)
        y_pred.dropna(inplace=True)
    else:
        raise Exception('There are non-overlapping null values in y_true and y_pred')

    results=[['Metric','Value']]
    metric_list = []
    if metrics is None:
        metrics=['r2','rmse','u']

    else:
        for metric in metrics:
            if isinstance(metric,str):
                metric_list.append(metric.lower())
            elif inspect.isfunction(metric):
                custom_res = metric(y_true,y_pred)
                results.append([metric.__name__,custom_res])
                metric_list.append(metric.__name__)
        metrics=metric_list

    # metrics = [m.lower() for m in metrics]

    if any(m in metrics for m in ('r2','r squared','R_squared')): #'r2' in metrics: #any(m in metrics for m in ('r2','r squared','R_squared'))
        r2 = r2_score(y_true, y_pred)
        results.append(['R Squared',r2])##f'R\N{SUPERSCRIPT TWO}',r2])

    if any(m in metrics for m in ('RMSE','rmse','root_mean_squared_error','root mean squared error')): #'RMSE' in metrics:
        RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
        results.append(['Root Mean Squared Error',RMSE])

    if any(m in metrics for m in ('MSE','mse','mean_squared_error','mean squared error')):
        MSE = mean_squared_error(y_true,y_pred)
        results.append(['Mean Squared Error',MSE])

    if any(m in metrics for m in ('MAE','mae','mean_absolute_error','mean absolute error')):#'MAE' in metrics or 'mean_absolute_error' in metrics:
        MAE = mean_absolute_error(y_true,y_pred)
        results.append(['Mean Absolute Error',MAE])


    if any(m in metrics for m in ('u',"thiel's u")):# in metrics:
        if display_thiels_u_info is True:
            show_eqn=True
            show_table=True
        else:
            show_eqn=False
            show_table=False

        U = thiels_U(y_true, y_pred,display_equation=show_eqn,display_table=show_table )
        results.append(["Thiel's U", U])

    results_df = list2df(results)#, index_col='Metric')
    results_df.set_index('Metric', inplace=True)
    if show_results:
        from IPython.display import display
        dfs = results_df.round(3).reset_index().style.hide_index().set_caption('Evaluation Metrics')
        display(dfs)
    return results_df.round(4)




def plot_confusion_matrix(conf_matrix, classes = None, normalize=False,
                          title='Confusion Matrix', cmap=None,
                          print_raw_matrix=False,fig_size=(5,5), show_help=False):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds and modified"""
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    cm = conf_matrix
    ## Set plot style properties
    if cmap==None:
        cmap = plt.get_cmap("Blues")

    ## Text Properties
    fmt = '.2f' if normalize else 'd'

    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }


    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()

    if classes is None:
        classes = ['negative','positive']

    tick_marks = np.arange(len(classes))


    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])


    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), color='darkgray',**fontDict['data_labels'])#color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])
    fig = plt.gcf()
    plt.show()

    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix)

    if show_help:
        print('''For binary classifications:
        [[0,0(true_neg),  0,1(false_pos)]
        [1,0(false_neg), 1,1(true_pos)] ]

        to get vals as vars:
        >>  tn,fp,fn,tp=confusion_matrix(y_test,y_hat_test).ravel()
                ''')

    return fig


def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U


# def my_rmse(y_true,y_pred):
#     """RMSE calculation using keras.backend"""
#     from keras import backend as kb
#     sq_err = kb.square(y_pred - y_true)
#     mse = kb.mean(sq_err,axis=-1)
#     rmse =kb.sqrt(mse)
#     return rmse



def quick_ref_pandas_freqs():
    from IPython.display import Markdown, display
    mkdwn_notes = """
    - **Pandas Frequency Abbreviations**<br><br>

    | Alias | 	Description |
    |----|-----|
    |B|	business day frequency|
    |C|	custom business day frequency|
    |D|	calendar day frequency|
    |W|	weekly frequency|
    |M|	month end frequency|
    |SM|	semi-month end frequency (15th and end of month)|
    |BM|	business month end frequency|
    |CBM|	custom business month end frequency|
    |MS|	month start frequency|
    |SMS|	semi-month start frequency (1st and 15th)|
    |BMS|	business month start frequency|
    |CBMS|	custom business month start frequency|
    |Q|	quarter end frequency|
    |BQ|	business quarter end frequency|
    |QS|	quarter start frequency|
    |BQS|	business quarter start frequency|
    |A|, Y	year end frequency|
    |BA|, BY	business year end frequency|
    |AS|, YS	year start frequency|
    |BAS|, BYS	business year start frequency|
    |BH|	business hour frequency|
    |H|	hourly frequency|
    |T|, min	minutely frequency|
    |S|	secondly frequency|
    |L|, ms	milliseconds|
    |U|, us	microseconds|
    |N|	nanoseconds|
    """

    # **Time/data properties of Timestamps**<br><br>

    # |Property|	Description|
    # |---|---|
    # |year|	The year of the datetime|
    # |month|	The month of the datetime|
    # |day|	The days of the datetime|
    # |hour|	The hour of the datetime|
    # |minute|	The minutes of the datetime|
    # |second|	The seconds of the datetime|
    # |microsecond|	The microseconds of the datetime|
    # |nanosecond|	The nanoseconds of the datetime|
    # |date|	Returns datetime.date (does not contain timezone information)|
    # |time|	Returns datetime.time (does not contain timezone information)|
    # |timetz|	Returns datetime.time as local time with timezone information|
    # |dayofyear|	The ordinal day of year|
    # |weekofyear|	The week ordinal of the year|
    # |week|	The week ordinal of the year|
    # |dayofweek|	The number of the day of the week with Monday=0, Sunday=6|
    # |weekday|	The number of the day of the week with Monday=0, Sunday=6|
    # |weekday_name|	The name of the day in a week (ex: Friday)|
    # |quarter|	Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.|
    # |days_in_month|	The number of days in the month of the datetime|
    # |is_month_start|	Logical indicating if first day of month (defined by frequency)|
    # |is_month_end|	Logical indicating if last day of month (defined by frequency)|
    # |is_quarter_start|	Logical indicating if first day of quarter (defined by frequency)|
    # |is_quarter_end|	Logical indicating if last day of quarter (defined by frequency)|
    # |is_year_start|	Logical indicating if first day of year (defined by frequency)|
    # |is_year_end|	Logical indicating if last day of year (defined by frequency)|
    # |is_leap_year|	Logical indicating if the date belongs to a leap year|
    # """
    display(Markdown(mkdwn_notes))
    return


## REFERNCE FOR CONTENTS OF CONFIG (for writing function below)
def make_model_menu(model1, multi_index=True):

    import pandas as pd
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, interactive_output
    from bs_ds import list2df

    def get_model_config_df(model1, multi_index=True):
        model_config_dict = model1.get_config()
        model_layer_list=model_config_dict['layers']
        output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

        for num,layer_dict in enumerate(model_layer_list):
        #     layer_dict = model_layer_list[0]


            # layer_dict['config'].keys()
            # config_keys = list(layer_dict.keys())
            # combine class and name into 1 column
            layer_class = layer_dict['class_name']
            layer_name = layer_dict['config'].pop('name')
            col_000 = f"{num}: {layer_class}"
            col_00 = layer_name#f"{layer_class} ({layer_name})"

            # get layer's config dict
            layer_config = layer_dict['config']


            # config_keys = list(layer_config.keys())


            # for each parameter in layer_config
            for param_name,col2_v_or_dict in layer_config.items():
                # col_1 is the key( name of param)
            #     col_1 = param_name


                # check the contents of col2_:

                # if list, append col2_, fill blank cols
                if isinstance(col2_v_or_dict,dict)==False:
                    col_0 = 'top-level'
                    col_1 = param_name
                    col_2 = col2_v_or_dict

                    output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


                # else, set col_2 as the param name,
                if isinstance(col2_v_or_dict,dict):

                    param_sub_type = col2_v_or_dict['class_name']
                    col_0 = param_name +'  ('+param_sub_type+'):'

                    # then loop through keys,vals of col_2's dict for col3,4
                    param_dict = col2_v_or_dict['config']

                    for sub_param,sub_param_val in param_dict.items():
                        col_1 =sub_param
                        col_2 = sub_param_val
                        col_3 = ''


                        output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])

        df = list2df(output)
        if multi_index==True:
            df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
            df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        return df


    # https://blog.ouseful.info/2016/12/29/simple-view-controls-for-pandas-dataframes-using-ipython-widgets/
    def model_layer_config_menu(df):
        import ipywidgets as widgets
        from IPython.display import display
        from ipywidgets import interact, interactive
        import pandas as pd
        # from IPython.html.widgets import interactive


        ## SOLUION for getting values https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
        layer_names = pd.MultiIndex.get_level_values(df.index,level=0).unique().to_list()
        param_level = pd.MultiIndex.get_level_values(df.index, level=2).unique().to_list()

        # items = ['All']+sorted(df['layer_name'].unique().tolist())
        # layer_names.append('All') # 'All'+layer_names[:]#+param_level]
        layer_names=sorted(layer_names)
        layer_names.append('All')

        layer_levels = param_level
        layer_levels.append('All')


        def view(df=df,layer_num='',param_level=''):
            import pandas as pd
            idx = pd.IndexSlice

            if layer_num=='All':
                # df = df.sort_index(by='#')
                if param_level=='All':
                    # return display(df.sort_index(by='#'))
                    # return display(df.sort_index(by='#'))
                    df_out=df
                else:
                    # return display(df.loc[idx[:,:,param_level],:])#display(df.xs(param_level,level=2).sort_index(by='#'))
                    # return display(df.loc[idx[:,:,param_level],:].sort_index(by='#'))#display(df.xs(param_level,level=2).sort_index(by='#'))
                    df_out = df.loc[idx[:,:,param_level],:]

            else:
                if param_level=='All':
                    # return display(df.loc[idx[layer_num,:,:],:])#display(df.xs(layer_num,level=0))
                    # return display(df.loc[idx[layer_num,:,:],:].sort_index(by='#'))#display(df.xs(layer_num,level=0))
                    df_out = df.loc[idx[layer_num,:,:],:]
                else:
                    # return display(df.loc[idx[layer_num,:, param_level],:])#display(df.loc[layer][:][level]) #[df.xs(layer)])
                    # return display(df.loc[idx[layer_num,:, param_level],:].sort_index(by='#'))#display(df.loc[layer][:][level]) #[df.xs(layer)])
                    df_out = df.loc[idx[layer_num,:, param_level],:]

            display(df_out.sort_index(by='#').style.set_caption('Model Layer Parameters'))
            return df_out


        w = widgets.Select(options=layer_names,value='All',description='Layer #')
        # interactive(view,layer=w)

        w2 = widgets.Select(options=layer_levels,value='All',desription='Level')
        # interactive(view,layer=w,level=w2)

        out= widgets.interactive_output(view,{'layer_num':w,'param_level':w2})
        return widgets.VBox([widgets.HBox([w,w2]),out])

    ## APPLYING FUNCTIONS
    df = get_model_config_df(model1,multi_index=True)

    return model_layer_config_menu(df)

# interactive(view, Menu) #layer=Menu.children[0],level=Menu.children[1])

# df.head()
def make_qgrid_model_menu(model, return_df = False):

    df=get_model_config_df(model)
    import qgrid
    from IPython.display import display
    import pandas as pd

    pd.set_option('display.max_rows',None)

    qgrid_menu = qgrid.show_grid(df,  grid_options={'highlightSelectedCell':True}, show_toolbar=True)

    display(qgrid_menu)
    if return_df:
        return df
    else:
        return



def get_model_config_df(model1, multi_index=True):

    import pandas as pd
    from bs_ds import list2df
    pd.set_option('display.max_rows',None)

    model_config_dict = model1.get_config()
    model_layer_list=model_config_dict['layers']
    output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

    for num,layer_dict in enumerate(model_layer_list):
    #     layer_dict = model_layer_list[0]


        # layer_dict['config'].keys()
        # config_keys = list(layer_dict.keys())
        # combine class and name into 1 column
        layer_class = layer_dict['class_name']
        layer_name = layer_dict['config'].pop('name')

        # col_000 = f"{num}: {layer_class}"
        # col_00 = layer_name#f"{layer_class} ({layer_name})"

        # get layer's config dict
        layer_config = layer_dict['config']


        # config_keys = list(layer_config.keys())


        # for each parameter in layer_config
        for param_name,col2_v_or_dict in layer_config.items():
            # col_1 is the key( name of param)
        #     col_1 = param_name


            col_000 = f"{num}: {layer_class}"

            ### DETERMINE LAYER_NAME WITH UNITS OF
            if 'units' in layer_config.keys():
                units = layer_config['units'] #col2_v_or_dict
                col_00 = layer_name+' ('+str(units)+' units)'

            elif 'batch_input_shape' in layer_config.keys():
                input_length =  layer_config['input_length']
                output_dim =  layer_config['output_dim']
                col_00 = layer_name+' \n('+str(input_length)+' words, '+str(output_dim)+')'
            else:
                col_00 = layer_name#+' '+f"({}"#f"{layer_class} ({layer_name})"

            # check the contents of col2_:

            # if list, append col2_, fill blank cols
            if isinstance(col2_v_or_dict,dict)==False:


                col_0 = 'top-level'
                col_1 = param_name
                col_2 = col2_v_or_dict

                output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


            # else, set col_2 as the param name,
            if isinstance(col2_v_or_dict,dict):

                param_sub_type = col2_v_or_dict['class_name']
                col_0 = param_name +'  ('+param_sub_type+'):'

                # then loop through keys,vals of col_2's dict for col3,4
                param_dict = col2_v_or_dict['config']

                for sub_param,sub_param_val in param_dict.items():
                    col_1 =sub_param
                    col_2 = sub_param_val
                    # col_3 = ''


                    output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])

    df = list2df(output)
    if multi_index==True:
        df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
        df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        df.sort_index(level=0, inplace=True)
    return df


from sklearn.model_selection._split import _BaseKFold
class BlockTimeSeriesSplit(_BaseKFold): #sklearn.model_selection.TimeSeriesSplit):
    """A variant of sklearn.model_selection.TimeSeriesSplit that keeps train_size and test_size
    constant across folds.
    Requires n_splits,train_size,test_size. train_size/test_size can be integer indices or float ratios """
    def __init__(self, n_splits=5,train_size=None, test_size=None, step_size=None, method='sliding'):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        if 'sliding' in method or 'normal' in method:
            self.method = method
        else:
            raise  Exception("Method may only be 'normal' or 'sliding'")

    def split(self,X,y=None, groups=None):
        import numpy as np
        import math
        method = self.method
        ## Get n_samples, trian_size, test_size, step_size
        n_samples = len(X)
        test_size = self.test_size
        train_size =self.train_size


        ## If train size and test sze are ratios, calculate number of indices
        if train_size<1.0:
            train_size = math.floor(n_samples*train_size)

        if test_size <1.0:
            test_size = math.floor(n_samples*test_size)

        ## Save the sizes (all in integer form)
        self._train_size = train_size
        self._test_size = test_size

        ## calcualte and save k_fold_size
        k_fold_size = self._test_size + self._train_size
        self._k_fold_size = k_fold_size



        indices = np.arange(n_samples)

        ## Verify there is enough data to have non-overlapping k_folds
        if method=='normal':
            import warnings
            if n_samples // self._k_fold_size <self.n_splits:
                warnings.warn('The train and test sizes are too big for n_splits using method="normal"\n\
                switching to method="sliding"')
                method='sliding'
                self.method='sliding'



        if method=='normal':

            margin = 0
            for i in range(self.n_splits):

                start = i * k_fold_size
                stop = start+k_fold_size

                ## change mid to match my own needs
                mid = int(start+self._train_size)
                yield indices[start: mid], indices[mid + margin: stop]


        elif method=='sliding':

            step_size = self.step_size
            if step_size is None: ## if no step_size, calculate one
                ## DETERMINE STEP_SIZE
                last_possible_start = n_samples-self._k_fold_size #index[-1]-k_fold_size)\
                step_range =  range(last_possible_start)
                step_size = len(step_range)//self.n_splits
            self._step_size = step_size


            for i in range(self.n_splits):
                if i==0:
                    start = 0
                else:
                    start = prior_start+self._step_size #(i * step_size)

                stop =  start+k_fold_size
                ## change mid to match my own needs
                mid = int(start+self._train_size)
                prior_start = start
                yield indices[start: mid], indices[mid: stop]



def get_model_preds_from_gen(model,test_generator, true_test_data, model_params=None,
                       n_input=None, n_features=None, suffix=None, verbose=0,return_df=True):
        """
        Gets prediction from model using the generator's timeseries using model.predict_generator()
        Must provide a model_params dictionary with 'input_params' OR must define ('n_input','n_features').

        """
        import pandas as pd
        import numpy as np

        if model_params is not None:
            n_input= model_params['input_params']['n_input']
            n_features = model_params['input_params']['n_features']

        if model_params is None:
            if n_input is None:
                n_input= test_generator.length
            if n_features is None:
                n_features=test_generator.data[0].shape[0]

        # GET TRUE VALUES AND DATETIME INDEX FROM GENERATOR

        # Get true time index from the generator's start_index and end_index
        gen_index = true_test_data.index[test_generator.start_index:test_generator.end_index+1]
        gen_true_targets = test_generator.targets[test_generator.start_index:test_generator.end_index+1]

        # Generate predictions from the test_generator
        gen_preds = model.predict_generator(test_generator)
        gen_preds_flat = gen_preds.ravel()
        gen_true_targets = gen_true_targets.ravel()


        # RETURN OUTPUT AS DATAFRAME OR ARRAY OF PREDS
        if return_df == False:
            return gen_preds

        else:
            # Combine the outputs
            if verbose>0:
                print(len(gen_index),len(gen_true_targets), len(gen_preds_flat))

            gen_pred_df = pd.DataFrame({'index':gen_index,'true':gen_true_targets,'pred':gen_preds_flat})
            gen_pred_df['index'] = pd.to_datetime(gen_pred_df['index'])
            gen_pred_df.set_index('index',inplace=True)

            if suffix is not None:
                colnames = [name+suffix for name in gen_pred_df.columns]
            else:
                colnames = gen_pred_df.columns
            gen_pred_df.columns=colnames
            return gen_pred_df




def save_ihelp_to_file(function,save_help=False,save_code=True, 
                        as_md=False,as_txt=True,
                        folder='readme_resources/ihelp_outputs/',
                        filename=None,file_mode='w'):
    """Saves the string representation of the ihelp source code as markdown. 
    Filename should NOT have an extension. .txt or .md will be added based on
    as_md/as_txt.
    If filename is None, function name is used."""

    if as_md & as_txt:
        raise Exception('Only one of as_md / as_txt may be true.')

    import sys
    from io import StringIO
    ## save original output to restore
    orig_output = sys.stdout
    ## instantiate io stream to capture output
    io_out = StringIO()
    ## Redirect output to output stream
    sys.stdout = io_out
    
    if save_code:
        print('### SOURCE:')
        help_md = get_source_code_markdown(function)
        ## print output to io_stream
        print(help_md)
        
    if save_help:
        print('### HELP:')
        help(function)
        
    ## Get printed text from io stream
    text_to_save = io_out.getvalue()
    

    ## MAKE FULL FILENAME
    if filename is None:

        ## Find the name of the function
        import re
        func_names_exp = re.compile('def (\w*)\(')
        func_name = func_names_exp.findall(text_to_save)[0]    
        print(f'Found code for {func_name}')

        save_filename = folder+func_name#+'.txt'
    else:
        save_filename = folder+filename

    if as_md:
        ext = '.md'
    elif as_txt:
        ext='.txt'

    full_filename = save_filename + ext
    
    with open(full_filename,file_mode) as f:
        f.write(text_to_save)
        
    print(f'Output saved as {full_filename}')
    
    sys.stdout = orig_output



def get_source_code_markdown(function):
    """Retrieves the source code as a string and appends the markdown
    python syntax notation"""
    import inspect
    from IPython.display import display, Markdown
    source_DF = inspect.getsource(function)            
    output = "```python" +'\n'+source_DF+'\n'+"```"
    return output

def save_ihelp_menu_to_file(function_list, filename,save_help=False,save_code=True, 
    folder='readme_resources/ihelp_outputs/',as_md=True, as_txt=False,verbose=1):
    """Accepts a list of functions and uses save_ihelp_to_file with mode='a' 
    to combine all outputs. Note: this function REQUIRES a filename"""
    if as_md:
        ext='.md'
    elif as_txt:
        ext='.txt'

    for function in function_list:
        save_ihelp_to_file(function=function,save_help=save_help, save_code=save_code,
                              as_md=as_md, as_txt=as_txt,folder=folder,
                              filename=filename,file_mode='a')

    if verbose>0:
        print(f'Functions saved as {folder+filename+ext}')


class Clock(object):
    """A clock meant to be used as a timer for functions using local time.
    Clock.tic() starts the timer, .lap() adds the current laps time to clock._list_lap_times, .toc() stops the timer.
    If user initiializes with verbose =0, only start and final end times are displays.
        If verbose=1, print each lap's info at the end of each lap.
        If verbose=2 (default, display instruction line, return datafarme of results.)
    """

    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone
    from bs_ds import list2df
    # from bs_ds import list2df

    def get_time(self,local=True):
        """Returns current time, in local time zone by default (local=True)."""
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _now_utc_=datetime.now(timezone('UTC'))
        _now_local_=_now_utc_.astimezone(self._timezone_)
        if local==True:
            time_now = _now_local_

            return time_now#_now_local_
        else:
            return _now_utc_


    def __init__(self, display_final_time_as_minutes=True, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        self._strformat_ = []
        self._timezone_ = []
        self._timezone_ = get_localzone()
        self._start_time_ = []
        self._lap_label_ = []
        self._lap_end_time_ = []
        self._verbose_ = verbose
        self._lap_duration_ = []
        self._verbose_ = verbose
        self._prior_start_time_ = []
        self._display_as_minutes_ = display_final_time_as_minutes

        strformat = "%m/%d/%y - %I:%M:%S %p"
        self._strformat_ = strformat

    def mark_lap_list(self, label=None):
        """Used internally, appends the current laps' information when called by .lap()
        self._lap_times_list_ = [['Lap #' , 'Start Time','Stop Time', 'Stop Label', 'Duration']]"""
        import import bs_ds_local as bs bs
#         print(self._prior_start_time_, self._lap_end_time_)

        if label is None:
            label='--'

        duration = self._lap_duration_.total_seconds()
        self._lap_times_list_.append([ self._lap_counter_ , # Lap #
                                      (self._prior_start_time_).strftime(self._strformat_), # This Lap's Start Time
                                      self._lap_end_time_,#.strftime(self._strformat_), # stop clock time
                                      label,#self._lap_label_, # The Label passed with .lap()
                                      f'{duration:.3f} sec']) # the lap duration


    def tic(self, label=None ):
        "Start the timer and display current time, appends label to the _list_lap_times."
        from datetime import datetime
        from pytz import timezone

        self._start_time_ = self.get_time()
        self._start_label_ = label
        self._lap_counter_ = 0
        self._prior_start_time_=self._start_time_
        self._lap_times_list_=[]

        # Initiate lap counter and list
        self._lap_times_list_ = [['Lap #','Start Time','Stop Time', 'Label', 'Duration']]
        self._lap_counter_ = 0
        self._decorate_ = '--- '
        decorate=self._decorate_
        base_msg = f'{decorate}CLOCK STARTED @: {self._start_time_.strftime(self._strformat_):>{25}}'

        if label == None:
            display_msg = base_msg+' '+ decorate
            label='--'
        else:
            spacer = ' '
            display_msg = base_msg+f'{spacer:{10}} Label: {label:{10}} {decorate}'
        if self._verbose_>0:
            print(display_msg)#f'---- Clock started @: {self._start_time_.strftime(self._strformat_):>{25}} {spacer:{10}} label: {label:{20}}  ----')

    def toc(self,label=None, summary=True):
        """Stop the timer and displays results, appends label to final _list_lap_times entry"""
        if label == None:
            label='--'
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone
        from bs_ds import list2df
        if label is None:
            label='--'

        _final_end_time_ = self.get_time()
        _total_time_ = _final_end_time_ - self._start_time_
        _end_label_ = label

        self._lap_counter_+=1
        self._final_end_time_ = _final_end_time_
        self._lap_label_=_end_label_
        self._lap_end_time_ = _final_end_time_.strftime(self._strformat_)
        self._lap_duration_ = _final_end_time_ - self._prior_start_time_
        self._total_time_ = _total_time_

        decorate=self._decorate_
        # Append Summary Line
        if self._display_as_minutes_ == True:
            total_seconds = self._total_time_.total_seconds()
            total_mins = int(total_seconds // 60)
            sec_remain = total_seconds % 60
            total_time_to_display = f'{total_mins} min, {sec_remain:.3f} sec'
        else:

            total_seconds = self._total_time_.total_seconds()
            sec_remain = round(total_seconds % 60,3)

            total_time_to_display = f'{sec_remain} sec'
        self._lap_times_list_.append(['TOTAL',
                                      self._start_time_.strftime(self._strformat_),
                                      self._final_end_time_.strftime(self._strformat_),
                                      label,
                                      total_time_to_display]) #'Total Time: ', total_time_to_display])

        if self._verbose_>0:
            print(f'--- TOTAL DURATION   =  {total_time_to_display:>{15}} {decorate}')

        if summary:
            self.summary()

    def lap(self, label=None):
        """Records time, duration, and label for current lap. Output display varies with clock verbose level.
        Calls .mark_lap_list() to document results in clock._list_lap_ times."""
        from datetime import datetime
        if label is None:
            label='--'
        _end_time_ = self.get_time()

        # Append the lap attribute list and counter
        self._lap_label_ = label
        self._lap_end_time_ = _end_time_.strftime(self._strformat_)
        self._lap_counter_+=1
        self._lap_duration_ = (_end_time_ - self._prior_start_time_)
        # Now update the record
        self.mark_lap_list(label=label)

        # Now set next lap's new _prior_start
        self._prior_start_time_=_end_time_
        spacer = ' '

        if self._verbose_>0:
            print(f'       - Lap # {self._lap_counter_} @:  \
            {self._lap_end_time_:>{25}} {spacer:{5}} Dur: {self._lap_duration_.total_seconds():.3f} sec.\
            {spacer:{5}}Label:  {self._lap_label_:{20}}')

    def summary(self):
        """Display dataframe summary table of Clock laps"""
        from bs_ds import list2df
        import pandas as pd
        from IPython.display import display
        df_lap_times = list2df(self._lap_times_list_)#,index_col='Lap #')
        df_lap_times.drop('Stop Time',axis=1,inplace=True)
        df_lap_times = df_lap_times[['Lap #','Start Time','Duration','Label']]
        dfs = df_lap_times.style.hide_index().set_caption('Summary Table of Clocked Processes').set_properties(subset=['Start Time','Duration'],**{'width':'140px'})
        display(dfs.set_table_styles([dict(selector='table, th', props=[('text-align', 'center')])]))

def list2df(list, index_col=None, set_caption=None, return_df=True,df_kwds=None): #, sort_values='index'):
    
    """ Quick turn an appened list with a header (row[0]) into a pretty dataframe.

        
        Args
            list (list of lists):
            index_col (string): name of column to set as index; None (Default) has integer index.
            set_caption (string):
            show_and_return (bool):
    
    EXAMPLE USE:
    >> list_results = [["Test","N","p-val"]] 
    
    # ... run test and append list of result values ...
    
    >> list_results.append([test_Name,length(data),p])
    
    ## Displays styled dataframe if caption:
    >> df = list2df(list_results, index_col="Test",
                     set_caption="Stat Test for Significance")
    

    """
    from IPython.display import display
    import pandas as pd
    df_list = pd.DataFrame(list[1:],columns=list[0],**df_kwds)
    
        
    if index_col is not None:
        df_list.reset_index(inplace=True)
        df_list.set_index(index_col, inplace=True)
        
    if set_caption is not None:
        dfs = df_list.style.set_caption()
        display(dfs)
    return df_list


# print('my_keras_functions loaded')
def my_rmse(y_true,y_pred):
    """RMSE calculation using keras.backend"""
    from keras import backend as kb
    sq_err = kb.square(y_pred - y_true)
    mse = kb.mean(sq_err,axis=-1)
    rmse =kb.sqrt(mse)
    return rmse

def quiet_mode(filter_warnings=True, filter_keras=True,in_function=True,verbose=0):
    """Convenience function to execute commands to silence warnings:
    - filter_warnings:
        - warnings.filterwarnings('ignore')
    - filter_keras:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    """

    cmd_warnings = "import warnings\nwarnings.filterwarnings('ignore')"
    cmd_keras = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    cmd_combined = '\n'.join((cmd_warnings,cmd_keras))

    if filter_warnings and filter_keras:
        if verbose>0: 
            print(cmd_combined)
        output = cmd_combined

    elif filter_warnings and filter_keras is False:
        if verbose>0: 
            print(cmd_warnings)
        output = cmd_warnings

    elif filter_warnings is False and filter_keras:
        if verbose>0: 
            print(cmd_keras)
        output = cmd_keras
    
    if in_function:
        # exec_string = output #scaled_test_data#"exec('"+output+"')"
        return output

    else:
        return exec(output)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# my_keras_functions
def def_data_params(stock_df, num_test_days=45, num_train_days=365,days_for_x_window=5,verbose=0):
    """
    data_params['num_test_days']     =  45 # Number of days for test data
    data_params['num_train_days']    = 365 # Number of days for training data - 5 days/week * 52 weeks
    data_params['days_for_x_window'] =   5 # Number of days to include as 1 X sequence for predictions
    data_params['periods_per_day'] = ji.get_day_window_size_from_freq( stock_df, ji.custom_BH_freq() )
    """
    import functions_combined_BEST as ji
    
    data_params={}
    data_params['num_test_days']     =  num_test_days # Number of days for test data
    data_params['num_train_days']    = num_train_days # Number of days for training data - 5 days/week * 52 weeks
    data_params['days_for_x_window'] =   days_for_x_window # Number of days to include as 1 X sequence for predictions
    
    # Calculate number of rows to bin for x_windows
    periods_per_day = ji.get_day_window_size_from_freq( stock_df, ji.custom_BH_freq() ) # get the # of rows that == 1 day
    x_window = periods_per_day * days_for_x_window#data_params['days_for_x_window'] 
    
    # Update data_params
    data_params['periods_per_day'] = periods_per_day
    data_params['x_window'] = x_window    
    # days_for_x_window = #data_params['days_for_x_window']


    if verbose>1:
        print(f'X_window size = {x_window} -- ({days_for_x_window} day(s) * {periods_per_day} rows/day)\n')
    
    if verbose>0:
#         ji.display_dict_dropdown(data_params)
        from pprint import pprint
        print("data_params.items()={\t")
        pprint(data_params)
        print('}\n')
    return data_params


def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator



def reshape_train_data_and_target(train_data_series, X_cols=None, y_cols='price',
                                  n_features=1,debug=False):
    """Reshapes the X_col and y_col data into proper shape for timeseris generator"""
    train_data = []
    train_targets = []
    import pandas as pd
    import numpy as np

    if isinstance(train_data_series, pd.DataFrame):
        # train_data_test = train_data_series.values
        train_data = train_data_series.values
        
        # Train X_data to array (if not specified, use all columns)
        if X_cols is None:
            train_data = train_data_series.values
        else:
            train_data = train_data_series[X_cols].values

        # if not specified, assume 'price' is taret_col
        if y_cols is None:
            y_cols = 'price'
        ## Train Target y_values
        train_targets = train_data_series[y_cols].values

    elif isinstance(train_data_series, pd.Series): #        
        train_data = train_data_series.values.reshape(-1,1)
        train_targets = train_data

    
    ## Reshape as neded
    if train_targets.ndim <2: 
        train_targets = train_targets.reshape(-1,1)

    if debug==True:
        print('train_data[0]=\n',train_data[0])
        print('train_targets[0]=\n',train_targets[0])
        
    return train_data, train_targets





def def_callbacks_and_params(model_params=None,loss_function='my_rmse',checkpoint_mode='min',filepath=None,
                             stop_mode='min',patience=1,min_delta=.001,verbose=1):
    import functions_combined_BEST as ji
    if 'my_rmse' in loss_function:
        def my_rmse(y_true,y_pred):
            """RMSE calculation using keras.backend"""
            from keras import backend as kb
            sq_err = kb.square(y_pred - y_true)
            mse = kb.mean(sq_err,axis=-1)
            rmse =kb.sqrt(mse)
            return rmse
        my_rmse=my_rmse
        loss_function = my_rmse

        
    ########## Define loss function and callback params ##########
    callback_params ={}
    callback_params['custom_loss_function'] = loss_function
    callback_params['custom_loss_function'] = loss_function
    callback_params['ModelCheckpoint'] = {'monitor': loss_function, 'mode':checkpoint_mode}
    callback_params['EarlyStopping'] = {'monitor':loss_function, 'mode':stop_mode, 
                                        'patience':patience, 'min_delta':min_delta}

    # CREATING CALLBACKS
    from keras import callbacks

    if filepath is None:
        filepath = f"models/checkpoints/model1_weights_{ji.auto_filename_time(prefix=None)}.hdf5"

    # Create ModelCheckPoint
    fun_params=callback_params['ModelCheckpoint']
    checkpoint = callbacks.ModelCheckpoint(filepath=filepath, monitor=fun_params['monitor'], mode=fun_params['mode'],
                                           save_best_only=False, verbose=verbose)
    # Create EarlyStopping
    fun_params=callback_params['EarlyStopping']
    early_stop = callbacks.EarlyStopping(monitor=my_rmse, mode=fun_params['mode'], patience=fun_params['patience'],
                                         min_delta=fun_params['min_delta'],verbose=verbose)
    callbacks = [checkpoint,early_stop]

    if model_params is None:
        model_params=callback_params
    else:
        model_params['callbacks'] = callback_params
    return callbacks, model_params


def def_compile_params_optimizer(loss='my_rmse',metrics=['acc','my_rmse'],optimizer='optimizers.Nadam()',model_params=None):
    ####### Specify additional model parameters
    from keras import optimizers
    
    if 'my_rmse' in loss or 'my_rmse' in metrics:
        def my_rmse(y_true,y_pred):
            """RMSE calculation using keras.backend"""
            from keras import backend as kb
            sq_err = kb.square(y_pred - y_true)
            mse = kb.mean(sq_err,axis=-1)
            rmse =kb.sqrt(mse)
            return rmse
        my_rmse=my_rmse
        
        
        # replace string with function in loss
        loss = my_rmse
        
        # replace string with function in metrics
        idx = metrics.index('my_rmse')
        metrics[idx]=my_rmse
        
    compile_params={}
    compile_params['loss']= loss#{'my_rmse':my_rmse}
    compile_params['metrics'] = metrics#['acc',my_rmse]

    if type(optimizer) is str:
        optimizer_name = optimizer
        optimizer = eval(optimizer_name)
    else:
        optimizer_name = optimizer.__class__().__str__()
        
    compile_params['optimizer'] = optimizer
    compile_params['optimizer_name'] = optimizer_name#'optimizers.Nadam()'

    if model_params is not None:
        model_params['compile_params'] = compile_params
    else:
        model_params=compile_params
    
    return model_params



def make_model1(model_params, summary=True):
    from keras.models import Sequential
    from keras.layers import Bidirectional, Dense, LSTM, Dropout
    from IPython.display import display
    from keras.regularizers import l2

    # Specifying input shape (size of samples, rank of samples?)
    n_input = model_params['input_params']['n_input']
    n_features = model_params['input_params']['n_features']
    
    input_shape=(n_input, n_features)
    
    # Create model architecture
    model = Sequential()
    model.add(LSTM(units=50, input_shape =input_shape,return_sequences=True))#,  kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(1))

    # load compile params and compile
    comp_params = model_params['compile_params']
    # metrics = comp_params['metrics']
    model.compile(loss=comp_params['loss'], metrics=comp_params['metrics'],
                  optimizer=comp_params['optimizer'])##eval(comp_params['optimizer']), metrics=metrics)#optimizer=optimizers.Nadam()
    
    if summary is True:
        display(model.summary())

    return model



def fit_model(model,train_generator,model_params=None,epochs=5,callbacks=None,verbose=2,workers=3):
    import import bs_ds_local as bs bs
    import functions_combined_BEST as ji
    from IPython.display import display

    quiet_command = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    exec(quiet_command)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # cmd_keras = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"

    
    if model_params is None:
        model_params={}
    model_params['fit_params'] = {'epochs':epochs,'callbacks':callbacks}

    # Instantiating clock timer
    clock = bs.Clock()

    print('---'*20)
    print('\tFITTING MODEL:')
    print('---'*20,'\n')     
    
    # start the timer
    clock.tic('')

    # Fit the model
    fit_params = model_params['fit_params']
    if callbacks is None:
        
        history = model.fit_generator(train_generator,epochs=fit_params['epochs'], 
                                       verbose=2, use_multiprocessing=True, workers=3)
    else:
        
        history = model.fit_generator(train_generator,epochs=fit_params['epochs'],
                                       callbacks=callbacks,
                                       verbose=2,use_multiprocessing=True, workers=3)

    # model_results = model.history.history
    clock.toc('')
    
    return model,model_params,history


def evaluate_model_plot_history(model, train_generator, test_generator,as_df=False, plot=True):
    """ Takes a keras model fit using fit_generator(), a train_generator and test generator.
    Extracts and plots Keras model.history's metrics."""
    from IPython.display import display
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import functions_combined_BEST as ji
    print('\n')
    print('---'*28)
    print('\tEVALUATE MODEL:')
    print('---'*28)
        # duration = print(clock._lap_duration_)
    model_results = model.history.history
    
    if plot==True and len(model.history.epoch)>1:

        # ji.plot_keras_history()
        fig, ax = plt.subplots(figsize=(6,3))

        for k,v in model_results.items():
            ax.plot(range(len(v)),v, label=k)
                
        plt.title('Model Training History')
        ax.set_xlabel('Epoch #',**{'size':12,'weight':70})
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        plt.legend()
        plt.show()


    # # EVALUATE MODEL PREDICTIONS FROM GENERATOR 
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)
    # print(model_metrics_test)

    eval_gen_dict = {}
    eval_gen_dict['Train Data'] = dict(zip(model.metrics_names,model_metrics_train))
    eval_gen_dict['Test Data'] = dict(zip(model.metrics_names,model_metrics_test))
    df_eval = pd.DataFrame(eval_gen_dict).round(4).T
    display(df_eval.style.set_caption('Model Evaluation Results'))

    if as_df:
        return df_eval
    else:
        return  eval_gen_dict


def get_model_config_df(model1, multi_index=True):

    import import bs_ds_local as bs bs
    import functions_combined_BEST as ji
    import pandas as pd
    pd.set_option('display.max_rows',None)

    model_config_dict = model1.get_config()
    try:
        model_layer_list=model_config_dict['layers']
    except:
        return model_config_dict
        raise Exception()
    output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

    for num,layer_dict in enumerate(model_layer_list):
    #     layer_dict = model_layer_list[0]


        # layer_dict['config'].keys()
        # config_keys = list(layer_dict.keys())
        # combine class and name into 1 column
        layer_class = layer_dict['class_name']
        layer_name = layer_dict['config'].pop('name')
        col_000 = f"{num}: {layer_class}"
        col_00 = layer_name#f"{layer_class} ({layer_name})"

        # get layer's config dict
        layer_config = layer_dict['config']


        # config_keys = list(layer_config.keys())


        # for each parameter in layer_config
        for param_name,col2_v_or_dict in layer_config.items():
            # col_1 is the key( name of param)
        #     col_1 = param_name


            # check the contents of col2_:

            # if list, append col2_, fill blank cols
            if isinstance(col2_v_or_dict,dict)==False:
                col_0 = 'top-level'
                col_1 = param_name
                col_2 = col2_v_or_dict

                output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


            # else, set col_2 as the param name,
            if isinstance(col2_v_or_dict,dict):

                param_sub_type = col2_v_or_dict['class_name']
                col_0 = param_name +'  ('+param_sub_type+'):'

                # then loop through keys,vals of col_2's dict for col3,4
                param_dict = col2_v_or_dict['config']

                for sub_param,sub_param_val in param_dict.items():
                    col_1 =sub_param
                    col_2 = sub_param_val
                    # col_3 = ''


                    output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])
        
    df = bs.list2df(output)    
    if multi_index==True:
        df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
        df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        df.sort_index(level=0, inplace=True)
    return df





def save_model_weights_params(model,model_params=None, filename_prefix = 'models/model', filename_suffix='', check_if_exists = True,
 auto_increment_name=True, auto_filename_suffix=True, save_model_layer_config_xlsx=True, sep='_', suffix_time_format = '%m-%d-%Y_%I%M%p'):
    """Saves a fit Keras model and its weights as a .json file and a .h5 file, respectively.
    auto_filename_suffix will use the date and time to give the model a unique name (avoiding overwrites).
    Returns the model_filename and weight_filename"""
    import json
    import pickle
    from functions_combined_BEST import auto_filename_time
    import functions_combined_BEST as ji

    # create base model filename 
    if auto_filename_suffix:
        filename = auto_filename_time(prefix=filename_prefix, sep=sep,timeformat=suffix_time_format )
    else:
        filename=filename_prefix
    

    ## Add suffix to filename
    full_filename = filename + filename_suffix
    full_filename = full_filename+'.json'


    ## check if file exists
    if check_if_exists:
        import os
        import pandas as pd
        current_files = os.listdir()

        # check if file already exists
        if full_filename in current_files and auto_increment_name==False:
            raise Exception('Filename already exists')
        
        elif full_filename in current_files and auto_increment_name==True:
        
            # check if filename ends in version #
            import re
            num_ending = re.compile(r'[vV].?(\d+).json')
            
            curr_file_num = num_ending.findall(full_filename)
            if len(curr_file_num)==0:
                v_num = '_v01'
            else:
                v_num = f"_{int(curr_file_num)+1}"

            full_filename = filename + v_num + '.json'

            print(f'{filename} already exists... incrementing filename to {full_filename}.')
    
    ## SAVE MODEL AS JSON FILE
    # convert model to json
    model_json = model.to_json()

    ji.create_required_folders(full_filename)
    # save json model to json file
    with open(full_filename, "w") as json_file:
        json.dump(model_json,json_file)
    print(f'Model saved as {full_filename}')


    ## GET BASE FILENAME WITHOUT EXTENSION
    file_ext=full_filename.split('.')[-1]
    filename = full_filename.replace(f'.{file_ext}','')    

    ## SAVE MODEL WEIGHTS AS HDF5 FILE
    weight_filename = filename+'_weights.h5'
    model.save_weights(weight_filename)
    print(f'Weights saved as {weight_filename}') 


    ## SAVE MODEL LAYER CONFIG TO EXCEL FILE 
    if save_model_layer_config_xlsx == True:

        excel_filename=filename+'_model_layers.xlsx'
        df_model_config = get_model_config_df(model)

        try:
            # Get modelo config df
            df_model_config.to_excel(excel_filename, sheet_name='Keras Model Config')
            print(f"Model configuration table saved as {excel_filename }")
        except:
            print('ERROR:df_model_config = get_model_config_df(model)')
            print(type(df_model_config))
            # print(df_model_config)

            


    ## SAVE MODEL PARAMS TO PICKLE 
    if model_params is not None:
        # import json
        import inspect
        import pickle# as pickle        
        
        def replace_function(function):
            import inspect
            return inspect.getsource(function)
        
        ## Select good model params to save
        model_params_to_save = {}
        model_params_to_save['data_params'] = model_params['data_params']
        model_params_to_save['input_params'] = model_params['input_params']
        
        model_params_to_save['compile_params'] = {}
        model_params_to_save['compile_params']['loss'] = model_params['compile_params']['loss']

        ## Check for and replace functins in metrics
        metric_list =  model_params['compile_params']['metrics']
        
        # replace functions in metric list with source code
        for i,metric in enumerate(metric_list):
            if inspect.isfunction(metric):
                metric_list[i] = replace_function(metric)
        metric_list =  model_params['compile_params']['metrics']


        # model_params_to_save['compile_params']['metrics'] = model_params['compile_params']['metrics']
        model_params_to_save['compile_params']['optimizer_name'] = model_params['compile_params']['optimizer_name']
        model_params_to_save['fit_params'] = model_params['fit_params']

        ## save model_params_to_save to pickle
        model_params_filename=filename+'_model_params.pkl'
        try:
            with open(model_params_filename,'wb') as param_file:
                pickle.dump(model_params_to_save, param_file) #sort_keys=True,indent=4)
        except:
            print('Pickling failed')
    else:
        model_params_filename=''

    filename_dict = {'model':filename,'weights':weight_filename,'excel':excel_filename,'params':model_params_filename}
    return filename_dict#[filename, weight_filename, excel_filename, model_params_filename]


def load_model_weights_params(base_filename = 'models/model_',load_model_params=True, load_model_layers_excel=True, trainable=False, 
model_filename=None,weight_filename=None, model_params_filename = None, excel_filename=None, verbose=1):
    """Loads in Keras model from json file and loads weights from .h5 file.
    optional set model layer trainability to False"""
    from IPython.display import display
    from keras.models import model_from_json
    import json
    
    ## Set model and weight filenames from base_filename if None:
    if model_filename is None:
        model_filename = base_filename+'.json'

    if weight_filename is None:
        weight_filename = base_filename+'_weights.h5'
    
    if model_params_filename is None:
        model_params_filename = base_filename + '_model_params.pkl'
    
    if excel_filename is None:
        excel_filename = base_filename + '_model_layers.xlsx'


    ## LOAD JSON MODEL
    with open(model_filename, 'r') as json_file:
        loaded_model_json = json.loads(json_file.read())
    loaded_model = model_from_json(loaded_model_json)

    ## LOAD MODEL WEIGHTS 
    loaded_model.load_weights(weight_filename)
    print(f"Loaded {model_filename} and loaded weights from {weight_filename}.")

    # SET LAYER TRAINABILITY
    if trainable is False:
        for i, model_layer in enumerate(loaded_model.layers):
            loaded_model.get_layer(index=i).trainable=False
        if verbose>0:
            print('All model.layers.trainable set to False.')
        if verbose>1:
            print(model_layer,loaded_model.get_layer(index=i).trainable)
    
    # IF VERBOSE, DISPLAY SUMMARY
    if verbose>0:
        display(loaded_model.summary())
        print("Note: Model must be compiled again to be used.")

    
    ## START RETURN LIST WITH MODEL
    return_list = [loaded_model]

    ## LOAD MODEL_PARAMS PICKLE
    if load_model_params:
        import pickle
        model_params = pickle.load(model_params_filename)
        return_list.append(model_params)

    ## LOAD EXCEL OF MODEL LAYERS CONFIG
    if load_model_layers_excel:
        import pandas as pd
        df_model_layers = pd.read_excel(excel_filename)
        return_list.append(df_model_layers)

    return return_list[:]
    #     return loaded_model, model_params
    # else:
    #     return loaded_model 


def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U


def plot_confusion_matrix(conf_matrix, classes = None, normalize=False,
                          title='Confusion Matrix', cmap=None,
                          print_raw_matrix=False,fig_size=(5,5), show_help=False):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds and modified"""
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    cm = conf_matrix
    ## Set plot style properties
    if cmap==None:
        cmap = plt.get_cmap("Blues")

    ## Text Properties
    fmt = '.2f' if normalize else 'd'

    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',   
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }


    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()
    
    if classes is None:
        classes = ['negative','positive']
        
    tick_marks = np.arange(len(classes))


    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])

    
    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), color='darkgray',**fontDict['data_labels'])#color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])
    fig = plt.gcf()
    plt.show()
    
    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix)

    if show_help:
        print('''For binary classifications:
        [[0,0(true_neg),  0,1(false_pos)]
        [1,0(false_neg), 1,1(true_pos)] ]
        
        to get vals as vars:
        >>  tn,fp,fn,tp=confusion_matrix(y_test,y_hat_test).ravel()
                ''')

    return fig


def evaluate_regression(y_true, y_pred, metrics=None, show_results=False, display_thiels_u_info=False):
    """Calculates and displays any of the following evaluation metrics: (passed as strings in metrics param)
    r2, MAE,MSE,RMSE,U 
    if metrics=None:
        metrics=['r2','RMSE','U']
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    from bs_ds import list2df
    import inspect
    
    import functions_combined_BEST as ji
    idx_true_null = ji.find_null_idx(y_true)
    idx_pred_null = ji.find_null_idx(y_pred)
    if all(idx_true_null == idx_pred_null):
        y_true.dropna(inplace=True)
        y_pred.dropna(inplace=True)
    else:
        raise Exception('There are non-overlapping null values in y_true and y_pred')

    results=[['Metric','Value']]
    metric_list = []
    if metrics is None:
        metrics=['r2','rmse','u']

    else:
        for metric in metrics:
            if isinstance(metric,str):
                metric_list.append(metric.lower())
            elif inspect.isfunction(metric):
                custom_res = metric(y_true,y_pred)
                results.append([metric.__name__,custom_res])
                metric_list.append(metric.__name__)
        metrics=metric_list

    # metrics = [m.lower() for m in metrics]

    if any(m in metrics for m in ('r2','r squared','R_squared')): #'r2' in metrics: #any(m in metrics for m in ('r2','r squared','R_squared'))
        r2 = r2_score(y_true, y_pred)
        results.append(['R Squared',r2])##f'R\N{SUPERSCRIPT TWO}',r2])
    
    if any(m in metrics for m in ('RMSE','rmse','root_mean_squared_error','root mean squared error')): #'RMSE' in metrics:
        RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
        results.append(['Root Mean Squared Error',RMSE])

    if any(m in metrics for m in ('MSE','mse','mean_squared_error','mean squared error')):
        MSE = mean_squared_error(y_true,y_pred)
        results.append(['Mean Squared Error',MSE])

    if any(m in metrics for m in ('MAE','mae','mean_absolute_error','mean absolute error')):#'MAE' in metrics or 'mean_absolute_error' in metrics:
        MAE = mean_absolute_error(y_true,y_pred)
        results.append(['Mean Absolute Error',MAE])

    
    if any(m in metrics for m in ('u',"thiel's u")):# in metrics:
        if display_thiels_u_info is True:
            show_eqn=True
            show_table=True
        else:
            show_eqn=False 
            show_table=False

        U = thiels_U(y_true, y_pred,display_equation=show_eqn,display_table=show_table )
        results.append(["Thiel's U", U])
    
    results_df = list2df(results)#, index_col='Metric')
    results_df.set_index('Metric', inplace=True)
    if show_results:
        from IPython.display import display
        dfs = results_df.round(3).reset_index().style.hide_index().set_caption('Evaluation Metrics')
        display(dfs)
    return results_df.round(4)



def res_dict_to_merged_df(dict_of_dfs, key_index_name='Prediction Source', old_col_index_name=None):
    import pandas as pd
    res_dict = dict_of_dfs
    # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
    rename_mapper = {'R_squared':'R^2','R Squared':'R^2','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
    if len(res_dict.keys())==1:
        
        res_df = res_dict[list(res_dict.keys())[0]]

        # res_df.set_index('Metric',inplace=True)
        res_df.rename(mapper=rename_mapper, axis='index',inplace=True)
        res_df=res_df.transpose()
        # caption='Evaluation Metrics'

    else:
        res_df= pd.concat(res_dict.values(), axis=1,keys=res_dict.keys())
        res_df.columns = res_df.columns.levels[0]
        res_df.columns.name=key_index_name
        res_df.index.name=old_col_index_name
        res_df = res_df.transpose()#inplace=True)
    
        # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
        res_df.rename(mapper= rename_mapper, axis='columns',inplace=True)

    return res_df


def get_evaluate_regression_dict(df,  metrics=['r2','RMSE','U'], show_results_dict = False, show_results_df=True,return_as_styled_df=False, return_as_df =True): #, return_col_names=False):
    """Calculates and displays any of the following evaluation metrics (passed as strings in metrics param) for each true/pred pair of columns in df:
    r2, MAE,MSE,RMSE,U """
    import re
    import functions_combined_BEST as ji
    from IPython.display import display
    import pandas as pd

    col_list = df.columns
    from_where = re.compile('(true|pred)_(from_\w*_?\w+?)')
    found = [from_where.findall(col) for col in col_list]
    found = [x[0] for x in found if len(x)>0]


    pairs_of_cols = {}
    df_dict = {}

    if 'true_test_price' in col_list:
        use_single_true_column = True
        true_test_series = df['true_test_price']
    else:
        use_single_true_column = False

#     results =[['preds_from','metric','value']]
    # for _,where in found:  
    for true_pred, from_source in found: 

        if use_single_true_column:
            true_series = true_test_series #.dropna()
            true_series_name = true_series.name
        else:
            true_series = df['true_'+from_source]#.dropna()
            true_series_name = true_series.name

        pred_series = df['pred_'+from_source]#.dropna()
        pred_series_name = pred_series.name

        # combine true_series and pred_series and then dropna()
        df_eval = pd.concat([true_series,pred_series],axis=1)
        df_eval.dropna(inplace=True)

        pairs_of_cols[from_source] = {}
        pairs_of_cols[from_source]=[true_series_name,pred_series_name]#['col_names']
        
        df_dict[from_source] = ji.evaluate_regression(df_eval[true_series_name],df_eval[pred_series_name],metrics=metrics) #.reset_index().set_index('Metric')#,inplace=True)
#         pairs_of_cols[where]['results']=res_df

    # # combine into one dataframe
    # df_results = pd.DataFrame.from_dict(df_dict,)

    if show_results_dict:
        ji.display_df_dict_dropdown(df_dict)
    
    ## Combine dataframes from dictionary into one output table 
    if return_as_df or show_results_df:
        
        # if only 1 set of results, just rename metrics


        def res_dict_to_merged_df(dict_of_dfs, key_index_name='Prediction Source', old_col_index_name=None):

            res_dict = dict_of_dfs
            # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
            rename_mapper = {'R_squared':'R^2','R Squared':'R^2','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}

            if len(res_dict.keys())==1:
                
                res_df = res_dict[list(res_dict.keys())[0]]

                # res_df.set_index('Metric',inplace=True)
                res_df.rename(mapper=rename_mapper, axis='index',inplace=True)
                res_df=res_df.transpose()
                # caption='Evaluation Metrics'

            else:
                res_df= pd.concat(res_dict.values(), axis=1,keys=res_dict.keys())
                res_df.columns = res_df.columns.levels[0]
                res_df.columns.name=key_index_name
                res_df.index.name=old_col_index_name
                res_df = res_df.transpose()#inplace=True)
            
                # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
                res_df.rename(mapper= rename_mapper, axis='columns',inplace=True)
            ## new to fix exporting for dash
            res_df.reset_index(inplace=True)
            return res_df

        res_df = res_dict_to_merged_df(df_dict)


    if show_results_df:
        res_df_s = res_df.style.hide_index().set_caption('Evaluation Metrics')# by Prediction Source'))
        display(res_df_s)

    if return_as_styled_df:
        return res_df_s
    elif return_as_df:
        return res_df
    else:
        return df_dict
        

def compare_eval_metrics_for_shifts(true_series,pred_series, shift_list=[-2,-1,0,1,2], true_train_series_to_add=None,
color_coded=True, return_results=False, return_styled_df=False, return_shifted_df=True, display_results=True, display_U_info=False):
    
    ## SHIFT THE TRUE VALUES, PLOT, AND CALC THIEL's U
    import functions_combined_BEST as ji
    from bs_ds import list2df
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    import pandas as pd

    true_colname = true_series.name#'true'
    pred_colname = pred_series.name#'pred'

    # combine true and preds into one dataframe
    df = pd.concat([true_series, pred_series], axis=1)
    df.columns=[true_colname, pred_colname]#.dropna(axis=0,subset=[[true_colname,pred_colname]])

    # Create Empty Resuts containers
    results=[['Bins Shifted','Metric','Value']]
    combined_results = pd.DataFrame(columns=results[0])
    shift_results_dict= {}
    
    # Loop through shifts, add to df_shifted, calc thiel's U
    df_shifted=df.copy()
    for shift in shift_list:

        # create df for current shift
        df_shift=pd.DataFrame()
        df_shift['pred'] = df[pred_colname].shift(shift)
        df_shift['true'] = df[true_colname]
        
        # Add shifted columns to df_shifted
        df_shifted['pred_shift'+str(shift)] =  df_shift['pred']
        
        # drop null values from current shit to calc metrics
        df_shift.dropna(inplace=True)

        #[!] ### DIFFERENT THAN U COMPARE U FUNCTION
        shift_results = evaluate_regression(df_shift['true'],df_shift['pred']).reset_index() #[true_colname],df_shift[pred_colname]).reset_index()
        shift_results.insert(0,'Bins Shifted',shift)
        

        ## ADD RESULTS TO VARIOUS OUTPUT METHODS
        results.append(shift_results)
        combined_results = pd.concat([combined_results,shift_results], axis=0)
        shift_results_dict[shift] =  shift_results.drop('Bins Shifted',axis=1).set_index('Metric')


    # if 
    if true_train_series_to_add is not None:
        df_shifted['true_train_price'] = true_train_series_to_add
    # Turn results into dataframe when complete
    # df_results = list2df(results)#
    # df_results.set_index('# of Bins Shifted', inplace=True)
    


    # Restructure DataFrame for ouput
    df_results = res_dict_to_merged_df(shift_results_dict, key_index_name='Pred Shifted')
    df_results.reset_index(inplace=True)

    if display_results:
        
        # Dispaly Thiel's U info
        if display_U_info:
            _ = thiels_U(None,None,True,True)
        
        
        # Display dataframe results
        if color_coded is True:
            dfs_results = ji.color_cols(df_results, subset=['RMSE','U'], rev=True)
            dfs_results.set_caption("Evaluation Metrics for Shifted Preds")

        else:
            df_results.style.set_caption('Evaluation Metrics for Shifted Preds')

        dfs_results.hide_index().set_properties(**{'text-align':'center'})
        display(dfs_results)


    ## Return requested oututs
    return_list = []


    if return_results:
        return_list.append(df_results)

    if return_styled_df:
        return_list.append(dfs_results)

    if return_shifted_df:
        return_list.append(df_shifted)

    return return_list[:]
    


def plot_best_shift(df,df_results, true_colname='true',pred_colname='pred',  col_to_check='U', best='min'):
    
    import matplotlib.pyplot as plt
    import pandas as pd
    if 'min' in best:
        best_shift = df_results[col_to_check].idxmin()#[0]
    elif 'max' in best:
        best_shift = df_results[col_to_check].idxmax()#[0]

    df[true_colname].plot(label = 'True Values')
    df[pred_colname].shift(best_shift).plot(ls='--',label = f'Predicted-Shifted({best_shift})')
    plt.legend()
    plt.title(f"Best {col_to_check} for Shifted Time Series")
    plt.tight_layout()
    return 


def compare_u_for_shifts(true_series,pred_series, shift_list=[-2,-1,0,1,2],
    plot_all=False,plot_best=True, color_coded=True, return_results=False, return_shifted_df=True,
    display_U_info=False):
    
    import functions_combined_BEST as ji
    from bs_ds import list2df
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    true_colname = true_series.name#'true'
    pred_colname = pred_series.name#'pred'
    
    # combine the series into one dataframe and rename
    df = pd.concat([true_series, pred_series],axis=1)
    # df.columns=[true_colname, pred_colname]#.dropna(axis=0,subset=[[true_colname,pred_colname]])
    
    # create results list
    results=[['# of Bins Shifted','U']]
    
    if plot_all or plot_best:
        plt.figure()

    if plot_all is True:
        df[true_colname].plot(color='black',lw=3,label = 'True Values')
        plt.legend()
        plt.title('Shifted Time Series vs Predicted')
        

    # Loop through shifts, add to df_shifted, calc thiel's U
    df_shifted = df.copy()        

    for shift in shift_list:
        if plot_all==True:
            df[pred_colname].shift(shift).plot(label = f'Predicted-Shifted({shift})')

        # create df for current shift
        df_shift=pd.DataFrame()
        df_shift['pred'] = df[pred_colname].shift(shift)
        df_shift['true'] = df[true_colname]
        
        # add to df_shifted
        df_shifted['pred_shift'+str(shift)] =  df_shift['pred']

        # Drop null values and calcualte Thiels U
        df_shift.dropna(inplace=True)
        U = thiels_U(df_shift['true'], df_shift['pred'],False,False)

        # Append results to results list
        results.append([shift,U])
    
    # Turn results into dataframe when complete
    df_results = list2df(results)#
    df_results.set_index('# of Bins Shifted', inplace=True)

    # if plot+nest
    if plot_best==True:
        plot_best_shift(df,df_results,true_colname=true_colname, pred_colname=pred_colname)

        # # def plot_best_shift(df_results,col_to_check):
        # best_shift = df_results['U'].idxmin()#[0]

        # df[true_colname].plot(label = 'True Values')
        # df[pred_colname].shift(best_shift).plot(ls='--',label = f'Predicted-Shifted({best_shift})')
        # plt.legend()
        # plt.title("Best Thiel's U for Shifted Time Series")
        # plt.tight_layout()

    # Dispaly Thiel's U info
    if display_U_info:
        _ = thiels_U(None,None,True,True)

    # Display dataframe results
    if color_coded is True:
        dfs_results = ji.color_cols(df_results, rev=True)
        display(dfs_results.set_caption("Thiel's U - Shifting Prediction Time bins"))

    else:
        display(df_results.style.set_caption("Thiel's U - Shifting Prediction Time bins"))
        
    # Return requested oututs
    return_list = []

    if return_results:
        return_list.append(df_results)

    if return_shifted_df:
        return_list.append(df_shifted)

    return return_list[:]


def compare_time_shifted_model(df_model,true_colname='true test',pred_colname='pred test',
                               shift_list=[-4,-3,-2,-1,0,1,2,3,4],show_results=True,show_U_info=True,
                               caption=''):
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # GET EVALUATION METRICS FROM PREDICTIONS
    true_test_series = df_model[true_colname].dropna()
    pred_test_series = df_model[pred_colname].dropna()

    # Comparing Shifted Timebins
    res_df, shifted_df = compare_eval_metrics_for_shifts(true_test_series.rename(true_colname),
    pred_test_series.rename(pred_colname),shift_list=np.arange(-4,4,1,))

    res_df = res_df.pivot(index='Bins Shifted', columns='Metric',values='Value')
    res_df.columns.rename(None, inplace=True)
    
    
    if show_results:
        
        res_dfs = res_df.copy().style
        res_dfs = ji.color_cols(res_df,subset=["Thiel's U"],rev=True) #OLD
        display(res_dfs.set_caption(caption))
    
    if show_U_info:
        _ = thiels_U(None,None,True,True)
        
#         metric_best_crit = {'R_squared':'max', "Thiel's U":'min','Root Mean Squared Error':'min'}    
#         for k,v in metric_best_crit.items():
#             res_dfs = res_dfs.apply(lambda x: highlight_best(x,v),axis=0)        
#         display(res_dfs)
    return res_df

# def get_model_preds_df(model, train_generator, test_generator, true_train_data, true_test_data,
# preds_from_gen=True, preds_from_train_preds =True, preds_from_test_preds=True, model_params=None, train_data_index=None, test_data_index=None, x_window=None, return_combined=False):
#     """Accepts a model, the training and testing data TimeseriesGenerators, the test_index and train_index.
#     Returns a dataframe with True and Predicted Values for Both the Training and Test Datasets."""
#     import pandas as pd
#     import functions_combined_BEST as ji

#     if model_params is not None:
#         train_data_index = model_params['input_params']['train_data_index']
#         test_data_index = model_params['input_params']['test_data_index']
#         x_window =  model_params['data_params']['x_window']


#     if true_test_data is None:
#         raise Exception("true_test_data = df_test['price']")

#     if true_train_data is None:        
#         raise Exception("true_train_data=df_train['price']")
    



#     #### ADD SWITCH DEPENDING ON TRUE CONDITIONS

#     if preds_from_gen == True:
#         get_model_preds_from_gen()

#         # #  GET INDICES BASED ON GENERATOR START AND END
#         # gen_index = true_test_data.index[test_generator.start_index:test_generator.end_index+1]

        
#         # # GET PREDICTIONS FOR TRAINING DATA AND TEST DATA
#         # test_predictions = ji.arr2series( model.predict_generator(test_generator),
#         #                             test_data_index[x_window:], 'pred test')
        
#         # train_predictions = ji.arr2series( model.predict_generator(train_generator),
#         #                             train_data_index[x_window:], 'pred train')

    

#     # GET TRUE TEST AND TRAIN DATA AS SERIES
#     true_test_price = pd.Series( true_test_data.iloc[x_window:],
#                                 index= test_data_index[x_window:], name='true test')
    
#     true_train_price = pd.Series(true_train_data.iloc[x_window:],
#                                  index = train_data_index[x_window:], name='true train')

    
#     # COMBINE TRAINING DATA AND TESTING DATA INTO 2 DFS (with correct date axis)
#     df_true_v_preds_train = pd.concat([true_train_price, train_predictions],axis=1)
#     df_true_v_preds_test= pd.concat([true_test_price, test_predictions],axis=1)
    
#     # RETURN ONE OR TWO DATAFRAMES
#     if return_combined is False:
#         return df_true_v_preds_train, df_true_v_preds_test
#     elif return_combined is True:
#         df_combined = pd.concat([df_true_v_preds_train, df_true_v_preds_test],axis=1)
#         df_combined.columns=['true train','pred train','true test','pred test']
#         return df_combined
# def get_true_vs_preds_df(model, true_test_series=None,test_generator=None):
#     import pandas as pd
#     import functions_combined_BEST as ji
#     import bs_ds  as bs


#     pass

def get_model_preds_df(model, true_train_series,true_test_series, test_generator,model_params=None,
x_window=None, n_features=None, inverse_tf=False, scaler=None, include_train_data=True,
 preds_from_gen=True, preds_from_train_preds =False, preds_from_test_preds=False, 
 iplot=False,iplot_title=None,verbose=1):#  train_data_index=None, test_data_index=None
    """ Gets predictions for training data from the 3 options: 
    1) from generator  --  len(output) = (len(true_test_series)-n_input)
    2) from predictions on test data  --  len(output) = (len(true_test_series)-n_input)
    3) from predictions on train data -- len(true_test_series)
    """
    import pandas as pd
    import functions_combined_BEST as ji
    import bs_ds  as bs
    # x_window=n_input

    ## If no model params
    if model_params is None:

        ## get the seires indices from the true input series
        train_data_index = true_train_series.index
        test_data_index = true_test_series.index

        ## Get x_window and n_features from the generator
        if x_window is None:
            x_window = test_generator.length
        if n_features is None:
            n_features=test_generator.data[0].shape[0]

        if inverse_tf and scaler is None:
            raise Exception('if inverse_tf, must provide previously fit scaler.')

    
    if model_params is not None:
        if scaler is None and inverse_tf == True:
            scaler = model_params['scaler_library']['price']
        ## get n_features,x_window from model_params
        n_features = model_params['input_params']['n_features']
        x_window = model_params['input_params']['n_input']

        ## get indices from model_params
        train_data_index = model_params['input_params']['train_data_index']
        test_data_index = model_params['input_params']['test_data_index']
        # if model_params['data_params']['x_window'] == model_params['input_params']['n_input']:
        #     x_window = model_params['input_params']['n_input']
        # else:
        #     print('x_window and n_input params are not the same, using n_input as x_window...')
            
    if (preds_from_gen == True) and (test_generator == None):
        raise Exception('If from_gen=True, must provide generator.')

            
    ### GET the 3 DIFERENT TYPES OF PREDICTIONS    
    df_list = []
    if preds_from_gen:
        ## get predictions from generator and return gen_df with correct data indices
        gen_df = get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
         model_params=model_params, n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

        df_list.append(gen_df)



    if preds_from_test_preds:
        
        func_df_from_test = get_model_preds_from_preds(model=model, true_train_data=true_train_series, true_test_data=true_test_series,
        model_params=model_params, x_window=x_window, n_features=n_features,
         suffix='_from_test_preds',build_preds_from_train=False, return_df=True)

        df_list.append(func_df_from_test)



    if preds_from_train_preds:

        func_df_from_train  = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=x_window, n_features=n_features,
        suffix='_from_train_preds', build_preds_from_train=True,return_df=True)
        df_list.append(func_df_from_train)


    # bs.display_side_by_side(func_df,func_df_from_train)
    ## combine into df
    df_all_preds = pd.concat([df for df in df_list],axis=1)
    df_all_preds = bs.drop_cols(df_all_preds,['i_']);


    ## ADD TRAINING DATA TO DATAFRAME IF REQUESTED
    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,df_all_preds],axis=1)
        
    ## INVERSE TRANSOFRM BACK TO PRICE
    if inverse_tf:
        df_out = ji.transform_cols_from_library(df_all_preds,single_scaler=scaler,inverse=True)
    else:
        df_out = df_all_preds

    
    def get_plot_df_with_one_true_series(df_out,train_data = true_train_series, include_train_data=include_train_data):
        
        # print(df_out.columns)
        df_plot = pd.DataFrame()

        ## Check columns for true_train_price, then make list of other colnames
        cols = df_out.columns.to_list()

        if 'true_train_price' in cols:
            if include_train_data:
                df_plot['true_train_price'] = df_out['true_train_price']
            
        # remove from the col_list to be looped through.
        col_list = [x for x in cols if x !='true_train_price']

        ## Use regexp to separate 'true_from_source','pred_from_source' columns
        import re
        from_where = re.compile('(true|pred)_(from_\w*_?\w+?)')
        # found = [from_where.findall(col)[0] for col in col_list]
        found = [from_where.findall(col) for col in col_list]
        found = [x[0] for x in found if len(x)>0]
        
        pairs_of_cols = {}
        true_col_data = []
        true_col_name = []
        check_true_col_name = []
        check_true_col_data = []
        # Check tuple for true/pred and from_source
        for true_pred, from_source in found: 

            if 'true' in true_pred:
                
                if len(true_col_data)==0:
                    
                    true_col_name = f"{true_pred}_{from_source}"
                    true_col_data = df_out[true_col_name]

                    df_plot['true_test_price'] = true_col_data

                else:
                    check_true_col_name = f"{true_pred}_{from_source}"
                    check_true_col_data = df_out[check_true_col_data]

                    if all(check_true_col_data == true_col_data):
                        continue
                    else:
                        print(f'Warning: true data from {true_col_name} and {check_true_col_name} do not match!')
                        # name_recon = f"{true_pred}_{from_source}"
                        # df_plot['true_test_price'] = df_out[name_recon]#true_series_to_plot
                        # continue #break?
                
            elif 'pred' in true_pred:
                name_recon = f"{true_pred}_{from_source}"
                df_plot[name_recon] = df_out[name_recon]
                # continue            

        return df_plot 
    
    df_plot = get_plot_df_with_one_true_series(df_out,train_data=true_train_series, include_train_data=include_train_data ) 

    ## display head if verbose
    if verbose>0:
        ji.disp_df_head_tail(df_plot)

    if iplot==False:
        return df_plot
    else:
        # from plotly.offline import 
        # df_plot = get_plot_df_with_one_true_series(df_out,train_data=true_train_series, include_train_data=include_train_data ) 
        pred_columns = [x for x in df_plot.columns if 'pred' in x]
        if iplot_title is None:
            iplot_title='S&P 500 True Price Vs Predictions ($)'
        fig = ji.plotly_true_vs_preds_subplots(df_plot, title=iplot_title,true_train_col='true_train_price',
            true_test_col='true_test_price', pred_test_columns=pred_columns)

        return df_plot




def get_eval_dict_for_paired_cols(df,col_regex_tokens='(true|pred)_(from_\w*_?\w+?)'):
    import re
    
    col_list = df.columns
    from_where = re.compile(col_regex_tokens) #'(true|pred)_(from_\w*_?\w+?)')
    found = [from_where.findall(col) for col in col_list]
    found = [x[0] for x in found if len(x)>0]


    pairs_of_cols = {}
    for _,where in found: 

        true_series = df['true_'+where].dropna()
        pred_series = df['pred_'+where].dropna()
        pairs_of_cols[where] = {}
        pairs_of_cols[where]['col_names']={'true':true_series.name,'pred':pred_series.name}
        pairs_of_cols[where]['results']=evaluate_regression(true_series,pred_series).reset_index()
    
    return pairs_of_cols
    

def get_predictions_df_and_evaluate_model(model, test_generator,
                                        true_train_data, true_test_data, model_params=None,
                                        train_data_index=None, test_data_index=None, 
                                        x_window=None, scaler=None, inverse_tf =True,
                                        return_separate=False, plot_results = True,
                                        iplot_results=False):

    import functions_combined_BEST as ji
    import pandas as pd
    from IPython.display import display
    n_input=x_window

    if model_params is not None:
        train_data_index = model_params['input_params']['train_data_index']
        test_data_index = model_params['input_params']['test_data_index']
        x_window =  model_params['data_params']['x_window']
        scaler_library = model_params['scaler_library']

    if true_test_data is None:
        raise Exception("true_test_data = df_test['price']")

    if true_test_data is None:        
        raise Exception("true_train_data=df_train['price']")


    # Call helper to get predictions and return as dataframes 
    # df_true_v_preds_train, df_true_v_preds_test 
    df_model = get_model_preds_df(model,  test_generator=test_generator,
    true_train_series=true_train_data, true_test_series = true_test_data, model_params=model_params,
        x_window=None, inverse_tf = inverse_tf)
    ## Concatenate into one dataframe
    # df_model_preds = pd.concat([df_true_v_preds_train, df_true_v_preds_test],axis=1)
    
    # ## CONVERT BACK TO DOLLARS AND PLOT
    # if inverse_tf==True:
    #     df_model = pd.DataFrame()
    #     for col in df_model_preds.columns:
    #         df_model[col] = ji.transform_series(df_model_preds[col],scaler_library['price'], inverse=True) 
    # else:
    #     df_model = df_model_preds

        
    if plot_results:
        # PLOTTING TRAINING + TRUE/PRED TEST DATA
        ji.plot_true_vs_preds_subplots(df_model['true train'],df_model['true test'], 
                                    df_model['pred test'], subplots=True)
    if iplot_results:
        df_plot = df_model.copy().drop(['pred train','true_from_test_preds','true_from_train_preds'],axis=1)
        ji.plotly_time_series(df_plot) 


    

    # prepare display_of_results

    # # GET EVALUATION METRICS FROM PREDICTIONS
    # true_test_series = df_model['true test'].dropna()
    # pred_test_series = df_model['pred test'].dropna()
    
    # # Get and display regression statistics
    # results_tf = evaluate_regression(true_test_series, pred_test_series)
    pairs_of_cols = ji.get_evaluate_regression_dict(df_model)
    display(pairs_of_cols)

    return df_model




def get_model_preds_from_preds(model,true_train_data, true_test_data,
                         model_params=None, x_window=None, n_features=None, 
                         build_preds_from_train=True, return_df=True,suffix=None, debug=False):
    
    """ Gets predictions from model using using its own predictions as the subsequent input.
    Must provide a model_params dictionary with 'input_params' OR must define ('n_input','n_features').
    
    * IF build_preds_from_train is True:
        1. starting true time series for predictions is the last rows [-n_input:] from training data.
        2. output predicitons will be the same length as the input scaled_test_data
    
    * IF build_preds_from_train is False:
        1. starting true time series for predictions is the first rows [:n_input] from test data.
        2. output predicitons will be shorter by n_input # of rows
    """
    scaled_train_data = true_train_data
    scaled_test_data = true_test_data
    import import bs_ds_local as bs bs
    import numpy as np
    import pandas as pd
    test_predictions = []
    first_eval_batch=[]

    n_input = x_window

    if model_params is not None:
        if n_input is None:
            n_input= model_params['input_params']['n_input']

        if n_features is None:
            n_features = model_params['input_params']['n_features']
    


    preds_out = [['i','index','pred']]
    
    # SAVING COPY OF INPUT TEST DATA
    scaled_test_series = scaled_test_data.copy() 

    ## SAVING TRAIN AND TEST DATA INDICES AND VALUES
    train_data_index = scaled_train_data.index
    scaled_train_data = scaled_train_data.values
    test_data_index = scaled_test_data.index
    scaled_test_data = scaled_test_data.values
    
    
    ## PREPARE THE FIRST EVAL BATCH TIMESERIES FROM TRAIN OR TEST DATA
    # Change parameters depending on if from train or test data
    if build_preds_from_train:
        
        # If using trianing data loop through full test data
        loop_length = range(len(scaled_test_data))
        
        # take the last window size of data from training data 
        first_eval_batch = scaled_train_data[-1*n_input:]
        # first_batch_idx = train_data_index[-n_input:]
        
        # set the true index to test_data_index
        true_index_out = test_data_index
        
    # Set the loop to be from n_input # of rows into test_data
    else:
        loop_length = range(n_input,len(scaled_test_data))
        first_eval_batch = scaled_test_data[:n_input]
        true_index_out =  test_data_index[n_input:]
      
    
    # reshape first batch of data for model.predict 
    first_batch_pre_reshape = first_eval_batch.shape    
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    first_batch_shape = current_batch.shape

    
    ## LOOP THROUGH REMAINING TIMEBINS USING CURRENT PREDICITONS AS NEW DATA FOR NEXT
    for i in loop_length:

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]        
        # store prediction
        test_predictions.append(current_pred) 

        # update batch to now include prediction and drop first value
        # try:
            # current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        # except:
            # print('COMMAND FAILED: "current_batch = np.append(current_batch[:,1:,:],[current_pred],axis=1)"' )
        #     print(f'\nn_features={n_features}')
        #     print(f"n_input={n_input}")
        #     print(f"first_batch_shape={first_batch_shape}; current_batch.shape={current_batch.shape}; current_pred.shape={current_pred.shape}" )
        # finally:
        #     from pprint import pprint
        #     print('current_batch:',current_batch,'\ncurrent_pred:',current_pred)
        
        ## Append the data to the output df list
        preds_out.append([i,test_data_index[i],current_pred[0]])


    ## If returning a dataframe,prepare and rename with suffix
    if return_df:
        res_df = bs.list2df(preds_out)
        res_df['index'] = pd.to_datetime(res_df['index'])
        res_df.set_index('index',inplace=True)
        
        # adding true price
        res_df['true'] = scaled_test_series.loc[true_index_out]
        res_df=res_df[['i','true','pred']]
        
        if suffix is not None:
            colnames = [name+suffix for name in res_df.columns]
        else:
            colnames = res_df.columns
            
        res_df.columns=colnames
        return res_df #test_predictions, checks
    
    # Else just return array of predictions
    else:
        return np.array(test_predictions)



def get_model_preds_from_gen(model,test_generator, true_test_data, model_params=None,
                       n_input=None, n_features=None, suffix=None, verbose=0,return_df=True):
        """
        Gets prediction from model using the generator's timeseries using model.predict_generator()
        Must provide a model_params dictionary with 'input_params' OR must define ('n_input','n_features').

        """
        import pandas as pd
        import numpy as np
        import import bs_ds_local as bs bs
        import functions_combined_BEST as ji
        if model_params is not None:
            n_input= model_params['input_params']['n_input']
            n_features = model_params['input_params']['n_features']

        if model_params is None:
            if n_input is None:
                n_input= test_generator.length
            if n_features is None:
                n_features=test_generator.data[0].shape[0]

        # GET TRUE VALUES AND DATETIME INDEX FROM GENERATOR
        
        # Get true time index from the generator's start_index and end_index 
        gen_index = true_test_data.index[test_generator.start_index:test_generator.end_index+1]
        gen_true_targets = test_generator.targets[test_generator.start_index:test_generator.end_index+1]
        
        # Generate predictions from the test_generator
        gen_preds = model.predict_generator(test_generator)
        gen_preds_flat = gen_preds.ravel()
        gen_true_targets = gen_true_targets.ravel()
        
        
        # RETURN OUTPUT AS DATAFRAME OR ARRAY OF PREDS
        if return_df == False:
            return gen_preds

        else:
            # Combine the outputs
            if verbose>0:
                print(len(gen_index),len(gen_true_targets), len(gen_preds_flat))

            gen_pred_df = pd.DataFrame({'index':gen_index,'true':gen_true_targets,'pred':gen_preds_flat})
            gen_pred_df['index'] = pd.to_datetime(gen_pred_df['index'])
            gen_pred_df.set_index('index',inplace=True)

            if suffix is not None:
                colnames = [name+suffix for name in gen_pred_df.columns]
            else:
                colnames = gen_pred_df.columns
            gen_pred_df.columns=colnames
            return gen_pred_df


def compare_model_pred_methods(model, true_train_series,true_test_series, test_generator=None,
                               model_params=None, n_input=None, n_features=None, from_gen=True,
                               from_train_series = True, from_test_series=True, 
                               iplot=True, plot_with_train_data=True,return_df=True, inverse_tf=True):
    """ Gets predictions for training data from the 3 options: 
    1) from generator  --  len(output) = (len(true_test_series)-n_input)
    2) from predictions on test data  --  len(output) = (len(true_test_series)-n_input)
    3) from predictions on train data -- len(true_test_series)
    """
    import pandas as pd
    import functions_combined_BEST as ji
    import bs_ds  as bs
    if model_params is not None:
        n_input= model_params['input_params']['n_input']
        n_features = model_params['input_params']['n_features']

    if model_params is None:
        if n_input is None or n_features is None:
            raise Exception('Must provide model params or define n_input and n_features')
            
    if from_gen is True and test_generator is None:
        raise Exception('If from_gen=True, must provide generator.')

            
    ### GET the 3 DIFERENT TYPES OF PREDICTIONS    
    df_list = []
    #(model, test_generator, true_test_data, model_params=None, n_input=None, n_features=None, suffix=None, return_df=True)
    if from_gen:
        gen_df = get_model_preds_from_gen(model=model, test_generator=test_generator,
        true_test_data=true_test_series, model_params=model_params, 
        n_input=n_input, n_features=n_features,  suffix='_from_gen',return_df=True)    
        df_list.append(gen_df)
    #s(model, scaled_train_data, scaled_test_data, model_params=None, n_input=None, n_features=None, build_preds_from_train=True, return_df=True, suffix=None)
    if from_test_series:

        func_df_from_test = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=n_input, n_features=n_features,
         suffix='_from_test',build_preds_from_train=False, return_df=True)
        df_list.append(func_df_from_test)

    if from_train_series:
        func_df_from_train  = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=n_input, n_features=n_features,
        suffix='_from_train', build_preds_from_train=True,return_df=True)
        df_list.append(func_df_from_train)

    # bs.display_side_by_side(func_df,func_df_from_train)

    df_all_preds = pd.concat([df for df in df_list],axis=1)
    df_all_preds = bs.drop_cols(df_all_preds,['i_'])
    # print(df_all_preds.shape)
    if plot_with_train_data:
        df_all_preds=pd.concat([true_train_series.rename('true_train_price'),df_all_preds],axis=1)

    if inverse_tf:
        df_out = ji.transform_cols_from_library(df_all_preds,single_scaler=model_params['scaler_library']['price'],inverse=True)
    else:
        df_out = df_all_preds

    if iplot:
        ji.plotly_time_series(df_out)

    if return_df:
        return df_out


def extract_true_vs_pred_cols(df_model1, rename_cols = True, from_gen=True, from_test_preds=False,
from_train_preds=False):
    import pandas as pd
    if sum([from_gen, from_test_preds, from_train_preds]) >1:
        raise Exception('Only 1 of the "from_source" inputs may ==True: ')
    
    list_of_possible_cols = ['true_from_gen', 'pred_from_gen',
     'true_from_test_preds','pred_from_test_preds', 'true_from_train_preds',
       'pred_from_train_preds']

    if from_gen:
        true_col = 'true_from_gen'
        pred_col = 'pred_from_gen'
    
    if from_test_preds:
        true_col = 'true_from_test_preds'
        pred_col = 'pred_from_test_preds'

    if from_test_preds:
        true_col = 'true_from_train_preds'
        pred_col = 'pred_from_train_preds'


    true_series = df_model1[true_col].rename('true')
    pred_series = df_model1[pred_col].rename('pred')

    
    df_model_out = pd.concat([true_series, pred_series],axis=1)

    if rename_cols==True:
        df_model_out.columns = ['true','pred']

    return df_model_out 


def evaluate_classification(*args,**kwargs):
    raise Exception('Use `evaluate_classification_model` instead of `evaluate_classification`.')

# def evaluate_classification(model, history, X_train,X_test,y_train,y_test,report_as_df=True, binary_classes=True,
#                             conf_matrix_classes= ['Decrease','Increase'],
#                             normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
#                             history_filename ='results/keras_history.png', save_conf_matrix_png=False,
#                             conf_mat_filename= 'results/confusion_matrix.png',save_summary=False, 
#                             summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

#     """Evaluates kera's model's performance, plots model's history,displays classification report,
#     and plots a confusion matrix. 
#     conf_matrix_classes are the labels for the matrix. [negative, positive]
#     Returns df of classification report and fig object for  confusion matrix's plot."""

#     from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
#     import import bs_ds_local as bs bs
#     import functions_combined_BEST as ji
#     from IPython.display import display
#     import pandas as pd
#     import matplotlib as mpl
#     numFmt = '.4f'
#     num_dashes = 30

#     # results_list=[['Metric','Value']]
#     # metric_list = ['accuracy','precision','recall','f1']
#     print('---'*num_dashes)
#     print('\tTRAINING HISTORY:')
#     print('---'*num_dashes)

#     if auto_unique_filenames:
#         ## Get same time suffix for all files
#         time_suffix = ji.auto_filename_time(fname_friendly=True)

#         filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
#         ## update filenames 
#         for filetype,filename in filename_dict.items():
#             if '.' in filename:
#                 filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
#             else:
#                 if filetype =='summary':
#                     ext='.txt'
#                 else:
#                     ext='.png'
#                 filename_dict[filetype] = filename+time_suffix + ext


#         history_filename = filename_dict['history']
#         conf_mat_filename = filename_dict['conf_mat']
#         summary_filename = filename_dict['summary']


#     ## PLOT HISTORY
#     ji.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

#     print('\n')
#     print('---'*num_dashes)
#     print('\tEVALUATE MODEL:')
#     print('---'*num_dashes)

#     print('\n- Evaluating Training Data:')
#     loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
#     print(f'    - Accuracy:{accuracy_train:{numFmt}}')
#     print(f'    - Loss:{loss_train:{numFmt}}')

#     print('\n- Evaluating Test Data:')
#     loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
#     print(f'    - Accuracy:{accuracy_test:{numFmt}}')
#     print(f'    - Loss:{loss_test:{numFmt}}\n')


#     ## Get model predictions
    
#     if hasattr(model, 'predict_classes'):
#         y_hat_train = model.predict_classes(X_train)
#         y_hat_test = model.predict_classes(X_test)
#     elif hasattr(model,'predict'):
#         y_hat_train = model.predict(X_train)
#         y_hat_test = model.predict(X_test)
#     else:
#         raise Exception('model has neither "predict" nor "predict_classes" methods')

#     if y_test.ndim>1 or y_hat_test.ndim>1 or binary_classes==False:
#         if binary_classes==False: 
#             pass
#         else:
#             binary_classes = False
#             print(f"[!] y_test was >1 dim, setting binary_classes to False")

#         ## reduce dimensions of y_train and y_test
#         # y_train = y_train.argmax(axis=1)
#         # y_test = y_test.argmax(axis=1)
#         if y_test.ndim>1:            
#             y_test = y_test.argmax(axis=1)
#         if y_hat_test.ndim>1:
#             y_hat_test = y_hat_test.argmax(axis=1)
#         # for var in ['y_test', 'y_hat_test', 'y_train', 'y_hat_train']:
#         #     real_var = eval(var)
#         #     print('real_var shape:',real_var.shape)
#         #     if real_var.ndim>1:
#         #         ## reduce dimensions
#         #         cmd =  var+'= real_var.argmax(axis=1)'
#         #         # eval(cmd)
#         #         eval(var+'=') real_var.argymax(axis=1)
#         #         # exec(cmd)
#         #         cmd =f'print("argmax shape:",{var}.shape)' 
#         #         eval(cmd)
#         #         # exec(cmd)
        
        
        

#     print('---'*num_dashes)
#     print('\tCLASSIFICATION REPORT:')
#     print('---'*num_dashes)

#     # get both versions of classification report output
#     report_str = classification_report(y_test,y_hat_test)
#     report_dict = classification_report(y_test,y_hat_test,output_dict=True)
#     if report_as_df:
#         try:
#             ## Create and display classification report
#             # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
#             df_report_temp = pd.DataFrame(report_dict)
#             df_report_temp = df_report_temp.T#reset_index(inplace=True)

#             df_report = df_report_temp[['precision','recall','f1-score','support']]
#             display(df_report.round(4).style.set_caption('Classification Report'))
#             print('\n')
        
#         except:
#             print(report_str)
#             # print(report_dict)
#             df_report = pd.DataFrame()
#     else:
#         print(report_str)

#     if save_summary:
#         with open(summary_filename,'w') as f:
#             model.summary(print_fn=lambda x: f.write(x+"\n"))
#             f.write(f"\nSaved at {time_suffix}\n")
#             f.write(report_str)

#     ## Create and plot confusion_matrix
#     conf_mat = confusion_matrix(y_test, y_hat_test)
#     mpl.rcParams['figure.figsize'] = conf_matrix_figsize
#     fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes,
#                                    normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
#     if save_conf_matrix_png:
#         fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)

#     if report_as_df:
#         return df_report, fig
#     else:
#         return report_str,fig



def evaluate_classification_model(model,  X_train,X_test,y_train,y_test, history=None,binary_classes=True,
                            conf_matrix_classes= ['Decrease','Increase'], plot_training_conf_mat = False,
                            normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
                            history_filename ='results/keras_history.png', save_conf_matrix_png=False,
                            conf_mat_filename= 'results/confusion_matrix.png',save_summary=False,
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Updated version of `evaluate_classification` from bs-ds. 
    Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix.
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
    import import bs_ds_local as bs bs
    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = bs.auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
        ## update filenames
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        conf_mat_filename = filename_dict['conf_mat']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    if history is not None:
        bs.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)
    
    ## Show model.evaluate for training and test data
    if hasattr(model,'evaluate'):

        print('\n- Evaluating Training Data:')
        loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
        print(f'    - Accuracy:{accuracy_train:{numFmt}}')
        print(f'    - Loss:{loss_train:{numFmt}}')

        print('\n- Evaluating Test Data:')
        loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
        print(f'    - Accuracy:{accuracy_test:{numFmt}}')
        print(f'    - Loss:{loss_test:{numFmt}}\n')

    
     



    ## Get predictions from model
    if hasattr(model,'predict_classes'):
        ## Get model predictions
        y_hat_train = model.predict_classes(X_train)
        y_hat_test = model.predict_classes(X_test)
    elif hasattr(model,'predict'):
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
    else: 
        raise Exception('Model does not have .predict_classes or .predict methods')
    
    
    
        
    ## Flatten any multi-column keras targets
    if y_test.ndim>1:
        y_test = y_test.argmax(axis=1)

    if y_train.ndim>1:
        y_train =y_train.argmax(axis=1)        
        
    if y_hat_test.ndim>1:
        y_hat_test = y_hat_test.argmax(axis=1)

    if y_hat_train.ndim>1:
        y_hat_train =y_hat_train.argmax(axis=1)            
        # or y_train.ndim>1:#binary_classes==False:
        # if binary_classes==False: 
        #     pass
        
        ## reduce dimensions of y_train and y_test
        # y_train = y_train.argmax(axis=1)
        
        # if binary_classes == True:
        #     binary_classes = False
        #     print(f"[!] y_test was >1 dim, setting binary_classes to False")
            
        
        
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)

    try:
        ## Get sklearn classification report 
        report_str = classification_report(y_test,y_hat_test)
        report_dict = classification_report(y_test,y_hat_test,output_dict=True)
    except:
        print('y_test:',y_test.shape)
        print('y_hat_test:',y_hat_test.shape)
        print('y_train:',y_train.shape)
        print('y_hat_train:',y_hat_train.shape)
    
    try:
        ## Create and display classification report
        # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
        df_report_temp = pd.DataFrame(report_dict)
        df_report_temp = df_report_temp.T#reset_index(inplace=True)

        df_report = df_report_temp[['precision','recall','f1-score','support']]
        display(df_report.round(4).style.set_caption('Classification Report'))
        print('\n')
    
    except:
        print(report_str)
        # print(report_dict)
        df_report = pd.DataFrame()

    ## if saving the model.summary() printout 
    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(report_str)

    def get_and_plot_conf_mat(y_true, y_pred,title='Confusion Matrix',save_png = save_conf_matrix_png):#, save_conf_matrix_png=save_conf_matrix_png,
                            #   conf_mat_filename=conf_mat_filename, normalize_conf_matrix=):
        ## Create and plot confusion_matrix
        import matplotlib.pyplot as plt
        conf_mat = confusion_matrix(y_true, y_pred)
        with plt.rc_context(rc={'figure.figsize':conf_matrix_figsize}): # rcParams['figure.figsize']
            fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes, title=title,
                                        normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
        if save_png:
            fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)
            
        fig.show()
        return fig
            
    fig = get_and_plot_conf_mat(y_test,y_hat_test,title='Confusion Matrix: Test Data')    
        
    if plot_training_conf_mat:
        fig_cm_train = get_and_plot_conf_mat(y_train,y_hat_train,save_png=False,title='Confusion Matrix: Training Data')    

    return df_report, fig



def evaluate_regression_model(model, history, train_generator, test_generator,true_train_series,
true_test_series,include_train_data=True,return_preds_df = False, save_history=False, history_filename ='results/keras_history.png', save_summary=False, 
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix. 
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
    import import bs_ds_local as bs bs
    import functions_combined_BEST as ji
    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = ji.auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'summary':summary_filename}
        ## update filenames 
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    ji.plot_keras_history( history,filename_base=history_filename,no_val_data=True, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)

        # # EVALUATE MODEL PREDICTIONS FROM GENERATOR 
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_train[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_train[0]:{numFmt}}')

    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_test[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_test[0]:{numFmt}}')

    x_window = test_generator.length
    n_features = test_generator.data[0].shape[0]
    gen_df = ji.get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
        n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

    regr_results = evaluate_regression(y_true=gen_df['true_from_gen'], y_pred=gen_df['pred_from_gen'],show_results=True,
                                metrics=['r2', 'RMSE', 'U'])


    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(regr_results.__repr__())


    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,gen_df],axis=1)
    else:
        df_all_preds = gen_df

    if return_preds_df:
        return df_all_preds




# def evaluate_gridsearch_classification(model, X_train,X_test,y_train,y_test,report_as_df=True, binary_classes=True,
#                             conf_matrix_classes= ['Decrease','Increase'],
#                             normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
#                             history_filename ='results/keras_history.png', save_conf_matrix_png=False,
#                             conf_mat_filename= 'results/confusion_matrix.png',save_summary=False, 
#                             summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

#     """Evaluates kera's model's performance, plots model's history,displays classification report,
#     and plots a confusion matrix. 
#     conf_matrix_classes are the labels for the matrix. [negative, positive]
#     Returns df of classification report and fig object for  confusion matrix's plot."""

#     from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
#     import import bs_ds_local as bs bs
#     import functions_combined_BEST as ji
#     from IPython.display import display
#     import pandas as pd
#     import matplotlib as mpl
#     numFmt = '.4f'
#     num_dashes = 30

#     # results_list=[['Metric','Value']]
#     # metric_list = ['accuracy','precision','recall','f1']
#     print('---'*num_dashes)
#     print('\tTRAINING HISTORY:')
#     print('---'*num_dashes)

#     if auto_unique_filenames:
#         ## Get same time suffix for all files
#         time_suffix = ji.auto_filename_time(fname_friendly=True)

#         filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
#         ## update filenames 
#         for filetype,filename in filename_dict.items():
#             if '.' in filename:
#                 filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
#             else:
#                 if filetype =='summary':
#                     ext='.txt'
#                 else:
#                     ext='.png'
#                 filename_dict[filetype] = filename+time_suffix + ext


#         history_filename = filename_dict['history']
#         conf_mat_filename = filename_dict['conf_mat']
#         summary_filename = filename_dict['summary']


#     ## PLOT HISTORY
#     ji.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

#     print('\n')
#     print('---'*num_dashes)
#     print('\tEVALUATE MODEL:')
#     print('---'*num_dashes)

#     print('\n- Evaluating Training Data:')
#     loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
#     print(f'    - Accuracy:{accuracy_train:{numFmt}}')
#     print(f'    - Loss:{loss_train:{numFmt}}')

#     print('\n- Evaluating Test Data:')
#     loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
#     print(f'    - Accuracy:{accuracy_test:{numFmt}}')
#     print(f'    - Loss:{loss_test:{numFmt}}\n')


#     ## Get model predictions
    
#     if hasattr(model, 'predict_classes'):
#         y_hat_train = model.predict_classes(X_train)
#         y_hat_test = model.predict_classes(X_test)
#     elif hasattr(model,'predict'):
#         y_hat_train = model.predict(X_train)
#         y_hat_test = model.predict(X_test)
#     else:
#         raise Exception('model has neither "predict" nor "predict_classes" methods')

#     if y_test.ndim>1 or y_hat_test.ndim>1 or binary_classes==False:
#         if binary_classes==False: 
#             pass
#         else:
#             binary_classes = False
#             print(f"[!] y_test was >1 dim, setting binary_classes to False")

#         ## reduce dimensions of y_train and y_test
#         # y_train = y_train.argmax(axis=1)
#         # y_test = y_test.argmax(axis=1)
#         if y_test.ndim>1:            
#             y_test = y_test.argmax(axis=1)
#         if y_hat_test.ndim>1:
#             y_hat_test = y_hat_test.argmax(axis=1)
#         # for var in ['y_test', 'y_hat_test', 'y_train', 'y_hat_train']:
#         #     real_var = eval(var)
#         #     print('real_var shape:',real_var.shape)
#         #     if real_var.ndim>1:
#         #         ## reduce dimensions
#         #         cmd =  var+'= real_var.argmax(axis=1)'
#         #         # eval(cmd)
#         #         eval(var+'=') real_var.argymax(axis=1)
#         #         # exec(cmd)
#         #         cmd =f'print("argmax shape:",{var}.shape)' 
#         #         eval(cmd)
#         #         # exec(cmd)
        
        
        

#     print('---'*num_dashes)
#     print('\tCLASSIFICATION REPORT:')
#     print('---'*num_dashes)

#     # get both versions of classification report output
#     report_str = classification_report(y_test,y_hat_test)
#     report_dict = classification_report(y_test,y_hat_test,output_dict=True)
#     if report_as_df:
#         try:
#             ## Create and display classification report
#             # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
#             df_report_temp = pd.DataFrame(report_dict)
#             df_report_temp = df_report_temp.T#reset_index(inplace=True)

#             df_report = df_report_temp[['precision','recall','f1-score','support']]
#             display(df_report.round(4).style.set_caption('Classification Report'))
#             print('\n')
        
#         except:
#             print(report_str)
#             # print(report_dict)
#             df_report = pd.DataFrame()
#     else:
#         print(report_str)

#     if save_summary:
#         with open(summary_filename,'w') as f:
#             model.summary(print_fn=lambda x: f.write(x+"\n"))
#             f.write(f"\nSaved at {time_suffix}\n")
#             f.write(report_str)

#     ## Create and plot confusion_matrix
#     conf_mat = confusion_matrix(y_test, y_hat_test)
#     mpl.rcParams['figure.figsize'] = conf_matrix_figsize
#     fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes,
#                                    normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
#     if save_conf_matrix_png:
#         fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)

#     if report_as_df:
#         return df_report, fig
#     else:
#         return report_str,fig

   

def my_custom_scorer(y_true,y_pred, model=None,method='sum', **kwargs):
    """My custom score function to use with sklearn's GridSearchCV
    - Method may be 'mean' or 'sum'
    
    - Maximizes the average accuracy per class using a normalized confusion matrix
    [i] Note: To use my_custom_scorer in GridSearch:
    >> from sklearn.metrics import make_scorer
    >> grid = GridSearch(estimator, parameter_grid)
    """
    from sklearn.metrics import make_scorer,confusion_matrix#, acc
    import numpy as np
    import functions_combined_BEST as ji    

    # set labels for confusion matrix
    labels = ['Decrease','No Change', 'Increase']

    
    ## If y_true is a multi-column one-hotted target
    if y_true.ndim>1 or y_pred.ndim>1:

        ## reduce dimensions of y_train and y_test
        if y_true.ndim>1:            
            y_true = y_true.argmax(axis=1)
            
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)

    
    ## Get confusion matrx
    cm = confusion_matrix(y_true, y_pred)

    ## Normalize confusion matrix
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

    ## Get diagonals for class accuracy
    diag = cm_norm.diagonal()
    
    print('\n')
    ## Choose Score Method
    if 'mean' in method:
        score = np.mean(diag)
        print(f'Mean Class Accuracy = {score}')

        
    elif 'sum' in method:
        score = np.sum(diag)
        print(f'Summed Class Accuracy = {score}')    
    
    ## Display results for user
    print('')
    print(f'Class Accuracy Values:')
    print(diag)    

    ## Plot confusion matrix
    ji.plot_confusion_matrix(cm,normalize=True,classes=labels);

    return score


# # Create HyperParaemeter Space
# params_to_search ={'filter_size':[2,4,6],
#                    'activation':['relu','tanh','linear'],
#                    'n_filters':[100,200],#,300,400],
#                   'dropout':[0.2,0.4],
#                   'optimizer':['adam','rmsprop','adadelta'],
#                 'epochs':[10]}

# def fit_gridsearch(build_fn,parameter_grid,X_train,y_train,score_fn=None,verbose=1,send_email=False):
#     """Builds a Keras model from build_fn, then wraps it in KerasClassifier 
#     for use with sklearn's GridSearchCV. Can score GridSearch with built-in 
#     metric from sklearn, or can pass a custom functions to be used with make_scorer().
#     Upon completion, emails best parameters to gmail account. 
    
#     Args:
#         build_fn (func): Build function for model with parameters to tune as arguments.
#         parameter_grid (dict): Dict of build_fn parameters (keys) and lists of parameters (values)
#         X_train, y_train (numpy array): training dataset
#         score_fn (func or str): Scoring function to use with GridSearchCV. 
#             - For builtin sklearn metrics, pass their name as a string.
#                 - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#             - For custom function, pass function itself. Function must accept, y_true,y_pred
#                 and must return a value to maximize. 
#             - Default(None)=ji.my_custom_scorer().
            
#     Returns:
#         model: (KerasClassifier) The return value. True for success, False otherwise.
#         results
#     """
#     from keras.wrappers.scikit_learn import KerasClassifier#, KerasRegressor
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.metrics import make_scorer
#     import pandas as pd
    
#     import functions_combined_BEST as ji
#     import import bs_ds_local as bs bs
    

#     ## Wrap create_model with KerasClassifier
#     neural_network = KerasClassifier(build_fn=build_fn,verbose=verbose)
    
    
#     ## Run GridSearch
#     import types
#     if score_fn is None:
#         score_func = make_scorer(ji.my_custom_scorer)
#     elif isinstance(score_fn, types.FunctionType):
#         score_func = make_scorer(score_fn)
#     elif isinstance(score_fn, str):
#         score_func =  score_fn
        

#     grid = GridSearchCV(estimator=neural_network,param_grid=parameter_grid, 
#                         scoring=score_func)

#     ## Start Timer
#     tune_clock = bs.Clock()
#     tune_clock.tic()
    
#     ## Fit GridSearch
#     grid_result = grid.fit(X_train, y_train)
#     tune_clock.toc()

#     ## Print Best Params
#     best_params = grid_result.best_params_
#     print(best_params)

#     if send_email:
#         ## Send Email with completion time and best parameters found. 
#         time_completed = pd.datetime.now()
#         fmt = '%m/%d%Y-%T'
#         msg = f"GridSearch Completed at {time_completed.strftime(fmt)}\n GridSearchResults:\n{best_params}"
#         email_notification(msg=msg)
    
#     return grid_result


# class EncryptedPassword():
#     """Class that can be used to either provide a password to be encrypted or load an encypted password from file.
#     The string representations of the unencrypted password are shielded from displaying, when possible. 
    
#     - If encrypting a password, a key file and a password file will be saved to disk. 
#         - Default Key Filename: '..\\encryption_key.bin',
#         - Default Password Filename: '..\\encrypted_pwd.bin'
#     - If opening and decrypting key and password files, pass filenames during initialization. 
    
    
#     Example Usage:
#     >> # To Encrypt, with default folders:
#     >> my_pwd EncryptedPassword('my_password')
    
#     >> # To Encrypt With custom folders
#     >> my_pwd = EncryptedPassword('my_password',filename_for_key='..\folder_outside_repo\key.bin',
#                                     filename_for_password = '..\folder_outside_repo\key.bin')
                                    
                                    
#     >> # To open and decrypt files (from default folders):
#     >> my_pwd = EncryptedPassword(from_file=True)
    
#     >> # To open and decrypt files (from custom folders):
#     >> my_pwd = EncryptedPassword(from_file=True, 
#                                 filename_for_key='..\folder_outside_repo\key.bin',
#                                 filename_for_password = '..\folder_outside_repo\key.bin')
                                    
        
#     """
#     username = 'NOT PROVIDED'
    
#     @property
#     def password(self):
#         if hasattr(self,'_encrypted_password_'):
#             print('Encrypted Password:')
#             return self._encrypted_password_
#         else:
#             raise Exception('Password not yet encrypted.')
            
#     @password.setter
#     def password(self,password):
#         raise Exception('.password is read only.')
        
           
#     def __init__(self,username=None,password=None,from_file=False, encrypt=True,
#                  load_filenames_from_txt_file=None,
#                 filename_for_key='..\\encryption_key.bin',
#                 filename_for_password='..\\encrypted_pwd.bin',
#                 filename_for_username = '..\\encrypted_username.bin'):
        
#         if load_filenames_from_txt_file is None:
#             self.filename_for_key = filename_for_key
#             self.filename_for_password = filename_for_password
#             self.filename_for_username = filename_for_username
#         else:
#             print(f'Overriding input filenames for those saved in {load_filenames_from_txt_file}')
#             self.load_filenames_from_txt()
        
#         if username is not None:
#             self.username = username
        
#         if (password is None):
#             if (from_file==True):
#                 try:
#                     self.load_from_file(key_filename=filename_for_key,
#                                     password_filename=filename_for_password,
#                                         username_filename=filename_for_username)
#                 except:
#                     raise Exception('Something went wrong. Do the key and password files exist?')
                
#             else:
#                 raise Exception('Must either provide a password to encrypt, or set from_file=True')
            
#         else:
# #             _password_
#             self._password_ = password
#             if encrypt:
#                 self.encrypt_password()
                
#     def __repr__(self):
#         password = self._password_
#         msg = f'[i] Password is {len(password)} chars long.'
#         return msg

#     def __str__(self):
#         password = self._password_
#         msg = f'[i] Password is {len(password)} chars long.'
#         return msg        

    
#     def _get_password(self):
#         return self._password_
    
#     def load_filenames_from_txt(self,filename_file='password_filenames.txt'):
#         with open(filename_file,'r') as file:
#             filenames = file.read()
            
#         self.filename_for_key= filenames[0]
#         self.filename_for_username= filenames[1]
#         self.filename_for_password= filenames[2]
#         print(f'Filenames loaded and saved from {filename_file}')
            
    
#     def save_filenames_to_txt(self,filename_file='encrypted_password_filenames.txt'):
#         with open(filename_file,'w') as file:
#             file.write(self.filename_for_key)
#             file.write(self.filename_for_username)
#             file.write(self.filename_for_password)
#         print(f'Key/Password/Username filepaths saved to {filename_file}')
            
    
#     def load_from_file(self,key_filename,password_filename,
#                       username_filename):
    
#         from cryptography.fernet import Fernet
#         ## Load Key
#         with open(key_filename,'rb') as file:
#             for line in file:
#                 key = line


#         cipher_suite = Fernet(key)
#         self._cipher_suite_ = cipher_suite

#         ## Load password
#         with open(password_filename,'rb') as file:
#             for line in file:
#                 encryptedpwd = line
#         self._encrypted_password_ = encryptedpwd
        
#         ## Decrypt password
#         unciphered_text = (cipher_suite.decrypt(encryptedpwd))
#         plain_text_encrypted_password = bytes(unciphered_text).decode('utf-8')
#         self._password_ = plain_text_encrypted_password
        
#         ## Load username
#         with open(username_filename,'rb') as file:
#             for line in file:
#                 username = line
#         self.username = username
        
    
    
#     def encrypt_password(self, show_encrypted_password=False):
     
#         filename_for_key= self.filename_for_key
#         filename_for_password=self.filename_for_password
#         filename_for_username = self.filename_for_username

#         ## Import cryptography and generate encryption key
#         from cryptography.fernet import Fernet
#         key = Fernet.generate_key()
#         self._key_ = key

#         ## Create the cipher_suit for encrypting/decrypting
#         cipher_suite = Fernet(key)
#         self._cipher_suite_ = cipher_suite
 
        
#         ## Encrypt password
#         password = self._password_
#         text_to_encrypt = bytes(password,'utf-8')
#         ciphered_text = cipher_suite.encrypt(text_to_encrypt)#password goes here
#         self._encrypted_password_ = bytes(ciphered_text).decode('utf-8')
        
#         if show_encrypted_password:
#             print('Encrypyted Password:')
#             print(self._encrypted_password_)
        
        
#         ## Encrypt username
#         username = self.username
#         username_to_encrypt = bytes(username,'utf-8')
#         ciphered_username = cipher_suite.encrypt(username_to_encrypt)
#         self._encrypted_username_ = bytes(ciphered_username).decode('utf-8')
        

#         ## Test decryption
#         unciphered_text = cipher_suite.decrypt(ciphered_text)
#         unciphered_username = cipher_suite.decrypt(ciphered_username)
#         password_decoded = unciphered_text.decode('utf-8')
#         username_decoded = unciphered_username.decode('utf-8')
        
#         check_pwd = password_decoded==password
#         check_user = username_decoded==username
                    
#         if  check_pwd &check_user:
#             self._password_ = password_decoded 
#             print('[!] Make sure to delete typed password above from class instantiation.')
#         else:
#             raise Exception('Decrypted password and input password/username do not match. Something went wrong.')

#         ## Specify binary files (outside of repo) for storing key and password files
#         with open(filename_for_key,'wb') as file:
#             file.write(key)

#         with open(filename_for_password,'wb') as file:
#             file.write(ciphered_text)
            
#         with open(filename_for_username,'wb') as file:
#             file.write(ciphered_username)

#         print(f'[io] Encryption Key saved as {filename_for_key}')
#         print(f'[io] Encrypted Password saved as {filename_for_password}')
#         print(f'[io] Encrypted Username saved as {filename_for_username}')
        
        
class EncryptedPassword():
    """Class that can be used to either provide a password/username to be encrypted 
    OR to load a previously encypted password from file.    
    NOTE: Once you have encrypted your password and saved to bin files, you do not need to provide the password again. 
    Make sure to delete your password from the notebook after. 
    - If encrypting a password, a key file and a password file will be saved to disk. 
        - Default Key Filename: '..\\encryption_key.bin',
        - Default Password Filename: '..\\encrypted_pwd.bin'
        - Default Username Filename: '..\\encrypted_username.bin'
    
    The string representations of the unencrypted password are shielded from displaying, when possible. 
    


    - If opening and decrypting key and password files, pass filenames during initialization. 
    
    
    Example Usage:
    >> # To Encrypt, with default folders:
    >> my_pwd EncryptedPassword('my_password')
    
    >> # To Encrypt With custom folders
    >> my_pwd = EncryptedPassword('my_password',filename_for_key='..\folder_outside_repo\key.bin',
                                    filename_for_password = '..\folder_outside_repo\key.bin')
                                    
                                    
    >> # To open and decrypt files (from default folders):
    >> my_pwd = EncryptedPassword(from_file=True)
    
    >> # To open and decrypt files (from custom folders):
    >> my_pwd = EncryptedPassword(from_file=True, 
                                filename_for_key='..\folder_outside_repo\key.bin',
                                filename_for_password = '..\folder_outside_repo\key.bin')
                                    
        
    """
    
    ## Default username
    username = 'NOT PROVIDED'
    
    ## the .password property is designed so it will not display an unencrypted password. 
    @property ## password getter 
    def password(self):
        # if the encrypyted password already exists, print the encrypted pwd (unusable without key)
        if hasattr(self,'_encrypted_password_'):
            print('Encrypted Password:')
            return self._encrypted_password_
        else:
            raise Exception('Password not yet encrypted.')
    
    ## the .password property cannot be set by a user
    @password.setter ## password setter
    def password(self,password):
        raise Exception('.password is read only.')
        
               
    ## 
    def __init__(self,username=None,password=None,from_file=False, encrypt=True,
                filename_for_key='..\\encryption_key.bin',
                filename_for_password='..\\encrypted_pwd.bin',
                filename_for_username = '..\\encrypted_username.bin'):
        """Accepts either a username and password to encyrypt, 
        or loads a previously encrypyed password from file.
        
        Args:
            username (str): email username.
            password (str): email password (note: if have 2-factor authentication on email account, 
                will need app-specific password).
            from_file (bool): whether to load the user credentials from file
            encrypt (bool): whether to encrypt provided password. Default=True
            
            filename_for_key (str): filepath for key.bin (default is'..\\encryption_key.bin')
            filename_for_password: filepath for password.bin (default is'..\\encryption_pwd.bin')
            filename_for_username: filepath for username.bin (default is'..\\encrypted_username.bin')
            """
        
        ## Save filenames 
        self.filename_for_key = filename_for_key
        self.filename_for_password = filename_for_password
        self.filename_for_username = filename_for_username
        
        ## If user passed a username, set username
        if username is not None:
            self.username = username
        
        ## If no password is provided:
        if (password is None):
            
            ##  if load from file if `from_file`=True
            if (from_file==True):
                
                try: ## Load in the key, password, username files
                    self.load_from_file(key_filename=filename_for_key,
                                    password_filename=filename_for_password,
                                        username_filename=filename_for_username)
                except:
                    raise Exception('Something went wrong. Do the key and password files exist?')
            
            ## If no password provided, and from_file=False, raise error
            else:
                raise Exception('Must either provide a password to encrypt, or set from_file=True')
        
        
        ## If the user DOES provide a password
        else:
            self._password_ = password # set the private attribute for password
            
            ## Encrypt the password
            if encrypt:
                self.encrypt_password()
                
                
    def encrypt_password(self, show_encrypted_password=False):
        """Encrypt the key, username, and password and save to external files."""
         ## Get filenames to use.
        filename_for_key= self.filename_for_key
        filename_for_password=self.filename_for_password
        filename_for_username = self.filename_for_username

        ## Import cryptography and generate encryption key
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        self._key_ = key

        ## Create the cipher_suit from key for encrypting/decrypting
        cipher_suite = Fernet(key)
        self._cipher_suite_ = cipher_suite
 
        ## ENCRYPT PASSWORD
        # Get password and change to byte encoding
        password = self._password_
        password_to_encrypt = bytes(password,'utf-8') #password must be in bytes format
        
        # Use the encryption suite to encrypt the password and save to self
        ciphered_pwd = cipher_suite.encrypt(password_to_encrypt)
        self._encrypted_password_ = bytes(ciphered_pwd).decode('utf-8')
        
        # Print encrypyted password if true
        if show_encrypted_password:
            print('Encrypyted Password:')
            print(self._encrypted_password_)
        
        
        ## ENCRYPT USERNAME
        username = self.username
        username_to_encrypt = bytes(username,'utf-8')
        ciphered_username = cipher_suite.encrypt(username_to_encrypt)
        self._encrypted_username_ = bytes(ciphered_username).decode('utf-8')
        
        ## TEST DECRYPTION
        # decrypt password and username
        unciphered_pwd = cipher_suite.decrypt(ciphered_pwd)
        unciphered_username = cipher_suite.decrypt(ciphered_username)
        
        ## Decode from bytes to utf-8
        password_decoded = unciphered_pwd.decode('utf-8')
        username_decoded = unciphered_username.decode('utf-8')
        
        # Check if decoded text matches input text
        check_pwd = password_decoded==password
        check_user = username_decoded==username
        
        ## If everything matches, warn user to delete their exposed password
        if  check_pwd & check_user:
            self._password_ = password_decoded 
            print('[!] Make sure to delete typed password above from class instantiation.')
        else:
            raise Exception('Decrypted password and input password/username do not match. Something went wrong.')

        ## SAVE KEY, PASSWORD, AND USERNAME TO BIN FILES
        ## Specify binary files (outside of repo) for storing key and password files
        with open(filename_for_key,'wb') as file:
            file.write(key)

        with open(filename_for_password,'wb') as file:
            file.write(ciphered_pwd)
            
        with open(filename_for_username,'wb') as file:
            file.write(ciphered_username)

        # Display filepaths for user.
        print(f'[io] Encryption Key saved as {filename_for_key}')
        print(f'[io] Encrypted Password saved as {filename_for_password}')
        print(f'[io] Encrypted Username saved as {filename_for_username}')

            
    
    def load_from_file(self,key_filename,password_filename,
                      username_filename):
        """Load in the encrypted password from file. """
        
        from cryptography.fernet import Fernet
        
        ## Load Key 
        with open(key_filename,'rb') as file:
            for line in file:
                key = line

        ## Make ciphere suite from key
        cipher_suite = Fernet(key)
        self._cipher_suite_ = cipher_suite

        ## Load password
        with open(password_filename,'rb') as file:
            for line in file:
                encryptedpwd = line
        self._encrypted_password_ = encryptedpwd
        
        ## Decrypt password
        unciphered_text = (cipher_suite.decrypt(encryptedpwd))
        plain_text_encrypted_password = bytes(unciphered_text).decode('utf-8')
        self._password_ = plain_text_encrypted_password
        
        ## Load username
        with open(username_filename,'rb') as file:
            for line in file:
                username = line
        unciphered_username = (cipher_suite.decrypt(username))
        plan_text_username = bytes(unciphered_username).decode('utf-8')
        self.username = plan_text_username
        
    def __repr__(self):
        """Controls the printout when the object is the final command in a cell.
        i.e:
        >> pwd =EncrypytedPassword(username='me',password='secret')
        >> pwd
        """
        password = self._password_
        msg = f'[i] Password is {len(password)} chars long.'
        return msg

    def __str__(self):
        """Controls the printout when the object is printed.
        i.e:
        >> pwd =EncrypytedPassword(username='me',password='secret')
        >> print(pwd)
        """
        password = self._password_
        msg = f'[i] Password is {len(password)} chars long.'
        return msg 
    
            
def email_notification(password_obj=None,subject='GridSearch Finished',msg='The GridSearch is now complete.'):
    """Sends email notification from gmail account using prevouisly encrypyter password.
    Args:
        password_obj (EncryptedPassword object): EncryptedPassword object with username/password.
        subject (str):Text for subject line.
        msg (str): Text for body of email. 

    Returns:
        bool: The return value. True for success, False otherwise.

    Loads encrypted key and password from previously exported cipher from cryptography.fernet"""
    if password_obj is None:
        print('Must pass an EncrypytedPassword object.')
        print('>> pwd_obj = EncryptedPassword(username="my_username",password="my_password")')
        print('>> send_email(encrypted_password_obj=pwd_obj)')
        raise Exception('Must pass an EncryptedPassword.')
    
    import smtplib
    from email.message import EmailMessage
    from email.headerregistry import Address
    from email.utils import make_msgid
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    
    
    
    gmail_user = password_obj.username
    gmail_password = password_obj._password_
    
    ## WRITE EMAIL
    message = MIMEMultipart()
    message['Subject'] =subject
    message['To'] = gmail_user
    message['From']=gmail_user
    body = msg
    message.attach(MIMEText(body,'plain'))
    text_message = message.as_string()


    # emails end request
    try:
        with  smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
            server.login(gmail_user,gmail_password)
            server.sendmail(gmail_user,gmail_user, text_message)
            server.close()
            print('Email sent!')

        
    except Exception as e:
        print(e)
        print('Something went wrong')
        
        


## DEFINE FUNCTION FOR FITTING AND EVALUATING SINGLE MODELS (Not GridSearch)
def fit_and_eval_model(model, X_train, X_test, X_val, y_train, y_test, y_val,
                      epochs = 10, batch_size=100,plot_conf_mat_train=False, verbose=1,**kwargs):
    """Fits Kera's model with X_train, y_train data, using (X_val, y_val) for validation_data.
    - Then evaluates model with training/test data
    - Plots Keras Training History
    - Plots Confusion Matrix
    """
    import import bs_ds_local as bs bs
    import functions_combined_BEST as ji
    if 'epochs' in kwargs:
        epochs=kwargs['epochs']
        
    if 'batch_size' in kwargs:
        batch_size=kwargs['batch_size']
    clock = bs.Clock()
    clock.tic()
    dashes = '---'*20
    print(f"{dashes}\n\tFITTING MODEL:\n{dashes}")
    
    

    history = model.fit(X_train, y_train, 
                          epochs=epochs,
                          verbose=verbose, 
                          validation_data=(X_val,y_val),#validation_split=validation_split,
                          batch_size=batch_size)#,
    #                       callbacks=callbacks)

    clock.toc()

    df_report,fig=evaluate_classification_model(model=model,
                                                       X_train=X_train, X_test=X_test,
                                                       y_train=y_train, y_test=y_test, history=history, 
                                                       binary_classes=False,
                                                       plot_training_conf_mat=plot_conf_mat_train,
                                                       conf_matrix_classes=['Decrease','No Change','Increase'])
    return model, df_report



def fit_gridsearch(build_fn,parameter_grid,X_train,y_train,score_fn=None,verbose=1,send_email=False,encrypted_password=None):
    """Builds a Keras model from build_fn, then wraps it in KerasClassifier 
    for use with sklearn's GridSearchCV. Can score GridSearch with built-in 
    metric from sklearn, or can pass a custom functions to be used with make_scorer().
    Upon completion, emails best parameters to gmail account. 
    
    Args:
        build_fn (func): Build function for model with parameters to tune as arguments.
        parameter_grid (dict): Dict of build_fn parameters (keys) and lists of parameters (values)
        X_train, y_train (numpy array): training dataset
        score_fn (func or str): Scoring function to use with GridSearchCV. 
            - For builtin sklearn metrics, pass their name as a string.
                - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            - For custom function, pass function itself. Function must accept, y_true,y_pred
                and must return a value to maximize. 
            - Default(None)=ji.my_custom_scorer().
            
    Returns:
        grid_result:The output from the grid.fit.
    """
    from keras.wrappers.scikit_learn import KerasClassifier#, KerasRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    import pandas as pd
    
    import functions_combined_BEST as ji
    import import bs_ds_local as bs bs
    

    ## Wrap create_model with KerasClassifier
    neural_network = KerasClassifier(build_fn=build_fn,verbose=verbose)
    
    
    ## Run GridSearch
    import types
    if score_fn is None:
        score_func = make_scorer(ji.my_custom_scorer)
    elif isinstance(score_fn, types.FunctionType):
        score_func = make_scorer(score_fn)
    elif isinstance(score_fn, str):
        score_func =  score_fn
        

    grid = GridSearchCV(estimator=neural_network,param_grid=parameter_grid, 
                        scoring=score_func)

    ## Start Timer
    tune_clock = bs.Clock()
    tune_clock.tic()
    
    ## Fit GridSearch
    grid_result = grid.fit(X_train, y_train)
    tune_clock.toc()

    ## Print Best Params
    best_params = grid_result.best_params_
    print(best_params)

    if send_email:
        ## Send Email with completion time and best parameters found. 
        time_completed = pd.datetime.now()
        fmt = '%m/%d%Y-%T'
        msg = f"GridSearch Completed at {time_completed.strftime(fmt)}\n GridSearchResults:\n{best_params}"
        email_notification(password_obj=encrypted_password, msg=msg)
        
    
    return grid_result

