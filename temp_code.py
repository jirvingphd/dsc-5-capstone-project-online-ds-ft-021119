

def search_for_tweets_with_word(twitter_df,word, display_n=10,from_column='content', ascending=False,
                                as_md=True):
    """Searches the df's `from_column` for the specified `word`
    - `as_md`
        - If True(default):
            - Call `df_to_md` to get Markdown string from resulting df
        - if False:
            - Return as dataframe
        
        - display the most-recent or oldest tweets using `ascending` parameter.
    - if return_index: return the datetimeindex of the tweets containing the word."""
    import pandas as pd
    import functions_combined_BEST as ji
    from IPython.display import display
    import numpy as np
    
    word_in = word

    ## Make list of cols starting with from_column and adding display_cols
    select_cols = [from_column]
#     [select_cols.append(x) for x in display_cols]
    
    # Create new df copy with select_cols
    df = twitter_df[select_cols].copy()
    
    ## Check from_column for word.lower() in text.lower()
    check_word = df[from_column].apply(lambda x: True if word.lower() in x.lower() else False)
    
    # Tally number of tweets containing word
    found_words = int(np.sum([1 for x in check_word if x ==True]))
    
    ## Get rows with the searched word
    res_df = df.loc[check_word]
    
    ## Sort res_df_ by datetime index, before resetting index
    res_df.sort_index(inplace=True, ascending=ascending)
    res_df.reset_index(inplace=True)
    
    
    ## If as_md, get md_text from df_to_md
    if as_md:
        df_to_show = res_df.iloc[:display_n]
        header = f'### Showing {display_n} of {found_words} instances of "{word_in}"\n'
        md_tweets = df_to_md(df_to_show)
        text_out = header + md_tweets
        
        return text_out
    
    else:
        return res_df
    

def search_for_tweets_by_date(twitter_df,date, display_n=10,from_column='content', ascending=False,
                                as_md=True):
    """Searches the df's `from_column` for the specified `word`
    - `as_md`
        - If True(default):
            - Call `df_to_md` to get Markdown string from resulting df
        - if False:
            - Return as dataframe
        
        - display the most-recent or oldest tweets using `ascending` parameter.
    - if return_index: return the datetimeindex of the tweets containing the word."""
    import pandas as pd
    import functions_combined_BEST as ji
    from IPython.display import display
    import numpy as np
    ## Make list of cols starting with from_column and adding display_cols
    select_cols = [from_column]
#     [select_cols.append(x) for x in display_cols]
    
    # Create new df copy with select_cols
    df = twitter_df[select_cols].copy()
    
    ## Get rows with the searched word
    res_df = df.loc[date]
        # res_df = df.loc[check_word]
    
    ## Sort res_df_ by datetime index, before resetting index
    res_df.sort_index(inplace=True, ascending=ascending)
    res_df.reset_index(inplace=True)
    
    
    ## If as_md, get md_text from df_to_md
    if as_md:
        df_to_show = res_df.iloc[:display_n]
        if df_to_show.shape[0]>0:
            header = f'### Showing {display_n} Tweets from "{date}"\n'
        else:
            header = f'### No Tweets found for "{date}"\n'
        md_tweets = df_to_md(df_to_show,from_column=from_column)
        text_out = header + md_tweets
        
        return text_out
    
    else:
        return res_df
    


    
def df_to_md(res_df,show=False,from_column='content'):
    from IPython.display import display, Markdown
    import pandas as pd
    df_md = res_df.copy()
    
    date_format = '%m/%d/%Y - %T'
    df_md['md_tweet'] = res_df[from_column].apply(lambda x: str(x))
    df_md['md_date'] = res_df['date'].apply(lambda x: f'* ***Tweet from {x.strftime(date_format )}:***\n >')

    df_md['out_text'] = df_md['md_date'] + df_md['md_tweet']

    display_text = '\n'.join(df_md['out_text'])
    
    if show:
        display(Markdown(display_text))

    return display_text
        

def search_for_tweets_prior_hour(twitter_df,stock_hour, from_column='content', ascending=False,
                                as_md=True):
    """Searches the df's `from_column` for the specified `word`
    - `as_md`
        - If True(default):
            - Call `df_to_md` to get Markdown string from resulting df
        - if False:
            - Return as dataframe
        
        - display the most-recent or oldest tweets using `ascending` parameter.
    - if return_index: return the datetimeindex of the tweets containing the word."""
    import pandas as pd
    import functions_combined_BEST as ji
    from IPython.display import display
    import numpy as np
    fmt = '%m/%d/%Y %T'

    ## Make list of cols starting with from_column and adding display_cols
    select_cols = [from_column]
#     [select_cols.append(x) for x in display_cols]
    
    # Create new df copy with select_cols
    df = twitter_df[select_cols].copy()
    
    
    ## Make a timedelta of 1 hour to create daterange slice from [date+hr_ofst:date]
    hr_ofst = pd.to_timedelta('-1 hour')
    idx_end = stock_hour
    idx_start = idx_end+hr_ofst
    
    ## Convert back to strings for pandas
    idx_end = idx_end.strftime(fmt)
      
    idx_start = idx_start.strftime(fmt)
    ## Get rows with the searched word
    res_df = df.loc[idx_start:idx_end]
        # res_df = df.loc[check_word]
    
    ## Sort res_df_ by datetime index, before resetting index
    res_df.sort_index(inplace=True, ascending=ascending)
    res_df.reset_index(inplace=True)
    

    ## If as_md, get md_text from df_to_md
    if as_md:

        df_to_show = res_df #.iloc[:display_n]
        if df_to_show.shape[0]>0:
            header = f'### Showing {df_to_show.shape[0]} Tweets from "{idx_start} to {idx_end}"\n'
        
        else:
            header = ''#f'### No Tweets found for "{idx_start} to {idx_end}"\n'
        
        md_tweets = df_to_md(df_to_show,from_column=from_column)
        text_out = header + md_tweets
        
        return text_out
    
    else:
        return res_df    