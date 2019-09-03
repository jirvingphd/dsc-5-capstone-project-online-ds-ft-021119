
def search_for_tweets_with_word(twitter_df,word, from_column='content_min_clean',display_n=5, ascending=False,
                                return_index=False, display_df=False,as_md=False,as_df=True, display_cols = [
                                    'retweet_count','favorite_count','source',
                                    'compound_score','sentiment_class']):
    """Searches the df's `from_column` for the specified `word`.
    - if display_df: Displays first `n` rows of styled dataframe with, columns=`display_cols`.
        - display the most-recent or oldest tweets using `ascending` parameter.
    - if return_index: return the datetimeindex of the tweets containing the word."""
    import pandas as pd
    import functions_combined_BEST as ji
    from IPython.display import display
    import numpy as np
    n=display_n

    ## Make list of cols starting with from_column and adding display_cols
    select_cols = [from_column]
    [select_cols.append(x) for x in display_cols]
    
    # Create new df copy with select_cols
    df = twitter_df[select_cols].copy()
    
    ## Check from_column for word.lower() in text.lower()
    check_word = df[from_column].apply(lambda x: True if word.lower() in x.lower() else False)
    # Tally number of tweets containing word
    found_words = np.sum([1 for x in check_word if x ==True])
    
    ## Get rows with the searched word
    res_df_ = df.loc[check_word]
    
    # Save datetime index to output before changing
    output_index = res_df_.index.to_series()
    
    ## Sort res_df_ by datetime index, before resetting index
    res_df_.sort_index(inplace=True, ascending=ascending)
    res_df_.reset_index(inplace=True)
    
    
    # Take n # of rows from the top of the dataframe
    
    ## Set table_style for display df.styler
    table_style =[{'selector':'caption',
                'props':[('font-size','1.3em'),
                         ('color','darkblue'),
                         ('font-weight','semibold'),
                        ('text-align','left')]},
                 {'selector':'th',
                  'props':[('font-size','1.1em'),('text-align','center')]}]
#                  {'selector':'td','props':[('pad','0.1em')]}]
    


    if display_n is None:
        n=res_df_.shape[0]-1
        
    res_df = res_df_.iloc[:n,:]
    # full_index = res_df_.index
    
    ## Create styled df with caption, table_style, hidden index, and text columns formatted
    if as_md==False and as_df==False:
        df_to_show = res_df
    else:
        df_to_show = res_df_

    ## Caption for df
    capt_text = f'Tweets Containing "{word}" ({display_n} of {found_words})'
    
    dfs = df_to_show.style.hide_index().\
    set_caption(capt_text).set_properties(subset=[from_column],
                                          **{'width':'400px',
                                            'text-align':'center',
                                            'padding':'2em',
                                            'font-size':'1.2em'}).set_table_styles(table_style)


    if display_df:
        display(dfs)
        remaining_tweets = found_words - n
        next_index = res_df_['date'].iloc[n+1]

        print(f'\t * there are {remaining_tweets} tweets not shown. Next index = {next_index}')
    
    ## Return datetimeindex of all found tweets with the word
    if return_index==True:
        return output_index

    num_tweets = len(output_index)
    ## Display dataframe if display_df
    if as_df:
        return df
    if as_md:
        # return dfs
        ## to make mardown 
        md_to_show = []
        for ii in range(len(output_index)):#range(num_tweets):
            i = output_index[ii]
            tweet_to_print = twitter_df.loc[i]
        
            if num_tweets>1:
                md_to_show.append("<br>")
                md_to_show.append(f'### Tweet #{ii+1} of {len(output_index)}: sent @ *{i}*:')#index[i]
            else:
                md_to_show.append(f'#### TWEET FROM {i}:')#index[i]
            # print(f'* TWEETED ON {df_sampled.index[i]}')
            for col in display_cols:

                col_name = f'* **"{col}" column:**<p><blockquote>***" {tweet_to_print[col]} "***'
                md_to_show.append(col_name)

        md_combined = '\n'.join(md_to_show)
        output_md=md_combined
    return output_md


# ## FROM YELP API LAB _ EXAMPLE PAGINATIONS
# def page_results(twtiter_df,iloc_index,n,page,displayed_index=None):
#     num = len(idx_search)
#     print(f'{num} total matches found.')#.format(num))
#     dfs = []

#     if displayed_index is not None:
#         last_displayed = displayed_index.iloc[-1]

#         cur = iloc_index.loc[]
#     else:
#         cur = 0


#     while cur < num: #and cur < nun:
#         # url_params['offset'] = cur
#         idx_show = idx_search.iloc[cur:cur+n]
#         dfs.append(yelp_call(url_params, api_key))
#         time.sleep(1) #Wait a second
#         cur += 50
#     df = pd.concat(dfs, ignore_index=True)
#     return df


##
'''
def display_results_as_md_df(tweet,as_df=False,as_md=True):
    if as_df == False and as_md==False:

            ## for each tweet:
            for i in range(num_tweets):

                print(f'\n>> TWEETED ON {print_index[i] }') ##tweet.index[i]}')
                if is_series:
                    tweet_to_print = tweet
                else:
                    tweet_to_print=tweet.loc[print_index[i]]

                # print each column for dataframes
                for col in columns:
                    print(f"\n\t[col='{col}']:")
                    # col_text = tweet.loc[col] 
                    print('\t\t"',tweet_to_print[col],'"')

                if num_tweets>1:
                    print('\n','---'*10)


        elif as_md:

            for ii in range(len(print_index)):#range(num_tweets):
                i = print_index[ii]
                tweet_to_print = df_sampled.loc[i]

                if num_tweets>1:
                    display(Markdown("<br>"))
                    display(Markdown(f'### Tweet #{ii+1} of {len(print_index)}: sent @ *{i}*:'))#index[i]
                else:
                    display(Markdown(f'#### TWEET FROM {i}:'))#index[i]
                # print(f'* TWEETED ON {df_sampled.index[i]}')
                for col in columns:

                    col_name = f'* **["{col}"] column:**<p><blockquote>***"{tweet_to_print[col]}"***'
                    
                    display(Markdown(col_name))

                
        elif as_df:

            # for i in range(num_tweets):
            for ii in range(len(print_index)):#range(num_tweets):
                i = print_index[ii]
                tweet_to_print = df_sampled.loc[i]

                df = pd.DataFrame(columns=['tweet'],index=columns)
                df.index.name = 'column'
                for col in columns:
                    df['tweet'].loc[col] = tweet_to_print[col]#[i]
                
                with pd.option_context('display.max_colwidth',0, 'display.colheader_justify','left'):#,'colheader_justify','left'):
                    # caption = f'Tweet #{ii+1}  = {i}'
                    capt_text  = f'Tweet #{ii+1} of {len(print_index)}: sent @ *{i}*:'
                    table_style =[{'selector':'caption',
                    'props':[('font-size','1.2em'),('color','darkblue'),('font-weight','bold'),
                    ('vertical-align','0%')]}]
                    dfs = df.style.set_caption(capt_text).set_properties(subset=['tweet'],
                    **{'width':'600px',
                    'text-align':'center',
                    'padding':'1em',
                    'font-size':'1.2em'}).set_table_styles(table_style)
                    display(dfs)
        return 
        '''