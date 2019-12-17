### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,
 use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)"""
    # Save 'hashtags' column containing all hastags
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None or 'content_min_clean' in use_col_for_case_ratio:
        use_col_for_case_ratio='content_min_clean'
        df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
        print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: apply_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
    series_neu = series_df.apply(lambda x: x['neu'])

```
### SOURCE:
```python
    twitter_df['sentiment_class'] = twitter_df['compound_score'].apply(lambda score: 'pos' if score >=0 else 'neg')

```
### SOURCE:
```python
        # df_full['content_urls'] = df_full[check_content_col].apply(lambda x: urls.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

```
### SOURCE:
```python
    def lemmatize_tweet(x):
        
        import functions_combined_BEST as ji
        if isinstance(x,str):
            from nltk import regexp_tokenize
            pattern = ji.make_regexp_pattern()
            x = regexp_tokenize(x,pattern)
            
            
        from nltk.stem import WordNetLemmatizer
        lemmatizer=WordNetLemmatizer()
        output = []
        for word in x:
            output.append(lemmatizer.lemmatize(word))
        output = ' '.join(output)
        return output

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df

```
### SOURCE:
```python
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list

```
### SOURCE:
```python
def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

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
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



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


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df

```
### SOURCE:
```python
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

```
