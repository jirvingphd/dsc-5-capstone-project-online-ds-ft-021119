# ABSTRACT:

> Stock Market prices are notoriously difficult to model, but advances in machine learning algorithms in recent years provide renewed possibilities in accurately modeling market performance. One notable addition in modern machine learning is that of Natural Language Processing (NLP). For those modeling a specific stock, performing NLP feature extraction and analysis on the collection of news headlines, shareholder documents, or social media postings that mention the company can provide additional information about the human/social elements to predicting market behaviors. These insights could not be captured by historical price data and technical indicators alone.

> President Donald J. Trump is one of the most prolific users of social media, specifically Twitter, using it as a direct messaging channel to his followers, avoiding the traditional filtering and restriction that normally controls the public influence of the President of the United States. An additional element of the presidency that Trump has avoided is that of financial transparency and divesting of assets. Historically, this is done in order to avoid conflicts of interest, apparent or actual. The president is also known to target companies directly with his Tweets, advocating for specific changes/decisions by the company, or simply airing his greivances. This leads to the natural question, how much influence *does* President Trump exert over the financial markets? 

> To explore this question, we built multiple types of models attempting to answer this question, using the S&P500 as our market index. First, we built a classification model to predict the change in stock price 60 mins after the tweet. We trained Word2Vec embeddings on President Trump's tweets since his election, which we used as the embedding layer for LSTM and GRU neural networks. 

> We next build a baseline time series regression model, using historical price data alone to predict price by trading-hour. We then built upon this, adding several technical indicators of market performance as additional features. 
Finally, we combined the predicitons of our classification model, as well as several other metrics about the tweets (sentiment scores, # of retweets/favorites, upper-to-lowercase ratio,etc.) to see if combining all of these sources of information could explain even more of the variance in stock market prices. 


## MAIN QUESTION:
> #### **Can the Twitter activity of Donald Trump explain fluctuations in the stock market?**

**We will use a combination of traditional stock market forecasting combined with Natural Language Processing and word embeddings from President Trump's tweets to predict fluctuations in the stock market (using S&P 500 as index).**

- **Question 1: Can we predict if stock prices will go up or down at a fixed time point, based on the language in Trump's tweets?**
    - [NLP Model 0](#Model-0)<br><br>
    
- **Question 2: How well can explain stock market fluctuations using only historical price data?**
    - [Stock Market Model 1](#Model-1:-Using-Price-as-only-feature)<br><br>
- **Question 3: Does adding technical market indicators to our model improve its ability to predict stock prices?**
    - [Stock Market Model 2](#Model-2:-Stock-Price-+-Technical-Indicators)<br><br>
- **Question 4: Can the NLP predictions from Question 1, combined with all of the features from Question 3, as well as additional information regarding Trump's Tweets explain even more of the stock market fluctuations?**
    - Stock Market Model 3
    - Stock Market Model X<br><br>

    
### REFERENCES / INSPIRATION:

1. **Stanford Scientific Poster Using NLP ALONE to predict if stock prices increase or decrease 5 mins after Trump tweets.**  
    - [Poster PDF LINK](http://cs229.stanford.edu/proj2017/final-posters/5140843.pdf)
    - Best accuracy was X, goal 1 is to create a classifier on a longer timescale with superior results.
    

2. **TowardsDataScience Blog Plost on "Using the latest advancements in deep learning to predict stock price movements."** 
     - [Blog Post link](https://towardsdatascience.com/aifortrading-2edd6fac689d)


## OVERVIEW OF DATA/FEATURES USED PER MODEL


#### TWITTER DATA - CLASSIFICATION MODEL
**Trained Word2Vec embeddings on collection of Donal Trump's Tweets.**
- Used negative skip-gram method and negative sampling to best represent infrequently used words.
    
**Classified tweets based on change in stock price (delta_price)**
- Calculated price change from time of tweet to 60 mins later.
    - "No Change" if the delta price was < \\$0.05 
    - "Increase" if delta price was >+\\$0.05
    - "Decrease if delta price was >-\\$0.05
    
*NOTE: This model's predictions will become a feature in our final model.*



#### STOCK MARKET (S&P 500) DATA :
##### TIME SERIES FORECASTING USING MARKET DATA
**Model 1: Use price alone to forecast hourly price.**
- Train model using time sequences of 7-trading-hours (1 day) to predict the following hour. 
    * [x] ~~SARIMAX model~~
    * [x] LSTM neural network 

**Model 2: Use price combined with technical indicators.**
    * LSTM neural network
- **Calculate 7 technical indicators from S&P 500 hourly closing price.**
    * [x] 7 days moving average 
    * [x] 21 days moving average
    * [x] exponential moving average
    * [x] momentum
    * [x] Bollinger bands
    * [x] MACD
    
  


#### FINAL MODEL: COMBINING STOCK MARKET DATA,  NLP CLASSIFICATION, AND OTHER TWEET METRICS
- **FEATURES FOR FINAL MODEL:**<br><br>
    - **Stock Data:**
        * [x] 7 days moving average 
        * [x] 21 days moving average
        * [x] exponential moving average
        * [x] momentum
        * [x] Bollinger bands
        * [x] MACD<br><br>
    - **Tweet Data:**
        * [x] 'delta_price' prediction classification for body of tweets from prior hour (model 0)
        * [x] Number of tweets in hour
        * [x] Ratio of uppercase:lowercase ratio (case_ratio)
        * [x] Total # of favorites for the tweets
        * [x] Total # of retweets for the tweets
        * [x] Sentiment Scores:
            - [x] Individual negative, neutral, and positive sentiment scores
            - [x] Compound Sentiment Score (combines all 3)
            - [x] sentiment class (+/- compound score)    


## OSEMN FRAMEWORK

### [OBTAIN](#OBTAIN)
- Obtaining 1-min resolution stock market data (S&P 500 Index)
- Obtain batch of historical tweets by President Trump 

### [SCRUB](#SCRUB)
1. **[Tweets](#TRUMP'S-TWEETS)**
    - Preprocessing for Natural Language Processing<br><br>
2. **[Stock Market](#Loading-&-Processing-Stock-Data-(SCRUB))**
    - Time frequency conversion
    - Technical Indicator Calculation

### [EXPLORE / VISUALIZE](#EXPLORE/VISUALIZE)
- [Tweet Delta Price Classes](#Delta-Price-Classes) 
- [NLP Figures / Example Tweets](#Natural-Language-Processing)
- [S&P 500 Price](#Model-1:-Using-Price-as-only-feature)
- [S&P 500 Technical Indicators](#Technical-Indicator-Details)

### [MODELING (Initial)](#INITIAL-MODELING)
- [Delta-Stock-Price NLP Classifier](#TWEET-DELTA-PRICE-CLASSIFICATON)
- [S&P 500 Neural Network (price only)] ( )

### iNTERPRETATION 
- Delta-Stock-Price NLP Models
    - Model 0A Summary
    - Model 0B Summary
    
- Stock-Market-Forecasting
    - Model 1 Summary
    - Model 2 Summary
    - Model 3 Summary
    - Model 4 Summary
- Final Summary



# OBTAIN


### DATA SOURCES:

* **All Donald Trump tweets from 12/01/2016 (pre-inaugaration day) to end of 08/23/2018**
    *          Extracted from http://www.trumptwitterarchive.com/

* **Minute-resolution data for the S&P500 covering the same time period.**
    *         IVE S&P500 Index from - http://www.kibot.com/free_historical_data.aspx
    
    
* NOTE: Both sources required manual extraction and both 1-min historical stock data and batch-historical-tweet data are difficult to obtain without paying \\$150-\\$2000 monthly developer memberships. 


## TRUMP'S TWEETS

### Natural Language Processing Info

To prepare Donal Trump's tweets for modeling, **it is essential to preprocess the text** and simplify its contents.
<br><br>
1. **At a minimum, things like:**
    - punctuation
    - numbers
    - upper vs lowercase letters<br>
    ***must*** be addressed before any initial analyses. I refer tho this initial cleaning as **"minimal cleaning"** of the text content<br>
    
> Version 1 of the tweet processing removes these items, as well as the removal of any urls in a tweet. The resulting data column is referred to here as "content_min_clean".

<br><br>
2. It is **always recommended** that go a step beyond this and<br> remove **commonly used words that contain little information** <br>for our machine learning algorithms. Words like: (the,was,he,she, it,etc.)<br> are called **"stopwords"**, and it is critical to address them as well.

> Version 2 of the tweet processing removes these items and the resulting data column is referred here as `cleaned_stopped_content`

<br>

3. Additionally, many analyses **need the text tokenzied** into a list of words<br> and not in a natural sentence format. Instead, they are a list of words (**tokens**) separated by ",", which tells the algorithm what should be considered one word.<br><br>For the tweet processing, I used a version of tokenization, called `regexp_tokenziation` <br>which uses pattern of letters and symbols (the `expression`) <br>that indicate what combination of alpha numeric characters should be considered a single token.<br><br>The pattern I used was `"([a-zA-Z]+(?:'[a-z]+)?)"`, which allows for words such as "can't" that contain "'" in the middle of word. This processes was actually applied in order to process Version 1 and 2 of the Tweets, but the resulting text was put back into sentence form. 

> Version 3 of the tweets keeps the text in their regexp-tokenized form and is reffered to as `cleaned_stopped_tokens`
<br>

4. While not always required, it is often a good idea to reduce similar words down to a shared core.
There are often **multiple variants of the same word with the same/simiar meaning**,<br> but one may plural **(i.e. "democrat" and "democrats")**, or form of words is different **(i.e. run, running).**<br> Simplifying words down to the basic core word (or word *stem*) is referred to as **"stemming"**. <br><br> A more advanced form of this also understands things like words that are just in a **different tense** such as  i.e.  **"ran", "run", "running"**. This process is called  **"lemmatization**, where the words are reduced to their simplest form, called "**lemmas**"<br>  
> Version 4 of the tweets are all reduced down to their word lemmas, futher aiding the algorithm in learning the meaning of the texts.


#### EXAMPLE TWEETS AND PROCESSING STEPS:
**TWEET FROM 08-25-2017 12:25:10:**
* **["content"] column:**<p><blockquote>***"Strange statement by Bob Corker considering that he is constantly asking me whether or not he should run again in '18. Tennessee not happy!"***
* **["content_min_clean"] column:**<p><blockquote>***"strange statement by bob corker considering that he is constantly asking me whether or not he should run again in  18  tennessee not happy "***
* **["cleaned_stopped_content"] column:**<p><blockquote>***"strange statement bob corker considering constantly asking whether run tennessee happy"***
* **["cleaned_stopped_tokens"] column:**<p><blockquote>***"['strange', 'statement', 'bob', 'corker', 'considering', 'constantly', 'asking', 'whether', 'run', 'tennessee', 'happy']"***
* **["cleaned_stopped_lemmas"] column:**<p><blockquote>***"strange statement bob corker considering constantly asking whether run tennessee happy"***



## Twitter Processing 

```python
def full_twitter_df_processing(df, raw_tweet_col='content', 
name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None,
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