# import nltk
## SENTIMENT ANALYSIS WITH VADER
import bs_ds as bs
# import mod4functions_JMI as jmi
from bs_ds.imports import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
compound_scores=pd.DataFrame()

df_small = pd.read_csv('df_small_equal_phrase_sampled_tweets.csv',encoding='utf-8',index_col=0)


from nltk.sentiment.vader import SentimentIntensityAnalyzer
# df_tokenize = df_small

# Instantiate sid
sid = SentimentIntensityAnalyzer()

# Create a column of sentiment_scores
df_small['sentiment_scores'] = df_small['content_min_clean'].apply(lambda x: sid.polarity_scores(x))
# Returns:
# {'neg': 0.03, 'neu':0.2, 'pos':0.45, 'compound':0.34}

# To extract the compound scores (overall score)
df_small['compound_score'] = df_small['sentiment_scores'].apply(lambda dict: dict['compound'])

# TO simplify to a sentiment_class
df_small['sentiment_class'] = df_small['compound_score'].apply(lambda score: 'pos' if score >=0 else 'neg')
# Write a function to extract the group scores from the dataframe
def get_group_sentiment_scores(df, score_col='sentiment_scores', groupby_col='troll_tweet', group_dict={0:'controls',1:'trolls'}):
    import pandas as pd
    series_df = df[score_col]
    series_neg = series_df.apply(lambda x: x['neg'])
    series_pos = series_df.apply(lambda x: x['pos'])
    series_neu = series_df.apply(lambda x: x['neu'])
    
    series_neg.name='neg'
    series_pos.name='pos'
    series_neu.name='neu'
    
    df = pd.concat([df,series_neg,series_neu,series_pos],axis=1)

#     troll_tweet = pd.Series(df[groupby_col])

#     group_scores = pd.concat([troll_tweet,series_pos, series_neu, series_neg], axis=1)  
#     group_scores.set_index(df.index)
#     group_scores.columns = [['troll_tweet','pos','neu','neg']]
    
    return df
    
with plt.style.context('seaborn-poster'):
    fig,ax =plt.subplots()
    from scipy.stats import sem
    df_troll_res = compound_scores.groupby('troll_tweet')['neg','neu','pos'].get_group(1)
    df_contr_res = compound_scores.groupby('troll_tweet')['neg','neu','pos'].get_group(0)

    y_bars1 = np.mean(df_contr_res[['neg','neu','pos']])
    y_errbars1 = sem(df_contr_res[['neg','neu','pos']])
    y_bars2 = np.mean(df_troll_res[['neg','neu','pos']])
    y_errbars2 = sem(df_troll_res[['neg','neu','pos']])

    bar_width = 0.25
    bar1 = np.arange(len(y_bars1))
    bar2 = [x + bar_width for x in bar1]

    # bar_labels = bar1+0.5
    # bar3 = [x + bar_width for x in bar2]

    ax.bar(x=bar1,height=y_bars1, color='blue', width=bar_width, label = 'Control Tweets',yerr=y_errbars1)

    ax.bar(x=bar2,height=y_bars2,color='orange', width=bar_width, label ='Troll Tweets', yerr=y_errbars2)
    plt.ylim([0,1])
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Scores by Sentiment Type')
    plt.xticks([r + bar_width for r in range(len(y_bars1))],['Negative','Neutral','Positive'])
    plt.legend()