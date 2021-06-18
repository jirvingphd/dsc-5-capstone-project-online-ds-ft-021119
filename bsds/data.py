## Import data downloading packages
import datetime as dt
import requests
import pandas as pd
import json
import os,sys,glob,time
from IPython.display import display


def download_stock_data(fpath='data/ive_minute_tick_bidask_API_2021.csv',
                       verbose=True,append_date=True):
    """Downloads up-to-date IVE S&P 500 1-min aggregate data from 
    http://www.kibot.com/free_historical_data.aspx
    
    Args:
        fpath (str): csv filepath to save (Default='data/ive_minute_tick_bidask_API.csv')
        verbose (bool): Display file info (Default=True)
        
    Returns:
        stock_df: DataFrame with correct headers and datetime index
        
    Example Use:
    >> # stocks = download_stock_data()
    >> stocks.to_csv('data/ive_minute_tick_bidask_API_06-16-21.csv.gz',
              compression='gzip',index=True)
    >> stock_df = pd.read_csv('data/ive_minute_tick_bidask_API_06-16-21.csv.gz',
                      parse_dates = ['datetime'], index_col='datetime')
    stock_df    
    """
    agg_url = 'http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask1&bp=1&user=guest'
    response = requests.get(agg_url,
                            allow_redirects=True)

    ## Save output to csv file
    with open(fpath,'wb') as file:
        file.write(response.content)
        
        
    ## Load in Stock Data Frame with headers (then save)
    headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
    stock_df = pd.read_csv(fpath,names=headers)

# 
    ## Make Combined Date Time column and Drop Origs
    stock_df['datetime'] = pd.to_datetime(stock_df['Date'].astype(str)+' '+stock_df['Time'].astype(str))
    
    if append_date:
        suffix = dt.date.today().strftime('%m-%d-%y')
        fpath = f"{fpath.split('.csv')[0]}_{suffix}.csv.gz"
        
    print(f'Saving as {fpath}')
    stock_df.to_csv(fpath,index=False, compression='gzip')
        
    if verbose:
        print('[i] Stock data successfully downloaded and saved as:')
        print(' - ',fpath)
        
    return pd.read_csv(fpath,parse_dates=['datetime'],index_col='datetime')




def download_trump_tweets(fpath='data/trump_tweets_2021.csv',append_date=True,
                          verbose=True,return_data=True):
    """Downloads the most recent data from the trumptwittearchive v2.
    new url = "https://drive.google.com/file/d/1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6/view?usp=sharing"

    Args:
        fpath (str): filepath for data that ends with .csv
        append_date (bool): Whether to save today's date as part of filename(Default=True)
        verbose (bool): Whether to print the file name (Default=True)
        return_data (bool): Whether to return the data as a df (Default=True)"""
#     url = "https://www.thetrumparchive.com/latest-tweets"
    # url="https://drive.google.com/uc?export=download&id=1JZnhB0Nq_2RKtDb-IOnd0XxnD5c7x-nQ"
    url="https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6"
    response = requests.get(url)
    
    if append_date:
        suffix = "_"+dt.date.today().strftime('%m-%d-%y')
        filepath = f"{fpath.split('.')[0]}{suffix}.{fpath.split('.')[-1]}"
    else:
        filepath=fpath
        
        
    ## Save output to csv file
    with open(filepath,'wb') as file:
        file.write(response.content)  
        
#     with open(filepath,'w') as f:
# #         f.write(response.content)
#         f.write(json.dumps(response.json()))
    
#     if fpath.endswith('.csv'):
#         tweets = pd.read_json(filepath)
#         tweets.to_csv(filepath)
#     else: 
#         tweets = pd.read_json(filepath)
        
    if verbose:
        print('[i] Tweet data successfully downloaded and saved as:')
        print('- ',filepath)
        
    if return_data:

        return pd.read_csv(filepath,index_col='date',parse_dates=['date'])
#tweets#,parse_dates=['created_at'])



def get_most_recent_file(fpath='data/', ftype='csv', find_str='tweet',show_all=False,load=True,
parse_dates=None, index_col = None):
    """Check the {fpath} for the all {ftype} files that contain {find_str}

    Args:
        fpath (str): fpath ending in '/'
        ftype (str): file extension (do not include the'.' e.g. "csv")
        find_str (str): common str in filenames to check [Orig options: 'ive_minute'/'tweet']
        show_all (bool; Default=False): whether to display series of all found files.
        load (bool; Default = True): whether to load and return the dataframe.
            - If True, return the dataframe
            - If False. return the Series of filenames

        parse_dates (str, Defaults to None): argument for pd.read_csv's parse_dates
            - if None and find_str=='ive_minute': will use 'datetime'
            - if None and find_str=='tweet': will use 'date'

        index_col (str, Defaults to None): argument for pd.read_csv's index_col
            - if None and find_str=='ive_minute': will use 'datetime'
            - if None and find_str=='tweet': will use 'date'       
    Returns:

    """
    ## Set parse_dates col
    if parse_dates is None:
        if find_str=='ive_minute':
            parse_dates = ['datetime']
        elif find_str=='tweet':
            parse_dates = ['date']
        else:
            parse_dates=False


    if index_col is None:
        if find_str=='ive_minute':
            index_col = 'datetime'
        elif find_str=='tweet':
            index_col = 'date'


    ## Check for prx-existing files
    files_glob = glob.glob(f'{fpath}/*.{ftype}')
    file_list = list(filter(lambda x: find_str in x, files_glob))
    # tweet_files = list(filter(lambda x: 'tweet' in x, files_glob))


    ## Get Time Files Modified 
    FILES = pd.Series({f:pd.to_datetime(time.ctime(os.path.getmtime(f))) for f in file_list})

    if show_all:
        display(FILES.sort_values())

    ## Get most recent files using idxmin
    recent_file = FILES.idxmax()
    
    ## Load in the csvs with datetime indices
    if load:

        df = pd.read_csv(recent_file,parse_dates=parse_dates,index_col=index_col)
        
        ## Sort timeseries
        df.sort_index(inplace=True)

        return df
    else:
        return FILES