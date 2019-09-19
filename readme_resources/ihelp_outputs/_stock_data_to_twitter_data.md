### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
### SOURCE:
```python
def load_raw_stock_data_from_txt(filename='IVE_bidask1min_08_23_2019.csv', 
                               folderpath=None,
                               start_index = '2016-12-01',
                                 clean=True,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    """- Loads in the IVE_bidask1min data from text file, adds column headers.
        - Original stock data file: 'IVE_bidask1min.txt', 'IVE_bidask1min_08_23_2019.txt'
    - Creates datetimeindex from Date/Time cols, but keeps the 'datetime_index` column in the df.  
        - Limits data to specified `start_index` date(defualt='2016-12-01').

    - If clean=True, addresses rare occurance of '0' values for stock price using `fill_or_drop_null` param. 

    - Sets the frequency of the data and handles null values with ji.set_timeindex_freq. 
        - `freq` = any pandas frequency offset abbreviation (i.e. 'B','BD','H','T','BH',etc.)
            - Default value of `CBH` creates custom business hour (market open @ 09:30am to 4:30pm
        - `fill_method`= method used to resolve null values created during frequency resampling

    - Verbose controls the level of detail regarding number datetimeindex creation, # of null value addressed, etc.
        - Default = 2(maximum)
        - Default >=1 will display stock_df.head()
     """
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # Load in the text file and set headers
    if folderpath is not None:
        fullfilename= folderpath+filename
    else:
        fullfilename = filename
    print(f"[i] Loading {fullfilename}...\n")

    ## IF USING TRUE RAW TXT FILES:
    ext = filename.split('.')[-1]
    if 'txt' in ext:

        headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
        stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)

        # Create datetime index
        date_time_index = (stock_df['Date']+' '+stock_df['Time']).rename('date_time_index')
        stock_df['date_time_index'] = pd.to_datetime(date_time_index)

        # stock_df.set_index('date_time_index', inplace=True, drop=False)
        # stock_df.sort_index(inplace=True, ascending=True)
    
    elif 'csv' in ext: # USING THE PARTIAL PROCESSED (size reduced, datetime index)
        stock_df = pd.read_csv(fullfilename, parse_dates=True)
        stock_df['date_time_index'] = pd.to_datetime( stock_df['date_time_index'])
    else:
        raise Exception('file extension not csv or txt')
    
    stock_df.set_index('date_time_index', inplace=True, drop=False)

    # Select only the days after start_index
    if verbose>0: 
        print(f"[i] Exlcuding data earlier than {start_index}.")
    
    ## Continue processing
    stock_df.sort_index(inplace=True, ascending=True)  
    # remove earlier data
    stock_df = stock_df.loc[start_index:]
    

    
    # Remove 0's from BidClose
    if clean==True:
        # Get number of zeros
        num_zeros= len(stock_df.loc[stock_df['BidClose']==0])

        # Replacing 0's with np.nan
        stock_df.loc[stock_df['BidClose']==0] = np.nan

        # get count of null values
        num_null = stock_df['BidClose'].isna().sum()

        if verbose>1:
            print(f'[i] Cleaning 0 values:\n\t -replaced {num_zeros} zeroes in "BidClose" with np.nan ...".')
        
        ## handle the new null values
        if fill_or_drop_null=='drop':
            if verbose>0:
                print("\t- dropping null values (`fill_or_drop_null`).")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)

        elif fill_or_drop_null=='fill':
            if verbose>0:
                print(f'\t- filling null values using fill_method: "{fill_method}"')
            # print(f'\tsince fill_or_drop_null=fill, using fill_method={fill_method} to fill BidClose.')
            stock_df.sort_index(inplace=True,ascending=True)
            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        # if verbose>0:
            # print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            # print(f"Filling 0 values using method = {fill_method}")

    # call set_timeindex_freq to specify proper frequency
    if freq is not None:
        # Set the time index .
        if verbose>0:
            print(f'[i] Converting the datetimeindex to `freq` "{freq}"')
            print(f'\t- addressing resulting null values using fill_method {fill_method}...')

        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=0)

    # sort data so newest at top
    stock_df.sort_index(inplace=True, ascending=False)           

    # Display feedback


    if verbose>0:
        try:
            index_report(stock_df)
        except:
            print('[!] Error in index_report')
            print(stock_df.index)
        # print(stock_df.index[[0,-1]],stock_df.index.freq)

    if verbose>1:
        cap = f"Data Loaded from {fullfilename}"
        display(stock_df.head().style.set_caption(cap))

    return stock_df

```
### SOURCE:
```python
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_nulls = True, fill_with_val_or_method='method',fill_val= None, fill_method='ffill',
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
    ive_df = ive_df.asfreq(freq)#'min')
    
    #     # Set timezone
    #     if set_tz==True:
    #         ive_df.tz_localize()
    #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
    # Report Success / Details
    if verbose>1:
        print(f"[i] Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


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
    ive_df.sort_index(inplace=True, ascending=True)     

    if fill_nulls:
        if 'method' in fill_with_val_or_method:# =='fill':
            if fill_method is not None:
                ive_df.fillna(method=fill_method, inplace=True)
            else:
                raise Exception('[!] fill_method not specified')

        elif 'val' in fill_with_val_or_method:
            if fill_val is not None:
                ive_df.fillna(fill_val,inplace=True)
            else:
                raise Exception('[!] fill_val not specified')

    # Report # filled
    if verbose>0:
        check_fill = ive_df.loc[ive_df['filled_timebin']>0]
        print(f'\n[i] Filled {len(check_fill==True)}# of rows using method {fill_method}')
    
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

```
### SOURCE:
```python
def load_twitter_df_stock_price(twitter_df = None, stock_price_file = 'IVE_bidask1min_08_23_2019.csv', display_header=False,
get_stock_prices_per_tweet=False,price_mins_after_tweet=60):# del stock_price
    """Loads in stock_price data from original text souce in minute frequency using 
    load_stock_price_series(filename= #'IVE_bidask1min_08_23_2019.csv)
    if twitter_df not provided, it is created from scratch using processing functions. 
    if get_stock_prices_per_tweet == True, runs the follow-up function to add several stock_price columns matched to each tweet.
    price_mins_after_tweet = how many minutes after the tweet should the change in stock price be calculated. 
    
    Returns twitter_df only if `get_stock_prices_per_tweet`==True.
    Returns twitter_df, stock_price_series if it is False
    """
    from IPython.display import display
    import functions_combined_BEST as ji
    try: 
        stock_price
    except NameError: 
        stock_price = None

    if stock_price is  None:    
        print('[io] Loading 1-minute-resolution stock_prices...')
        stock_price = ji.load_stock_price_series()

    else:
        print('[i] Using pre-existing stock_price.')

    # Make sure stock_price is loaded as minute data
    stock_price = stock_price.asfreq('T')
    stock_price.dropna(inplace=True)
    stock_price.sort_index(inplace=True)

    ## LOAD TWEETS, SELECT THE PROPER DATE RANGE AND COLUMNS
    # twitter_df = load_twitter_df(verbose=0)
    if twitter_df is None: 
        print('[i] Creating twitter_df using `load_raw_twitter_file()`, `full_twitter_df_processing`')
        twitter_df= ji.load_raw_twitter_file()
        twitter_df = ji.full_twitter_df_processing(twitter_df)

    stock_price.sort_index(inplace=True)
    twitter_df.sort_index(inplace=True)
    
    if display_header:
        display(twitter_df.head(3))
        ji.index_report(twitter_df,label='twitter_df')
        ji.index_report(stock_price, label='stock_price')
        # print(stock_price.index[0],stock_price.index[-1])
        # print(twitter_df.index[0],twitter_df.index[-1])
    
    if get_stock_prices_per_tweet:

        cols_created_by_func = ['B_day', 'B_shifted', 'B_time','B_dt_index', 'time_shift', 'B_dt_minutes', 'stock_price_results','null_results', 'B_ts_rounded', 'pre_tweet_price', 'mins_after_tweet',
        'B_ts_post_tweet', 'post_tweet_price', 'delta_time', 'delta_price','delta_price_class', 'delta_price_class_int']
        
        if any([True for x in cols_created_by_func if x in  twitter_df.columns]):
           print(f'[!] Found columns created by `get_stock_prices_for_twitter_data`\nReturning input df.')
           return twitter_df
        else:
            print(f'[i] Adding stock_price data for {price_mins_after_tweet} mins post-tweets using `get_stock_prices_for_twitter_data`...')
            print('[i] Limiting twitter_df timeindex to match stock_price.')
            twitter_df = twitter_df.loc[stock_price.index[0]:stock_price.index[-1]]

            twitter_df = get_stock_prices_for_twitter_data(twitter_df=twitter_df, stock_prices=stock_price,
            time_after_tweet=price_mins_after_tweet)
            return twitter_df

    else:
        return twitter_df, stock_price

```
