### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def load_processed_stock_data(processed_data_filename = 'data/_stock_df_with_technical_indicators.csv', force_from_raw=False, verbose=0):
    import functions_combined_BEST as ji
    import os
    import pandas as pd
    
    # check if csv file already exists
    current_files = os.listdir()

    # Run all processing on raw data if file not found.
    if (force_from_raw == True):# or (processed_data_filename not in current_files):
        
        print(f'[!] File not found. Processing raw data using custom ji functions...')
        print('1) ji.load_raw_stock_data_from_text\n2) ji.get_technical_indicators,dropping na from column "ma21"')

        stock_df = ji.load_raw_stock_data_from_txt(
            filename="IVE_bidask1min.txt", 
            start_index='2016-12-01',
            clean=True, fill_or_drop_null='drop', 
            freq='CBH',verbose=1)

        ## CALCULATE TECHNICAL INDICATORS FOR STOCK MARKET
        stock_df = ji.get_technical_indicators(stock_df, make_price_from='BidClose')

        ## Clean up stock_df 
        # Remove beginning null values for moving averages
        na_idx = stock_df.loc[stock_df['ma21'].isna() == True].index # was 'upper_band'
        stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]


    # load processed_data_filename if found
    else:# processed_data_filename in current_files:

        print(f'>> File found. Loading {processed_data_filename}')
        
        stock_df=pd.read_csv(processed_data_filename, index_col=0, parse_dates=True)
        stock_df['date_time_index'] = stock_df.index.to_series()
        stock_df.index.freq=ji.custom_BH_freq()
    
    if verbose>0:
        # -print(stock_df.index[[0,-1]],stock_df.index.freq)
        index_report(stock_df)
#     display(stock_df.head(3))
    stock_df.sort_index(inplace=True)

    return stock_df        

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
def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH

```
### SOURCE:
```python
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

```
