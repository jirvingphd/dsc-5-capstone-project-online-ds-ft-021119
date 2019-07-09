# functions_combined.py

def ihelp(any_function, show_help=True, show_code=True, get_source=True): 
    """Call on any module or functon to display:
    - help(any_function)
    - source_df = inspect.getsource(any_function)
    inspect.get)"""
    import inspect
    import pprint as pp
    if show_help:
        
        print("---"*40)
        print("---"*3,'\tHELP:\t',"---"*30)
        print("---"*40)
        help(any_function)
#         print('\n\n',"---"*20,'\n')
        
    if show_code or get_source:
        
        import inspect
        source_DF = inspect.getsource(any_function)
        print("---"*40)
        print("---"*3,'\tSOURCE:\t',"---"*30)
        print("---"*40)
        print(source_DF)    



def df_column_report(twitter_df, sort_column=None, ascending=True, interactive=True):
    from ipywidgets import interact
    import pandas as pd
    df_dtypes=pd.DataFrame()
    df_dtypes = pd.DataFrame({'Column #': range(len(twitter_df.columns)),'Column Name':twitter_df.columns,
                              'Data Types':twitter_df.dtypes.astype('str')}).set_index('Column Name') #.set_index('Column Name')
    
    decision_map = {'object':'join','int64':'sum','bool':'to_list()?','float64':'drop and recalculate'}
    
    df_dtypes['action'] = df_dtypes['Data Types'].map(decision_map)#column_list
#     df_dtypes.style.set_caption('DF Columns, Dtypes, and Course of Action')
    
    if sort_column is not None:
        df_dtypes.sort_values(by =sort_column,ascending=ascending, axis=0, inplace=True)
    if interactive==False:
        return df_dtypes
    else: 
        
        @interact(column= df_dtypes.columns,direction={'ascending':True,'descending':False})
        def sort_df(column, direction):
            return df_dtypes.sort_values(by=column,axis=0,ascending=direction)        



def make_half_hour_range(twitter_df):
    import pandas as pd
    # Get timebin before the first timestamp that starts at 30m into the hour
    ofst_30m_early=pd.offsets.Minute(-30)
    start_idx = ofst_30m_early(twitter_df['date'].iloc[0].floor('H'))

    # Get timbin after last timestamp that starts 30m into the hour.
    ofst_30m_late =pd.offsets.Minute(30)
    end_idx= ofst_30m_late(twitter_df['date'].iloc[-1].ceil('H'))


    # Make time bins using the above start and end points 
    half_hour_range = pd.date_range(start =start_idx, end = end_idx, freq='30T')#.to_period()
    half_hour_intervals = pd.interval_range(start=start_idx, end=end_idx,freq='30T',name='half_hour_bins',closed='left')
    
    return half_hour_intervals



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
def bin_df_by_date_intervals(test_df,half_hour_intervals,column='date'):
    """"""
    # Cut The Date column into interval bins, 
    cut_date = pd.cut(test_df[column], bins=half_hour_intervals)#,labels=list(range(len(half_hour_intervals))), retbins=True)
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
def collapse_df_by_group_indices(twitter_df,group_indices, new_col_order=None):
    """Loops through the group_indices provided to concatenate each group into
    a single row and combine into one dataframe with the ______ as the index"""


    # Create a Panel to temporarily hold the group series and dataframes
    # group_dict_to_df = {}
    # create a dataframe with same columns as twitter_df, and index=group ids from twitter_groups
    group_df_index = [x[0] for x in group_indices]
    
    
    twitter_grouped = pd.DataFrame(columns=twitter_df.columns, index=group_df_index)
    twitter_grouped['TweetFreq'] =0

    for (idx,group_members) in group_indices:

        group_df = twitter_df.loc[group_members]

        combined_series = concatenate_group_data(group_df)

#         twitter_grouped.loc[idx,:] = combined_series
        twitter_grouped.loc[idx] = combined_series#.values

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

    try: twitter_df
    except NameError: twitter_df = None
    if twitter_df is not None:
        print('twitter_df already exists.')
        if overwrite==True:
            print('Overwrite=True. deleting original...')
            del(twitter_df)
            
    if twitter_df is None:
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
def get_day_window_size_from_freq(dataset):#, freq='CBH'):
    
    if dataset.index.freq == custom_BH_freq():
        return 7
    
    if dataset.index.freq=='T':
        day_window_size = 60*24
    elif dataset.index.freq=='BH':
        day_window_size = 8
#     elif dataset.index.freq=='CBH':
#         day_window_size = 7
    elif dataset.index.freq=='B':
        day_window_size=1
    elif dataset.index.freq=='D':
        day_window_size=1
        
    else:
        raise Exception('dataset freq=None')
        
    return day_window_size
    

def custom_BH_freq():
    import pandas as pd
    CBH = pd.tseries.offsets.CustomBusinessHour(start='09:30',end='16:30')
    return CBH
    
    
def  set_timeindex_freq(ive_df, col_to_fill=None, freq='CBH',fill_method='ffill',
                        verbose=3): #set_tz=True,
    
    import pandas as pd
    import numpy as np
    
    
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
                               start_index = '2017-01-23',
                                 clean=False,fill_or_drop_null='drop',fill_method='ffill',
                                 freq='CBH',verbose=2):
    import pandas as pd
    
    # Load in the text file and set headers
    fullfilename= folderpath+filename
    headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
    stock_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)
    
    # Create datetime index
    date_time_index = stock_df['Date']+' '+stock_df['Time']
    date_time_index = pd.to_datetime(date_time_index)
    stock_df.index=date_time_index
    
    # Select only the days after start_index
    stock_df = stock_df[start_index:]
    
    # Remove 0's from BidClose
    if clean==True:
        
        stock_df.loc[stock_df['BidClose']==0] = np.nan
        num_null = stock_df['BidClose'].isna().sum()
        print(f'{num_null} null values to address.')
        
        if fill_or_drop_null=='drop':
            print("Since fill_or_drop_null=drop, dropping null values.")
            stock_df.dropna(subset=['BidClose'],axis=0, inplace=True)
        elif fill_or_drop_null=='fill':
            print(f"Since fill_or_drop_null=fill, using fill_method={fill_method} to fill.")

            stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
        if verbose>0:
            print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
            print(f"Filling 0 values using method = {fill_method}")
            


                  
    # call set_timeindex_freq to specify proper frequency
    if freq!=None:
        # Set the time index .
        stock_df = set_timeindex_freq(stock_df, freq=freq, fill_method = fill_method, verbose=verbose)
                  
    # Display feedback
    if verbose>0:
        display(stock_df.head())
    if verbose>1:
        print(stock_df.index)

    return stock_df





def load_stock_df_from_csv(filename='ive_sp500_min_data_match_twitter_ts.csv',
                           folderpath='/content/drive/My Drive/Colab Notebooks/Mod 5 Project/data/',
                          clean=True,freq='T',fill_method='ffill',verbose=2):
    import os
    import pandas as pd

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


######## SEASONAL DECOMPOSITION    
def plot_decomposition(TS, decomposition, figsize=(12,8),window_used=None):
    """ Plot the original data and output decomposed components"""
    
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

        from ipywidgets import interact, interactive, interaction, Label, Box, Layout
        import ipywidgets as iw
        from datetime import datetime

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
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST
    day_freq = periods_per_day
    start_train_day = stock_df.index[-1] - num_train_days*day_freq
    last_day = stock_df.index[-1] - num_test_days*day_freq
    
    train_data = stock_df.loc[start_train_day:last_day]#,'price']
    test_data = stock_df.loc[last_day:]#,'price']
    
    if verbose>0:
        print(f'Data split on index:\t{last_day}.')

    if verbose>1:
        print(f'Train Data Date Range:\t{start_train_day} \t {last_day}.')
        print(f'Test Data Date Range:\t{last_day} \t {stock_df.index[-1]}.')
        print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if 'price' in stock_df.columns:
            plot_col ='price'
        elif 'price_labels' in stock_df.columns:
            plot_col = 'price_labels'
        
        train_data[plot_col].plot(label='Training')
        test_data[plot_col].plot(label='Test')
        plt.title('Training and Test Data for S&P500')
        plt.ylabel('Price')
        plt.xlabel('Trading Date/Hour')
        plt.legend();
    
    return train_data, test_data

# train_data, test_data = train_test_split_by_last_days(stock_df)



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

    
    
def plot_true_vs_preds_subplots(train_price, test_price, pred_price, subplots=False, figsize=(14,5)):
    
    from sklearn.metrics import mean_squared_error

    # Check for null values
    train_null = train_price.isna().sum()
    test_null = test_price.isna().sum()
    pred_null = pred_price.isna().sum()
    null_test = train_null + test_null+pred_null
    if null_test>0:
        print(f'Dropping {null_test} null values.')
        train_price.dropna(inplace=True)
        test_price.dropna(inplace=True)
        pred_price.dropna(inplace=True)

    ## CREATE FIGURE AND AX(ES)
    if subplots==True:
        fig = plt.figure(figsize=figsize)#, constrained_layout=True)
        ax1 = plt.subplot2grid((2, 9), (0, 0), rowspan=2, colspan=5)
        ax2 = plt.subplot2grid((2, 9),(0,5), rowspan=2, colspan=4)
#         plt.tight_layout()
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

        
    ## Define plot styles by train/test/pred data type
    style_dict = {'train':{},'test':{},'pred':{}}
    style_dict['train']={'lw':2,'color':'blue','ls':'-', 'alpha':1}
    style_dict['test']={'lw':1,'color':'orange','ls':'-', 'alpha':1}
    style_dict['pred']={'lw':2,'color':'green','ls':'-', 'alpha':0.7}
    
    
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

    
    
    # ANNOTATING RMSE
    RMSE = np.sqrt(mean_squared_error(test_price,pred_price))
    bbox_props = dict(boxstyle="square,pad=0.5", fc="white", ec="k", lw=0.5)
    
    plt.annotate(f"RMSE: {RMSE.round(3)}",xycoords='figure fraction', xy=(0.085,0.85),bbox=bbox_props)
    plt.tight_layout()
    return fig, ax

# fig, ax = plot_price_vs_preds(df_train_price['train_price'],df_test_price['test_price'],df_test_price['pred_price'])

def print_array_info(X, name='Array'):
    """Test function for verifying shapes and data ranges of input arrays"""
    Xt=X
    print('X type:',type(Xt))
    print(f'X.shape = {Xt.shape}')
    print(f'\nX[0].shape = {Xt[0].shape}')
    print(f'X[0] contains:\n\t',Xt[0])

    
def get_true_vs_model_pred_df(model, test_generator, test_index, train_generator, train_index, scaler=None,
                              inverse_tf=True, plot=True, verbose=2):
    """Accepts a model, the training and testing data TimeseriesGenerators, the test_index and train_index.
    Returns a dataframe with True and Predicted Values for Both the Training and Test Datasets."""
    
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


def thiels_U(ys_true, ys_pred):
    sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U        


def get_u_for_shifts(shift_list,plot_all=False,plot_best=True):
    results=[['true_data_shift','U']]
    
    if plot_all==True:
        df_U['preds_from_gen'].plot(label = 'Prediction')
        plt.legend()
        plt.title('Shifted Time Series vs Predicted')
        
        
    for i,shift in enumerate(shift_list):
        if plot_all==True:
            df_U['true_from_gen'].shift(shift).plot(label = f'True + Shift({shift})')

        df_shift=pd.DataFrame()
        df_shift['true'] = df_U['true_from_gen'].shift(shift)
        df_shift['pred'] =df_U['preds_from_gen']
        df_shift.dropna(inplace=True)

        U =thiels_U(df_shift['true'], df_shift['pred'])
        results.append([shift,U])
    
    
    df_results = bs.list2df(results, index_col='true_data_shift')
    if plot_best==True:
        shift = df_results.idxmin()[0]
        df_U['preds_from_gen'].plot(label = 'Prediction')
        df_U['true_from_gen'].shift(shift).plot(ls='--',label = f'True + Shift({shift})')
        plt.legend()
        plt.title("Best Thiel's U for Shifted Time Series")
#         plt.show()
    return df_results