
from .functions_combined import *
## SAVE AND LOAD STOCK MARKET MINUTE DATA
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

##################### FILE & VAR MANAGEMENT FUNCTIONS #####################

# def mount_google_drive(force_remount=True):
#     from google.colab import drive
#     print('drive_filepath="drive/My Drive/"')
#     return drive.mount('/content/drive', force_remount=force_remount)


# def check_for_google_drive(mount_if_not=True):
#     import os
#     # Check if google drive is already mounted
#     x=[]
#     try: os.listdir('drive/My Drive/')
#     except FileNotFoundError: x=None

#     # if GDrive not mounted, mount it
#     if x==None:
#         print('Drive not mounted.')
#         if mount_if_not==False:
#             return False
#         else:
#             print('Mounting google drive...')
#             return mount_google_drive()
#     else:
#         return True

# def cd_project_folder(folder_path ='/content/drive/My Drive/Colab Notebooks/Mod 5 Project/data/'):
#     import os
#     return os.chdir(folder_path),print('Cur Dir:', os.getcwd())
#     # print('Dir Contents:\n',os.listdir())
    
    
# #################### GENERAL HELPER FUNCTIONS #####################
# def is_var(name):
#     x=[]
#     try: eval(name)
#     except NameError: x = None
        
#     if x is None:
#         return False
#     else:
#         return True      
    
# #################### TIMEINDEX FUNCTIONS #####################
# def get_day_window_size_from_freq(dataset):
#     if dataset.index.freq=='T':
#         day_window_size = 1440
#     elif dataset.index.freq=='BH':
#         day_window_size = 8
#     elif dataset.index.freq=='BD':
#         day_window_size=1
#     elif dataset.index.freq=='D':
#         day_window_size=1
        
#     else:
#         raise Exception('dataset freq=None')
        
#     return day_window_size

    
# def  set_timeindex_freq(ive_df, col_to_fill=None, freq='T',fill_method='ffill',
#                         verbose=3): #set_tz=True,
    
#     import pandas as pd
#     import numpy as np
    
    
#     if verbose>1:
#         # print(f"{'Index When:':>{10}}\t{'Freq:':>{20}}\t{'Index Start:':>{40}}\t{'Index End:':>{40}}")
#         print(f"{'Index When:'}\t{'Freq:'}\t{'Index Start'}\t\t{'Index End:'}")
#         print(f"Pre-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")
        
#     # Change frequency to frewq
#     ive_df = ive_df.asfreq(freq)#'min')
    
#     #     # Set timezone
#     #     if set_tz==True:
#     #         ive_df.tz_localize()
#     #         ive_df.index = ive_df.index.tz_convert('America/New_York')
    
#     # Report Success / Details
#     if verbose>1:
#         print(f"Post-Change\t{ive_df.index.freq}\t{ive_df.index[0]}\t{ive_df.index[-1]}")


#     ## FILL AND TRACK TIMEPOINTS WITH MISSING DATA    
    
#     # Helper Function for adding column to track the datapoints that were filled
#     def check_null_times(x):
#         import numpy as np
#         if np.isnan(x):
#             return True
#         else:
#             return False

#     ## CREATE A COLUMN TO TRACK ROWS TO BE FILLED
#     # If col_to_fill provided, use that column to create/judge ive_df['filled_timebin'] 
#     if col_to_fill!=None:
#         ive_df['filled_timebin'] = ive_df[col_to_fill].apply(lambda x: check_null_times(x))#True if ive_df.isna().any()
        
#     # if not provided, use all columns and sum results
#     elif col_to_fill == None:
#         # Prefill fol with 0's
#         ive_df['filled_timebin']=0
        
#         # loop through all columns and add results of check_null_times from each loop
#         for col in ive_df.columns:
            
#             #ive_df['filled_timebin'] = ive_df[target_col].apply(lambda x: check_null_times(x))#True if ive_df.isna().any()
#             curr_filled_timebin_col = ive_df[col].apply(lambda x: check_null_times(x))#True if ive_df.isna().any() 
            
#             # add results
#             ive_df['filled_timebin'] +=  curr_filled_timebin_col

            
#     ## FILL IN NULL VALUES
#     ive_df.fillna(method=fill_method, inplace=True)

#     # Report # filled
#     if verbose>0:
#         check_fill = ive_df.loc[ive_df['filled_timebin']>0]
#         print(f'\nFilled {len(check_fill==True)}# of rows using method {fill_method}')
    
#     # Report any remaning null values
#     if verbose>0:
#         res = ive_df.isna().sum()
#         if res.any():
#             print(f'Cols with Nulls:')
#             print(res[res>0])
#         else:
#             print('No Remaining Null Values')   
            
#     # display header
#     if verbose>2:
#         display(ive_df.head())
    
#     return ive_df


############### TIMESERIES TESTS AND VISUALS ###############

# def plot_time_series(stocks_df, freq=None, fill_method='ffill',figsize=(12,4)):
    
#     df = stocks_df.copy()
#     df.fillna(method=fill_method, inplace=True)
#     df.dropna(inplace=True)
    
#     if (df.index.freq==None) & (freq == None):
#         xlabels=f'Time'
    
#     elif (df.index.freq==None) & (freq != None):
#         df = df.asfreq(freq)
#         df.fillna(method=fill_method, inplace=True)
#         df.dropna(inplace=True)
#         xlabels=f'Time - Frequency = {freq}'

#     else:
#         xlabels=f'Time - Frequency = {df.index.freq}'
        
#     ylabels="Price"

#     raw_plot = df.plot(figsize=figsize)
#     raw_plot.set_title('Stock Bid Closing Price ')
#     raw_plot.set_ylabel(ylabels)
#     raw_plot.set_xlabel(xlabels)
    
    
# def stationarity_check(df, col='BidClose', window=80, freq='BH'):
#     """From learn.co lesson: use ADFuller Test for Stationary and Plot"""
#     import matplotlib.pyplot as plt
#     TS = df[col].copy()
#     TS = TS.asfreq(freq)
#     TS.fillna(method='ffill',inplace=True)
#     TS.dropna(inplace=True)
#     # Import adfuller
#     from statsmodels.tsa.stattools import adfuller
#     import pandas as pd
#     import numpy as np
    
#     # Calculate rolling statistics
#     rolmean = TS.rolling(window = window, center = False).mean()
#     rolstd = TS.rolling(window = window, center = False).std()
    
#     # Perform the Dickey Fuller Test
#     dftest = adfuller(TS) # change the passengers column as required 
    
#     #Plot rolling statistics:
#     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
#     ax[0].set_title('Rolling Mean & Standard Deviation')

#     ax[0].plot(TS, color='blue',label='Original')
#     ax[0].plot(rolmean, color='red', label='Rolling Mean',alpha =0.6)
#     ax[1].plot(rolstd, color='black', label = 'Rolling Std')
#     ax[0].legend()
#     ax[1].legend()
# #     plt.show(block=False)
#     plt.tight_layout()
    
#     # Print Dickey-Fuller test results
#     print ('Results of Dickey-Fuller Test:')
#     print('\tIf p<.05 then timeseries IS stationary.')

#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
#     print (dfoutput)
    
#     return None

# ##################### DATASET LOADING FUNCTIONS #####################   
# def load_raw_stock_data_from_txt(filename='IVE_bidask1min.txt', 
#                                folderpath='/content/drive/My Drive/Colab Notebooks/Mod 5 Project/data/',
#                                verbose=2):
    
#     check_for_google_drive()
    
#     import pandas as pd
    
#     headers = ['Date','Time','BidOpen','BidHigh','BidLow','BidClose','AskOpen','AskHigh','AskLow','AskClose']
    
#     fullfilename= folderpath+filename
#     ive_df = pd.read_csv(fullfilename, names=headers,parse_dates=True)
    
#     # Create datetime index
#     date_time_index = ive_df['Date']+' '+ive_df['Time']
    
#     #ive_df.index=date_time_index
#     #ive_df.index = pd.to_datetime(ive_df.index)
#     date_time_index = pd.to_datetime(date_time_index)
#     ive_df.index=date_time_index
    
#     if verbose>0:
#         display(ive_df.head())
#     if verbose>1:
#         print(ive_df.index)
        
#     return ive_df



# def load_stock_df_from_csv(filename='ive_sp500_min_data_match_twitter_ts.csv',
#                            folderpath='/content/drive/My Drive/Colab Notebooks/Mod 5 Project/data/',
#                           clean=True,freq='T',fill_method='ffill',verbose=2):
#     import os
#     import pandas as pd

#     #         check_for_google_drive()
        
#     # Check if user provided folderpath to append to filename
#     if len(folderpath)>0:
#         fullfilename = folderpath+filename
#     else:
#         fullfilename=filename
        
#     # load in csv by fullfilename
#     stock_df = pd.read_csv(fullfilename,index_col=0, parse_dates=True)
# #     stock_df = set_timeindex_freq(stock_df,['BidClose'],freq=freq, fill_method=fill_method)
    
#     if clean==True:
        
#         if verbose>0:
#             print(f"Number of 0 values:\n{len(stock_df.loc[stock_df['BidClose']==0])}")
#             print(f"Filling 0 values using method = {fill_method}")
            
#         stock_df.loc[stock_df['BidClose']==0] = np.nan
#         stock_df['BidClose'].fillna(method=fill_method, inplace=True)
        
    
#     # Set the time index 
#     stock_df = set_timeindex_freq(stock_df,'BidClose',freq=freq, fill_method = fill_method, verbose=verbose)
        

#     # Display info depending on verbose level
#     if verbose>0:
#         display(stock_df.head())
    
#     if verbose>1:
#         print(stock_df.index)
        
#     return stock_df   

# ######## SEASONAL DECOMPOSITION    
# def plot_decomposition(TS, decomposition, figsize=(12,8),window_used=None):
#     """ Plot the original data and output decomposed components"""
    
#     # Gather the trend, seasonality and noise of decomposed object
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid

#     fontdict_axlabels = {'fontsize':12}#,'fontweight':'bold'}
    
#     # Plot gathered statistics
#     fig, ax = plt.subplots(nrows=4, ncols=1,figsize=figsize)
    
#     ylabel = 'Original'
#     ax[0].plot(np.log(TS), color="blue")
#     ax[0].set_ylabel(ylabel, fontdict=fontdict_axlabels)
    
#     ylabel = label='Trend'
#     ax[1].plot(trend, color="blue")
#     ax[1].set_ylabel(ylabel, fontdict=fontdict_axlabels)
    
#     ylabel='Seasonality'
#     ax[2].plot(seasonal, color="blue")
#     ax[2].set_ylabel(ylabel, fontdict=fontdict_axlabels)
    
#     ylabel='Residuals'
#     ax[3].plot(residual, color="blue")
#     ax[3].set_ylabel(ylabel, fontdict=fontdict_axlabels)
#     ax[3].set_xlabel('Time', fontdict=fontdict_axlabels)
    
#     # Add title with window 
#     if window_used == None:
#         plt.suptitle('Seasonal Decomposition', y=1.02)
#     else:
#         plt.suptitle(f'Seasonal Decomposition - Window={window_used}', y=1.02)
    
#     # Adjust aesthetics
#     plt.tight_layout()
    
#     return ax
    
    
# def seasonal_decompose_and_plot(ive_df,col='BidClose',freq='H',
#                           fill_method='ffill',window=144,
#                          model='multiplicative', two_sided=False,
#                                plot_components=True):##WIP:
#     """Perform seasonal_decompose from statsmodels.tsa.seasonal.
#     Plot Output Decomposed Components"""
#     import pandas as pd
#     import numpy as np
#     from statsmodels.tsa.seasonal import seasonal_decompose


#     # TS = ive_df['BidClose'].asfreq('BH')
#     TS = pd.DataFrame(ive_df[col])
#     TS = TS.asfreq(freq)
#     TS[TS==0]=np.nan
#     TS.fillna(method='ffill',inplace=True)

#     # Perform decomposition
#     decomposition = seasonal_decompose(np.log(TS),freq=window, model=model, two_sided=two_sided)
    
#     if plot_components==True:
#         ax = plot_decomposition(TS, decomposition, window_used=window)
    
#     return decomposition



### FOR TRANSFORMING TWITTER DATA
#***#a
# twitter_df = load_twitter_df()

# def make_half_hour_range(twitter_df):
    
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



# #***#
# def int_to_ts(int_list, as_datetime=False, as_str=True):
#     """Accepts one Panda's interval and returns the left and right ends as either strings or Timestamps."""
#     if as_datetime & as_str:
#         raise Exception('Only one of `as_datetime`, or `as_str` can be True.')
    
#     left_edges =[]
#     right_edges= []
    
#     for interval in int_list:
#         int_str = interval.__str__()[1:-1]
#         left,right = int_str.split(',')
#         left_edges.append(left)
#         right_edges.append(right)
        
    
#     if as_str:
#         return left_edges, right_edges
    
#     elif as_datetime:
#         left = pd.to_datetime(left)
#         right = pd.to_datetime(right)
#         return left,right
    
    
# # Step 1:     
# def bin_df_by_date_intervals(test_df,half_hour_intervals,column='date'):
#     """"""
#     # Cut The Date column into interval bins, 
#     cut_date = pd.cut(test_df[column], bins=half_hour_intervals)#,labels=list(range(len(half_hour_intervals))), retbins=True)
#     test_df['int_times'] = cut_date    
    
#     # convert to str to be used as group names/codes
#     unique_bins = cut_date.astype('str').unique()
#     num_code = list(range(len(unique_bins)))
    
#     # Dictioanry of number codes to be used for interval groups
#     bin_codes = dict(zip(num_code,unique_bins))#.astype('str')

    
#     # Mapper dictionary to convert intervals into number codes
#     bin_codes_mapper = {v:k for k,v in bin_codes.items()}

    
#     # Add column to the dataframe, then map integer code onto it
#     test_df['int_bins'] = test_df['int_times'].astype('str').map(bin_codes_mapper)
    
    
#     # Get the left edge of the bins to use later as index (after grouped)
#     left_out, _ =int_to_ts(test_df['int_times'])#.apply(lambda x: int_to_ts(x))    
#     test_df['left_edge'] = pd.to_datetime(left_out)

#     # bin codes to labels 
#     bin_codes = [(k,v) for k,v in bin_codes.items()]
    
#     return test_df, bin_codes


# def concatenate_group_data(group_df_or_series):
#     """Accepts a series or dataframe from a groupby.get_group() loop.
#     Adds TweetFreq column for # of rows concatenate. If input is series, 
#     TweetFreq=1 and series is returned."""
    
#     import pandas as pd
#     from pandas.api import types as tp
    
#     if isinstance(group_df_or_series, pd.Series):
        
#         group_data = group_df_or_series
        
# #         group_data.index = group_df_or_series.index
#         group_data['TweetFreq'] = 1

#         return group_data
    
#     # if the group is a dataframe:
#     elif isinstance(group_df_or_series, pd.DataFrame):
        
#         df = group_df_or_series
        
#         # create an output series to collect combined data
#         group_data = pd.Series(index=df.columns)
#         group_data['TweetFreq'] = df.shape[0]
        

#         for col in df.columns:
            
#             combined=[]
#             col_data = []
            
#             col_data = df[col]
#             combined=col_data.values
            
#             group_data[col] = combined

#     return group_data


# #***#
# def collapse_df_by_group_indices(twitter_df,group_indices, new_col_order=None):
#     """Loops through the group_indices provided to concatenate each group into
#     a single row and combine into one dataframe with the ______ as the index"""


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
