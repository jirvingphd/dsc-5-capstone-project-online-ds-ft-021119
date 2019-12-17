### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
### SOURCE:
```python
def train_test_split_by_last_days(stock_df, periods_per_day=7,num_test_days = 90, num_train_days=180,verbose=1, plot=False,iplot=True,plot_col='price'):
    """Takes the last num_test_days of the time index to use as testing data, and take shte num_Trian_days prior to that date
    as the training data."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd

    if verbose>1:
        print(f'Data index (freq={stock_df.index.freq}')
        print(f'index[0] = {stock_df.index[0]}, index[-1]={stock_df.index[-1]}')
    
    # DETERMINING DAY TO USE TO SPLIT DATA INTO TRAIN AND TEST

    # first get integer indices for the days to select
    idx_start_train = -((num_train_days+num_test_days )*periods_per_day) 
    idx_stop_train = idx_start_train +(num_train_days*periods_per_day)

    idx_start_test = idx_stop_train-1 
    idx_stop_test = idx_start_test+(num_test_days*periods_per_day)

    # select train dates and corresponding rows from stock_df
    train_days = stock_df.index[idx_start_train:idx_stop_train]
    train_data = stock_df.loc[train_days]

    # select test_dates and corresponding rows from stock_df
    if idx_stop_test == 0:
        test_days = stock_df.index[idx_start_test:]
    else:
        test_days = stock_df.index[idx_start_test:idx_stop_test]
    test_data = stock_df.loc[test_days]

    
    if verbose>0:
        # print(f'Data split on index:\t{last_train_day}:')
        print(f'training dates:\t{train_data.index[0]} \t {train_data.index[-1]} = {len(train_data)} rows')
        print(f'test dates:\t{test_data.index[0]} \t {test_data.index[-1]} = {len(test_data)} rows')
        # print(f'\ttrain_data.shape:\t{train_data.shape}, test_data.shape:{test_data.shape}')
        
    if verbose>1:
        display(train_data.head(3).style.set_caption('Training Data'))
        display(test_data.head(3).style.set_caption('Test Data'))

                
    if plot==True:
        if plot_col in stock_df.columns:
            # plot_col ='price'
        # elif 'price_labels' in stock_df.columns:
        #     plot_col = 'price_labels'
            
            fig = plt.figure(figsize=(8,4))
            train_data[plot_col].plot(label='Training')
            test_data[plot_col].plot(label='Test')
            plt.title('Training and Test Data for S&P500')
            plt.ylabel('Price')
            plt.xlabel('Trading Date/Hour')
            plt.legend()
            plt.show()
        else:
            raise Exception('plot_col not found')

    if iplot==True:
        df_plot=pd.concat([train_data[plot_col].rename('train price'),test_data[plot_col].rename('test_price')],axis=1)
        from plotly.offline import iplot
        fig = plotly_time_series(df_plot,show_fig=False)
        iplot(fig)
        # display()#, as_figure=True)

    return train_data, test_data

```
### SOURCE:
```python
def make_scaler_library(df,transform=True,columns=None,use_model_params=False,model_params=None,verbose=1):
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
 
    if columns is None:
        columns = df.columns
         
    # select only compatible columns
    columns_filtered = df[columns].select_dtypes(exclude=['datetime','object','bool']).columns
 
    if len(columns) != len(columns_filtered):
        removed = [x for x in columns if x not in columns_filtered]
        if verbose>0:
            print(f'[!] These columns were excluded due to incompatible dtypes.\n{removed}\n')
 
    # Loop throught columns and fit a scaler to each
    scaler_dict = {}
    # scaler_dict['index'] = df.index
 
    for col in columns_filtered:
        scaler = MinMaxScaler()
        scaler.fit(df[col].values.reshape(-1,1))
        scaler_dict[col] = scaler 
 
    # Add the scaler dictionary to return_list
    return_list = [scaler_dict]
 
    # # transform and return outputs
    if transform == True:
        df_out = transform_cols_from_library(df, col_list=columns_filtered, scaler_library=scaler_dict)
        return_list.append(df_out)
 
    # add scaler to model_params
    if use_model_params:
        if model_params is not None:
            model_params['scaler_library']=scaler_dict
            return_list.append(model_params)
    
    return return_list[:]

```
### SOURCE:
```python
def transform_cols_from_library(df,col_list=None,col_scaler_dict=None, scaler_library=None,single_scaler=None, inverse=False, verbose=1):
    """Accepts a df and:
    1. col_list: a list of column names to transform (that are also the keys in scaler_library)
        --OR-- 
        col_scaler_dict: a dictionary with keys:
         column names to transform and values: scaler_library key for that column.

    2. A fit scaler_library (from make_scaler_library) with names matching col_list or the values in col_scaler_dict
    --OR-- 
    a fit single_scaler (whose name does  not matter) and will be applied to all columns in col_list

    3. inverse: if True, columns will be inverse_transformed.

    Returns a dataframe with all columns of original df.
    Can pyob"""
    import functions_combined_BEST as ji
    # replace_df_column = ji.inverse_transform_series

    # def test_if_fit(scaler):
    #     """Checks if scaler is fit by looking for scaler.data_range_"""
    #     # check if scaler is fit
    #     if hasattr(scaler, 'data_range_'): 
    #         return True 
    #     else:
    #         print('scaler not fit...')
    #         return False


    # MAKE SURE SCALER PROVIDED
    if (scaler_library is None) and (single_scaler is None):
        raise Exception('Must provide a scaler_library with or single_scaler')

    # MAKE COL_LIST FROM DICT OR COLUMNS
    if (col_list is None) and (col_scaler_dict is None):
        print('[i] Using all columns...')
        col_list = df.columns

    if col_scaler_dict is not None:
        col_list = [k for k in col_scaler_dict.keys()]
        # print('Using col_scaler_')


    ## copy df to replace columns
    df_out = df.copy()
    
    # Filter out incompatible data types
    for col in col_list:
        # if single_scaler, use it in tuple with col
        if single_scaler is not None:
            scaler = single_scaler

        elif scaler_library is not None:
            # get the column from scaler_librray
            if col in scaler_library.keys():
                scaler = scaler_library[col]

            else:
                print(f'[!] Key: scaler_library["{col}"] does not exist. Skipping column...')
                continue
        # send series to transform_series to be checked and transform
        series = df[col]
        df_out[col] = transform_series(series=series, scaler=scaler,check_fit=True, check_results=False, inverse = inverse)

    return df_out

```
### SOURCE:
```python
def make_train_test_series_gens(train_data_series,test_data_series,x_window,X_cols = None, y_cols='price',n_features=1,batch_size=1, model_params=None,debug=False,verbose=1):
    
    import functions_combined_BEST as ji

    if model_params is not None:
        if 'data_params' in model_params.keys():
            data_params = model_params['data_params']
            model_params['input_params'] = {}
        else:
            print('data_params not found in model_params')
    ########## Define shape of data by specifing these vars 
    if model_params is not None:
        input_params = {}        
        input_params['n_input'] = data_params['x_window']  # Number of timebins to analyze at once. Note: try to choose # greater than length of seasonal cycles 
        input_params['n_features'] = n_features # Number of columns
        input_params['batch_size'] = batch_size # Generally 1 for sequence data
        n_input =  data_params['x_window']
    else:
        # Get params from data_params_dict
        n_input = x_window


    import functions_combined_BEST as ji
    from keras.preprocessing.sequence import TimeseriesGenerator

    # RESHAPING TRAINING AND TRAINING DATA 
    train_data_index =  train_data_series.index

    if model_params is not None:
        input_params['train_data_index'] = train_data_index

    def reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols,debug=debug):
        # if only 1 column (aka is a series)
        train_data = []
        train_targets = []
        import pandas as pd
        import numpy as np

        if isinstance(train_data_series, pd.DataFrame):
            # train_data_test = train_data_series.values
            train_data = train_data_series.values
                # if not specified, use all columns for x data
            if X_cols is None:
                train_data = train_data_series.values
            
            else:
                train_data = train_data_series[X_cols].values


            # if not specified, assume 'price' is taret_col
            if y_cols is None:
                y_cols = 'price'
            
            train_targets = train_data_series[y_cols].values
            
            # print(train_targets.ndim)
        elif isinstance(train_data_series, pd.Series): #        if train_data_test.ndim<2: #shape[1] < 2:

            train_data = train_data_series.values.reshape(-1,1)
            train_targets = train_data

            # return train_data, train_targets

        # else: # if train_data_series really a df

        if train_targets.ndim <2: #ndim<2:
            train_targets = train_targets.reshape(-1,1)

        if debug==True:
            print('train_data[0]=\n',train_data[0])
            print('train_targets[0]=\n',train_targets[0])
        return train_data, train_targets


    ## CREATE TIMESERIES GENERATOR FOR TRAINING DATA
    train_data, train_targets = reshape_train_data_and_target(train_data_series, X_cols=X_cols, y_cols=y_cols)

    train_generator = TimeseriesGenerator(data=train_data, targets=train_targets,
                                          length=n_input, batch_size=batch_size )



    # RESHAPING TRAINING AND TEST DATA 
    # test_data = test_data_series.values.reshape(-1,1)
    test_data_index =test_data_series.index
    if model_params is not None:
        input_params['test_data_index'] = test_data_index


    ## CREATE TIMESERIES GENERATOR FOR TEST DATA
    test_data, test_targets = reshape_train_data_and_target(test_data_series, X_cols=X_cols, y_cols=y_cols)

    test_generator = TimeseriesGenerator(data=test_data, targets=test_targets,
                                         length=n_input, batch_size=batch_size )
    
    if model_params is not None:
        model_params['input_params'] = input_params

    if verbose>1:
        ji.display_dict_dropdown(model_params)

    if verbose>0:
        # What does the first batch look like?
        X,y = train_generator[0]
        print(f'Given the Array: \t(with shape={X.shape}) \n{X.flatten()}')
        print(f'\nPredict this y: \n {y}')

    if model_params is not None:
        return train_generator,test_generator, model_params
    else:
        return  train_generator, test_generator

```
