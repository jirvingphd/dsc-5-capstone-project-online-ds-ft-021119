# print('my_keras_functions loaded')
def my_rmse(y_true,y_pred):
    """RMSE calculation using keras.backend"""
    from keras import backend as kb
    sq_err = kb.square(y_pred - y_true)
    mse = kb.mean(sq_err,axis=-1)
    rmse =kb.sqrt(mse)
    return rmse

def quiet_mode(filter_warnings=True, filter_keras=True,in_function=True,verbose=0):
    """Convenience function to execute commands to silence warnings:
    - filter_warnings:
        - warnings.filterwarnings('ignore')
    - filter_keras:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    """

    cmd_warnings = "import warnings\nwarnings.filterwarnings('ignore')"
    cmd_keras = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    cmd_combined = '\n'.join((cmd_warnings,cmd_keras))

    if filter_warnings and filter_keras:
        if verbose>0: 
            print(cmd_combined)
        output = cmd_combined

    elif filter_warnings and filter_keras is False:
        if verbose>0: 
            print(cmd_warnings)
        output = cmd_warnings

    elif filter_warnings is False and filter_keras:
        if verbose>0: 
            print(cmd_keras)
        output = cmd_keras
    
    if in_function:
        # exec_string = output #scaled_test_data#"exec('"+output+"')"
        return output

    else:
        return exec(output)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# my_keras_functions
def def_data_params(stock_df, num_test_days=45, num_train_days=365,days_for_x_window=5,verbose=0):
    """
    data_params['num_test_days']     =  45 # Number of days for test data
    data_params['num_train_days']    = 365 # Number of days for training data - 5 days/week * 52 weeks
    data_params['days_for_x_window'] =   5 # Number of days to include as 1 X sequence for predictions
    data_params['periods_per_day'] = ji.get_day_window_size_from_freq( stock_df, ji.custom_BH_freq() )
    """
    import functions_combined_BEST as ji
    
    data_params={}
    data_params['num_test_days']     =  num_test_days # Number of days for test data
    data_params['num_train_days']    = num_train_days # Number of days for training data - 5 days/week * 52 weeks
    data_params['days_for_x_window'] =   days_for_x_window # Number of days to include as 1 X sequence for predictions
    
    # Calculate number of rows to bin for x_windows
    periods_per_day = ji.get_day_window_size_from_freq( stock_df, ji.custom_BH_freq() ) # get the # of rows that == 1 day
    x_window = periods_per_day * days_for_x_window#data_params['days_for_x_window'] 
    
    # Update data_params
    data_params['periods_per_day'] = periods_per_day
    data_params['x_window'] = x_window    
    # days_for_x_window = #data_params['days_for_x_window']


    if verbose>1:
        print(f'X_window size = {x_window} -- ({days_for_x_window} day(s) * {periods_per_day} rows/day)\n')
    
    if verbose>0:
#         ji.display_dict_dropdown(data_params)
        from pprint import pprint
        print("data_params.items()={\t")
        pprint(data_params)
        print('}\n')
    return data_params


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



def reshape_train_data_and_target(train_data_series, X_cols=None, y_cols='price',
                                  n_features=1,debug=False):
    """Reshapes the X_col and y_col data into proper shape for timeseris generator"""
    train_data = []
    train_targets = []
    import pandas as pd
    import numpy as np

    if isinstance(train_data_series, pd.DataFrame):
        # train_data_test = train_data_series.values
        train_data = train_data_series.values
        
        # Train X_data to array (if not specified, use all columns)
        if X_cols is None:
            train_data = train_data_series.values
        else:
            train_data = train_data_series[X_cols].values

        # if not specified, assume 'price' is taret_col
        if y_cols is None:
            y_cols = 'price'
        ## Train Target y_values
        train_targets = train_data_series[y_cols].values

    elif isinstance(train_data_series, pd.Series): #        
        train_data = train_data_series.values.reshape(-1,1)
        train_targets = train_data

    
    ## Reshape as neded
    if train_targets.ndim <2: 
        train_targets = train_targets.reshape(-1,1)

    if debug==True:
        print('train_data[0]=\n',train_data[0])
        print('train_targets[0]=\n',train_targets[0])
        
    return train_data, train_targets





def def_callbacks_and_params(model_params=None,loss_function='my_rmse',checkpoint_mode='min',filepath=None,
                             stop_mode='min',patience=1,min_delta=.001,verbose=1):
    import functions_combined_BEST as ji
    if 'my_rmse' in loss_function:
        def my_rmse(y_true,y_pred):
            """RMSE calculation using keras.backend"""
            from keras import backend as kb
            sq_err = kb.square(y_pred - y_true)
            mse = kb.mean(sq_err,axis=-1)
            rmse =kb.sqrt(mse)
            return rmse
        my_rmse=my_rmse
        loss_function = my_rmse

        
    ########## Define loss function and callback params ##########
    callback_params ={}
    callback_params['custom_loss_function'] = loss_function
    callback_params['custom_loss_function'] = loss_function
    callback_params['ModelCheckpoint'] = {'monitor': loss_function, 'mode':checkpoint_mode}
    callback_params['EarlyStopping'] = {'monitor':loss_function, 'mode':stop_mode, 
                                        'patience':patience, 'min_delta':min_delta}

    # CREATING CALLBACKS
    from keras import callbacks

    if filepath is None:
        filepath = f"models/checkpoints/model1_weights_{ji.auto_filename_time(prefix=None)}.hdf5"

    # Create ModelCheckPoint
    fun_params=callback_params['ModelCheckpoint']
    checkpoint = callbacks.ModelCheckpoint(filepath=filepath, monitor=fun_params['monitor'], mode=fun_params['mode'],
                                           save_best_only=False, verbose=verbose)
    # Create EarlyStopping
    fun_params=callback_params['EarlyStopping']
    early_stop = callbacks.EarlyStopping(monitor=my_rmse, mode=fun_params['mode'], patience=fun_params['patience'],
                                         min_delta=fun_params['min_delta'],verbose=verbose)
    callbacks = [checkpoint,early_stop]

    if model_params is None:
        model_params=callback_params
    else:
        model_params['callbacks'] = callback_params
    return callbacks, model_params


def def_compile_params_optimizer(loss='my_rmse',metrics=['acc','my_rmse'],optimizer='optimizers.Nadam()',model_params=None):
    ####### Specify additional model parameters
    from keras import optimizers
    
    if 'my_rmse' in loss or 'my_rmse' in metrics:
        def my_rmse(y_true,y_pred):
            """RMSE calculation using keras.backend"""
            from keras import backend as kb
            sq_err = kb.square(y_pred - y_true)
            mse = kb.mean(sq_err,axis=-1)
            rmse =kb.sqrt(mse)
            return rmse
        my_rmse=my_rmse
        
        
        # replace string with function in loss
        loss = my_rmse
        
        # replace string with function in metrics
        idx = metrics.index('my_rmse')
        metrics[idx]=my_rmse
        
    compile_params={}
    compile_params['loss']= loss#{'my_rmse':my_rmse}
    compile_params['metrics'] = metrics#['acc',my_rmse]

    if type(optimizer) is str:
        optimizer_name = optimizer
        optimizer = eval(optimizer_name)
    else:
        optimizer_name = optimizer.__class__().__str__()
        
    compile_params['optimizer'] = optimizer
    compile_params['optimizer_name'] = optimizer_name#'optimizers.Nadam()'

    if model_params is not None:
        model_params['compile_params'] = compile_params
    else:
        model_params=compile_params
    
    return model_params



def make_model1(model_params, summary=True):
    from keras.models import Sequential
    from keras.layers import Bidirectional, Dense, LSTM, Dropout
    from IPython.display import display
    from keras.regularizers import l2

    # Specifying input shape (size of samples, rank of samples?)
    n_input = model_params['input_params']['n_input']
    n_features = model_params['input_params']['n_features']
    
    input_shape=(n_input, n_features)
    
    # Create model architecture
    model = Sequential()
    model.add(LSTM(units=50, input_shape =input_shape,return_sequences=True))#,  kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(1))

    # load compile params and compile
    comp_params = model_params['compile_params']
    # metrics = comp_params['metrics']
    model.compile(loss=comp_params['loss'], metrics=comp_params['metrics'],
                  optimizer=comp_params['optimizer'])##eval(comp_params['optimizer']), metrics=metrics)#optimizer=optimizers.Nadam()
    
    if summary is True:
        display(model.summary())

    return model



def fit_model(model,train_generator,model_params=None,epochs=5,callbacks=None,verbose=2,workers=3):
    import bs_ds_local as bs
    import functions_combined_BEST as ji
    from IPython.display import display

    quiet_command = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
    exec(quiet_command)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # cmd_keras = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"

    
    if model_params is None:
        model_params={}
    model_params['fit_params'] = {'epochs':epochs,'callbacks':callbacks}

    # Instantiating clock timer
    clock = bs.Clock()

    print('---'*20)
    print('\tFITTING MODEL:')
    print('---'*20,'\n')     
    
    # start the timer
    clock.tic('')

    # Fit the model
    fit_params = model_params['fit_params']
    if callbacks is None:
        
        history = model.fit_generator(train_generator,epochs=fit_params['epochs'], 
                                       verbose=2, use_multiprocessing=True, workers=3)
    else:
        
        history = model.fit_generator(train_generator,epochs=fit_params['epochs'],
                                       callbacks=callbacks,
                                       verbose=2,use_multiprocessing=True, workers=3)

    # model_results = model.history.history
    clock.toc('')
    
    return model,model_params,history


def evaluate_model_plot_history(model, train_generator, test_generator,as_df=False, plot=True):
    """ Takes a keras model fit using fit_generator(), a train_generator and test generator.
    Extracts and plots Keras model.history's metrics."""
    from IPython.display import display
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import functions_combined_BEST as ji
    print('\n')
    print('---'*28)
    print('\tEVALUATE MODEL:')
    print('---'*28)
        # duration = print(clock._lap_duration_)
    model_results = model.history.history
    
    if plot==True and len(model.history.epoch)>1:

        # ji.plot_keras_history()
        fig, ax = plt.subplots(figsize=(6,3))

        for k,v in model_results.items():
            ax.plot(range(len(v)),v, label=k)
                
        plt.title('Model Training History')
        ax.set_xlabel('Epoch #',**{'size':12,'weight':70})
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        plt.legend()
        plt.show()


    # # EVALUATE MODEL PREDICTIONS FROM GENERATOR 
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)
    # print(model_metrics_test)

    eval_gen_dict = {}
    eval_gen_dict['Train Data'] = dict(zip(model.metrics_names,model_metrics_train))
    eval_gen_dict['Test Data'] = dict(zip(model.metrics_names,model_metrics_test))
    df_eval = pd.DataFrame(eval_gen_dict).round(4).T
    display(df_eval.style.set_caption('Model Evaluation Results'))

    if as_df:
        return df_eval
    else:
        return  eval_gen_dict


def get_model_config_df(model1, multi_index=True):

    import bs_ds_local as bs
    import functions_combined_BEST as ji
    import pandas as pd
    pd.set_option('display.max_rows',None)

    model_config_dict = model1.get_config()
    try:
        model_layer_list=model_config_dict['layers']
    except:
        return model_config_dict
        raise Exception()
    output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

    for num,layer_dict in enumerate(model_layer_list):
    #     layer_dict = model_layer_list[0]


        # layer_dict['config'].keys()
        # config_keys = list(layer_dict.keys())
        # combine class and name into 1 column
        layer_class = layer_dict['class_name']
        layer_name = layer_dict['config'].pop('name')
        col_000 = f"{num}: {layer_class}"
        col_00 = layer_name#f"{layer_class} ({layer_name})"

        # get layer's config dict
        layer_config = layer_dict['config']


        # config_keys = list(layer_config.keys())


        # for each parameter in layer_config
        for param_name,col2_v_or_dict in layer_config.items():
            # col_1 is the key( name of param)
        #     col_1 = param_name


            # check the contents of col2_:

            # if list, append col2_, fill blank cols
            if isinstance(col2_v_or_dict,dict)==False:
                col_0 = 'top-level'
                col_1 = param_name
                col_2 = col2_v_or_dict

                output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


            # else, set col_2 as the param name,
            if isinstance(col2_v_or_dict,dict):

                param_sub_type = col2_v_or_dict['class_name']
                col_0 = param_name +'  ('+param_sub_type+'):'

                # then loop through keys,vals of col_2's dict for col3,4
                param_dict = col2_v_or_dict['config']

                for sub_param,sub_param_val in param_dict.items():
                    col_1 =sub_param
                    col_2 = sub_param_val
                    # col_3 = ''


                    output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])
        
    df = bs.list2df(output)    
    if multi_index==True:
        df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
        df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        df.sort_index(level=0, inplace=True)
    return df





def save_model_weights_params(model,model_params=None, filename_prefix = 'models/model', filename_suffix='', check_if_exists = True,
 auto_increment_name=True, auto_filename_suffix=True, save_model_layer_config_xlsx=True, sep='_', suffix_time_format = '%m-%d-%Y_%I%M%p'):
    """Saves a fit Keras model and its weights as a .json file and a .h5 file, respectively.
    auto_filename_suffix will use the date and time to give the model a unique name (avoiding overwrites).
    Returns the model_filename and weight_filename"""
    import json
    import pickle
    from functions_combined_BEST import auto_filename_time
    import functions_combined_BEST as ji

    # create base model filename 
    if auto_filename_suffix:
        filename = auto_filename_time(prefix=filename_prefix, sep=sep,timeformat=suffix_time_format )
    else:
        filename=filename_prefix
    

    ## Add suffix to filename
    full_filename = filename + filename_suffix
    full_filename = full_filename+'.json'


    ## check if file exists
    if check_if_exists:
        import os
        import pandas as pd
        current_files = os.listdir()

        # check if file already exists
        if full_filename in current_files and auto_increment_name==False:
            raise Exception('Filename already exists')
        
        elif full_filename in current_files and auto_increment_name==True:
        
            # check if filename ends in version #
            import re
            num_ending = re.compile(r'[vV].?(\d+).json')
            
            curr_file_num = num_ending.findall(full_filename)
            if len(curr_file_num)==0:
                v_num = '_v01'
            else:
                v_num = f"_{int(curr_file_num)+1}"

            full_filename = filename + v_num + '.json'

            print(f'{filename} already exists... incrementing filename to {full_filename}.')
    
    ## SAVE MODEL AS JSON FILE
    # convert model to json
    model_json = model.to_json()

    ji.create_required_folders(full_filename)
    # save json model to json file
    with open(full_filename, "w") as json_file:
        json.dump(model_json,json_file)
    print(f'Model saved as {full_filename}')


    ## GET BASE FILENAME WITHOUT EXTENSION
    file_ext=full_filename.split('.')[-1]
    filename = full_filename.replace(f'.{file_ext}','')    

    ## SAVE MODEL WEIGHTS AS HDF5 FILE
    weight_filename = filename+'_weights.h5'
    model.save_weights(weight_filename)
    print(f'Weights saved as {weight_filename}') 


    ## SAVE MODEL LAYER CONFIG TO EXCEL FILE 
    if save_model_layer_config_xlsx == True:

        excel_filename=filename+'_model_layers.xlsx'
        df_model_config = get_model_config_df(model)

        try:
            # Get modelo config df
            df_model_config.to_excel(excel_filename, sheet_name='Keras Model Config')
            print(f"Model configuration table saved as {excel_filename }")
        except:
            print('ERROR:df_model_config = get_model_config_df(model)')
            print(type(df_model_config))
            # print(df_model_config)

            


    ## SAVE MODEL PARAMS TO PICKLE 
    if model_params is not None:
        # import json
        import inspect
        import pickle# as pickle        
        
        def replace_function(function):
            import inspect
            return inspect.getsource(function)
        
        ## Select good model params to save
        model_params_to_save = {}
        model_params_to_save['data_params'] = model_params['data_params']
        model_params_to_save['input_params'] = model_params['input_params']
        
        model_params_to_save['compile_params'] = {}
        model_params_to_save['compile_params']['loss'] = model_params['compile_params']['loss']

        ## Check for and replace functins in metrics
        metric_list =  model_params['compile_params']['metrics']
        
        # replace functions in metric list with source code
        for i,metric in enumerate(metric_list):
            if inspect.isfunction(metric):
                metric_list[i] = replace_function(metric)
        metric_list =  model_params['compile_params']['metrics']


        # model_params_to_save['compile_params']['metrics'] = model_params['compile_params']['metrics']
        model_params_to_save['compile_params']['optimizer_name'] = model_params['compile_params']['optimizer_name']
        model_params_to_save['fit_params'] = model_params['fit_params']

        ## save model_params_to_save to pickle
        model_params_filename=filename+'_model_params.pkl'
        try:
            with open(model_params_filename,'wb') as param_file:
                pickle.dump(model_params_to_save, param_file) #sort_keys=True,indent=4)
        except:
            print('Pickling failed')
    else:
        model_params_filename=''

    filename_dict = {'model':filename,'weights':weight_filename,'excel':excel_filename,'params':model_params_filename}
    return filename_dict#[filename, weight_filename, excel_filename, model_params_filename]


def load_model_weights_params(base_filename = 'models/model_',load_model_params=True, load_model_layers_excel=True, trainable=False, 
model_filename=None,weight_filename=None, model_params_filename = None, excel_filename=None, verbose=1):
    """Loads in Keras model from json file and loads weights from .h5 file.
    optional set model layer trainability to False"""
    from IPython.display import display
    from keras.models import model_from_json
    import json
    
    ## Set model and weight filenames from base_filename if None:
    if model_filename is None:
        model_filename = base_filename+'.json'

    if weight_filename is None:
        weight_filename = base_filename+'_weights.h5'
    
    if model_params_filename is None:
        model_params_filename = base_filename + '_model_params.pkl'
    
    if excel_filename is None:
        excel_filename = base_filename + '_model_layers.xlsx'


    ## LOAD JSON MODEL
    with open(model_filename, 'r') as json_file:
        loaded_model_json = json.loads(json_file.read())
    loaded_model = model_from_json(loaded_model_json)

    ## LOAD MODEL WEIGHTS 
    loaded_model.load_weights(weight_filename)
    print(f"Loaded {model_filename} and loaded weights from {weight_filename}.")

    # SET LAYER TRAINABILITY
    if trainable is False:
        for i, model_layer in enumerate(loaded_model.layers):
            loaded_model.get_layer(index=i).trainable=False
        if verbose>0:
            print('All model.layers.trainable set to False.')
        if verbose>1:
            print(model_layer,loaded_model.get_layer(index=i).trainable)
    
    # IF VERBOSE, DISPLAY SUMMARY
    if verbose>0:
        display(loaded_model.summary())
        print("Note: Model must be compiled again to be used.")

    
    ## START RETURN LIST WITH MODEL
    return_list = [loaded_model]

    ## LOAD MODEL_PARAMS PICKLE
    if load_model_params:
        import pickle
        model_params = pickle.load(model_params_filename)
        return_list.append(model_params)

    ## LOAD EXCEL OF MODEL LAYERS CONFIG
    if load_model_layers_excel:
        import pandas as pd
        df_model_layers = pd.read_excel(excel_filename)
        return_list.append(df_model_layers)

    return return_list[:]
    #     return loaded_model, model_params
    # else:
    #     return loaded_model 


def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U


def plot_confusion_matrix(conf_matrix, classes = None, normalize=False,
                          title='Confusion Matrix', cmap=None,
                          print_raw_matrix=False,fig_size=(5,5), show_help=False):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds_local and modified"""
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    cm = conf_matrix
    ## Set plot style properties
    if cmap==None:
        cmap = plt.get_cmap("Blues")

    ## Text Properties
    fmt = '.2f' if normalize else 'd'

    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',   
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }


    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()
    
    if classes is None:
        classes = ['negative','positive']
        
    tick_marks = np.arange(len(classes))


    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])

    
    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), color='darkgray',**fontDict['data_labels'])#color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])
    fig = plt.gcf()
    plt.show()
    
    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix)

    if show_help:
        print('''For binary classifications:
        [[0,0(true_neg),  0,1(false_pos)]
        [1,0(false_neg), 1,1(true_pos)] ]
        
        to get vals as vars:
        >>  tn,fp,fn,tp=confusion_matrix(y_test,y_hat_test).ravel()
                ''')

    return fig


def evaluate_regression(y_true, y_pred, metrics=None, show_results=False, display_thiels_u_info=False):
    """Calculates and displays any of the following evaluation metrics: (passed as strings in metrics param)
    r2, MAE,MSE,RMSE,U 
    if metrics=None:
        metrics=['r2','RMSE','U']
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    # from bs_ds_local import list2df
    from bs_ds_local import list2df
    import inspect
    
    import functions_combined_BEST as ji
    idx_true_null = ji.find_null_idx(y_true)
    idx_pred_null = ji.find_null_idx(y_pred)
    if all(idx_true_null == idx_pred_null):
        y_true.dropna(inplace=True)
        y_pred.dropna(inplace=True)
    else:
        raise Exception('There are non-overlapping null values in y_true and y_pred')

    results=[['Metric','Value']]
    metric_list = []
    if metrics is None:
        metrics=['r2','rmse','u']

    else:
        for metric in metrics:
            if isinstance(metric,str):
                metric_list.append(metric.lower())
            elif inspect.isfunction(metric):
                custom_res = metric(y_true,y_pred)
                results.append([metric.__name__,custom_res])
                metric_list.append(metric.__name__)
        metrics=metric_list

    # metrics = [m.lower() for m in metrics]

    if any(m in metrics for m in ('r2','r squared','R_squared')): #'r2' in metrics: #any(m in metrics for m in ('r2','r squared','R_squared'))
        r2 = r2_score(y_true, y_pred)
        results.append(['R Squared',r2])##f'R\N{SUPERSCRIPT TWO}',r2])
    
    if any(m in metrics for m in ('RMSE','rmse','root_mean_squared_error','root mean squared error')): #'RMSE' in metrics:
        RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
        results.append(['Root Mean Squared Error',RMSE])

    if any(m in metrics for m in ('MSE','mse','mean_squared_error','mean squared error')):
        MSE = mean_squared_error(y_true,y_pred)
        results.append(['Mean Squared Error',MSE])

    if any(m in metrics for m in ('MAE','mae','mean_absolute_error','mean absolute error')):#'MAE' in metrics or 'mean_absolute_error' in metrics:
        MAE = mean_absolute_error(y_true,y_pred)
        results.append(['Mean Absolute Error',MAE])

    
    if any(m in metrics for m in ('u',"thiel's u")):# in metrics:
        if display_thiels_u_info is True:
            show_eqn=True
            show_table=True
        else:
            show_eqn=False 
            show_table=False

        U = thiels_U(y_true, y_pred,display_equation=show_eqn,display_table=show_table )
        results.append(["Thiel's U", U])
    
    results_df = list2df(results)#, index_col='Metric')
    results_df.set_index('Metric', inplace=True)
    if show_results:
        from IPython.display import display
        dfs = results_df.round(3).reset_index().style.hide_index().set_caption('Evaluation Metrics')
        display(dfs)
    return results_df.round(4)



def res_dict_to_merged_df(dict_of_dfs, key_index_name='Prediction Source', old_col_index_name=None):
    import pandas as pd
    res_dict = dict_of_dfs
    # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
    rename_mapper = {'R_squared':'R^2','R Squared':'R^2','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
    if len(res_dict.keys())==1:
        
        res_df = res_dict[list(res_dict.keys())[0]]

        # res_df.set_index('Metric',inplace=True)
        res_df.rename(mapper=rename_mapper, axis='index',inplace=True)
        res_df=res_df.transpose()
        # caption='Evaluation Metrics'

    else:
        res_df= pd.concat(res_dict.values(), axis=1,keys=res_dict.keys())
        res_df.columns = res_df.columns.levels[0]
        res_df.columns.name=key_index_name
        res_df.index.name=old_col_index_name
        res_df = res_df.transpose()#inplace=True)
    
        # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
        res_df.rename(mapper= rename_mapper, axis='columns',inplace=True)

    return res_df


def get_evaluate_regression_dict(df,  metrics=['r2','RMSE','U'], show_results_dict = False, show_results_df=True,return_as_styled_df=False, return_as_df =True): #, return_col_names=False):
    """Calculates and displays any of the following evaluation metrics (passed as strings in metrics param) for each true/pred pair of columns in df:
    r2, MAE,MSE,RMSE,U """
    import re
    import functions_combined_BEST as ji
    from IPython.display import display
    import pandas as pd

    col_list = df.columns
    from_where = re.compile('(true|pred)_(from_\w*_?\w+?)')
    found = [from_where.findall(col) for col in col_list]
    found = [x[0] for x in found if len(x)>0]


    pairs_of_cols = {}
    df_dict = {}

    if 'true_test_price' in col_list:
        use_single_true_column = True
        true_test_series = df['true_test_price']
    else:
        use_single_true_column = False

#     results =[['preds_from','metric','value']]
    # for _,where in found:  
    for true_pred, from_source in found: 

        if use_single_true_column:
            true_series = true_test_series #.dropna()
            true_series_name = true_series.name
        else:
            true_series = df['true_'+from_source]#.dropna()
            true_series_name = true_series.name

        pred_series = df['pred_'+from_source]#.dropna()
        pred_series_name = pred_series.name

        # combine true_series and pred_series and then dropna()
        df_eval = pd.concat([true_series,pred_series],axis=1)
        df_eval.dropna(inplace=True)

        pairs_of_cols[from_source] = {}
        pairs_of_cols[from_source]=[true_series_name,pred_series_name]#['col_names']
        
        df_dict[from_source] = ji.evaluate_regression(df_eval[true_series_name],df_eval[pred_series_name],metrics=metrics) #.reset_index().set_index('Metric')#,inplace=True)
#         pairs_of_cols[where]['results']=res_df

    # # combine into one dataframe
    # df_results = pd.DataFrame.from_dict(df_dict,)

    if show_results_dict:
        ji.display_df_dict_dropdown(df_dict)
    
    ## Combine dataframes from dictionary into one output table 
    if return_as_df or show_results_df:
        
        # if only 1 set of results, just rename metrics


        def res_dict_to_merged_df(dict_of_dfs, key_index_name='Prediction Source', old_col_index_name=None):

            res_dict = dict_of_dfs
            # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
            rename_mapper = {'R_squared':'R^2','R Squared':'R^2','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}

            if len(res_dict.keys())==1:
                
                res_df = res_dict[list(res_dict.keys())[0]]

                # res_df.set_index('Metric',inplace=True)
                res_df.rename(mapper=rename_mapper, axis='index',inplace=True)
                res_df=res_df.transpose()
                # caption='Evaluation Metrics'

            else:
                res_df= pd.concat(res_dict.values(), axis=1,keys=res_dict.keys())
                res_df.columns = res_df.columns.levels[0]
                res_df.columns.name=key_index_name
                res_df.index.name=old_col_index_name
                res_df = res_df.transpose()#inplace=True)
            
                # rename_mapper = {'R_squared':f'R\N{SUPERSCRIPT TWO}','R Squared':f'R\N{SUPERSCRIPT TWO}','Root Mean Squared Error':'RMSE','Mean Absolute Error':'MAE',"Thiel's U":'U'}
                res_df.rename(mapper= rename_mapper, axis='columns',inplace=True)
            ## new to fix exporting for dash
            res_df.reset_index(inplace=True)
            return res_df

        res_df = res_dict_to_merged_df(df_dict)


    if show_results_df:
        res_df_s = res_df.style.hide_index().set_caption('Evaluation Metrics')# by Prediction Source'))
        display(res_df_s)

    if return_as_styled_df:
        return res_df_s
    elif return_as_df:
        return res_df
    else:
        return df_dict
        

def compare_eval_metrics_for_shifts(true_series,pred_series, shift_list=[-2,-1,0,1,2], true_train_series_to_add=None,
color_coded=True, return_results=False, return_styled_df=False, return_shifted_df=True, display_results=True, display_U_info=False):
    
    ## SHIFT THE TRUE VALUES, PLOT, AND CALC THIEL's U
    import functions_combined_BEST as ji
    from bs_ds_local import list2df
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    import pandas as pd

    true_colname = true_series.name#'true'
    pred_colname = pred_series.name#'pred'

    # combine true and preds into one dataframe
    df = pd.concat([true_series, pred_series], axis=1)
    df.columns=[true_colname, pred_colname]#.dropna(axis=0,subset=[[true_colname,pred_colname]])

    # Create Empty Resuts containers
    results=[['Bins Shifted','Metric','Value']]
    combined_results = pd.DataFrame(columns=results[0])
    shift_results_dict= {}
    
    # Loop through shifts, add to df_shifted, calc thiel's U
    df_shifted=df.copy()
    for shift in shift_list:

        # create df for current shift
        df_shift=pd.DataFrame()
        df_shift['pred'] = df[pred_colname].shift(shift)
        df_shift['true'] = df[true_colname]
        
        # Add shifted columns to df_shifted
        df_shifted['pred_shift'+str(shift)] =  df_shift['pred']
        
        # drop null values from current shit to calc metrics
        df_shift.dropna(inplace=True)

        #[!] ### DIFFERENT THAN U COMPARE U FUNCTION
        shift_results = evaluate_regression(df_shift['true'],df_shift['pred']).reset_index() #[true_colname],df_shift[pred_colname]).reset_index()
        shift_results.insert(0,'Bins Shifted',shift)
        

        ## ADD RESULTS TO VARIOUS OUTPUT METHODS
        results.append(shift_results)
        combined_results = pd.concat([combined_results,shift_results], axis=0)
        shift_results_dict[shift] =  shift_results.drop('Bins Shifted',axis=1).set_index('Metric')


    # if 
    if true_train_series_to_add is not None:
        df_shifted['true_train_price'] = true_train_series_to_add
    # Turn results into dataframe when complete
    # df_results = list2df(results)#
    # df_results.set_index('# of Bins Shifted', inplace=True)
    


    # Restructure DataFrame for ouput
    df_results = res_dict_to_merged_df(shift_results_dict, key_index_name='Pred Shifted')
    df_results.reset_index(inplace=True)

    if display_results:
        
        # Dispaly Thiel's U info
        if display_U_info:
            _ = thiels_U(None,None,True,True)
        
        
        # Display dataframe results
        if color_coded is True:
            dfs_results = ji.color_cols(df_results, subset=['RMSE','U'], rev=True)
            dfs_results.set_caption("Evaluation Metrics for Shifted Preds")

        else:
            df_results.style.set_caption('Evaluation Metrics for Shifted Preds')

        dfs_results.hide_index().set_properties(**{'text-align':'center'})
        display(dfs_results)


    ## Return requested oututs
    return_list = []


    if return_results:
        return_list.append(df_results)

    if return_styled_df:
        return_list.append(dfs_results)

    if return_shifted_df:
        return_list.append(df_shifted)

    return return_list[:]
    


def plot_best_shift(df,df_results, true_colname='true',pred_colname='pred',  col_to_check='U', best='min'):
    
    import matplotlib.pyplot as plt
    import pandas as pd
    if 'min' in best:
        best_shift = df_results[col_to_check].idxmin()#[0]
    elif 'max' in best:
        best_shift = df_results[col_to_check].idxmax()#[0]

    df[true_colname].plot(label = 'True Values')
    df[pred_colname].shift(best_shift).plot(ls='--',label = f'Predicted-Shifted({best_shift})')
    plt.legend()
    plt.title(f"Best {col_to_check} for Shifted Time Series")
    plt.tight_layout()
    return 


def compare_u_for_shifts(true_series,pred_series, shift_list=[-2,-1,0,1,2],
    plot_all=False,plot_best=True, color_coded=True, return_results=False, return_shifted_df=True,
    display_U_info=False):
    
    import functions_combined_BEST as ji
    from bs_ds_local import list2df
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    true_colname = true_series.name#'true'
    pred_colname = pred_series.name#'pred'
    
    # combine the series into one dataframe and rename
    df = pd.concat([true_series, pred_series],axis=1)
    # df.columns=[true_colname, pred_colname]#.dropna(axis=0,subset=[[true_colname,pred_colname]])
    
    # create results list
    results=[['# of Bins Shifted','U']]
    
    if plot_all or plot_best:
        plt.figure()

    if plot_all is True:
        df[true_colname].plot(color='black',lw=3,label = 'True Values')
        plt.legend()
        plt.title('Shifted Time Series vs Predicted')
        

    # Loop through shifts, add to df_shifted, calc thiel's U
    df_shifted = df.copy()        

    for shift in shift_list:
        if plot_all==True:
            df[pred_colname].shift(shift).plot(label = f'Predicted-Shifted({shift})')

        # create df for current shift
        df_shift=pd.DataFrame()
        df_shift['pred'] = df[pred_colname].shift(shift)
        df_shift['true'] = df[true_colname]
        
        # add to df_shifted
        df_shifted['pred_shift'+str(shift)] =  df_shift['pred']

        # Drop null values and calcualte Thiels U
        df_shift.dropna(inplace=True)
        U = thiels_U(df_shift['true'], df_shift['pred'],False,False)

        # Append results to results list
        results.append([shift,U])
    
    # Turn results into dataframe when complete
    df_results = list2df(results)#
    df_results.set_index('# of Bins Shifted', inplace=True)

    # if plot+nest
    if plot_best==True:
        plot_best_shift(df,df_results,true_colname=true_colname, pred_colname=pred_colname)

        # # def plot_best_shift(df_results,col_to_check):
        # best_shift = df_results['U'].idxmin()#[0]

        # df[true_colname].plot(label = 'True Values')
        # df[pred_colname].shift(best_shift).plot(ls='--',label = f'Predicted-Shifted({best_shift})')
        # plt.legend()
        # plt.title("Best Thiel's U for Shifted Time Series")
        # plt.tight_layout()

    # Dispaly Thiel's U info
    if display_U_info:
        _ = thiels_U(None,None,True,True)

    # Display dataframe results
    if color_coded is True:
        dfs_results = ji.color_cols(df_results, rev=True)
        display(dfs_results.set_caption("Thiel's U - Shifting Prediction Time bins"))

    else:
        display(df_results.style.set_caption("Thiel's U - Shifting Prediction Time bins"))
        
    # Return requested oututs
    return_list = []

    if return_results:
        return_list.append(df_results)

    if return_shifted_df:
        return_list.append(df_shifted)

    return return_list[:]


def compare_time_shifted_model(df_model,true_colname='true test',pred_colname='pred test',
                               shift_list=[-4,-3,-2,-1,0,1,2,3,4],show_results=True,show_U_info=True,
                               caption=''):
    import numpy as np
    from IPython.display import display
    import functions_combined_BEST as ji

    # GET EVALUATION METRICS FROM PREDICTIONS
    true_test_series = df_model[true_colname].dropna()
    pred_test_series = df_model[pred_colname].dropna()

    # Comparing Shifted Timebins
    res_df, shifted_df = compare_eval_metrics_for_shifts(true_test_series.rename(true_colname),
    pred_test_series.rename(pred_colname),shift_list=np.arange(-4,4,1,))

    res_df = res_df.pivot(index='Bins Shifted', columns='Metric',values='Value')
    res_df.columns.rename(None, inplace=True)
    
    
    if show_results:
        
        res_dfs = res_df.copy().style
        res_dfs = ji.color_cols(res_df,subset=["Thiel's U"],rev=True) #OLD
        display(res_dfs.set_caption(caption))
    
    if show_U_info:
        _ = thiels_U(None,None,True,True)
        
#         metric_best_crit = {'R_squared':'max', "Thiel's U":'min','Root Mean Squared Error':'min'}    
#         for k,v in metric_best_crit.items():
#             res_dfs = res_dfs.apply(lambda x: highlight_best(x,v),axis=0)        
#         display(res_dfs)
    return res_df

# def get_model_preds_df(model, train_generator, test_generator, true_train_data, true_test_data,
# preds_from_gen=True, preds_from_train_preds =True, preds_from_test_preds=True, model_params=None, train_data_index=None, test_data_index=None, x_window=None, return_combined=False):
#     """Accepts a model, the training and testing data TimeseriesGenerators, the test_index and train_index.
#     Returns a dataframe with True and Predicted Values for Both the Training and Test Datasets."""
#     import pandas as pd
#     import functions_combined_BEST as ji

#     if model_params is not None:
#         train_data_index = model_params['input_params']['train_data_index']
#         test_data_index = model_params['input_params']['test_data_index']
#         x_window =  model_params['data_params']['x_window']


#     if true_test_data is None:
#         raise Exception("true_test_data = df_test['price']")

#     if true_train_data is None:        
#         raise Exception("true_train_data=df_train['price']")
    



#     #### ADD SWITCH DEPENDING ON TRUE CONDITIONS

#     if preds_from_gen == True:
#         get_model_preds_from_gen()

#         # #  GET INDICES BASED ON GENERATOR START AND END
#         # gen_index = true_test_data.index[test_generator.start_index:test_generator.end_index+1]

        
#         # # GET PREDICTIONS FOR TRAINING DATA AND TEST DATA
#         # test_predictions = ji.arr2series( model.predict_generator(test_generator),
#         #                             test_data_index[x_window:], 'pred test')
        
#         # train_predictions = ji.arr2series( model.predict_generator(train_generator),
#         #                             train_data_index[x_window:], 'pred train')

    

#     # GET TRUE TEST AND TRAIN DATA AS SERIES
#     true_test_price = pd.Series( true_test_data.iloc[x_window:],
#                                 index= test_data_index[x_window:], name='true test')
    
#     true_train_price = pd.Series(true_train_data.iloc[x_window:],
#                                  index = train_data_index[x_window:], name='true train')

    
#     # COMBINE TRAINING DATA AND TESTING DATA INTO 2 DFS (with correct date axis)
#     df_true_v_preds_train = pd.concat([true_train_price, train_predictions],axis=1)
#     df_true_v_preds_test= pd.concat([true_test_price, test_predictions],axis=1)
    
#     # RETURN ONE OR TWO DATAFRAMES
#     if return_combined is False:
#         return df_true_v_preds_train, df_true_v_preds_test
#     elif return_combined is True:
#         df_combined = pd.concat([df_true_v_preds_train, df_true_v_preds_test],axis=1)
#         df_combined.columns=['true train','pred train','true test','pred test']
#         return df_combined
# def get_true_vs_preds_df(model, true_test_series=None,test_generator=None):
#     import pandas as pd
#     import functions_combined_BEST as ji
#     import bs_ds_local  as bs


#     pass

def get_model_preds_df(model, true_train_series,true_test_series, test_generator,model_params=None,
x_window=None, n_features=None, inverse_tf=False, scaler=None, include_train_data=True,
 preds_from_gen=True, preds_from_train_preds =False, preds_from_test_preds=False, 
 iplot=False,iplot_title=None,verbose=1):#  train_data_index=None, test_data_index=None
    """ Gets predictions for training data from the 3 options: 
    1) from generator  --  len(output) = (len(true_test_series)-n_input)
    2) from predictions on test data  --  len(output) = (len(true_test_series)-n_input)
    3) from predictions on train data -- len(true_test_series)
    """
    import pandas as pd
    import functions_combined_BEST as ji
    import bs_ds_local  as bs
    # x_window=n_input

    ## If no model params
    if model_params is None:

        ## get the seires indices from the true input series
        train_data_index = true_train_series.index
        test_data_index = true_test_series.index

        ## Get x_window and n_features from the generator
        if x_window is None:
            x_window = test_generator.length
        if n_features is None:
            n_features=test_generator.data[0].shape[0]

        if inverse_tf and scaler is None:
            raise Exception('if inverse_tf, must provide previously fit scaler.')

    
    if model_params is not None:
        if scaler is None and inverse_tf == True:
            scaler = model_params['scaler_library']['price']
        ## get n_features,x_window from model_params
        n_features = model_params['input_params']['n_features']
        x_window = model_params['input_params']['n_input']

        ## get indices from model_params
        train_data_index = model_params['input_params']['train_data_index']
        test_data_index = model_params['input_params']['test_data_index']
        # if model_params['data_params']['x_window'] == model_params['input_params']['n_input']:
        #     x_window = model_params['input_params']['n_input']
        # else:
        #     print('x_window and n_input params are not the same, using n_input as x_window...')
            
    if (preds_from_gen == True) and (test_generator == None):
        raise Exception('If from_gen=True, must provide generator.')

            
    ### GET the 3 DIFERENT TYPES OF PREDICTIONS    
    df_list = []
    if preds_from_gen:
        ## get predictions from generator and return gen_df with correct data indices
        gen_df = get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
         model_params=model_params, n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

        df_list.append(gen_df)



    if preds_from_test_preds:
        
        func_df_from_test = get_model_preds_from_preds(model=model, true_train_data=true_train_series, true_test_data=true_test_series,
        model_params=model_params, x_window=x_window, n_features=n_features,
         suffix='_from_test_preds',build_preds_from_train=False, return_df=True)

        df_list.append(func_df_from_test)



    if preds_from_train_preds:

        func_df_from_train  = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=x_window, n_features=n_features,
        suffix='_from_train_preds', build_preds_from_train=True,return_df=True)
        df_list.append(func_df_from_train)


    # bs.display_side_by_side(func_df,func_df_from_train)
    ## combine into df
    df_all_preds = pd.concat([df for df in df_list],axis=1)
    df_all_preds = bs.drop_cols(df_all_preds,['i_']);


    ## ADD TRAINING DATA TO DATAFRAME IF REQUESTED
    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,df_all_preds],axis=1)
        
    ## INVERSE TRANSOFRM BACK TO PRICE
    if inverse_tf:
        df_out = ji.transform_cols_from_library(df_all_preds,single_scaler=scaler,inverse=True)
    else:
        df_out = df_all_preds

    
    def get_plot_df_with_one_true_series(df_out,train_data = true_train_series, include_train_data=include_train_data):
        
        # print(df_out.columns)
        df_plot = pd.DataFrame()

        ## Check columns for true_train_price, then make list of other colnames
        cols = df_out.columns.to_list()

        if 'true_train_price' in cols:
            if include_train_data:
                df_plot['true_train_price'] = df_out['true_train_price']
            
        # remove from the col_list to be looped through.
        col_list = [x for x in cols if x !='true_train_price']

        ## Use regexp to separate 'true_from_source','pred_from_source' columns
        import re
        from_where = re.compile('(true|pred)_(from_\w*_?\w+?)')
        # found = [from_where.findall(col)[0] for col in col_list]
        found = [from_where.findall(col) for col in col_list]
        found = [x[0] for x in found if len(x)>0]
        
        pairs_of_cols = {}
        true_col_data = []
        true_col_name = []
        check_true_col_name = []
        check_true_col_data = []
        # Check tuple for true/pred and from_source
        for true_pred, from_source in found: 

            if 'true' in true_pred:
                
                if len(true_col_data)==0:
                    
                    true_col_name = f"{true_pred}_{from_source}"
                    true_col_data = df_out[true_col_name]

                    df_plot['true_test_price'] = true_col_data

                else:
                    check_true_col_name = f"{true_pred}_{from_source}"
                    check_true_col_data = df_out[check_true_col_data]

                    if all(check_true_col_data == true_col_data):
                        continue
                    else:
                        print(f'Warning: true data from {true_col_name} and {check_true_col_name} do not match!')
                        # name_recon = f"{true_pred}_{from_source}"
                        # df_plot['true_test_price'] = df_out[name_recon]#true_series_to_plot
                        # continue #break?
                
            elif 'pred' in true_pred:
                name_recon = f"{true_pred}_{from_source}"
                df_plot[name_recon] = df_out[name_recon]
                # continue            

        return df_plot 
    
    df_plot = get_plot_df_with_one_true_series(df_out,train_data=true_train_series, include_train_data=include_train_data ) 

    ## display head if verbose
    if verbose>0:
        ji.disp_df_head_tail(df_plot)

    if iplot==False:
        return df_plot
    else:
        # from plotly.offline import 
        # df_plot = get_plot_df_with_one_true_series(df_out,train_data=true_train_series, include_train_data=include_train_data ) 
        pred_columns = [x for x in df_plot.columns if 'pred' in x]
        if iplot_title is None:
            iplot_title='S&P 500 True Price Vs Predictions ($)'
        fig = ji.plotly_true_vs_preds_subplots(df_plot, title=iplot_title,true_train_col='true_train_price',
            true_test_col='true_test_price', pred_test_columns=pred_columns)

        return df_plot




def get_eval_dict_for_paired_cols(df,col_regex_tokens='(true|pred)_(from_\w*_?\w+?)'):
    import re
    
    col_list = df.columns
    from_where = re.compile(col_regex_tokens) #'(true|pred)_(from_\w*_?\w+?)')
    found = [from_where.findall(col) for col in col_list]
    found = [x[0] for x in found if len(x)>0]


    pairs_of_cols = {}
    for _,where in found: 

        true_series = df['true_'+where].dropna()
        pred_series = df['pred_'+where].dropna()
        pairs_of_cols[where] = {}
        pairs_of_cols[where]['col_names']={'true':true_series.name,'pred':pred_series.name}
        pairs_of_cols[where]['results']=evaluate_regression(true_series,pred_series).reset_index()
    
    return pairs_of_cols
    

def get_predictions_df_and_evaluate_model(model, test_generator,
                                        true_train_data, true_test_data, model_params=None,
                                        train_data_index=None, test_data_index=None, 
                                        x_window=None, scaler=None, inverse_tf =True,
                                        return_separate=False, plot_results = True,
                                        iplot_results=False):

    import functions_combined_BEST as ji
    import pandas as pd
    from IPython.display import display
    n_input=x_window

    if model_params is not None:
        train_data_index = model_params['input_params']['train_data_index']
        test_data_index = model_params['input_params']['test_data_index']
        x_window =  model_params['data_params']['x_window']
        scaler_library = model_params['scaler_library']

    if true_test_data is None:
        raise Exception("true_test_data = df_test['price']")

    if true_test_data is None:        
        raise Exception("true_train_data=df_train['price']")


    # Call helper to get predictions and return as dataframes 
    # df_true_v_preds_train, df_true_v_preds_test 
    df_model = get_model_preds_df(model,  test_generator=test_generator,
    true_train_series=true_train_data, true_test_series = true_test_data, model_params=model_params,
        x_window=None, inverse_tf = inverse_tf)
    ## Concatenate into one dataframe
    # df_model_preds = pd.concat([df_true_v_preds_train, df_true_v_preds_test],axis=1)
    
    # ## CONVERT BACK TO DOLLARS AND PLOT
    # if inverse_tf==True:
    #     df_model = pd.DataFrame()
    #     for col in df_model_preds.columns:
    #         df_model[col] = ji.transform_series(df_model_preds[col],scaler_library['price'], inverse=True) 
    # else:
    #     df_model = df_model_preds

        
    if plot_results:
        # PLOTTING TRAINING + TRUE/PRED TEST DATA
        ji.plot_true_vs_preds_subplots(df_model['true train'],df_model['true test'], 
                                    df_model['pred test'], subplots=True)
    if iplot_results:
        df_plot = df_model.copy().drop(['pred train','true_from_test_preds','true_from_train_preds'],axis=1)
        ji.plotly_time_series(df_plot) 


    

    # prepare display_of_results

    # # GET EVALUATION METRICS FROM PREDICTIONS
    # true_test_series = df_model['true test'].dropna()
    # pred_test_series = df_model['pred test'].dropna()
    
    # # Get and display regression statistics
    # results_tf = evaluate_regression(true_test_series, pred_test_series)
    pairs_of_cols = ji.get_evaluate_regression_dict(df_model)
    display(pairs_of_cols)

    return df_model




def get_model_preds_from_preds(model,true_train_data, true_test_data,
                         model_params=None, x_window=None, n_features=None, 
                         build_preds_from_train=True, return_df=True,suffix=None, debug=False):
    
    """ Gets predictions from model using using its own predictions as the subsequent input.
    Must provide a model_params dictionary with 'input_params' OR must define ('n_input','n_features').
    
    * IF build_preds_from_train is True:
        1. starting true time series for predictions is the last rows [-n_input:] from training data.
        2. output predicitons will be the same length as the input scaled_test_data
    
    * IF build_preds_from_train is False:
        1. starting true time series for predictions is the first rows [:n_input] from test data.
        2. output predicitons will be shorter by n_input # of rows
    """
    scaled_train_data = true_train_data
    scaled_test_data = true_test_data
    import bs_ds_local as bs
    import numpy as np
    import pandas as pd
    test_predictions = []
    first_eval_batch=[]

    n_input = x_window

    if model_params is not None:
        if n_input is None:
            n_input= model_params['input_params']['n_input']

        if n_features is None:
            n_features = model_params['input_params']['n_features']
    


    preds_out = [['i','index','pred']]
    
    # SAVING COPY OF INPUT TEST DATA
    scaled_test_series = scaled_test_data.copy() 

    ## SAVING TRAIN AND TEST DATA INDICES AND VALUES
    train_data_index = scaled_train_data.index
    scaled_train_data = scaled_train_data.values
    test_data_index = scaled_test_data.index
    scaled_test_data = scaled_test_data.values
    
    
    ## PREPARE THE FIRST EVAL BATCH TIMESERIES FROM TRAIN OR TEST DATA
    # Change parameters depending on if from train or test data
    if build_preds_from_train:
        
        # If using trianing data loop through full test data
        loop_length = range(len(scaled_test_data))
        
        # take the last window size of data from training data 
        first_eval_batch = scaled_train_data[-1*n_input:]
        # first_batch_idx = train_data_index[-n_input:]
        
        # set the true index to test_data_index
        true_index_out = test_data_index
        
    # Set the loop to be from n_input # of rows into test_data
    else:
        loop_length = range(n_input,len(scaled_test_data))
        first_eval_batch = scaled_test_data[:n_input]
        true_index_out =  test_data_index[n_input:]
      
    
    # reshape first batch of data for model.predict 
    first_batch_pre_reshape = first_eval_batch.shape    
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    first_batch_shape = current_batch.shape

    
    ## LOOP THROUGH REMAINING TIMEBINS USING CURRENT PREDICITONS AS NEW DATA FOR NEXT
    for i in loop_length:

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]        
        # store prediction
        test_predictions.append(current_pred) 

        # update batch to now include prediction and drop first value
        # try:
            # current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        # except:
            # print('COMMAND FAILED: "current_batch = np.append(current_batch[:,1:,:],[current_pred],axis=1)"' )
        #     print(f'\nn_features={n_features}')
        #     print(f"n_input={n_input}")
        #     print(f"first_batch_shape={first_batch_shape}; current_batch.shape={current_batch.shape}; current_pred.shape={current_pred.shape}" )
        # finally:
        #     from pprint import pprint
        #     print('current_batch:',current_batch,'\ncurrent_pred:',current_pred)
        
        ## Append the data to the output df list
        preds_out.append([i,test_data_index[i],current_pred[0]])


    ## If returning a dataframe,prepare and rename with suffix
    if return_df:
        res_df = bs.list2df(preds_out)
        res_df['index'] = pd.to_datetime(res_df['index'])
        res_df.set_index('index',inplace=True)
        
        # adding true price
        res_df['true'] = scaled_test_series.loc[true_index_out]
        res_df=res_df[['i','true','pred']]
        
        if suffix is not None:
            colnames = [name+suffix for name in res_df.columns]
        else:
            colnames = res_df.columns
            
        res_df.columns=colnames
        return res_df #test_predictions, checks
    
    # Else just return array of predictions
    else:
        return np.array(test_predictions)



def get_model_preds_from_gen(model,test_generator, true_test_data, model_params=None,
                       n_input=None, n_features=None, suffix=None, verbose=0,return_df=True):
        """
        Gets prediction from model using the generator's timeseries using model.predict_generator()
        Must provide a model_params dictionary with 'input_params' OR must define ('n_input','n_features').

        """
        import pandas as pd
        import numpy as np
        import bs_ds_local as bs
        import functions_combined_BEST as ji
        if model_params is not None:
            n_input= model_params['input_params']['n_input']
            n_features = model_params['input_params']['n_features']

        if model_params is None:
            if n_input is None:
                n_input= test_generator.length
            if n_features is None:
                n_features=test_generator.data[0].shape[0]

        # GET TRUE VALUES AND DATETIME INDEX FROM GENERATOR
        
        # Get true time index from the generator's start_index and end_index 
        gen_index = true_test_data.index[test_generator.start_index:test_generator.end_index+1]
        gen_true_targets = test_generator.targets[test_generator.start_index:test_generator.end_index+1]
        
        # Generate predictions from the test_generator
        gen_preds = model.predict_generator(test_generator)
        gen_preds_flat = gen_preds.ravel()
        gen_true_targets = gen_true_targets.ravel()
        
        
        # RETURN OUTPUT AS DATAFRAME OR ARRAY OF PREDS
        if return_df == False:
            return gen_preds

        else:
            # Combine the outputs
            if verbose>0:
                print(len(gen_index),len(gen_true_targets), len(gen_preds_flat))

            gen_pred_df = pd.DataFrame({'index':gen_index,'true':gen_true_targets,'pred':gen_preds_flat})
            gen_pred_df['index'] = pd.to_datetime(gen_pred_df['index'])
            gen_pred_df.set_index('index',inplace=True)

            if suffix is not None:
                colnames = [name+suffix for name in gen_pred_df.columns]
            else:
                colnames = gen_pred_df.columns
            gen_pred_df.columns=colnames
            return gen_pred_df


def compare_model_pred_methods(model, true_train_series,true_test_series, test_generator=None,
                               model_params=None, n_input=None, n_features=None, from_gen=True,
                               from_train_series = True, from_test_series=True, 
                               iplot=True, plot_with_train_data=True,return_df=True, inverse_tf=True):
    """ Gets predictions for training data from the 3 options: 
    1) from generator  --  len(output) = (len(true_test_series)-n_input)
    2) from predictions on test data  --  len(output) = (len(true_test_series)-n_input)
    3) from predictions on train data -- len(true_test_series)
    """
    import pandas as pd
    import functions_combined_BEST as ji
    import bs_ds_local  as bs
    if model_params is not None:
        n_input= model_params['input_params']['n_input']
        n_features = model_params['input_params']['n_features']

    if model_params is None:
        if n_input is None or n_features is None:
            raise Exception('Must provide model params or define n_input and n_features')
            
    if from_gen is True and test_generator is None:
        raise Exception('If from_gen=True, must provide generator.')

            
    ### GET the 3 DIFERENT TYPES OF PREDICTIONS    
    df_list = []
    #(model, test_generator, true_test_data, model_params=None, n_input=None, n_features=None, suffix=None, return_df=True)
    if from_gen:
        gen_df = get_model_preds_from_gen(model=model, test_generator=test_generator,
        true_test_data=true_test_series, model_params=model_params, 
        n_input=n_input, n_features=n_features,  suffix='_from_gen',return_df=True)    
        df_list.append(gen_df)
    #s(model, scaled_train_data, scaled_test_data, model_params=None, n_input=None, n_features=None, build_preds_from_train=True, return_df=True, suffix=None)
    if from_test_series:

        func_df_from_test = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=n_input, n_features=n_features,
         suffix='_from_test',build_preds_from_train=False, return_df=True)
        df_list.append(func_df_from_test)

    if from_train_series:
        func_df_from_train  = get_model_preds_from_preds(model=model, true_train_data=true_train_series,
        true_test_data=true_test_series,model_params=model_params, x_window=n_input, n_features=n_features,
        suffix='_from_train', build_preds_from_train=True,return_df=True)
        df_list.append(func_df_from_train)

    # bs.display_side_by_side(func_df,func_df_from_train)

    df_all_preds = pd.concat([df for df in df_list],axis=1)
    df_all_preds = bs.drop_cols(df_all_preds,['i_'])
    # print(df_all_preds.shape)
    if plot_with_train_data:
        df_all_preds=pd.concat([true_train_series.rename('true_train_price'),df_all_preds],axis=1)

    if inverse_tf:
        df_out = ji.transform_cols_from_library(df_all_preds,single_scaler=model_params['scaler_library']['price'],inverse=True)
    else:
        df_out = df_all_preds

    if iplot:
        ji.plotly_time_series(df_out)

    if return_df:
        return df_out


def extract_true_vs_pred_cols(df_model1, rename_cols = True, from_gen=True, from_test_preds=False,
from_train_preds=False):
    import pandas as pd
    if sum([from_gen, from_test_preds, from_train_preds]) >1:
        raise Exception('Only 1 of the "from_source" inputs may ==True: ')
    
    list_of_possible_cols = ['true_from_gen', 'pred_from_gen',
     'true_from_test_preds','pred_from_test_preds', 'true_from_train_preds',
       'pred_from_train_preds']

    if from_gen:
        true_col = 'true_from_gen'
        pred_col = 'pred_from_gen'
    
    if from_test_preds:
        true_col = 'true_from_test_preds'
        pred_col = 'pred_from_test_preds'

    if from_test_preds:
        true_col = 'true_from_train_preds'
        pred_col = 'pred_from_train_preds'


    true_series = df_model1[true_col].rename('true')
    pred_series = df_model1[pred_col].rename('pred')

    
    df_model_out = pd.concat([true_series, pred_series],axis=1)

    if rename_cols==True:
        df_model_out.columns = ['true','pred']

    return df_model_out 


def evaluate_classification(*args,**kwargs):
    raise Exception('Use `evaluate_classification_model` instead of `evaluate_classification`.')

# def evaluate_classification(model, history, X_train,X_test,y_train,y_test,report_as_df=True, binary_classes=True,
#                             conf_matrix_classes= ['Decrease','Increase'],
#                             normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
#                             history_filename ='results/keras_history.png', save_conf_matrix_png=False,
#                             conf_mat_filename= 'results/confusion_matrix.png',save_summary=False, 
#                             summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

#     """Evaluates kera's model's performance, plots model's history,displays classification report,
#     and plots a confusion matrix. 
#     conf_matrix_classes are the labels for the matrix. [negative, positive]
#     Returns df of classification report and fig object for  confusion matrix's plot."""

#     from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
#     import bs_ds_local as bs
#     import functions_combined_BEST as ji
#     from IPython.display import display
#     import pandas as pd
#     import matplotlib as mpl
#     numFmt = '.4f'
#     num_dashes = 30

#     # results_list=[['Metric','Value']]
#     # metric_list = ['accuracy','precision','recall','f1']
#     print('---'*num_dashes)
#     print('\tTRAINING HISTORY:')
#     print('---'*num_dashes)

#     if auto_unique_filenames:
#         ## Get same time suffix for all files
#         time_suffix = ji.auto_filename_time(fname_friendly=True)

#         filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
#         ## update filenames 
#         for filetype,filename in filename_dict.items():
#             if '.' in filename:
#                 filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
#             else:
#                 if filetype =='summary':
#                     ext='.txt'
#                 else:
#                     ext='.png'
#                 filename_dict[filetype] = filename+time_suffix + ext


#         history_filename = filename_dict['history']
#         conf_mat_filename = filename_dict['conf_mat']
#         summary_filename = filename_dict['summary']


#     ## PLOT HISTORY
#     ji.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

#     print('\n')
#     print('---'*num_dashes)
#     print('\tEVALUATE MODEL:')
#     print('---'*num_dashes)

#     print('\n- Evaluating Training Data:')
#     loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
#     print(f'    - Accuracy:{accuracy_train:{numFmt}}')
#     print(f'    - Loss:{loss_train:{numFmt}}')

#     print('\n- Evaluating Test Data:')
#     loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
#     print(f'    - Accuracy:{accuracy_test:{numFmt}}')
#     print(f'    - Loss:{loss_test:{numFmt}}\n')


#     ## Get model predictions
    
#     if hasattr(model, 'predict_classes'):
#         y_hat_train = model.predict_classes(X_train)
#         y_hat_test = model.predict_classes(X_test)
#     elif hasattr(model,'predict'):
#         y_hat_train = model.predict(X_train)
#         y_hat_test = model.predict(X_test)
#     else:
#         raise Exception('model has neither "predict" nor "predict_classes" methods')

#     if y_test.ndim>1 or y_hat_test.ndim>1 or binary_classes==False:
#         if binary_classes==False: 
#             pass
#         else:
#             binary_classes = False
#             print(f"[!] y_test was >1 dim, setting binary_classes to False")

#         ## reduce dimensions of y_train and y_test
#         # y_train = y_train.argmax(axis=1)
#         # y_test = y_test.argmax(axis=1)
#         if y_test.ndim>1:            
#             y_test = y_test.argmax(axis=1)
#         if y_hat_test.ndim>1:
#             y_hat_test = y_hat_test.argmax(axis=1)
#         # for var in ['y_test', 'y_hat_test', 'y_train', 'y_hat_train']:
#         #     real_var = eval(var)
#         #     print('real_var shape:',real_var.shape)
#         #     if real_var.ndim>1:
#         #         ## reduce dimensions
#         #         cmd =  var+'= real_var.argmax(axis=1)'
#         #         # eval(cmd)
#         #         eval(var+'=') real_var.argymax(axis=1)
#         #         # exec(cmd)
#         #         cmd =f'print("argmax shape:",{var}.shape)' 
#         #         eval(cmd)
#         #         # exec(cmd)
        
        
        

#     print('---'*num_dashes)
#     print('\tCLASSIFICATION REPORT:')
#     print('---'*num_dashes)

#     # get both versions of classification report output
#     report_str = classification_report(y_test,y_hat_test)
#     report_dict = classification_report(y_test,y_hat_test,output_dict=True)
#     if report_as_df:
#         try:
#             ## Create and display classification report
#             # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
#             df_report_temp = pd.DataFrame(report_dict)
#             df_report_temp = df_report_temp.T#reset_index(inplace=True)

#             df_report = df_report_temp[['precision','recall','f1-score','support']]
#             display(df_report.round(4).style.set_caption('Classification Report'))
#             print('\n')
        
#         except:
#             print(report_str)
#             # print(report_dict)
#             df_report = pd.DataFrame()
#     else:
#         print(report_str)

#     if save_summary:
#         with open(summary_filename,'w') as f:
#             model.summary(print_fn=lambda x: f.write(x+"\n"))
#             f.write(f"\nSaved at {time_suffix}\n")
#             f.write(report_str)

#     ## Create and plot confusion_matrix
#     conf_mat = confusion_matrix(y_test, y_hat_test)
#     mpl.rcParams['figure.figsize'] = conf_matrix_figsize
#     fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes,
#                                    normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
#     if save_conf_matrix_png:
#         fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)

#     if report_as_df:
#         return df_report, fig
#     else:
#         return report_str,fig



def evaluate_classification_model(model,  X_train,X_test,y_train,y_test, history=None,binary_classes=True,
                            conf_matrix_classes= ['Decrease','Increase'], plot_training_conf_mat = False,
                            normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
                            history_filename ='results/keras_history.png', save_conf_matrix_png=False,
                            conf_mat_filename= 'results/confusion_matrix.png',save_summary=False,
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Updated version of `evaluate_classification` from bs-ds. 
    Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix.
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
    import bs_ds_local as bs
    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = bs.auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
        ## update filenames
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        conf_mat_filename = filename_dict['conf_mat']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    if history is not None:
        bs.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)
    
    ## Show model.evaluate for training and test data
    if hasattr(model,'evaluate'):

        print('\n- Evaluating Training Data:')
        loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
        print(f'    - Accuracy:{accuracy_train:{numFmt}}')
        print(f'    - Loss:{loss_train:{numFmt}}')

        print('\n- Evaluating Test Data:')
        loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
        print(f'    - Accuracy:{accuracy_test:{numFmt}}')
        print(f'    - Loss:{loss_test:{numFmt}}\n')

    
     



    ## Get predictions from model
    if hasattr(model,'predict_classes'):
        ## Get model predictions
        y_hat_train = model.predict_classes(X_train)
        y_hat_test = model.predict_classes(X_test)
    elif hasattr(model,'predict'):
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
    else: 
        raise Exception('Model does not have .predict_classes or .predict methods')
    
    
    
        
    ## Flatten any multi-column keras targets
    if y_test.ndim>1:
        y_test = y_test.argmax(axis=1)

    if y_train.ndim>1:
        y_train =y_train.argmax(axis=1)        
        
    if y_hat_test.ndim>1:
        y_hat_test = y_hat_test.argmax(axis=1)

    if y_hat_train.ndim>1:
        y_hat_train =y_hat_train.argmax(axis=1)            
        # or y_train.ndim>1:#binary_classes==False:
        # if binary_classes==False: 
        #     pass
        
        ## reduce dimensions of y_train and y_test
        # y_train = y_train.argmax(axis=1)
        
        # if binary_classes == True:
        #     binary_classes = False
        #     print(f"[!] y_test was >1 dim, setting binary_classes to False")
            
        
        
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)

    try:
        ## Get sklearn classification report 
        report_str = classification_report(y_test,y_hat_test)
        report_dict = classification_report(y_test,y_hat_test,output_dict=True)
    except:
        print('y_test:',y_test.shape)
        print('y_hat_test:',y_hat_test.shape)
        print('y_train:',y_train.shape)
        print('y_hat_train:',y_hat_train.shape)
    
    try:
        ## Create and display classification report
        # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
        df_report_temp = pd.DataFrame(report_dict)
        df_report_temp = df_report_temp.T#reset_index(inplace=True)

        df_report = df_report_temp[['precision','recall','f1-score','support']]
        display(df_report.round(4).style.set_caption('Classification Report'))
        print('\n')
    
    except:
        print(report_str)
        # print(report_dict)
        df_report = pd.DataFrame()

    ## if saving the model.summary() printout 
    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(report_str)

    def get_and_plot_conf_mat(y_true, y_pred,title='Confusion Matrix',save_png = save_conf_matrix_png):#, save_conf_matrix_png=save_conf_matrix_png,
                            #   conf_mat_filename=conf_mat_filename, normalize_conf_matrix=):
        ## Create and plot confusion_matrix
        import matplotlib.pyplot as plt
        conf_mat = confusion_matrix(y_true, y_pred)
        with plt.rc_context(rc={'figure.figsize':conf_matrix_figsize}): # rcParams['figure.figsize']
            fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes, title=title,
                                        normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
        if save_png:
            fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)
            
        fig.show()
        return fig
            
    fig = get_and_plot_conf_mat(y_test,y_hat_test,title='Confusion Matrix: Test Data')    
        
    if plot_training_conf_mat:
        fig_cm_train = get_and_plot_conf_mat(y_train,y_hat_train,save_png=False,title='Confusion Matrix: Training Data')    

    return df_report, fig



def evaluate_regression_model(model, history, train_generator, test_generator,true_train_series,
true_test_series,include_train_data=True,return_preds_df = False, save_history=False, history_filename ='results/keras_history.png', save_summary=False, 
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix. 
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
    import bs_ds_local as bs
    import functions_combined_BEST as ji
    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = ji.auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'summary':summary_filename}
        ## update filenames 
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    ji.plot_keras_history( history,filename_base=history_filename,no_val_data=True, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)

        # # EVALUATE MODEL PREDICTIONS FROM GENERATOR 
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_train[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_train[0]:{numFmt}}')

    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)
    print(f'    - Accuracy:{model_metrics_test[1]:{numFmt}}')
    print(f'    - Loss:{model_metrics_test[0]:{numFmt}}')

    x_window = test_generator.length
    n_features = test_generator.data[0].shape[0]
    gen_df = ji.get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
        n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

    regr_results = evaluate_regression(y_true=gen_df['true_from_gen'], y_pred=gen_df['pred_from_gen'],show_results=True,
                                metrics=['r2', 'RMSE', 'U'])


    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")
            f.write(regr_results.__repr__())


    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,gen_df],axis=1)
    else:
        df_all_preds = gen_df

    if return_preds_df:
        return df_all_preds




# def evaluate_gridsearch_classification(model, X_train,X_test,y_train,y_test,report_as_df=True, binary_classes=True,
#                             conf_matrix_classes= ['Decrease','Increase'],
#                             normalize_conf_matrix=True,conf_matrix_figsize=(8,4),save_history=False,
#                             history_filename ='results/keras_history.png', save_conf_matrix_png=False,
#                             conf_mat_filename= 'results/confusion_matrix.png',save_summary=False, 
#                             summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

#     """Evaluates kera's model's performance, plots model's history,displays classification report,
#     and plots a confusion matrix. 
#     conf_matrix_classes are the labels for the matrix. [negative, positive]
#     Returns df of classification report and fig object for  confusion matrix's plot."""

#     from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
#     import bs_ds_local as bs
#     import functions_combined_BEST as ji
#     from IPython.display import display
#     import pandas as pd
#     import matplotlib as mpl
#     numFmt = '.4f'
#     num_dashes = 30

#     # results_list=[['Metric','Value']]
#     # metric_list = ['accuracy','precision','recall','f1']
#     print('---'*num_dashes)
#     print('\tTRAINING HISTORY:')
#     print('---'*num_dashes)

#     if auto_unique_filenames:
#         ## Get same time suffix for all files
#         time_suffix = ji.auto_filename_time(fname_friendly=True)

#         filename_dict= {'history':history_filename,'conf_mat':conf_mat_filename,'summary':summary_filename}
#         ## update filenames 
#         for filetype,filename in filename_dict.items():
#             if '.' in filename:
#                 filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
#             else:
#                 if filetype =='summary':
#                     ext='.txt'
#                 else:
#                     ext='.png'
#                 filename_dict[filetype] = filename+time_suffix + ext


#         history_filename = filename_dict['history']
#         conf_mat_filename = filename_dict['conf_mat']
#         summary_filename = filename_dict['summary']


#     ## PLOT HISTORY
#     ji.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

#     print('\n')
#     print('---'*num_dashes)
#     print('\tEVALUATE MODEL:')
#     print('---'*num_dashes)

#     print('\n- Evaluating Training Data:')
#     loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=True)
#     print(f'    - Accuracy:{accuracy_train:{numFmt}}')
#     print(f'    - Loss:{loss_train:{numFmt}}')

#     print('\n- Evaluating Test Data:')
#     loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=True)
#     print(f'    - Accuracy:{accuracy_test:{numFmt}}')
#     print(f'    - Loss:{loss_test:{numFmt}}\n')


#     ## Get model predictions
    
#     if hasattr(model, 'predict_classes'):
#         y_hat_train = model.predict_classes(X_train)
#         y_hat_test = model.predict_classes(X_test)
#     elif hasattr(model,'predict'):
#         y_hat_train = model.predict(X_train)
#         y_hat_test = model.predict(X_test)
#     else:
#         raise Exception('model has neither "predict" nor "predict_classes" methods')

#     if y_test.ndim>1 or y_hat_test.ndim>1 or binary_classes==False:
#         if binary_classes==False: 
#             pass
#         else:
#             binary_classes = False
#             print(f"[!] y_test was >1 dim, setting binary_classes to False")

#         ## reduce dimensions of y_train and y_test
#         # y_train = y_train.argmax(axis=1)
#         # y_test = y_test.argmax(axis=1)
#         if y_test.ndim>1:            
#             y_test = y_test.argmax(axis=1)
#         if y_hat_test.ndim>1:
#             y_hat_test = y_hat_test.argmax(axis=1)
#         # for var in ['y_test', 'y_hat_test', 'y_train', 'y_hat_train']:
#         #     real_var = eval(var)
#         #     print('real_var shape:',real_var.shape)
#         #     if real_var.ndim>1:
#         #         ## reduce dimensions
#         #         cmd =  var+'= real_var.argmax(axis=1)'
#         #         # eval(cmd)
#         #         eval(var+'=') real_var.argymax(axis=1)
#         #         # exec(cmd)
#         #         cmd =f'print("argmax shape:",{var}.shape)' 
#         #         eval(cmd)
#         #         # exec(cmd)
        
        
        

#     print('---'*num_dashes)
#     print('\tCLASSIFICATION REPORT:')
#     print('---'*num_dashes)

#     # get both versions of classification report output
#     report_str = classification_report(y_test,y_hat_test)
#     report_dict = classification_report(y_test,y_hat_test,output_dict=True)
#     if report_as_df:
#         try:
#             ## Create and display classification report
#             # df_report =pd.DataFrame.from_dict(report_dict,orient='columns')#'index')#class_rows,orient='index')
#             df_report_temp = pd.DataFrame(report_dict)
#             df_report_temp = df_report_temp.T#reset_index(inplace=True)

#             df_report = df_report_temp[['precision','recall','f1-score','support']]
#             display(df_report.round(4).style.set_caption('Classification Report'))
#             print('\n')
        
#         except:
#             print(report_str)
#             # print(report_dict)
#             df_report = pd.DataFrame()
#     else:
#         print(report_str)

#     if save_summary:
#         with open(summary_filename,'w') as f:
#             model.summary(print_fn=lambda x: f.write(x+"\n"))
#             f.write(f"\nSaved at {time_suffix}\n")
#             f.write(report_str)

#     ## Create and plot confusion_matrix
#     conf_mat = confusion_matrix(y_test, y_hat_test)
#     mpl.rcParams['figure.figsize'] = conf_matrix_figsize
#     fig = plot_confusion_matrix(conf_mat,classes=conf_matrix_classes,
#                                    normalize=normalize_conf_matrix, fig_size=conf_matrix_figsize)
#     if save_conf_matrix_png:
#         fig.savefig(conf_mat_filename,facecolor='white', format='png', frameon=True)

#     if report_as_df:
#         return df_report, fig
#     else:
#         return report_str,fig

   

def my_custom_scorer(y_true,y_pred, model=None,method='sum', **kwargs):
    """My custom score function to use with sklearn's GridSearchCV
    - Method may be 'mean' or 'sum'
    
    - Maximizes the average accuracy per class using a normalized confusion matrix
    [i] Note: To use my_custom_scorer in GridSearch:
    >> from sklearn.metrics import make_scorer
    >> grid = GridSearch(estimator, parameter_grid)
    """
    from sklearn.metrics import make_scorer,confusion_matrix#, acc
    import numpy as np
    import functions_combined_BEST as ji    

    # set labels for confusion matrix
    labels = ['Decrease','No Change', 'Increase']

    
    ## If y_true is a multi-column one-hotted target
    if y_true.ndim>1 or y_pred.ndim>1:

        ## reduce dimensions of y_train and y_test
        if y_true.ndim>1:            
            y_true = y_true.argmax(axis=1)
            
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)

    
    ## Get confusion matrx
    cm = confusion_matrix(y_true, y_pred)

    ## Normalize confusion matrix
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

    ## Get diagonals for class accuracy
    diag = cm_norm.diagonal()
    
    print('\n')
    ## Choose Score Method
    if 'mean' in method:
        score = np.mean(diag)
        print(f'Mean Class Accuracy = {score}')

        
    elif 'sum' in method:
        score = np.sum(diag)
        print(f'Summed Class Accuracy = {score}')    
    
    ## Display results for user
    print('')
    print(f'Class Accuracy Values:')
    print(diag)    

    ## Plot confusion matrix
    ji.plot_confusion_matrix(cm,normalize=True,classes=labels);

    return score


# # Create HyperParaemeter Space
# params_to_search ={'filter_size':[2,4,6],
#                    'activation':['relu','tanh','linear'],
#                    'n_filters':[100,200],#,300,400],
#                   'dropout':[0.2,0.4],
#                   'optimizer':['adam','rmsprop','adadelta'],
#                 'epochs':[10]}

# def fit_gridsearch(build_fn,parameter_grid,X_train,y_train,score_fn=None,verbose=1,send_email=False):
#     """Builds a Keras model from build_fn, then wraps it in KerasClassifier 
#     for use with sklearn's GridSearchCV. Can score GridSearch with built-in 
#     metric from sklearn, or can pass a custom functions to be used with make_scorer().
#     Upon completion, emails best parameters to gmail account. 
    
#     Args:
#         build_fn (func): Build function for model with parameters to tune as arguments.
#         parameter_grid (dict): Dict of build_fn parameters (keys) and lists of parameters (values)
#         X_train, y_train (numpy array): training dataset
#         score_fn (func or str): Scoring function to use with GridSearchCV. 
#             - For builtin sklearn metrics, pass their name as a string.
#                 - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#             - For custom function, pass function itself. Function must accept, y_true,y_pred
#                 and must return a value to maximize. 
#             - Default(None)=ji.my_custom_scorer().
            
#     Returns:
#         model: (KerasClassifier) The return value. True for success, False otherwise.
#         results
#     """
#     from keras.wrappers.scikit_learn import KerasClassifier#, KerasRegressor
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.metrics import make_scorer
#     import pandas as pd
    
#     import functions_combined_BEST as ji
#     import bs_ds_local as bs
    

#     ## Wrap create_model with KerasClassifier
#     neural_network = KerasClassifier(build_fn=build_fn,verbose=verbose)
    
    
#     ## Run GridSearch
#     import types
#     if score_fn is None:
#         score_func = make_scorer(ji.my_custom_scorer)
#     elif isinstance(score_fn, types.FunctionType):
#         score_func = make_scorer(score_fn)
#     elif isinstance(score_fn, str):
#         score_func =  score_fn
        

#     grid = GridSearchCV(estimator=neural_network,param_grid=parameter_grid, 
#                         scoring=score_func)

#     ## Start Timer
#     tune_clock = bs.Clock()
#     tune_clock.tic()
    
#     ## Fit GridSearch
#     grid_result = grid.fit(X_train, y_train)
#     tune_clock.toc()

#     ## Print Best Params
#     best_params = grid_result.best_params_
#     print(best_params)

#     if send_email:
#         ## Send Email with completion time and best parameters found. 
#         time_completed = pd.datetime.now()
#         fmt = '%m/%d%Y-%T'
#         msg = f"GridSearch Completed at {time_completed.strftime(fmt)}\n GridSearchResults:\n{best_params}"
#         email_notification(msg=msg)
    
#     return grid_result


# class EncryptedPassword():
#     """Class that can be used to either provide a password to be encrypted or load an encypted password from file.
#     The string representations of the unencrypted password are shielded from displaying, when possible. 
    
#     - If encrypting a password, a key file and a password file will be saved to disk. 
#         - Default Key Filename: '..\\encryption_key.bin',
#         - Default Password Filename: '..\\encrypted_pwd.bin'
#     - If opening and decrypting key and password files, pass filenames during initialization. 
    
    
#     Example Usage:
#     >> # To Encrypt, with default folders:
#     >> my_pwd EncryptedPassword('my_password')
    
#     >> # To Encrypt With custom folders
#     >> my_pwd = EncryptedPassword('my_password',filename_for_key='..\folder_outside_repo\key.bin',
#                                     filename_for_password = '..\folder_outside_repo\key.bin')
                                    
                                    
#     >> # To open and decrypt files (from default folders):
#     >> my_pwd = EncryptedPassword(from_file=True)
    
#     >> # To open and decrypt files (from custom folders):
#     >> my_pwd = EncryptedPassword(from_file=True, 
#                                 filename_for_key='..\folder_outside_repo\key.bin',
#                                 filename_for_password = '..\folder_outside_repo\key.bin')
                                    
        
#     """
#     username = 'NOT PROVIDED'
    
#     @property
#     def password(self):
#         if hasattr(self,'_encrypted_password_'):
#             print('Encrypted Password:')
#             return self._encrypted_password_
#         else:
#             raise Exception('Password not yet encrypted.')
            
#     @password.setter
#     def password(self,password):
#         raise Exception('.password is read only.')
        
           
#     def __init__(self,username=None,password=None,from_file=False, encrypt=True,
#                  load_filenames_from_txt_file=None,
#                 filename_for_key='..\\encryption_key.bin',
#                 filename_for_password='..\\encrypted_pwd.bin',
#                 filename_for_username = '..\\encrypted_username.bin'):
        
#         if load_filenames_from_txt_file is None:
#             self.filename_for_key = filename_for_key
#             self.filename_for_password = filename_for_password
#             self.filename_for_username = filename_for_username
#         else:
#             print(f'Overriding input filenames for those saved in {load_filenames_from_txt_file}')
#             self.load_filenames_from_txt()
        
#         if username is not None:
#             self.username = username
        
#         if (password is None):
#             if (from_file==True):
#                 try:
#                     self.load_from_file(key_filename=filename_for_key,
#                                     password_filename=filename_for_password,
#                                         username_filename=filename_for_username)
#                 except:
#                     raise Exception('Something went wrong. Do the key and password files exist?')
                
#             else:
#                 raise Exception('Must either provide a password to encrypt, or set from_file=True')
            
#         else:
# #             _password_
#             self._password_ = password
#             if encrypt:
#                 self.encrypt_password()
                
#     def __repr__(self):
#         password = self._password_
#         msg = f'[i] Password is {len(password)} chars long.'
#         return msg

#     def __str__(self):
#         password = self._password_
#         msg = f'[i] Password is {len(password)} chars long.'
#         return msg        

    
#     def _get_password(self):
#         return self._password_
    
#     def load_filenames_from_txt(self,filename_file='password_filenames.txt'):
#         with open(filename_file,'r') as file:
#             filenames = file.read()
            
#         self.filename_for_key= filenames[0]
#         self.filename_for_username= filenames[1]
#         self.filename_for_password= filenames[2]
#         print(f'Filenames loaded and saved from {filename_file}')
            
    
#     def save_filenames_to_txt(self,filename_file='encrypted_password_filenames.txt'):
#         with open(filename_file,'w') as file:
#             file.write(self.filename_for_key)
#             file.write(self.filename_for_username)
#             file.write(self.filename_for_password)
#         print(f'Key/Password/Username filepaths saved to {filename_file}')
            
    
#     def load_from_file(self,key_filename,password_filename,
#                       username_filename):
    
#         from cryptography.fernet import Fernet
#         ## Load Key
#         with open(key_filename,'rb') as file:
#             for line in file:
#                 key = line


#         cipher_suite = Fernet(key)
#         self._cipher_suite_ = cipher_suite

#         ## Load password
#         with open(password_filename,'rb') as file:
#             for line in file:
#                 encryptedpwd = line
#         self._encrypted_password_ = encryptedpwd
        
#         ## Decrypt password
#         unciphered_text = (cipher_suite.decrypt(encryptedpwd))
#         plain_text_encrypted_password = bytes(unciphered_text).decode('utf-8')
#         self._password_ = plain_text_encrypted_password
        
#         ## Load username
#         with open(username_filename,'rb') as file:
#             for line in file:
#                 username = line
#         self.username = username
        
    
    
#     def encrypt_password(self, show_encrypted_password=False):
     
#         filename_for_key= self.filename_for_key
#         filename_for_password=self.filename_for_password
#         filename_for_username = self.filename_for_username

#         ## Import cryptography and generate encryption key
#         from cryptography.fernet import Fernet
#         key = Fernet.generate_key()
#         self._key_ = key

#         ## Create the cipher_suit for encrypting/decrypting
#         cipher_suite = Fernet(key)
#         self._cipher_suite_ = cipher_suite
 
        
#         ## Encrypt password
#         password = self._password_
#         text_to_encrypt = bytes(password,'utf-8')
#         ciphered_text = cipher_suite.encrypt(text_to_encrypt)#password goes here
#         self._encrypted_password_ = bytes(ciphered_text).decode('utf-8')
        
#         if show_encrypted_password:
#             print('Encrypyted Password:')
#             print(self._encrypted_password_)
        
        
#         ## Encrypt username
#         username = self.username
#         username_to_encrypt = bytes(username,'utf-8')
#         ciphered_username = cipher_suite.encrypt(username_to_encrypt)
#         self._encrypted_username_ = bytes(ciphered_username).decode('utf-8')
        

#         ## Test decryption
#         unciphered_text = cipher_suite.decrypt(ciphered_text)
#         unciphered_username = cipher_suite.decrypt(ciphered_username)
#         password_decoded = unciphered_text.decode('utf-8')
#         username_decoded = unciphered_username.decode('utf-8')
        
#         check_pwd = password_decoded==password
#         check_user = username_decoded==username
                    
#         if  check_pwd &check_user:
#             self._password_ = password_decoded 
#             print('[!] Make sure to delete typed password above from class instantiation.')
#         else:
#             raise Exception('Decrypted password and input password/username do not match. Something went wrong.')

#         ## Specify binary files (outside of repo) for storing key and password files
#         with open(filename_for_key,'wb') as file:
#             file.write(key)

#         with open(filename_for_password,'wb') as file:
#             file.write(ciphered_text)
            
#         with open(filename_for_username,'wb') as file:
#             file.write(ciphered_username)

#         print(f'[io] Encryption Key saved as {filename_for_key}')
#         print(f'[io] Encrypted Password saved as {filename_for_password}')
#         print(f'[io] Encrypted Username saved as {filename_for_username}')
        
        
class EncryptedPassword():
    """Class that can be used to either provide a password/username to be encrypted 
    OR to load a previously encypted password from file.    
    NOTE: Once you have encrypted your password and saved to bin files, you do not need to provide the password again. 
    Make sure to delete your password from the notebook after. 
    - If encrypting a password, a key file and a password file will be saved to disk. 
        - Default Key Filename: '..\\encryption_key.bin',
        - Default Password Filename: '..\\encrypted_pwd.bin'
        - Default Username Filename: '..\\encrypted_username.bin'
    
    The string representations of the unencrypted password are shielded from displaying, when possible. 
    


    - If opening and decrypting key and password files, pass filenames during initialization. 
    
    
    Example Usage:
    >> # To Encrypt, with default folders:
    >> my_pwd EncryptedPassword('my_password')
    
    >> # To Encrypt With custom folders
    >> my_pwd = EncryptedPassword('my_password',filename_for_key='..\folder_outside_repo\key.bin',
                                    filename_for_password = '..\folder_outside_repo\key.bin')
                                    
                                    
    >> # To open and decrypt files (from default folders):
    >> my_pwd = EncryptedPassword(from_file=True)
    
    >> # To open and decrypt files (from custom folders):
    >> my_pwd = EncryptedPassword(from_file=True, 
                                filename_for_key='..\folder_outside_repo\key.bin',
                                filename_for_password = '..\folder_outside_repo\key.bin')
                                    
        
    """
    
    ## Default username
    username = 'NOT PROVIDED'
    
    ## the .password property is designed so it will not display an unencrypted password. 
    @property ## password getter 
    def password(self):
        # if the encrypyted password already exists, print the encrypted pwd (unusable without key)
        if hasattr(self,'_encrypted_password_'):
            print('Encrypted Password:')
            return self._encrypted_password_
        else:
            raise Exception('Password not yet encrypted.')
    
    ## the .password property cannot be set by a user
    @password.setter ## password setter
    def password(self,password):
        raise Exception('.password is read only.')
        
               
    ## 
    def __init__(self,username=None,password=None,from_file=False, encrypt=True,
                filename_for_key='..\\encryption_key.bin',
                filename_for_password='..\\encrypted_pwd.bin',
                filename_for_username = '..\\encrypted_username.bin'):
        """Accepts either a username and password to encyrypt, 
        or loads a previously encrypyed password from file.
        
        Args:
            username (str): email username.
            password (str): email password (note: if have 2-factor authentication on email account, 
                will need app-specific password).
            from_file (bool): whether to load the user credentials from file
            encrypt (bool): whether to encrypt provided password. Default=True
            
            filename_for_key (str): filepath for key.bin (default is'..\\encryption_key.bin')
            filename_for_password: filepath for password.bin (default is'..\\encryption_pwd.bin')
            filename_for_username: filepath for username.bin (default is'..\\encrypted_username.bin')
            """
        
        ## Save filenames 
        self.filename_for_key = filename_for_key
        self.filename_for_password = filename_for_password
        self.filename_for_username = filename_for_username
        
        ## If user passed a username, set username
        if username is not None:
            self.username = username
        
        ## If no password is provided:
        if (password is None):
            
            ##  if load from file if `from_file`=True
            if (from_file==True):
                
                try: ## Load in the key, password, username files
                    self.load_from_file(key_filename=filename_for_key,
                                    password_filename=filename_for_password,
                                        username_filename=filename_for_username)
                except:
                    raise Exception('Something went wrong. Do the key and password files exist?')
            
            ## If no password provided, and from_file=False, raise error
            else:
                raise Exception('Must either provide a password to encrypt, or set from_file=True')
        
        
        ## If the user DOES provide a password
        else:
            self._password_ = password # set the private attribute for password
            
            ## Encrypt the password
            if encrypt:
                self.encrypt_password()
                
                
    def encrypt_password(self, show_encrypted_password=False):
        """Encrypt the key, username, and password and save to external files."""
         ## Get filenames to use.
        filename_for_key= self.filename_for_key
        filename_for_password=self.filename_for_password
        filename_for_username = self.filename_for_username

        ## Import cryptography and generate encryption key
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        self._key_ = key

        ## Create the cipher_suit from key for encrypting/decrypting
        cipher_suite = Fernet(key)
        self._cipher_suite_ = cipher_suite
 
        ## ENCRYPT PASSWORD
        # Get password and change to byte encoding
        password = self._password_
        password_to_encrypt = bytes(password,'utf-8') #password must be in bytes format
        
        # Use the encryption suite to encrypt the password and save to self
        ciphered_pwd = cipher_suite.encrypt(password_to_encrypt)
        self._encrypted_password_ = bytes(ciphered_pwd).decode('utf-8')
        
        # Print encrypyted password if true
        if show_encrypted_password:
            print('Encrypyted Password:')
            print(self._encrypted_password_)
        
        
        ## ENCRYPT USERNAME
        username = self.username
        username_to_encrypt = bytes(username,'utf-8')
        ciphered_username = cipher_suite.encrypt(username_to_encrypt)
        self._encrypted_username_ = bytes(ciphered_username).decode('utf-8')
        
        ## TEST DECRYPTION
        # decrypt password and username
        unciphered_pwd = cipher_suite.decrypt(ciphered_pwd)
        unciphered_username = cipher_suite.decrypt(ciphered_username)
        
        ## Decode from bytes to utf-8
        password_decoded = unciphered_pwd.decode('utf-8')
        username_decoded = unciphered_username.decode('utf-8')
        
        # Check if decoded text matches input text
        check_pwd = password_decoded==password
        check_user = username_decoded==username
        
        ## If everything matches, warn user to delete their exposed password
        if  check_pwd & check_user:
            self._password_ = password_decoded 
            print('[!] Make sure to delete typed password above from class instantiation.')
        else:
            raise Exception('Decrypted password and input password/username do not match. Something went wrong.')

        ## SAVE KEY, PASSWORD, AND USERNAME TO BIN FILES
        ## Specify binary files (outside of repo) for storing key and password files
        with open(filename_for_key,'wb') as file:
            file.write(key)

        with open(filename_for_password,'wb') as file:
            file.write(ciphered_pwd)
            
        with open(filename_for_username,'wb') as file:
            file.write(ciphered_username)

        # Display filepaths for user.
        print(f'[io] Encryption Key saved as {filename_for_key}')
        print(f'[io] Encrypted Password saved as {filename_for_password}')
        print(f'[io] Encrypted Username saved as {filename_for_username}')

            
    
    def load_from_file(self,key_filename,password_filename,
                      username_filename):
        """Load in the encrypted password from file. """
        
        from cryptography.fernet import Fernet
        
        ## Load Key 
        with open(key_filename,'rb') as file:
            for line in file:
                key = line

        ## Make ciphere suite from key
        cipher_suite = Fernet(key)
        self._cipher_suite_ = cipher_suite

        ## Load password
        with open(password_filename,'rb') as file:
            for line in file:
                encryptedpwd = line
        self._encrypted_password_ = encryptedpwd
        
        ## Decrypt password
        unciphered_text = (cipher_suite.decrypt(encryptedpwd))
        plain_text_encrypted_password = bytes(unciphered_text).decode('utf-8')
        self._password_ = plain_text_encrypted_password
        
        ## Load username
        with open(username_filename,'rb') as file:
            for line in file:
                username = line
        unciphered_username = (cipher_suite.decrypt(username))
        plan_text_username = bytes(unciphered_username).decode('utf-8')
        self.username = plan_text_username
        
    def __repr__(self):
        """Controls the printout when the object is the final command in a cell.
        i.e:
        >> pwd =EncrypytedPassword(username='me',password='secret')
        >> pwd
        """
        password = self._password_
        msg = f'[i] Password is {len(password)} chars long.'
        return msg

    def __str__(self):
        """Controls the printout when the object is printed.
        i.e:
        >> pwd =EncrypytedPassword(username='me',password='secret')
        >> print(pwd)
        """
        password = self._password_
        msg = f'[i] Password is {len(password)} chars long.'
        return msg 
    
            
def email_notification(password_obj=None,subject='GridSearch Finished',msg='The GridSearch is now complete.'):
    """Sends email notification from gmail account using prevouisly encrypyter password.
    Args:
        password_obj (EncryptedPassword object): EncryptedPassword object with username/password.
        subject (str):Text for subject line.
        msg (str): Text for body of email. 

    Returns:
        bool: The return value. True for success, False otherwise.

    Loads encrypted key and password from previously exported cipher from cryptography.fernet"""
    if password_obj is None:
        print('Must pass an EncrypytedPassword object.')
        print('>> pwd_obj = EncryptedPassword(username="my_username",password="my_password")')
        print('>> send_email(encrypted_password_obj=pwd_obj)')
        raise Exception('Must pass an EncryptedPassword.')
    
    import smtplib
    from email.message import EmailMessage
    from email.headerregistry import Address
    from email.utils import make_msgid
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    
    
    
    gmail_user = password_obj.username
    gmail_password = password_obj._password_
    
    ## WRITE EMAIL
    message = MIMEMultipart()
    message['Subject'] =subject
    message['To'] = gmail_user
    message['From']=gmail_user
    body = msg
    message.attach(MIMEText(body,'plain'))
    text_message = message.as_string()


    # emails end request
    try:
        with  smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
            server.login(gmail_user,gmail_password)
            server.sendmail(gmail_user,gmail_user, text_message)
            server.close()
            print('Email sent!')

        
    except Exception as e:
        print(e)
        print('Something went wrong')
        
        


## DEFINE FUNCTION FOR FITTING AND EVALUATING SINGLE MODELS (Not GridSearch)
def fit_and_eval_model(model, X_train, X_test, X_val, y_train, y_test, y_val,
                      epochs = 10, batch_size=100,plot_conf_mat_train=False, verbose=1,**kwargs):
    """Fits Kera's model with X_train, y_train data, using (X_val, y_val) for validation_data.
    - Then evaluates model with training/test data
    - Plots Keras Training History
    - Plots Confusion Matrix
    """
    import bs_ds_local as bs
    import functions_combined_BEST as ji
    if 'epochs' in kwargs:
        epochs=kwargs['epochs']
        
    if 'batch_size' in kwargs:
        batch_size=kwargs['batch_size']
    clock = bs.Clock()
    clock.tic()
    dashes = '---'*20
    print(f"{dashes}\n\tFITTING MODEL:\n{dashes}")
    
    

    history = model.fit(X_train, y_train, 
                          epochs=epochs,
                          verbose=verbose, 
                          validation_data=(X_val,y_val),#validation_split=validation_split,
                          batch_size=batch_size)#,
    #                       callbacks=callbacks)

    clock.toc()

    df_report,fig=evaluate_classification_model(model=model,
                                                       X_train=X_train, X_test=X_test,
                                                       y_train=y_train, y_test=y_test, history=history, 
                                                       binary_classes=False,
                                                       plot_training_conf_mat=plot_conf_mat_train,
                                                       conf_matrix_classes=['Decrease','No Change','Increase'])
    return model, df_report



def fit_gridsearch(build_fn,parameter_grid,X_train,y_train,score_fn=None,verbose=1,send_email=False,encrypted_password=None):
    """Builds a Keras model from build_fn, then wraps it in KerasClassifier 
    for use with sklearn's GridSearchCV. Can score GridSearch with built-in 
    metric from sklearn, or can pass a custom functions to be used with make_scorer().
    Upon completion, emails best parameters to gmail account. 
    
    Args:
        build_fn (func): Build function for model with parameters to tune as arguments.
        parameter_grid (dict): Dict of build_fn parameters (keys) and lists of parameters (values)
        X_train, y_train (numpy array): training dataset
        score_fn (func or str): Scoring function to use with GridSearchCV. 
            - For builtin sklearn metrics, pass their name as a string.
                - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            - For custom function, pass function itself. Function must accept, y_true,y_pred
                and must return a value to maximize. 
            - Default(None)=ji.my_custom_scorer().
            
    Returns:
        grid_result:The output from the grid.fit.
    """
    from keras.wrappers.scikit_learn import KerasClassifier#, KerasRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    import pandas as pd
    
    import functions_combined_BEST as ji
    import bs_ds_local as bs
    

    ## Wrap create_model with KerasClassifier
    neural_network = KerasClassifier(build_fn=build_fn,verbose=verbose)
    
    
    ## Run GridSearch
    import types
    if score_fn is None:
        score_func = make_scorer(ji.my_custom_scorer)
    elif isinstance(score_fn, types.FunctionType):
        score_func = make_scorer(score_fn)
    elif isinstance(score_fn, str):
        score_func =  score_fn
        

    grid = GridSearchCV(estimator=neural_network,param_grid=parameter_grid, 
                        scoring=score_func)

    ## Start Timer
    tune_clock = bs.Clock()
    tune_clock.tic()
    
    ## Fit GridSearch
    grid_result = grid.fit(X_train, y_train)
    tune_clock.toc()

    ## Print Best Params
    best_params = grid_result.best_params_
    print(best_params)

    if send_email:
        ## Send Email with completion time and best parameters found. 
        time_completed = pd.datetime.now()
        fmt = '%m/%d%Y-%T'
        msg = f"GridSearch Completed at {time_completed.strftime(fmt)}\n GridSearchResults:\n{best_params}"
        email_notification(password_obj=encrypted_password, msg=msg)
        
    
    return grid_result