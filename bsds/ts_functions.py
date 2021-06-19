## Lab Function
# from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as tsa
import statsmodels

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def adfuller_test_df(ts,index=['AD Fuller Results']):
    """Returns the AD Fuller Test Results and p-values for the null hypothesis
    that there the data is non-stationary (that there is a unit root in the data)"""
    
    df_res = tsa.stattools.adfuller(ts)

    names = ['Test Statistic','p-value','#Lags Used','# of Observations Used']
    res  = dict(zip(names,df_res[:4]))
    
    res['p<.05'] = res['p-value']<.05
    res['Stationary?'] = res['p<.05']
    
    if isinstance(index,str):
        index = [index]
    res_df = pd.DataFrame(res,index=index)
    res_df = res_df[['Test Statistic','#Lags Used',
                     '# of Observations Used','p-value','p<.05',
                    'Stationary?']]
    return res_df



def stationarity_check(TS,window=8,plot=True,index=['AD Fuller Results']):
    """Adapted from https://github.com/learn-co-curriculum/dsc-removing-trends-lab/tree/solution"""
    
    # Calculate rolling statistics
    roll_mean = TS.rolling(window=window, center=False).mean()
    roll_std = TS.rolling(window=window, center=False).std()
    
    # Perform the Dickey Fuller Test
    dftest = adfuller_test_df(TS,index=index)
    
    if plot:
        
        ## Building in contingency if not a series with a freq
        try: 
            freq = TS.index.freq
        except:
            freq = 'N/A'
            
        # Plot rolling statistics:
        fig = plt.figure(figsize=(12,6))
        plt.plot(TS, color='blue',label=f'Original (freq={freq}')
        plt.plot(roll_mean, color='red', label=f'Rolling Mean (window={window})')
        plt.plot(roll_std, color='gray', label = f'Rolling Std (window={window})')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        display(dftest)
        plt.show(block=False)
        
    return dftest
    
    
def plot_acf_pacf(ts,figsize=(9,6),lags=52,suptitle=None,sup_y = 1.01):
    """Plot pacf and acf using statsmodels"""
    fig,axes=plt.subplots(nrows=2,figsize=figsize)
    
    tsa.graphics.plot_acf(ts,ax=axes[0],lags=lags);
    tsa.graphics.plot_pacf(ts,ax=axes[1],lags=lags);
    
    ## Add grid
    [ax.grid(axis='x',which='both') for ax in axes]
    
    if suptitle is not None:
        fig.suptitle(suptitle,y=sup_y,fontweight='bold',fontsize=15)
        
    fig.tight_layout()
    return fig,axes


## funtionize diagnosing
def diagnose_model(model):
    """Takes a fit statsmodels model and displays the .summary 
    and plots the built-in plot.diagnostics()"""
    display(model.summary())
    model.plot_diagnostics()
    plt.tight_layout()
    
    
def get_df_from_pred(forecast_or_pred,forecast_label='Forecast'):
    """Takes a PredictionResultsWrapper from statsmodels
    extracts the confidence intervals and predicted mean and returns in a df"""
    forecast_df = forecast_or_pred.conf_int()
    forecast_df.columns = ['Lower CI','Upper CI']
    forecast_df[forecast_label] = forecast_or_pred.predicted_mean
    return forecast_df



def plot_forecast_from_df(*args, **kwargs):
    raise Exception("This function has been replaced with plot_forecast.")


def plot_forecast(forecast_or_model, future_steps=35,
                          ts_train=None, train_label='Training Data',
                          ts_test=None, test_label='True Test Data',
                          forecast_label='Forecast',
                          last_n_lags=52*35,figsize=(10,4)):
    """Takes a forecast from get_df_from_pred and optionally 
    the training/original time series.
    
    Plots the original ts, the predicted mean and the 
    confidence invtervals (using fill between)"""
    
    if isinstance(forecast_or_model,statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper):
        
        if ts_test is not None: 
            print('[i] Using length of ts_test instead of future_steps')
            future_steps = len(ts_test)
        forecast_df = get_forecast(forecast_or_model,steps=future_steps)
    else:#elif isinstance(forecast_or_model,pd.DataFrame):
        forecast_df = forecast_or_model.copy()
        
        
    fig,ax = plt.subplots(figsize=figsize)

    if ts_train is not None:
        ts_train.iloc[-last_n_lags:].plot(label=train_label)
        
    if ts_test is not None:
        ts_test.plot(label=test_label)
        
    forecast_df['Forecast'].plot(ax=ax,label=forecast_label)
    ax.fill_between(forecast_df.index,
                    forecast_df['Lower CI'], 
                    forecast_df['Upper CI'],color='g',alpha=0.3)
    ax.legend()
    ax.set(title=f'Forecasted {ts_train.name}')
    return fig,ax
        
        
        
### FORECAST SPECIFIC FUNCTIONS

def get_forecast(model,steps=12):
    pred = model.get_forecast(steps=steps)
    forecast = pred.conf_int()
    forecast.columns = ['Lower CI','Upper CI']
    forecast['Forecast'] = pred.predicted_mean
    return forecast

    
# def plot_forecast(model,ts,last_n_lags=35*4,future_steps=10):
#     forecast_df = get_forecast(model,steps=future_steps)

#     fig,ax = plt.subplots(figsize=(12,5))
#     ts.iloc[-last_n_lags:].plot(label='True Data')
#     forecast_df['Forecast'].plot(ax=ax)
#     ax.fill_between(forecast_df.index,
#                     forecast_df['Lower CI'], forecast_df['Upper CI'],alpha=0.6)
#     ax.legend()
#     ax.set(title=f'Forecasted {ts.name}')
#     return fig,ax


# def evaluate_model(model,ts,last_n_lags =52,steps=12):
#     display(model.summary())
#     model.plot_diagnostics();
#     fig,ax=plot_forecast(model,ts,future_steps=steps,last_n_lags=last_n_lags)
#     return fig,ax

def evaluate_model(model,ts,test_ts=None, last_n_lags =52,steps=12):
    diagnose_model(model)
    
    forecast = model.get_forecast(steps=steps)
    forecast_df = get_df_from_pred(forecast,)
    
    fig, ax = plot_forecast_from_df(forecast_df,ts_train=ts,
                                    last_n_lags=last_n_lags)
    
    if test_ts:
        test_ts.plot(ax=ax)
    return fig,ax


def get_one_step_ahead_pred(model,pred_steps=12):
    pred = model.get_prediction(start=-pred_steps)
    prediction = pred.conf_int()
    prediction.columns = ['Lower CI','Upper CI']
    prediction['One Step Ahead Forecast'] = pred.predicted_mean
    return prediction

def plot_one_step_ahead_prediction(prediction_or_model, pred_steps=35,
                          ts_train=None, train_label='Training Data',
                          forecast_label='Forecast',
                          last_n_lags=52*35,figsize=(10,6),plot_kws={'marker':'.'}):
    """Takes a forecast from get_df_from_pred and optionally 
    the training/original time series.
    
    Plots the original ts, the predicted mean and the 
    confidence invtervals (using fill between)"""
    
    if isinstance(prediction_or_model,statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper):
        
        # if ts_test is not None: 
        #     print('[i] Using length of ts_test instead of future_steps')
        #     pred_steps = len(ts_test)
        pred_df = get_one_step_ahead_pred(prediction_or_model,pred_steps=pred_steps)
    else:#elif isinstance(forecast_or_model,pd.DataFrame):
        pred_df = prediction_or_model.copy()
        
        
    fig,axes = plt.subplots(figsize=figsize,nrows=2)
    ax1,ax2 =axes
    if ts_train is not None:
        ts_train.iloc[-last_n_lags:].plot(label=train_label,ax=ax1,**plot_kws)
        
    # if ts_test is not None:
    #     ts_test.plot(label=test_label)
        
    pred_df['One Step Ahead Forecast'].plot(ax=ax1,label=forecast_label,**plot_kws)
    ax1.fill_between(pred_df.index,
                    pred_df['Lower CI'], 
                    pred_df['Upper CI'],color='g',alpha=0.3)
    ax1.legend()
    ax1.set(title=f'One-Step-Ahead Predictions for  {ts_train.name}')
    
    ##
    pred_df['True TS'] = ts_train.iloc[-pred_steps:]
    pred_df.reset_index(drop=False,inplace=True)
    
    pred_df[["One Step Ahead Forecast","True TS"]].plot(marker='o',ax=ax2)
    ax2.fill_between(pred_df.index,
                    pred_df['Lower CI'], 
                    pred_df['Upper CI'],color='g',alpha=0.3)
    ax2.set(ylabel='Values',xlabel='Time Step #')
    ax2.set_title('Comparing Just Predicted Time Steps')
    plt.tight_layout()
    
    return fig,axes
        