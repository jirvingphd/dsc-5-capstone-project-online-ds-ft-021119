# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:13:24 2019

@author: james
"""
import os, sys 
print(os.getcwd())
sys.path.append(os.getcwd())
from spyder_functions import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bs_ds as bs

# LOAD IN DATASET
ive_df = load_stock_df_from_csv(freq='CBH',verbose=0)
print(ive_df.index)

# ive_df=ive_df['2019':]
# display(ive_df.head())
plot_time_series(ive_df.filter(regex='Bid'))



# Calculate technical indicators 
stock_df = get_technical_indicators(ive_df)

# Remove timepoints without enough time periods for all indicators
na_idx = stock_df.loc[stock_df['upper_band'].isna() == True].index
stock_df = stock_df.loc[na_idx[-1]+1*na_idx.freq:]

print(stock_df.index)

# Check for null values
res = stock_df.isna().sum()
print(res[res>0])

plot_t