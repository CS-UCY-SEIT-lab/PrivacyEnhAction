import openpyxl
from openpyxl import load_workbook
import csv
import xlrd
import pandas as pd
import numpy as np



def process_file(filename):
    
    
    df = pd.read_csv(filename)
    df['result_time'] = pd.to_datetime(df['result_time'])   
    df['result_time'] = pd.to_datetime(df['result_time'].dt.strftime('%dd/%mm/%yyyy %HH:%mm:%ss'))
    
    filtered_df=df.dropna() 
    
    filtered_df=filtered_df.dropna()
    
    filtered_df['pir_cleaned'] =  np.where(filtered_df['pir'] == 0, 0,  
                                           np.where( filtered_df['pir'] ==1 ,np.where 
                                                    (( (filtered_df['pir'].shift(-6)==1)
                                                      | (filtered_df['pir'].shift(6)==1) 
                                                      |(filtered_df['pir'].shift(-5)==1)
                                                      | (filtered_df['pir'].shift(5)==1) 
                                                      | (filtered_df['pir'].shift(-4)==1)
                                                      | (filtered_df['pir'].shift(4)==1) 
                                                      | (filtered_df['pir'].shift(-3)==1) 
                                                      | (filtered_df['pir'].shift(3)==1)
                                                      | (filtered_df['pir'].shift(-2)==1)
                                                      | (filtered_df['pir'].shift(2)==1) 
                                                      | (filtered_df['pir'].shift(-1)==1) 
                                                      | (filtered_df['pir'].shift(1)==1)  )    ,1,0),1)) 
    
    
    df_uci = filtered_df 
    df_uci = df_uci.drop(['pir'], axis=1) 
    df_uci['result_time'] = pd.to_datetime(df_uci['result_time'])   
    df_uci['Hour'] = df_uci['result_time'].dt.hour 
    df_uci['result_time1'] = pd.to_datetime(df_uci['result_time']) 
    df_uci['Weekday1'] = df_uci['result_time'].dt.day_name()

    df_uci['Weekday'] = df_uci['result_time1'].dt.dayofweek


    df_uci['Weekday'] = np.where(( 
        (df_uci['Weekday'] == 0) | (df_uci['Weekday'] == 1)
        |  (df_uci['Weekday'] == 2) | (df_uci['Weekday'] == 3)  | (df_uci['Weekday'] == 4)), 1,0)


    df_uci = df_uci.drop(['Hour'],axis=1)

    df_uci = df_uci.set_index('result_time') 
    df_uci_hourly = df_uci.resample('H').sum()
    df_uci_hourly['hour'] = df_uci_hourly.index.hour
    df_uci_hourly.index = df_uci_hourly.index.date
    
    
    X = df_uci_hourly
    X=X[['pir_cleaned',  'Weekday',  'hour']].copy()

    X=X.loc[X['Weekday'] > 0]

    X =X.drop(['Weekday'], axis=1)
    
    return(X)

