import os 
from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import priv_functions 
import pandas as pd
from sklearn.cluster import KMeans


 


def process_motion_file(mode,fileName):
  
    
    filename = os.path.join('uploads', fileName)
   
    df = pd.read_csv(filename)
    df['result_time'] = pd.to_datetime(df['result_time'])   

 
    
    filtered_df=df.dropna() 
    
      
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
    
    return X
    
 




def predict(X):

  test_data = X; 
  train_data = pd.read_csv('pir_train_wd.csv')
  

  clustering = KMeans(n_clusters=5, random_state=8675309)
  clustering.fit(train_data)

 
    
  # get centroids
    
    # apply the labels
  train_labels = clustering.labels_
  X_train_clstrs_wd = train_data.copy()
  X_train_clstrs_wd['clusters'] = train_labels
    
 # predict labels on the test set
  test_labels = clustering.predict(test_data)
 
  X_test_clstrs_wd = test_data.copy()
  X_test_clstrs_wd['clusters'] = test_labels
    
  centroids = clustering.cluster_centers_
  cen_x = [i[0] for i in centroids] 
  cen_y = [i[1] for i in centroids]


## print train
  X_train_clstrs_wd['cen_x'] = X_train_clstrs_wd.clusters.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2] , 3:cen_x[3], 4:cen_x[4]})
  X_train_clstrs_wd['cen_y'] = X_train_clstrs_wd.clusters.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2] , 3:cen_y[3], 4:cen_y[4]})
# define and map colors
  colors =  ['blue','red', 'green',  'yellow', 'pink']
  X_train_clstrs_wd['c'] = X_train_clstrs_wd.clusters.map({0:colors[0], 1:colors[1], 2:colors[2] , 3:colors[3], 4:colors[4]})

    
## print test
  X_test_clstrs_wd['cen_x'] = X_test_clstrs_wd.clusters.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2] , 3:cen_x[3], 4:cen_x[4]})
  X_test_clstrs_wd['cen_y'] = X_test_clstrs_wd.clusters.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2] , 3:cen_y[3], 4:cen_y[4]})
# define and map colors
  colors =  ['blue','red', 'green',  'yellow', 'pink']
  X_test_clstrs_wd['c'] = X_test_clstrs_wd.clusters.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]})


  greenY = X_test_clstrs_wd[X_test_clstrs_wd['c'] == 'green' ]
  yellowY=X_test_clstrs_wd[X_test_clstrs_wd['c'] == 'yellow' ]
  blueY=X_test_clstrs_wd[X_test_clstrs_wd['c'] == 'blue' ]
  redY=X_test_clstrs_wd[X_test_clstrs_wd['c'] == 'red' ]
  pinkY=X_test_clstrs_wd[X_test_clstrs_wd['c'] == 'pink' ]
  


  return (greenY, yellowY, blueY,redY,pinkY )



#inference 1: average wake up time
def pir_wr_inf1(re):
    
    
    inf1 = re.drop(['cen_x','cen_y','c'], axis=1)
    inf1['Night'] = [1 if 7>=x>=0 else 0 for x in inf1['hour']]
    inf1['Morning'] = [1 if 12>=x>=8 else 0 for x in inf1['hour']] 
    
    inf_night=inf1.loc[inf1['Night'] > 0]
 
    if len(inf_night) == 0:
       
       
        inf1['Morning'] = [1 if 12>=x>=8 else 0 for x in inf1['hour']] 
        inf_morning=inf1.loc[inf1['Morning'] > 0]
        ave_wake_up_time = inf_morning["hour"].mean() 
        ave_wake_up_time = round(ave_wake_up_time)
        return inf1, ave_wake_up_time
    else: 
        
        ave_wake_up_time = inf_night["hour"].mean()
        ave_wake_up_time = round(ave_wake_up_time)
        return  inf1, ave_wake_up_time

#inference 2: average sleep time
def pir_wr_inf2(bl):
    
    
    inf2 = bl.drop(['cen_x','cen_y','c'], axis=1)
    inf2['Morning'] = [1 if 12>=x>=8 else 0 for x in inf2['hour']] 
    inf2['Afternoon'] = [1 if 17>=x>=13 else 0 for x in inf2['hour']]
    inf2['Evening'] = [1 if 23>=x>=18 else 0 for x in inf2['hour']]
    inf2['Night'] = [1 if 7>=x>=0 else 0 for x in inf2['hour']]
    
    
    inf_sleep=inf2.loc[inf2['Evening'] > 0] 
  
    inf_sleep = inf_sleep.loc[inf_sleep['hour'] > 20 ]
    inf_sleep = inf_sleep.loc[inf_sleep['pir_cleaned'] <=20 ]
    ave_sleep_time = inf_sleep["hour"].mean()
    ave_sleep_time = round(ave_sleep_time)
    return  inf_sleep, ave_sleep_time

#inference 3: average time leave for work
def pir_wr_inf3(pi):
    
    
    inf3 = pi.drop(['cen_x','cen_y','c'], axis=1)
    
    inf_leave = inf3.loc[(inf3['hour'] >= 5) & (inf3['hour'] < 13)]
    
    ave_leave_time = inf_leave["hour"].mean()
    
    ave_leave_time = round(ave_leave_time)
    return  inf3, ave_leave_time


#inference 4: average time return home

def pir_wr_inf4(gr):
    
    
    inf4 = gr.drop(['cen_x','cen_y','c'], axis=1)
   
    inf_back = inf4.loc[(inf4['hour'] >= 13) & (inf4['hour'] <= 19)]
    
    
    inf_back = inf_back.loc[inf_back['pir_cleaned'] > 20 ] 
    


    ave_back_time = inf_back["hour"].mean()
    
    ave_back_time = round(ave_back_time)
    return  inf4, ave_back_time





#inference 5: Night wake up

def pir_wr_inf5(pi): 


  inf5 = pi.drop(['cen_x','cen_y','c'], axis=1)
 
  inf_wake = inf5.loc[(inf5['hour'] >= 0) & (inf5['hour'] <= 6)]
    
   
  inf_wake = inf_wake.loc[inf_wake['pir_cleaned'] > 0 ] 
  
  groupdata = inf_wake.groupby(['hour']).size()
  no_of_wakeup_times = len(groupdata)
  
  return  no_of_wakeup_times



