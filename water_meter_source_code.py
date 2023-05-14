import os 
from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import metrics


 


def process_water_file(mode,fileName):


  filename = os.path.join('uploads', fileName)


  mydata =   pd.read_csv(filename) 


  mydata['consumption'] = mydata['v1'] - mydata.shift(1)['v1']

  mydata = mydata.dropna()
  mydata['result_time'] = pd.to_datetime(mydata['result_time'])

  mydata['Occupancy'] = [1 if x >= 0.007 else 0 for x in mydata['consumption']] 
  mydata['result_time'] = pd.to_datetime(mydata['result_time'])
  mydata['Hour'] = mydata['result_time'].dt.hour 

  mydata['Morning'] = [1 if 10>=x>=6 else 0 for x in mydata['Hour']] 
  mydata['Afternoon'] = [1 if 17>=x>=12 else 0 for x in mydata['Hour']]
  mydata['Evening'] = [1 if 23>=x>=18 else 0 for x in mydata['Hour']]
  mydata['Night'] = [1 if 5>=x>=0 else 0 for x in mydata['Hour']]

  mydata['Weekday'] = mydata['result_time'].dt.dayofweek
  mydata['Weekend'] = mydata['result_time'].dt.dayofweek
  mydata['Weekday'] = np.where(mydata['Weekday'] == 0, 1, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 1, 1, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 2, 1, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 3, 1, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 4, 1, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 5, 0, mydata['Weekday'])
  mydata['Weekday'] = np.where(mydata['Weekday'] == 6, 0, mydata['Weekday'])

  mydata['Weekend'] = np.where(mydata['Weekend'] ==5, 1, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==6, 1, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==0, 0, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==1, 0, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==2, 0, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==3, 0, mydata['Weekend'])
  mydata['Weekend'] = np.where(mydata['Weekend'] ==4, 0, mydata['Weekend'])



  mydata=mydata.drop('Hour', axis = 1)
  mydata=mydata.drop('result_time', axis = 1)

  y=mydata.Occupancy
  x=mydata.drop('Occupancy',axis=1)


  X_train, X_test, y_train, y_test = train_test_split(x, y,
                            test_size=0.25, random_state=0)

  knn = KNeighborsClassifier(n_neighbors = 20) 

  knn.fit(X_train, y_train) 
  pred = knn.predict(X_test) 

  clf = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  result = (confusion_matrix(y_test, pred)) 

  accuracy= metrics.accuracy_score(y_test, pred)

  recall = metrics.recall_score(y_test, y_pred)

  f1_score = metrics.f1_score(y_test, y_pred)

  precision = metrics.precision_score(y_test, y_pred)

  acc_perc = accuracy * 100


  if accuracy >=0.85:
    occupancy_detection = "The occupancy status of your house can be infered and predicted from your smart water meter data with "+str(acc_perc)+" percent accuracy."
  else:
    occupancy_detection = "The occupancy status of your house cannot be infered or predicted from your smart water meter data."


  return (occupancy_detection)






def process_water_sleep_inf(mode,fileName):

  filename = os.path.join('uploads', fileName)

  water =   pd.read_csv(filename) 

  water['result_time'] = pd.to_datetime(water['result_time'], format="%d/%m/%Y %H:%M:%S")   
  water = water.dropna()
  water['consumption'] = water['v1'] - water.shift(1)['v1']
  uniquedays = len(pd.to_datetime(water['result_time']).dt.date.unique())
  water_hr = water
  water_hr = water_hr.drop(['v1'],axis=1)  
  water_hr['result_time'] = pd.to_datetime(water_hr['result_time'])   
  water_hr['dayname'] = water_hr['result_time'].dt.day_name()
  water_hr['day'] = water_hr['result_time'].dt.dayofweek

  water_hr['Weekday'] = np.where(( (water_hr['day'] == 0) |(water_hr['day'] == 1) |  (water_hr['day'] == 2) | (water_hr['day'] == 3) |(water_hr['day'] == 4)), 1,0)
  water_hr['Weekend'] = np.where(( (water_hr['day'] == 0) |(water_hr['day'] == 1) |  (water_hr['day'] == 2) | (water_hr['day'] == 3) |(water_hr['day'] == 4)), 0,1)

  water_hr['Hour'] = water_hr['result_time'].dt.hour 

  water_hr['Morning'] = [1 if 11>=x>=5 else 0 for x in water_hr['Hour']] 
  water_hr['Afternoon'] = [1 if 17>=x>=12 else 0 for x in water_hr['Hour']]
  water_hr['Evening'] = [1 if 23>=x>=18 else 0 for x in water_hr['Hour']]
  water_hr['Night'] = [1 if 4>=x>=0 else 0 for x in water_hr['Hour']]

  wdays = water_hr[water_hr.Weekday != 0]

  wends=  water_hr[water_hr.Weekend != 0]

  uniquedays_wd = len(pd.to_datetime(wdays['result_time']).dt.date.unique())

  uniquedays_we = len(pd.to_datetime(wends['result_time']).dt.date.unique())

  water_hr['date'] = water_hr['result_time']

  water_hr = water_hr.set_index('result_time') 
  water_hr_hourly = water_hr.resample('H').sum()
    
  water_hr_hourly['hour'] = water_hr_hourly.index.hour
  water_hr_hourly.index = water_hr_hourly.index.date

  water_hr_hourly = water_hr_hourly.drop(['day'],axis=1)
  weekday_water= water_hr_hourly[water_hr_hourly.Weekday != 0]
  weekend_water= water_hr_hourly[water_hr_hourly.Weekend != 0]

  weekday_water= weekday_water[weekday_water.consumption >= 0]
  weekend_water= weekend_water[weekend_water.consumption >= 0]

  weekday_water = weekday_water.drop(['Weekday'],axis=1)
  weekday_water = weekday_water.drop(['Weekend'],axis=1)

  weekend_water = weekend_water.drop(['Weekday'],axis=1)
  weekend_water = weekend_water.drop(['Weekend'],axis=1)
  weekend_sum = weekend_water.groupby('hour').sum()
  weekday_sum = weekday_water.groupby('hour').sum()

  weekend_sum['avg_con'] = round(weekend_sum['consumption']/uniquedays_we,2)
  weekday_sum['avg_con'] = round(weekday_sum['consumption']/uniquedays_wd,2)

  #find average wake up time during the week
  #inference 1: average wake up time and sleep time
   
  weekday_sum["time"] = weekday_sum.index.get_level_values(0).values  
  weekend_sum["time"] = weekend_sum.index.get_level_values(0).values  
  
  morn_weekday = weekday_sum[weekday_sum.Morning !=0]
  night_weekday = weekday_sum[weekday_sum.Night !=0]

  morn_weekend = weekend_sum[weekend_sum.Morning !=0]
  night_weekend = weekend_sum[weekend_sum.Night !=0]
  weekday_sum["avg_con_previous"] = weekday_sum["avg_con"].shift(1)
  weekday_sum["avg_con_next"] = weekday_sum["avg_con"].shift(-1)

  weekend_sum["avg_con_previous"] = weekend_sum["avg_con"].shift(1)
  weekend_sum["avg_con_next"] = weekend_sum["avg_con"].shift(-1)

  for index1, row1 in weekday_sum.iterrows():  

    prev_val = weekday_sum.loc[index1, 'avg_con_previous']
    next_val = weekday_sum.loc[index1, 'avg_con_next']
    cur_val = weekday_sum.loc[index1, 'avg_con']
     
     
    if index1 != 0:
       
      if cur_val >15:
            
        if ( (prev_val >15 ) and (next_val >15)):
          pass
        elif ( (prev_val >15 ) and (next_val <15)):
          pass
        else:
          wakeuptime_weekday = weekday_sum.loc[index1, 'time']
          break
      else:
        pass
        
    else:
      if cur_val > 15:

        if next_val >15:
          pass
        else:
          wakeuptime_weekday = weekday_sum.loc[index1, 'time']
          break
                     
       
    
    
  for index2, row2 in weekend_sum.iterrows(): 

    prev_val = weekend_sum.loc[index2, 'avg_con_previous']
    next_val = weekend_sum.loc[index2, 'avg_con_next']
    cur_val = weekend_sum.loc[index2, 'avg_con']
     
       
    if index2 != 0:

      if cur_val >15:

        if ( (prev_val >15 ) and (next_val >15)):
          pass
        elif ( (prev_val >15 ) and (next_val <15)):
          pass
        else:
          wakeuptime_weekend = weekend_sum.loc[index2, 'time']
          break
      else:
        pass
        
    else:

      if cur_val > 15:

        if next_val >15:
          pass
        else:
          wakeuptime_weekend = weekend_sum.loc[index2, 'time']
          break
                 
      
    



  for index3, row3 in weekday_sum.iterrows():

    prev_val = weekday_sum.loc[index3, 'avg_con_previous']
    next_val = weekday_sum.loc[index3, 'avg_con_next']
    cur_val = weekday_sum.loc[index3, 'avg_con']
    
    cur_time_ev = weekday_sum.loc[index3, 'Evening']
    cur_time_night = weekday_sum.loc[index3, 'Night']
    
          
    if index3 != 0:

      if ( (cur_val <=22) and ( ( cur_time_ev  >0) or (cur_time_night  >0)) ):
        if ( (prev_val<=22 ) and (next_val <=22) and ( ( cur_time_ev  >0  ) or (cur_time_night  >0)   )):
          pass
        elif ( (prev_val <=22 ) and (next_val >=22) and  ( ( cur_time_ev  >0  ) or (cur_time_night  >0)   )):
          pass
        elif ( (prev_val >=22 ) and (next_val >=22)and  ( ( cur_time_ev  >0  ) or (cur_time_night  >0)   )):
          pass
        else:
          sleeptime_weekday = weekday_sum.loc[index3, 'time']
          break
      else:
        pass
        
    else:
      if ( (cur_val <=22) and  ( ( cur_time_ev  >0) or (cur_time_night  >0))):
        if ((next_val <=22) and ( ( cur_time_ev  >0  ) or (cur_time_night  >0))):
          pass
        else:
          sleeptime_weekday = weekday_sum.loc[index3, 'time']
          break
                 
                 
       


  for index4, row4 in weekend_sum.iterrows():  

    prev_val = weekend_sum.loc[index4, 'avg_con_previous']
    next_val = weekend_sum.loc[index4, 'avg_con_next']
    cur_val = weekend_sum.loc[index4, 'avg_con']
    
    cur_time_ev = weekend_sum.loc[index4, 'Evening']
    cur_time_night = weekend_sum.loc[index4, 'Night']
    
     
       
    if index4 != 0:
      if ( (cur_val <=22) and ( ( cur_time_ev  >0) or (cur_time_night  >0)) ):
        if ( (prev_val<=22 ) and (next_val <=22) and (  ( cur_time_ev  >0  ) or (cur_time_night  >0) ) ):
          pass
        elif ( (prev_val <=22 ) and (next_val >=22) and (  ( cur_time_ev  >0  ) or (cur_time_night  >0) ) ):
          pass
        elif ( (prev_val >=22 ) and (next_val >=22)  and (  ( cur_time_ev  >0  ) or (cur_time_night  >0) ) ):
          pass
        else:
          sleeptime_weekend = weekend_sum.loc[index4, 'time']
          break
      else:
        pass
        
    else:
      if ( (cur_val <=22) and  ( ( cur_time_ev  >0) or (cur_time_night  >0))):
        if ((next_val <=22) and  ( ( cur_time_ev  >0  )  or (cur_time_night  >0))):
          pass
        else:
          #sleeptime_weekend = sleeptime_weekend.loc[index3, 'time']
          sleeptime_weekend = weekend_sum.loc[index3, 'time']
          break
                 
                 
 

#inference 2: nigh wake up 


  night_wakeup = water_hr[(water_hr['Hour'] == 0 ) | (water_hr['Hour'] ==1 ) | (water_hr['Hour'] ==2 ) | (water_hr['Hour'] ==3 ) | (water_hr['Hour'] ==4 )]
  night_wakeup = night_wakeup [(night_wakeup['consumption'] >= 20 ) ]
  night_wakeup.drop('dayname', inplace=True, axis=1)
  night_wakeup.drop('Weekday', inplace=True, axis=1)
  night_wakeup.drop('Weekend', inplace=True, axis=1)
  night_wakeup.drop('Morning', inplace=True, axis=1)
  night_wakeup.drop('Afternoon', inplace=True, axis=1)
  night_wakeup.drop('Evening', inplace=True, axis=1)
  night_wakeup.drop('Night', inplace=True, axis=1)
    
  uniquedays_nwu = len(pd.to_datetime(night_wakeup['date']).dt.date.unique())
  groupdata = night_wakeup.groupby(['Hour']).size().reset_index(name='Size')
  groupdata['freq'] = groupdata['Size']/uniquedays_nwu

  final_wp = groupdata[(groupdata['freq'] >= 0.25)]

  no_of_wakeup_times = len(final_wp)


  return ( wakeuptime_weekday, wakeuptime_weekend, sleeptime_weekday, sleeptime_weekend , no_of_wakeup_times)

    
 
