import os 
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 

import tensorflow as tf 
import numpy as np 

import pandas as pd
import json
import datetime
import math
import time
from time import process_time







def process_mi4c_activity_data(fileName1, fileName2, fileName3, fileName4):

    pr1_start = process_time() 


    filename1 = os.path.join('uploads', fileName1)
    filename2 = os.path.join('uploads', fileName2)
    filename3 = os.path.join('uploads', fileName3)
    filename4 = os.path.join('uploads', fileName4)
   
    
    activity_data = pd.read_csv(filename1) 
    pr1_rowcount = len(activity_data.index)


    activity_data = activity_data[(activity_data['calories'] > 50)]

    pregnancy_pos =0
    #Process for personal inferences - religion

    activity_data['date'] = pd.to_datetime(activity_data['date'])  
    activity_data['dayOfWeek'] = activity_data['date'].dt.dayofweek

    sunday =  activity_data[activity_data.dayOfWeek ==6]
    monday =  activity_data[activity_data.dayOfWeek == 0]
    tuesday =  activity_data[activity_data.dayOfWeek == 1]
    wednesday =  activity_data[activity_data.dayOfWeek ==2]
    thursday =  activity_data[activity_data.dayOfWeek == 3]
    friday =  activity_data[activity_data.dayOfWeek == 4]
    saturd =  activity_data[activity_data.dayOfWeek == 5]


    
    saturd.rename(columns={'steps': 'Saturday Steps'}, inplace=True)
    monday.rename(columns={'steps': 'Monday Steps'}, inplace=True)
    wednesday.rename(columns={'steps': 'Wednesday Steps'}, inplace=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    monday.set_index('date').plot(marker='x',  y = 'Monday Steps',  title= 'Steps', ax=ax)
    wednesday.set_index('date').plot(marker='o',  y = 'Wednesday Steps',   ax=ax)
    saturd.set_index('date').plot(marker='s',  y = 'Saturday Steps', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t1 = time.strftime("%Y%m%d-%H%M%S")

    n1 = 'misteps'

    fullnamemi = n1+t1+'.png'
    plt.savefig('static/'+fullnamemi)  
    plt.close()  


    



    sun_steps = sunday['steps'].sum()
    mon_steps = monday['Monday Steps'].sum()
    tue_steps = tuesday['steps'].sum()
    wed_steps = wednesday['Wednesday Steps'].sum()

    thu_steps = thursday['steps'].sum()
    fri_steps = friday['steps'].sum()
    sat_steps = saturd['Saturday Steps'].sum()

    mean_all = (sun_steps + mon_steps + tue_steps + wed_steps + thu_steps + fri_steps )/ 6


    dif_of_sat = mean_all - sat_steps

 

    sat_km = sat_steps / 1250 
    avr_daily_km =  mean_all / 1250 

    if sat_km >=  avr_daily_km:
         jew_rel = 0 #negative
    else:
        jew_rel = 1 # could be positive



        
    #Process for personal inferences - if general active or not  
        
    TotalSteps = activity_data['steps'].sum() 



    mean_steps = round((TotalSteps/7),2)

    if mean_steps < 5000:
        activity_level = 'sedentary lifestyle'
    elif ( (mean_steps >= 5000)  & ( mean_steps <= 7499)):
        activity_level = 'low active'
    elif ( (mean_steps >= 7500)  & ( mean_steps <= 9999)):
        activity_level = 'somewhat active'
    elif ( (mean_steps >= 10000)  & ( mean_steps < 12500)):
        activity_level = 'active'
    else:
        activity_level = 'highly active'





    activity_index=   {'activity level':  ['sedentary', 'low active', 'somewhat active', 'active','highly active'],
            'steps': [5000, 7499, 9999, 12500, 100000]}
    activity_index_df = pd.DataFrame(activity_index)

    fig, ax = plt.subplots(figsize=(5, 5))
            
    activity_index_df.set_index('activity level').plot.bar( title= 'Activity Index', ax=ax)
    ax.plot(activity_level,mean_steps , 'ro')
    plt.xticks(rotation=30)
    plt.xticks(fontsize=10)   

    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'miactivity'

    fullnamemiactivityindex = n2+t2+'.png'
    plt.savefig('static/'+fullnamemiactivityindex)  
    plt.close()  



    sleep_data = pd.read_csv(filename3) 
    hr_data = pd.read_csv(filename2) 
    user_data = pd.read_csv(filename4) 



    hr_data["date"] = pd.to_datetime(hr_data["date"])

    hr_data["time"] = pd.to_timedelta(hr_data["time"])
    hr_data["datehr"] = hr_data["date"] + hr_data["time"]
    hr_data = hr_data.drop('date', 1)
    hr_data = hr_data.drop('time', 1)

    sleep_data = sleep_data[(sleep_data['start'] != sleep_data['stop'])]


    sleep_data['start'] = pd.to_datetime(sleep_data['start'], unit='s')
    sleep_data['stop'] = pd.to_datetime(sleep_data['stop'], unit='s')

    user_data["birthday"] = pd.to_datetime(user_data["birthday"])

    user_data["age"] = user_data["birthday"].apply(lambda x : (pd.datetime.now().year - x.year))
    
   



    # For female users check pregnacy possibility
    
    hr_date_list = []
    hr_values_list = []

    for index, row in user_data.iterrows():
        gender = user_data.loc[index, 'gender'] 
        age = user_data.loc[index, 'age'] 

        if gender == 1:
            pregnancy_pos =2
            fullnameprpos = 'nodataplot.png'




        if gender == 0:

            for index1, row1 in sleep_data.iterrows():
                start_sleep_time = sleep_data.loc[index1, 'start'] 
                stop_sleep_time = sleep_data.loc[index1, 'stop'] 

                hr_rows = zip(hr_data["heartRate"], hr_data["datehr"])

                for index_hr, ((col1, col2)) in enumerate(hr_rows):

                    hr = col1
                    dt = col2 

                    if (dt >= start_sleep_time and dt <= stop_sleep_time ):
                        hrsp = dt
                        hrval = hr
                        hr_date_list.append(hrsp)
                        hr_values_list.append(hrval)
                    else:
                        pass 

            else:
                pass 

            final_hr = pd.DataFrame() 
            final_hr['heart_rate'] = hr_values_list
            final_hr['date']=hr_date_list

 
    
            final_hr['date']=pd.to_datetime(final_hr['date'])

            a = final_hr.groupby(final_hr.date.dt.date).mean()

    
            a_rows = a["heart_rate"]

            for index_a, cola in enumerate(a_rows):
                hra = cola        

                if (age >= 9 and age <= 55 ):
                    if (hra >90):
                        pregnancy_pos =1
                    else:
                        pregnancy_pos =0
                else:
                    pregnancy_pos =0


            fig, ax = plt.subplots(figsize=(5, 5))

            final_hr.set_index('date').plot(marker='o',   title= 'heart', ax=ax)
            plt.xticks(rotation=30)
            plt.xticks(fontsize=6)

            pr1 = time.strftime("%Y%m%d-%H%M%S")

            pr2 = 'pregn_pos'

            fullnameprpos = pr1+pr1+'.png'
            plt.savefig('static/'+fullnameprpos)  
            plt.close()  


      

    count = activity_data.groupby(activity_data['date'].dt.date, sort=False, as_index=False).size()

    days_in_data_pers  = len(count)




    pr1_stop = process_time()

    pr1_time =  pr1_stop-pr1_start

    
    return(pr1_rowcount,pr1_time, days_in_data_pers, jew_rel, activity_level, mean_steps , pregnancy_pos, fullnamemi, fullnamemiactivityindex, fullnameprpos)  





def process_mi4c_sleep_data(fileName):

   

    filename = os.path.join('uploads', fileName)

    
   
   
    
    sleep_data = pd.read_csv(filename) 

    sleep_data = sleep_data.drop('lastSyncTime', 1)
    

  


    sleep_data = sleep_data[(sleep_data['start'] != sleep_data['stop'])]


    sleep_data['start'] = pd.to_datetime(sleep_data['start'], unit='s')
    sleep_data['stop'] = pd.to_datetime(sleep_data['stop'], unit='s')


    sleep_data['MinutesInBed'] = sleep_data['stop']-sleep_data['start']
    sleep_data['MinutesInBed'] =abs(sleep_data['MinutesInBed'] /np.timedelta64(1,'m'))







    sleep_data['gotosleep_hour']=  sleep_data['start'].dt.strftime('%H:%M')
    sleep_data['wake_hour']=  sleep_data['stop'].dt.strftime('%H:%M')




    sleep_data['sleepstarthour'] = sleep_data.start.dt.hour
    sleep_data['sleepstartmin'] = sleep_data.stop.dt.minute


    sleep_data['wakeuphour'] = sleep_data.stop.dt.hour



    sleep_data['dateonly'] = sleep_data['date']


    sleep_data['day_of_week'] = sleep_data.stop.dt.weekday




    #How regular are your sleeping habits in weekday and weekends?
    sdweekend =  sleep_data[(sleep_data['day_of_week'] == 5) | (sleep_data['day_of_week'] == 6)] 
    sdweekday =  sleep_data[(sleep_data['day_of_week'] == 0) | (sleep_data['day_of_week'] == 1)| (sleep_data['day_of_week'] == 2)
                       | (sleep_data['day_of_week'] == 3) | (sleep_data['day_of_week'] == 4)] 




    avgSleepHours_wdays = round((sdweekday['MinutesInBed'].mean()/60),2)
   
   

    avgSleepHours_wends = round((sdweekend['MinutesInBed'].mean()/60),2)

    avgSleepHours_wends_up = math.ceil(avgSleepHours_wends)
   

    avgsleepstarthour_wdays = round(sdweekday['sleepstarthour'].mean(),1)
    avgsleepstarthour_wends = round(sdweekend['sleepstarthour'].mean(),1)

   
    avgwakeuptime_sleep = round(sleep_data['wakeuphour'].mean())
   


    avgwakeuptime_wdays_sleep = round(sdweekday['wakeuphour'].mean())
 

    avgwakeuptime_wends_sleep = round(sdweekend['wakeuphour'].mean(),2)
    avgwakeuptime_wends_sleep_up = math.ceil(avgwakeuptime_wends_sleep)

   

    #averages in hours 
    avgdeepsleeptime = round(sleep_data['deepSleepTime'].mean()/60,2)

    avgwaketime = round(sleep_data['wakeTime'].mean()/60,2)


    avgshallowsleeptime = round(sleep_data['shallowSleepTime'].mean()/60,2)

   


    avgminutessleep = round(sleep_data['MinutesInBed'].mean()/60,2)

  
    percentage_of_deep_sleep = round((avgdeepsleeptime*100)/avgminutessleep,2)
    percentage_of_shallow_sleep = round((avgshallowsleeptime*100)/avgminutessleep,2)
    percentage_of_waketime = round((avgwaketime*100)/avgminutessleep,2)



    wplot = sdweekday
    weplot = sdweekend
    fig, ax = plt.subplots(figsize=(5, 5))

    wplot['date']=  pd.to_datetime(wplot['sleepStartTime'], format="%m/%d/%Y")

    wplot['Hours of Sleep']=wplot['MinutesInBed']/60
            
    wplot.set_index('date').plot(marker='o',  y = 'Hours of Sleep',  title= 'Week daily hours of sleep', ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    d1 = time.strftime("%Y%m%d-%H%M%S")

    d2 = 'weekdayhoursofsleep'

    fullnamewdhoursofsleep = d1+d2+'.png'
    plt.savefig('static/'+fullnamewdhoursofsleep)  
    plt.close()

   


    return(avgSleepHours_wdays,  avgSleepHours_wends_up, avgwakeuptime_sleep, avgwakeuptime_wdays_sleep, 
        avgwakeuptime_wends_sleep_up, percentage_of_deep_sleep, percentage_of_shallow_sleep,
        percentage_of_waketime, fullnamewdhoursofsleep)  




def process_mi4c_sleep_data2(fileName):

    pr2_start = process_time() 


    

    filename = os.path.join('uploads', fileName)

   
   
    
    sleep_data = pd.read_csv(filename) 
    pr2_rowcount = len(sleep_data.index)

    sleep_data = sleep_data.drop('lastSyncTime', 1)
   

  


    sleep_data = sleep_data[(sleep_data['start'] != sleep_data['stop'])]


    sleep_data['start'] = pd.to_datetime(sleep_data['start'], unit='s')
    sleep_data['stop'] = pd.to_datetime(sleep_data['stop'], unit='s')


    sleep_data['MinutesInBed'] = sleep_data['stop']-sleep_data['start']
    sleep_data['MinutesInBed'] =abs(sleep_data['MinutesInBed'] /np.timedelta64(1,'m'))







    sleep_data['gotosleep_hour']=  sleep_data['start'].dt.strftime('%H:%M')
    sleep_data['wake_hour']=  sleep_data['stop'].dt.strftime('%H:%M')




    sleep_data['sleepstarthour'] = sleep_data.start.dt.hour
    sleep_data['sleepstartmin'] = sleep_data.stop.dt.minute


    sleep_data['wakeuphour'] = sleep_data.stop.dt.hour



    sleep_data['dateonly'] = sleep_data['date']


    sleep_data['day_of_week'] = sleep_data.stop.dt.weekday




    #How regular are your sleeping habits in weekday and weekends?
    sdweekend =  sleep_data[(sleep_data['day_of_week'] == 5) | (sleep_data['day_of_week'] == 6)] 
    sdweekday =  sleep_data[(sleep_data['day_of_week'] == 0) | (sleep_data['day_of_week'] == 1)| (sleep_data['day_of_week'] == 2)
                       | (sleep_data['day_of_week'] == 3) | (sleep_data['day_of_week'] == 4)] 



    
    sleepwd = sdweekday
    sleepwe = sdweekend


    fig, ax = plt.subplots(figsize=(5, 5))

    sleepwd['date']=  pd.to_datetime(sleepwd['start'], format="%m/%d/%Y")
    sleepwe['date']=  pd.to_datetime(sleepwe['start'], format="%m/%d/%Y")

   
            
    sleepwd.set_index('date').plot(marker='o',  y = 'sleepstarthour',  title= 'Go to sleep and wake up hour', ax=ax)
    sleepwe.set_index('date').plot(marker='o',  y = 'wakeuphour',   ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    e1 = time.strftime("%Y%m%d-%H%M%S")

    e2 = 'sleepstarthour'

    fullnamesleepstarthour = e1+e2+'.png'
    plt.savefig('static/'+fullnamesleepstarthour)  
    plt.close()





    avgSleepHours_wdays = round((sdweekday['MinutesInBed'].mean()/60),2)
   
    
  
    avgSleepHours_wends = round((sdweekend['MinutesInBed'].mean()/60),2)

    avgSleepHours_wends_up = math.ceil(avgSleepHours_wends)
   

    sdweekday['sleepstarthour']  = np.where (sdweekday['sleepstarthour'] == 0, 24,
     np.where( sdweekday['sleepstarthour'] == 1, 25, np.where( sdweekday['sleepstarthour'] == 2, 26, 
        np.where( sdweekday['sleepstarthour'] == 3, 27,np.where( sdweekday['sleepstarthour'] == 4, 28,
            np.where( sdweekday['sleepstarthour'] == 5, 29, np.where( sdweekday['sleepstarthour'] == 6, 30, sdweekday['sleepstarthour'] ) ))))))


    sdweekend['sleepstarthour']  = np.where (sdweekend['sleepstarthour'] == 0, 24,
        np.where( sdweekend['sleepstarthour'] == 1, 25, np.where( sdweekend['sleepstarthour'] == 2, 26, np.where( sdweekend['sleepstarthour'] == 3, 27,
            np.where( sdweekend['sleepstarthour'] == 4, 28, np.where( sdweekend['sleepstarthour'] == 5, 29,
                np.where( sdweekend['sleepstarthour'] == 6, 30,sdweekend['sleepstarthour'] ) ))))))





    avgsleepstarthour_wdays = round(sdweekday['sleepstarthour'].mean(),1)
    avgsleepstarthour_wends = round(sdweekend['sleepstarthour'].mean(),1)

    if avgsleepstarthour_wdays >= 24:
        avgsleepstarthour_wdays = round(avgsleepstarthour_wdays - 24)

    if avgsleepstarthour_wends >= 24:
        avgsleepstarthour_wends = round(avgsleepstarthour_wends -24)
    
   
    avgwakeuptime_sleep = round(sleep_data['wakeuphour'].mean()) 

    avgwakeuptime_wdays_sleep = round(sdweekday['wakeuphour'].mean())

 

    avgwakeuptime_wends_sleep = round(sdweekend['wakeuphour'].mean(),2)
    avgwakeuptime_wends_sleep_up = math.ceil(avgwakeuptime_wends_sleep)

 

     #averages in hours 
    avgdeepsleeptime = round(sleep_data['deepSleepTime'].mean()/60,2)

    avgwaketime = round(sleep_data['wakeTime'].mean()/60,2)

    avgshallowsleeptime = round(sleep_data['shallowSleepTime'].mean()/60,2)
    avgminutessleep = round(sleep_data['MinutesInBed'].mean()/60,2)
    percentage_of_deep_sleep = round((avgdeepsleeptime*100)/avgminutessleep,2)
    percentage_of_shallow_sleep = round((avgshallowsleeptime*100)/avgminutessleep,2)
    percentage_of_waketime = round((avgwaketime*100)/avgminutessleep,2)

    wplot = sdweekday
    welot = sdweekend


    fig, ax = plt.subplots(figsize=(5, 5))

    wplot['date']=  pd.to_datetime(wplot['start'], format="%m/%d/%Y")
    welot['date']=  pd.to_datetime(welot['start'], format="%m/%d/%Y")

    wplot['Hours of Sleep']=wplot['MinutesInBed']/60
    welot['Hours of Sleep']=welot['MinutesInBed']/60


    wplot.rename(columns={'Hours of Sleep': 'Weekday Hours of Sleep'}, inplace=True)
    welot.rename(columns={'Hours of Sleep': 'Weekend Hours of Sleep'}, inplace=True)
            
    wplot.set_index('date').plot(marker='o',  y = 'Weekday Hours of Sleep',  title= 'Daily hours of sleep', ax=ax)
    welot.set_index('date').plot(marker='o',  y = 'Weekend Hours of Sleep',   ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    d1 = time.strftime("%Y%m%d-%H%M%S")

    d2 = 'weekdayhoursofsleep'

    fullnamewdhoursofsleep = d1+d2+'.png'
    plt.savefig('static/'+fullnamewdhoursofsleep)  
    plt.close()




    fig, ax = plt.subplots(figsize=(5, 5))

   

    wplot.rename(columns={'shallowSleepTime': 'Weekday light sleep'}, inplace=True)
    welot.rename(columns={'shallowSleepTime': 'Weekend light sleep'}, inplace=True)
    wplot.rename(columns={'deepSleepTime': 'Weekday deep sleep'}, inplace=True)
    welot.rename(columns={'deepSleepTime': 'Weekend deep sleep'}, inplace=True)
    wplot.rename(columns={'wakeTime': 'Weekday REM sleep'}, inplace=True)
    welot.rename(columns={'wakeTime': 'Weekend REM sleep'}, inplace=True)
            
    wplot.set_index('date').plot(marker='o',  y = 'Weekday light sleep',  title= 'Weekday and weekend sleep facts', ax=ax)
    wplot.set_index('date').plot(marker='o',  y = 'Weekday deep sleep',   ax=ax)
    wplot.set_index('date').plot(marker='o',  y = 'Weekday REM sleep',   ax=ax)
    welot.set_index('date').plot(marker='o',  y = 'Weekend light sleep',   ax=ax)
    welot.set_index('date').plot(marker='o',  y = 'Weekend deep sleep',   ax=ax)
    welot.set_index('date').plot(marker='o',  y = 'Weekend REM sleep',   ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    f1 = time.strftime("%Y%m%d-%H%M%S")

    f2 = 'sleepfactsmi'

    fullnamesleepfactsmi = f1+f2+'.png'
    plt.savefig('static/'+fullnamesleepfactsmi)  
    plt.close()



   # find length n dates of dataset used

    count = sleep_data.groupby(sleep_data['start'].dt.date, sort=False, as_index=False).size()

    days_in_data  = len(count)


    pr2_stop = process_time()

    pr2_time =  pr2_stop-pr2_start



    return(pr2_rowcount, pr2_time,days_in_data, avgSleepHours_wdays, avgSleepHours_wends_up,avgsleepstarthour_wdays, avgsleepstarthour_wends, avgwakeuptime_wdays_sleep,
        avgwakeuptime_wends_sleep_up,percentage_of_deep_sleep, percentage_of_shallow_sleep, 
        percentage_of_waketime,fullnamewdhoursofsleep, fullnamesleepstarthour, fullnamesleepfactsmi)  




def process_mi4c_heart_rate_data(fileName):


    pr3_start = process_time()
    

    filename = os.path.join('uploads', fileName)
   
    
    heartrate_auto_data = pd.read_csv(filename) 
    pr3_rowcount = len(heartrate_auto_data.index)


    heartrate_auto_data["date"] = pd.to_datetime(heartrate_auto_data["date"])
    heartrate_auto_data["time"] = pd.to_timedelta(heartrate_auto_data["time"])




    heartrate_auto_data["datetime"] = heartrate_auto_data["date"] + heartrate_auto_data["time"]

    min_hr_date= min(heartrate_auto_data['datetime'])
    max_hr_date=max(heartrate_auto_data['datetime'])



    heartrate_auto_data = heartrate_auto_data.set_index('datetime')
     

    heartrate_auto_data=heartrate_auto_data.reset_index()

       
    #Elevated heart rate
    #https://towardsdatascience.com/when-your-fitbit-says-your-heart-is-exploding-should-you-care-4e47aa5bf452
    # new df where heart rate is greater than 160

    maxhr = heartrate_auto_data.loc[heartrate_auto_data['heartRate'] > (160)]

    if (maxhr.empty):
       
        sum_max_hr = 0  
        fullnamemihighhr = 'nodataplot.png'
        

    else:


        maxhrtoplot=maxhr
       
        fig, ax = plt.subplots(figsize=(5, 5))

        maxhrtoplot.set_index('date').plot(marker='o',  y = 'heartRate',  title= 'High heart rate measurements over time', ax=ax)
        plt.xticks(rotation=30)
        plt.xticks(fontsize=6)

        s1 = time.strftime("%Y%m%d-%H%M%S")

        s2 = 'mihighheartrate'

        fullnamemihighhr = s1+s2+'.png'
        plt.savefig('static/'+fullnamemihighhr)  
        plt.close()
                
              
        # make new column from the original df index
        maxhr = maxhr.assign(original_index = maxhr.index)
             
        # reset index to avoid copy warning
        maxhr = maxhr.reset_index(drop=True)
             
        # group consecutive time stamps together, start by measuring the difference between each index
        maxhr['change'] = (maxhr['original_index'] - maxhr['original_index'].shift(1) )
        # loop through df. If change increment is not 1, start a new group


        result = []
        grp = 0

        for value in maxhr['change']: 
            if value != 1: 
                grp += 1
                result.append(grp) 
            else: 
                result.append(grp)


        maxhr['group'] = result

        result = maxhr.groupby('group')['datetime'].agg(['max','min'])


           

        result['max'] = pd.to_datetime(result['max'])  
        result['min'] = pd.to_datetime(result['min'])  

        result['diff'] = result['max']-result['min']
        sum_max_hr = result['diff'].sum()

    


     #Low heart rate < 60

    lowhr = heartrate_auto_data.loc[heartrate_auto_data['heartRate']< 60]



    if (lowhr.empty):
       
        sum_low_hr = 0 
        fullnamemilowghhr = 'nodataplot.png'
        

    else:


        lowhrtoplot=lowhr
     
        fig, ax = plt.subplots(figsize=(5, 5))

        lowhrtoplot.set_index('date').plot(marker='o',  y = 'heartRate',  title= 'Low heart rate measurements over time', ax=ax)
        plt.xticks(rotation=30)
        plt.xticks(fontsize=6)

        m1 = time.strftime("%Y%m%d-%H%M%S")

        m2 = 'milowheartrate'

        fullnamemilowhr = m1+m2+'.png'
        plt.savefig('static/'+fullnamemilowhr)  
        plt.close()
              
        # make new column from the original df index
        lowhr = lowhr.assign(original_index = lowhr.index)
             
        # reset index to avoid copy warning
        lowhr = lowhr.reset_index(drop=True)
             
         # group consecutive time stamps together, start by measuring the difference between each index
        lowhr['change'] = (lowhr['original_index'] - lowhr['original_index'].shift(1) )
            
        # loop through df. If change increment is not 1, start a new group
            
        result2 = []
            
        grp2 = 0
            
        for value2 in lowhr['change']: 
              
            if value2 != 1: 
                 
                grp2 += 1
                result2.append(grp2) 
            else: 
                 
                result2.append(grp2)


        lowhr['group'] = result2

        result2 = lowhr.groupby('group')['datetime'].agg(['max','min'])


           

            
        result2['max'] = pd.to_datetime(result2['max'])  
            
        result2['min'] = pd.to_datetime(result2['min'])  

            
        result2['diff'] = result2['max']-result2['min']




           
        

           
        sum_low_hr = result2['diff'].sum()


        # find length n dates of dataset used

   

    heartrate_auto_data['Time'] = pd.to_datetime(heartrate_auto_data['datetime'], format="%m/%d/%Y %H:%M:%S")



    count = heartrate_auto_data.groupby(heartrate_auto_data['datetime'].dt.date, sort=False, as_index=False).size()

    days_in_data  = len(count)



    pr3_stop = process_time()

    pr3_time =  pr3_stop-pr3_start

    return( pr3_rowcount, pr3_time,sum_max_hr, sum_low_hr, days_in_data, fullnamemihighhr, fullnamemilowhr)  





