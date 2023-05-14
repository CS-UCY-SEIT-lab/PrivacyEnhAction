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
import priv_functions 
import pandas as pd
import json
import time as time
from time import process_time

import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time
from time import process_time

def process_fitbit_dailyActivity(fileName):

    t2_start = process_time()   
    
    filename = os.path.join('uploads', fileName)
   

    fb_dataset = pd.read_csv(filename) 
    rowcount2 = len(fb_dataset.index)


    fb_dataset['ActivityDate'] = pd.to_datetime(fb_dataset['ActivityDate'])  
    fb_dataset['day'] = fb_dataset['ActivityDate'].dt.dayofweek

    sunday =  fb_dataset[fb_dataset.day == 6]
    monday =  fb_dataset[fb_dataset.day == 0]
    tuesday =  fb_dataset[fb_dataset.day == 1]
    wednesday =  fb_dataset[fb_dataset.day == 2]
    thursday =  fb_dataset[fb_dataset.day == 3]
    friday =  fb_dataset[fb_dataset.day == 4]
    saturd =  fb_dataset[fb_dataset.day == 5]

    fig, ax = plt.subplots(figsize=(5, 5))

    monday.set_index('ActivityDate').plot(marker='o',  y = 'TotalSteps',  title= 'Total steps - Mondays', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t1 = time.strftime("%Y%m%d-%H%M%S")

    n1 = 'monday'

    fullnamemon = n1+t1+'.png'
    plt.savefig('static/'+fullnamemon)  
    plt.close()  


    fig, ax = plt.subplots(figsize=(5, 5))

    wednesday.set_index('ActivityDate').plot(marker='o',  y = 'TotalSteps',  title= 'Total steps - Wednesdays', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'wednesday'

    fullnamewed = n2+t2+'.png'
    plt.savefig('static/'+fullnamewed)  
    plt.close()  



    fig, ax = plt.subplots(figsize=(5, 5))

    saturd.set_index('ActivityDate').plot(marker='o',  y = 'TotalSteps',  title= 'Total steps - Saturdays', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t3 = time.strftime("%Y%m%d-%H%M%S")

    n3 = 'saturday'

    fullnamesat = n3+t3+'.png'
    plt.savefig('static/'+fullnamesat)  
    plt.close()  

    sun_steps = sunday['TotalSteps'].sum()
    mon_steps = monday['TotalSteps'].sum()
    tue_steps = tuesday['TotalSteps'].sum()
    wed_steps = wednesday['TotalSteps'].sum()

    thu_steps = thursday['TotalSteps'].sum()
    fri_steps = friday['TotalSteps'].sum()
    sat_steps = saturd['TotalSteps'].sum()


    mean_all = (sun_steps + mon_steps + tue_steps + wed_steps + thu_steps + fri_steps )/ 6

    dif_of_sat = mean_all - sat_steps

    sat_km = sat_steps / 1250 
    avr_daily_km =  mean_all / 1250 

    if sat_km >=  avr_daily_km:
        jew_rel = 0 #negative
    else:
        jew_rel = 1 # could be positive

    
    t2_stop = process_time()

    elapsed_time2 =  t2_stop-t2_start

    return(rowcount2, elapsed_time2, jew_rel, fullnamemon, fullnamewed, fullnamesat)




def process_fitbit_sleepdata(fileName):
   

    
    t3_start = process_time()
    filename = os.path.join('uploads', fileName)
   

    sleep_data = pd.read_csv(filename) 
    rowcount3 = len(sleep_data.index)


    sleep_data['date'] = pd.to_datetime(sleep_data['date'], format="%m/%d/%Y %H:%M")

    mdslist = []
    mlslist = []
    mrslist = []

            
    # Aggregate data based on logid and value and then on min and max dates
    group1 = sleep_data.groupby(['logId', 'value'])['date'].count().reset_index(drop=False) 

    result = sleep_data.groupby('logId')['date'].agg(['min','max']).reset_index(drop=False) 


    # Find min and max of both value 
    result['sleepStartTime'] = result['min']
    result['sleepEndTime'] = result['max']


    result.drop('min', axis=1, inplace=True)
    result.drop('max', axis=1, inplace=True)

    result['sleepStartTime'] = pd.to_datetime(result['sleepStartTime'], format="%d/%m/%Y %H:%M")  
    result['sleepEndTime'] = pd.to_datetime(result['sleepEndTime'], format="%d/%m/%Y %H:%M")  

    result['MinutesInBed'] = result['sleepEndTime']-result['sleepStartTime']
    result['MinutesInBed'] = abs(result['MinutesInBed'] /np.timedelta64(1,'m'))


    max = len(group1)

    for index , row in result.iterrows():

        a = row.loc['logId']
    
        for index2 , row2 in group1.iterrows():

            b = row2.loc['logId']
            
            if a == b:

                val = row2.loc['value']
                
            
                if val == 1:
                    mls = row2.loc['date']
                    mlslist.append(mls)
                
            
                if val == 2:
                    mds = row2.loc['date']
                    mdslist.append(mds)
                if val == 3:
                    mrs = row2.loc['date']
                    mrslist.append(mrs)
        
        lmls = len(mlslist)
        lmds = len(mdslist)
        lmrs = len(mrslist)
                

                
        value = 0   
        if ( (lmls> lmrs) and (lmds > lmrs)     ):
            mrslist.append(value)
        elif  ( (lmls> lmds) and (lmrs > lmds)     ):
            mdslist.append(value)
        elif ( (lmls> lmds) and (lmls > lmrs)     ):
            mdslist.append(value)
            mrslist.append(value)
        elif ( (lmrs> lmls) and (lmrs > lmds)     ):
            mlslist.append(value)
            mdslist.append(value)
        elif ( (lmds> lmls) and (lmds > lmrs)     ):
        
            mlslist.append(value) 
            mrslist.append(value)
        elif ( (lmds> lmls) and (lmrs > lmls)     ):
            mlslist.append(value)
    
      
    result['mds'] = mdslist  
    result['mls'] = mlslist 
    result['mrs'] = mrslist 

    #How regular are your sleeping habits?

    sleepDesc = pd.DataFrame(result['MinutesInBed']/60).describe().transpose()
    avgSleepHours = round(sleepDesc.at['MinutesInBed','mean'],2)

    result['gotosleep_hour']=  result['sleepStartTime'].dt.strftime('%H:%M')
    result['wake_hour']=  result['sleepEndTime'].dt.strftime('%H:%M')

    result['sleepstarthour'] = result.sleepStartTime.dt.hour
    result['wakeuphour'] = result.sleepEndTime.dt.hour

    result['dateonly'] = result.sleepEndTime.dt.date


    result['day_of_week'] = result.sleepEndTime.dt.weekday

    #5 and 6 are weekend days 

    max = len(result)

    my_list = []

    for index, row in result.iterrows():
        if index  == 0:
            cur_rec_date = result.loc[index, 'dateonly']
            next_rec_date = result.loc[index+1, 'dateonly'] 

            if cur_rec_date == next_rec_date:
                wake_cur = result.loc[index, 'wakeuphour']
                sleep_time_next = result.loc[index+1, 'sleepstarthour']
                wake_next =result.loc[index+1, 'wakeuphour']

                difr = sleep_time_next - wake_cur

                if ( difr <= 3):
                    my_list.append(wake_next)
                    result.at[index, 'new'] = 'equal dates difr less than 3'
                    result.at[index+1, 'flag'] = 'no'
                    result.at[index, 'flag'] = 'yes' 
                else:
                    my_list.append(999)
                    result.at[index, 'new'] = 'equal dates dift bigger than 3'
                    result.at[index+1, 'flag'] = 'no'
                    result.at[index, 'flag'] = 'no'

            else:
                my_list.append(999)
                result.at[index, 'new'] = 'not equal dates'

                if index == 0:
                    result.at[index, 'flag'] = 'yes'
                
                    result.at[index+1, 'flag'] = 'yes'


        elif index != max-1:
            cur_rec_date = result.loc[index, 'dateonly']
            next_rec_date = result.loc[index+1, 'dateonly'] 
            cur_flag = result.loc[index, 'flag'] 

            if cur_flag == 'no':
                my_list.append(999)
            else:

                if cur_rec_date == next_rec_date:
                    wake_cur = result.loc[index, 'wakeuphour']
                    sleep_time_next = result.loc[index+1, 'sleepstarthour']
                    wake_next =result.loc[index+1, 'wakeuphour']
                
                    difr = sleep_time_next - wake_cur

                    if ( difr <= 3     ):
                        my_list.append(wake_next)
                        result.at[index, 'new'] = 'equal dates difr less than 3'
                        result.at[index+1, 'flag'] = 'no'
                        result.at[index, 'flag'] = 'yes'
                    else:
                        my_list.append(999)
                        result.at[index, 'new'] = 'equal dates dift bigger than 3'
                        result.at[index+1, 'flag'] = 'no'
                        result.at[index, 'flag'] = 'no'
                else:
                    my_list.append(999)
                    result.at[index, 'new'] = 'not equal dates'
                    if index == 1:
                        result.at[index, 'flag'] = 'yes'
                    
                        result.at[index+1, 'flag'] = 'yes'

        else:
            my_list.append(999)
            result.at[index, 'new'] = 'last record'
            result.at[index, 'flag'] = 'yes'

    
    result['list'] = my_list
    result2 = result[(result['flag'] == 'yes')   ]     

    result2 =   result2[(result2['MinutesInBed'] >= 180)]

    result2 =result2.squeeze()

    result2['sleepStartTime'] = pd.to_datetime(result2['sleepStartTime'], format="%d/%m/%Y %H:%M:%S")  
    result2['sleepEndTime'] = pd.to_datetime(result2['sleepEndTime'], format="%d/%m/%Y %H:%M:%S")  
   
    result2['sleepstarthour']  = np.where (result2['sleepstarthour'] == 0, 24,
                                       np.where( result2['sleepstarthour'] == 1, 25,
                                                np.where( result2['sleepstarthour'] == 2, 26,
                                                         np.where( result2['sleepstarthour'] == 3, 27,
                                                                  np.where( result2['sleepstarthour'] == 4, 28,
                                                                           np.where( result2['sleepstarthour'] == 5, 29,
                                                                                    np.where( result2['sleepstarthour'] == 6, 30,
                                       
                                      
                                       result2['sleepstarthour'] ) ))))))
 




    weekend =  result2[(result2['day_of_week'] == 5) | (result2['day_of_week'] == 6)] 
    weekday =  result2[(result2['day_of_week'] == 0) | (result2['day_of_week'] == 1)| (result2['day_of_week'] == 2) | (result2['day_of_week'] == 3) | (result2['day_of_week'] == 4)] 

   

    #How regular are your sleeping habits?

    sleepDesc = pd.DataFrame(result2['MinutesInBed']/60).describe().transpose()
    avgSleepHours = round(sleepDesc.at['MinutesInBed','mean'],2)

    #How regular are your sleeping habits in weekdays?

    sleepDesc_wdays = pd.DataFrame(weekday['MinutesInBed']/60).describe().transpose()
    avgSleepHours_wdays = round(sleepDesc_wdays.at['MinutesInBed','mean'],2)


    #How regular are your sleeping habits in weekends?

    sleepDesc_wends = pd.DataFrame(weekend['MinutesInBed']/60).describe().transpose()
    avgSleepHours_wends = round(sleepDesc_wends.at['MinutesInBed','mean'],2)


    avgsleepstarthour = round(result2['sleepstarthour'].mean(),1)
    avgsleepstarthour_wdays = round(weekday['sleepstarthour'].mean(),1)
    avgsleepstarthour_wends = round(weekend['sleepstarthour'].mean(),1)


    if avgsleepstarthour >= 24:
        avgsleepstarthour = round(avgsleepstarthour -24)

    if avgsleepstarthour_wdays >= 24:
        avgsleepstarthour_wdays = round(avgsleepstarthour_wdays -24)

    if avgsleepstarthour_wends >= 24:
        avgsleepstarthour_wends = round(avgsleepstarthour_wends -24)
        
     
    avgwakeuptime = round(result2['wakeuphour'].mean())
    avgwakeuptime_wdays = round(weekday['wakeuphour'].mean())
    avgwakeuptime_wends = round(weekend['wakeuphour'].mean())

   
    #averages in hours 
    avgMls = round(result2['mls'].mean()/60,2)

   

    avgMds = round(result2['mds'].mean()/60,2)

   
    avgMrs = round(result2['mrs'].mean()/60,2)

 

    avgminutessleep = round(result2['MinutesInBed'].mean()/60,2)


    percentage_of_deep_sleep = round((avgMds*100)/avgminutessleep,2)
    percentage_of_light_sleep = round((avgMls*100)/avgminutessleep,2)
    percentage_of_rem_sleep = round((avgMrs*100)/avgminutessleep,2)


    toplot = result
    fig, ax = plt.subplots(figsize=(5, 5))

    toplot['date']=  pd.to_datetime(toplot['sleepStartTime'], format="%m/%d/%Y")

    toplot['Hours of Sleep']=toplot['MinutesInBed']/60
            
    toplot.set_index('date').plot(marker='o',  y = 'Hours of Sleep',  title= 'Daily hours of sleep', ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    t1 = time.strftime("%Y%m%d-%H%M%S")

    n1 = 'hoursofsleep'

    fullnamehoursofsleep = n1+t1+'.png'
    plt.savefig('static/'+fullnamehoursofsleep)  
    plt.close()

    weekday1 = weekday

    weekday1.rename(columns={'dateonly': 'Date'}, inplace=True)


    weekday1.Date = pd.to_datetime(weekday1.Date)
    weekday1.set_index("Date", drop=True, inplace=True)
    weekday1.sort_index(inplace=True)

    weekday1.drop(columns=([
        "logId", 
        "sleepStartTime", 
        "sleepEndTime", 
        "gotosleep_hour", 
        "wake_hour", 
        "sleepstarthour", 
        "wakeuphour",
        "day_of_week",
        "new",
        "flag",
        "list"
    ]), inplace=True)


    weekend1 = weekend
    weekend1.rename(columns={'dateonly': 'Date'}, inplace=True)

    weekend1.Date = pd.to_datetime(weekend1.Date)
    weekend1.set_index("Date", drop=True, inplace=True)
    weekend1.sort_index(inplace=True)

    weekend1.drop(columns=([
        "logId", 
        "sleepStartTime", 
        "sleepEndTime", 
        "gotosleep_hour", 
        "wake_hour", 
        "sleepstarthour", 
        "wakeuphour",
        "day_of_week",
        "new",
        "flag",
        "list"
    ]), inplace=True)




    weekend1.rename(columns={'MinutesInBed': 'Minutes sleeping'}, inplace=True)

    weekend1.rename(columns={'mds': 'Minutes in Deep Sleep cycle' }, inplace=True)

    weekend1.rename(columns={'mls': 'Minutes in Light Sleep cycle'}, inplace=True)
    weekend1.rename(columns={'mrs': 'Minutes in REM Sleep cycle'}, inplace=True)

    weekday1.rename(columns={'MinutesInBed': 'Minutes sleeping'}, inplace=True)

    weekday1.rename(columns={'mds': 'Minutes in Deep Sleep cycle' }, inplace=True)

    weekday1.rename(columns={'mls': 'Minutes in Light Sleep cycle'}, inplace=True)
    weekday1.rename(columns={'mrs': 'Minutes in REM Sleep cycle'}, inplace=True)


    fig, ax = plt.subplots(figsize=(5, 5))

    weekend1.plot(marker='o',   title= 'Weekend Sleep Data Facts', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'weekendsleepfacts'

    fullnameweekendsleepfacts = n2+t2+'.png'
    plt.savefig('static/'+fullnameweekendsleepfacts)  
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
            
    weekday1.plot(marker='o',   title= 'Weekday Sleep Data Facts', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t3 = time.strftime("%Y%m%d-%H%M%S")

    n3 = 'weekdaysleepfacts'

    fullnameweekdaysleepfacts = n3+t3+'.png'
    plt.savefig('static/'+fullnameweekdaysleepfacts)  
    plt.close()
            

    # find length n dates of dataset used


    sleep_data['date'] = pd.to_datetime(sleep_data['date'], format="%m/%d/%Y %H:%M:%S")

    count = sleep_data.groupby(sleep_data['date'].dt.date, sort=False, as_index=False).size()

    days_in_data  = len(count)

    t3_stop = process_time()

    elapsed_time3 =  t3_stop-t3_start
    return(rowcount3,elapsed_time3, days_in_data, avgSleepHours,avgSleepHours_wdays, avgSleepHours_wends,
        avgsleepstarthour, avgsleepstarthour_wdays, avgsleepstarthour_wends, 
        avgwakeuptime, avgwakeuptime_wdays, avgwakeuptime_wends, percentage_of_deep_sleep, 
        percentage_of_light_sleep,percentage_of_rem_sleep, fullnamehoursofsleep , 
        fullnameweekdaysleepfacts,fullnameweekendsleepfacts )  




def process_fitbit_heartrate(fileName, avgsleepstarthour, avgwakeuptime):
   
    
    t1_start = process_time()   
    filename = os.path.join('uploads', fileName)
   
    
    hr_dataset = pd.read_csv(filename) 

    rowcount = len(hr_dataset.index)

    min_hr_date= min(hr_dataset['Value'])
    max_hr_date=max(hr_dataset['Value'])
    hr_dataset = hr_dataset.set_index('Time')
    hr_dataset=hr_dataset.reset_index()


    #Elevated heart rate
    #https://towardsdatascience.com/when-your-fitbit-says-your-heart-is-exploding-should-you-care-4e47aa5bf452
    # new df where heart rate is greater than 160
    maxhr = hr_dataset.loc[hr_dataset['Value'] > (100)]

   

    if (maxhr.empty):
        
        sum_max_hr = 0
        rowcount_highhr = 0
        fullnamehhr = 'nodataplot.png'
        

    else:
        
        
       
        maxhr['Timeonly']=pd.to_datetime(maxhr['Time'], format="%m/%d/%Y %H:%M:%S")
      #  maxhr.Timeonly = pd.to_datetime(maxhr.Timeonly).dt.time
        
        maxhrtoplot=maxhr
        maxhrtoplot.rename(columns={'Value': 'Heart Rate'}, inplace=True)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        maxhrtoplot.set_index('Timeonly').plot(marker='o',  y = 'Heart Rate',  title= 'High heart rate measurements over time', ax=ax)
        plt.xticks(rotation=30)
        plt.xticks(fontsize=6)

        timestr1 = time.strftime("%Y%m%d-%H%M%S")

        name1 = 'highheartrate'

        fullnamehhr = name1+timestr1+'.png'
        plt.savefig('static/'+fullnamehhr)  
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

        result = maxhr.groupby('group')['Timeonly'].agg(['max','min'])


        result['max'] = pd.to_datetime(result['max'])  
        result['min'] = pd.to_datetime(result['min'])  

        result['diff'] = result['max']-result['min']
        rowcount_highhr= len(result.index)

        

        sum_max_hr = result['diff'].sum()
        
            
        
                             
            
    #Low heart rate < 60

    lowhr = hr_dataset.loc[hr_dataset['Value']< 60]

    if (lowhr.empty):
       
        sum_low_hr = 0 
        rowcount_lowhr = 0
        fullnamelowhr = 'nodataplot.png'

    else:
        
        lowhr['Timeonly']=pd.to_datetime(lowhr['Time'], format="%m/%d/%Y %H:%M:%S")
      #  lowhr.Timeonly = pd.to_datetime(lowhr.Timeonly).dt.time


        lowhrtoplot=lowhr
        lowhrtoplot.rename(columns={'Value': 'Heart Rate'}, inplace=True)
          
          
        fig, ax = plt.subplots(figsize=(5, 5))
          
          
        lowhrtoplot.set_index('Timeonly').plot(marker='o',  y = 'Heart Rate',  title= 'Low  heart rate measurements over time', ax=ax)
        plt.xticks(rotation=30)
          
        plt.xticks(fontsize=6)

        timestr5 = time.strftime("%Y%m%d-%H%M%S")
        name5 = 'lowheartrate'

        fullnamelowhr = name5+timestr5+'.png'

        plt.savefig('static/'+fullnamelowhr)  

        plt.close()

        # make new column from the original df index
        lowhr = lowhr.assign(original_index = lowhr.index)
         
        # reset index to avoid copy warning
        lowhr = lowhr.reset_index(drop=True)

            
        # group consecutive time stamps together, start by measuring the difference between each index
        lowhr['change'] = (lowhr['original_index'] - lowhr['original_index'].shift(1) )
        # loop through df. If change increment is not 1, start a new group
        result3 = []
        grp3 = 0
        for value3 in lowhr['change']: 
          if value3 != 1: 
             grp3 += 1
             result3.append(grp3) 
          else: 
             result3.append(grp3)


        lowhr['group'] = result3

        result3 = lowhr.groupby('group')['Time'].agg(['max','min'])


       

        result3['max'] = pd.to_datetime(result3['max'])  
        result3['min'] = pd.to_datetime(result3['min'])  

        result3['diff'] = result3['max']-result3['min']


        sum_low_hr = result3['diff'].sum()

        rowcount_lowhr= len(result3.index)
        
 
    # find length n dates of dataset used

   

    hr_dataset['Time'] = pd.to_datetime(hr_dataset['Time'], format="%m/%d/%Y %H:%M:%S")



    count = hr_dataset.groupby(hr_dataset['Time'].dt.date, sort=False, as_index=False).size()

    days_in_data  = len(count)

    t1_stop = process_time()

    elapsed_time =  t1_stop-t1_start

    return(rowcount,elapsed_time, sum_max_hr, days_in_data,fullnamehhr,rowcount_highhr, sum_low_hr,rowcount_lowhr, fullnamelowhr)
