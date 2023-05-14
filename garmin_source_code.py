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
import googlemaps

import tensorflow as tf 
import numpy as np 
import priv_functions 
import pandas as pd
import json
import time as time
import time
from time import process_time


def process_garmin_location_inf(fileName):

    g1_start = process_time()  
    
    
       
    filename = os.path.join('uploads', fileName)
   

    df = pd.read_csv(filename)

    rowcount1g = len(df.index)

    activities = df.filter([ 'name','beginTimestamp','activityType',
                        'duration', 'distance',
                       
                       'startLatitude', 'startLongitude'
                        ], axis=1)





    activities['beginTimestamp'] = pd.to_datetime(activities['beginTimestamp'], unit='ms')

    activities['beginTimestamp'] = pd.to_datetime(activities['beginTimestamp'], format="%Y/%m/%d %H:%M:%S") 

 
    
    fig, ax = plt.subplots(figsize=(5, 5))

    activities.activityType.value_counts().plot(kind = 'pie')
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=0)
    plt.yticks(fontsize=5)

    t1 = time.strftime("%Y%m%d-%H%M%S")

    n1 = 'garmin_activities'

    fullnamegarminact = n1+t1+'.png'
    plt.savefig('static/'+fullnamegarminact)  
    plt.close()  

    

    most_common_activity = activities['activityType']. value_counts(). idxmax()
    
    cm_act_t = (activities[activities.activityType == most_common_activity])

    from geopy.geocoders import Nominatim

    # initialize Nominatim API 
    #geolocator = Nominatim(user_agent="geoapiExercises")
    geolocator = Nominatim(user_agent="alexia.kounoudes@gmail.com")

    activity_location =  cm_act_t.dropna()

    val='not available'
    buildings = []
    amenities = []

    roads = []
    suburbs = []

    offices = []
    house_numbers = []
    towns = []
    municipalities = []
    cities = []
    state_districts = []
    states= []
    postcodes = []
    countries = []
    

    for index, row in activity_location.iterrows():

      
    
        Latitude = row.loc['startLatitude']
        Longitude = row.loc['startLongitude'] 
        
             
        latitudestr = str(Latitude)
        longitudestr = str(Longitude)
        location = geolocator.reverse(latitudestr+","+longitudestr)
     
        address = location.raw['address']
        
        building = address.get('building', '')
        amenity = address.get('amenity', '')
     
        road = address.get('road', '')
        suburb = address.get('suburb', '')
       
        office = address.get('office', '')
        house_number = address.get('house_number', '')
        town = address.get('town', '') 
        municipality = address.get('municipality', '')
        city = address.get('city', '')
        state_district = address.get('state_district', '')
        
        postcode = address.get('postcode', '')
        country = address.get('country', '')
        
        
        if building != '':
             buildings.append(building)         
        else:        
            buildings.append(val)
            
        if amenity != '':
             amenities.append(amenity)      
        else:        
            amenities.append(val) 
       
        if road != '':
             roads.append(road)         
        else:        
            roads.append(val)
        if suburb != '':
             suburbs.append(suburb)         
        else:        
            suburbs.append(val)
       
        if office != '':
             offices.append(office)
        else:
            offices.append(val)
        if house_number != '':
             house_numbers.append(house_number)
        else:
            house_numbers.append(val)
        if town != '':
             towns.append(town)
        else:
            towns.append(val)
        if municipality != '':
             municipalities.append(municipality)
        else:
            municipalities.append(val)
        if city != '':
             cities.append(city)
        else:
            cities.append(val)
        if state_district != '':
             state_districts.append(state_district)
        else:
            state_districts.append(val)
        
        if postcode != '':
             postcodes.append(postcode)
        else:
            postcodes.append(val)
        if country != '':
             countries.append(country)
        else:
            countries.append(val)
        
            
            
     

    activity_location['building'] = buildings
    activity_location['amenity'] = amenities

    activity_location['road'] = roads
    activity_location['suburb'] = suburbs

    activity_location['office'] = offices
    activity_location['house_number'] = house_numbers
    activity_location['town'] = towns
    activity_location['municipality'] = municipalities
    activity_location['city'] = cities
    activity_location['state_district'] = state_districts
  
    activity_location['postcode'] = postcodes
    activity_location['country'] = countries
    


    activity_location = activity_location.filter([ 'beginTimestamp', 'duration_in_mins', 'distance', 'building','amenity', 
                                                'road','suburb', 'office', 'house_number', 'town', 'municipality', 
                                                'city', 'state_district','postcode', 'country',
                                                'startLatitude','startLongitude','activityType'], axis=1)


    activity_location.loc[activity_location['building'] =='not available', 'building'] = ''
    activity_location.loc[activity_location['amenity'] =='not available', 'amenity'] = ''
    activity_location.loc[activity_location['road'] =='not available', 'road'] = ''
    activity_location.loc[activity_location['suburb'] =='not available', 'suburb'] = ''
    activity_location.loc[activity_location['office'] =='not available', 'office'] = ''
    activity_location.loc[activity_location['house_number'] =='not available', 'house_number'] = ''
    activity_location.loc[activity_location['town'] =='not available', 'town'] = ''
    activity_location.loc[activity_location['municipality'] =='not available', 'municipality'] = ''
    activity_location.loc[activity_location['city'] =='not available', 'city'] = ''
    activity_location.loc[activity_location['state_district'] =='not available', 'state_district'] = ''
    
    activity_location.loc[activity_location['postcode'] =='not available', 'postcode'] = ''
    activity_location.loc[activity_location['country'] =='not available', 'country'] = ''
    


    location2 = []
    location3 = []



    for index, row in activity_location.iterrows():
          bu = row.loc['building'] + ' ' +  row.loc['amenity']+ ' ' +  row.loc['road']+ ' ' +  row.loc['suburb']+ ' ' +  row.loc['office']+ ' ' +  row.loc['house_number']+ ' '  +  row.loc['municipality']    + ' '  +  row.loc['city'] + ' '  +  row.loc['state_district']+ ' '   +  row.loc['postcode']+ ' '  +  row.loc['country']
          pl =  row.loc['road']+ ' ' + row.loc['city']

          location2.append(bu)
          location3.append(pl)
         
          
    
    activity_location['location_inf'] = location2   

        
    activity_location['location_inf'] = activity_location['location_inf'].str.replace(r'  ', '')

    activity_location['locations'] = location3   

        
    activity_location['locations'] = activity_location['locations'].str.replace(r'  ', '')




   
    fig, ax = plt.subplots(figsize=(5, 5))

    activity_location.locations.value_counts().plot(kind = 'pie')

    ax.legend(loc='upper left')

    
    plt.xticks(rotation=0)
    plt.yticks(fontsize=2)
    plt.xticks(fontsize=2)

    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'garmin_locations'

    fullnamegarminloc = n2+t2+'.png'
    plt.savefig('static/'+fullnamegarminloc)  
    plt.close()  



    most_common_loc = activity_location['location_inf']. value_counts(). idxmax()



    # find length n dates of dataset used

   

    activities['beginTimestamp'] = pd.to_datetime(activities['beginTimestamp'], format="%m/%d/%Y %H:%M:%S")



    count = activities.groupby(activities['beginTimestamp'].dt.date, sort=False, as_index=False).size()

    days_in_data_g  = len(count)

    g1_stop = process_time()

    elapsed_time1_g =  g1_stop-g1_start


  
            
    return (most_common_loc , most_common_activity , fullnamegarminact, days_in_data_g, elapsed_time1_g, rowcount1g, fullnamegarminloc)




def process_garmin_fitness_inf(fileName):

    g2_start = process_time()  
    
    filename = os.path.join('uploads', fileName)
   

    df = pd.read_csv(filename) 
    rowcount2g = len(df.index)

    #get only rows with vo2maxvalues

    mydata = df[df['vO2MaxValue'].notnull()]

    mydata['start_time'] = pd.to_datetime(mydata['beginTimestamp'], unit='ms')

    mydata['duration2'] = mydata['duration'].apply(np.int64)
        
        
    #convert from ms to minutes
    mydata['duration_in_mins_from_ms'] = mydata['duration2']/60000
    mydata['distance_in_km'] = mydata['distance']/ 100000
    mydata['calories'] = mydata['calories']/10



    if mydata.empty:
        fitness_condition = 'No data available.'
        vo2value = ' not available'


    else: 

        

        vo2_dataset = mydata.filter(['activityType', 'sportType', 'start_time', 'avgHr', 'MaxHr', 'vO2MaxValue' ,'steps', 'calories', 'duration_in_mins_from_ms'], axis=1)
        
        #get min and max value of vO2maxvalue and correspondinates
        #this code fetches the rows

        fig, ax = plt.subplots(figsize=(5, 5))

        vo2_dataset.set_index('start_time').plot(marker='o',  y = 'vO2MaxValue',  title= 'vO2MaxValue', ax=ax)   

        plt.xticks(rotation=30)
        plt.xticks(fontsize=6)

        t1 = time.strftime("%Y%m%d-%H%M%S")

        n1 = 'vO2MaxValue'

        fullnamevO2MaxValue = n1+t1+'.png'
        plt.savefig('static/'+fullnamevO2MaxValue)  
        plt.close()  




        
        
        maxvo2 = vo2_dataset.loc[vo2_dataset['vO2MaxValue'].idxmax()]
        minvo2 = vo2_dataset.loc[vo2_dataset['vO2MaxValue'].idxmin()]
        
        maxvo2['start_time'] = pd.to_datetime(maxvo2['start_time'])
        minvo2['start_time'] = pd.to_datetime(minvo2['start_time'])
       

      
    
       
        #if date of max vo2 value is later thandate of min vo2 value then gitness has  improved, but have to check with current value of vo2   
        if maxvo2['start_time'] >= minvo2['start_time']: 
           
            
            latest_row = vo2_dataset.loc[vo2_dataset['start_time'].idxmax()]
            vo2value = latest_row['vO2MaxValue']
            if  maxvo2['vO2MaxValue'] == latest_row['vO2MaxValue']:

                fitness_condition = 'Your fitness condition has improved since '+str(maxvo2['start_time'].strftime("%d-%b-%y"))+' based on your vO2Max data.'
                
               
            else:
                fitness_condition = 'Your fitness condition has worsened since '+ str(maxvo2['start_time'].strftime("%d-%b-%y"))+' based on your vO2Max data.'
                

                
        
        else:
            latest_row = vo2_dataset.loc[vo2_dataset['start_time'].idxmin()]
            vo2value = latest_row['vO2MaxValue']
            fitness_condition = 'Your fitness condition has worsened since '+ str(maxvo2['start_time'].strftime("%d-%b-%y"))+' based on your vO2Max data.'

           
            


    mydata2 = df

    mydata2['start_time'] = pd.to_datetime(mydata2['beginTimestamp'], unit='ms')

    mydata2['duration2'] = mydata2['duration'].apply(np.int64)
        
        
    #convert from ms to minutes
    mydata2['duration_in_mins_from_ms'] = mydata2['duration2']/60000
    mydata2['distance_in_km'] = mydata2['distance']/ 100000
    mydata2['calories'] = mydata2['calories']/10
   

    mydata2['Date2'] = mydata2['start_time'].dt.date
  

    mydata2['diff'] = mydata2["Date2"].diff()
 
    mydata2['dif_in_days'] = mydata2["Date2"].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
  

    mydata2['dif_in_days'] = abs(mydata2['dif_in_days'])
 
    




    avrg_frequency = mydata2['dif_in_days'].value_counts().idxmax()
    
    mean_days_between_fitness = round(mydata2['dif_in_days'].mean())
    
   

    mean_duration = round(mydata2['duration_in_mins_from_ms'].mean())
   
                
    mydata2['start_time'] = pd.to_datetime(mydata2['start_time'])
    mydata2['dayOfWeek'] = mydata2['start_time'].dt.day_name()
    mydata2['nodayofweek'] = np.where(mydata2['dayOfWeek'] == 'Sunday', 0,
                         (np.where(mydata2['dayOfWeek'] == 'Monday', 1,
                                   (np.where(mydata2['dayOfWeek'] == 'Tuesday', 2,
                                             (np.where(mydata2['dayOfWeek'] == 'Wednesday', 3,
                                                       (np.where(mydata2['dayOfWeek'] == 'Thursday', 4,
                                                                 (np.where(mydata2['dayOfWeek'] == 'Friday', 5,
                                                                           (np.where(mydata2['dayOfWeek'] == 'Saturday', 6,
                                            
                                            
                                            
                                            10)))))))))))))


    most_common_day =  mydata2['dayOfWeek'].value_counts().idxmax()


    mydata2['hour'] = mydata2['start_time'].dt.hour
    most_common_hour =  mydata2['hour'].value_counts().idxmax()   




    mydata['start_time'] = pd.to_datetime(mydata['start_time'], infer_datetime_format=True)

    # find length n dates of dataset used

   

    fig, ax = plt.subplots(figsize=(5, 5))

    mydata2.dayOfWeek.value_counts().plot(kind = 'pie')
    
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=0)
    plt.yticks(fontsize=5)

    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'garmin_active_day'

    fullnamegarminactday = n2+t2+'.png'
    plt.savefig('static/'+fullnamegarminactday)  
    plt.close()  



    count = mydata.groupby(mydata['start_time'].dt.date, sort=False, as_index=False).size()

    days_in_data_gf  = len(count)

    g2_stop = process_time()

    elapsed_time2_g =  g2_stop-g2_start


    return (fitness_condition, vo2value, mean_days_between_fitness, mean_duration, most_common_day,most_common_hour, 
        fullnamevO2MaxValue,days_in_data_gf, elapsed_time2_g, rowcount2g ,fullnamegarminactday)






def process_garmin_sleep_inf(fileName):
    g3_start = process_time()  

    filename = os.path.join('uploads', fileName)

    sleep_dataset = pd.read_csv(filename) 
    rowcount3g = len(sleep_dataset.index)
    
    sleep_data = sleep_dataset[sleep_dataset.sleepWindowConfirmationType != 'UNCONFIRMED']


    sleep_data['sleepStartTime'] = pd.to_datetime(sleep_data['sleepStartTimestampGMT'].astype(str)) + pd.DateOffset(hours=2)
    sleep_data['sleepEndTime'] = pd.to_datetime(sleep_data['sleepEndTimestampGMT'].astype(str)) + pd.DateOffset(hours=2) 
    sleep_data['hoursofsleep'] = (sleep_data.sleepEndTime - sleep_data.sleepStartTime).astype('timedelta64[h]')

    sleep_data['sleepStartTime'] = pd.to_datetime(sleep_data['sleepStartTime'], format="%d/%m/%Y %H:%M")

    sleep_data['sleepEndTime'] = pd.to_datetime(sleep_data['sleepEndTime'], format="%d/%m/%Y %H:%M")  

    sleep_data['MinutesInBed'] = sleep_data['sleepEndTime']-sleep_data['sleepStartTime']
    sleep_data['MinutesInBed'] = abs(sleep_data['MinutesInBed'] /np.timedelta64(1,'m'))



    #How regular are your sleeping habits?

    sleepDesc = pd.DataFrame(sleep_data['MinutesInBed']/60).describe().transpose()
    avgSleepHours = round(sleepDesc.at['MinutesInBed','mean'],2)

    sleep_data['gotosleep_hour']=  sleep_data['sleepStartTime'].dt.strftime('%H:%M')
    sleep_data['wake_hour']=  sleep_data['sleepEndTime'].dt.strftime('%H:%M')

    sleep_data['sleepstarthour'] = sleep_data.sleepStartTime.dt.hour
    sleep_data['wakeuphour'] = sleep_data.sleepEndTime.dt.hour

    sleep_data['dateonly'] = sleep_data.sleepEndTime.dt.date
    sleep_data['day_of_week'] = sleep_data.sleepEndTime.dt.weekday

    sleep_data['sleepstarthour']  = np.where (sleep_data['sleepstarthour'] == 0, 24, np.where( sleep_data['sleepstarthour'] == 1, 25,                                        np.where( sleep_data['sleepstarthour'] == 2, 26,
                                                         np.where( sleep_data['sleepstarthour'] == 3, 27,
                                                                  np.where( sleep_data['sleepstarthour'] == 4, 28,
                                                                           np.where( sleep_data['sleepstarthour'] == 5, 29,
                                                                                    np.where( sleep_data['sleepstarthour'] == 6, 30, sleep_data['sleepstarthour'] ) ))))))
 


     
    
    fig, ax = plt.subplots(figsize=(5, 5))

    

    sleep_data['Hours of Sleep']=sleep_data['MinutesInBed']/60
            
    sleep_data.set_index('dateonly').plot(marker='o',  y = 'Hours of Sleep',  title= 'Daily hours of sleep', ax=ax)


    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    t1 = time.strftime("%Y%m%d-%H%M%S")

    n1 = 'garminhoursofsleep'

    fullnamegarminhoursofsleep = n1+t1+'.png'
    plt.savefig('static/'+fullnamegarminhoursofsleep)  
    plt.close()



    
    weekend =  sleep_data[(sleep_data['day_of_week'] == 5) | (sleep_data['day_of_week'] == 6)] 
    
    weekday =  sleep_data[(sleep_data['day_of_week'] == 0) | (sleep_data['day_of_week'] == 1)| (sleep_data['day_of_week'] == 2) | (sleep_data['day_of_week'] == 3) | (sleep_data['day_of_week'] == 4)] 

   

    #How regular are your sleeping habits?

    
    sleepDesc = pd.DataFrame(sleep_data['MinutesInBed']/60).describe().transpose()
    
    avgSleepHours = round(sleepDesc.at['MinutesInBed','mean'],2)

    #How regular are your sleeping habits in weekdays?

    
    sleepDesc_wdays = pd.DataFrame(weekday['MinutesInBed']/60).describe().transpose()
    
    avgSleepHours_wdays = round(sleepDesc_wdays.at['MinutesInBed','mean'],2)


    #How regular are your sleeping habits in weekends?

    
    sleepDesc_wends = pd.DataFrame(weekend['MinutesInBed']/60).describe().transpose()
    
    avgSleepHours_wends = round(sleepDesc_wends.at['MinutesInBed','mean'],2)


    
    avgsleepstarthour = round(sleep_data['sleepstarthour'].mean(),1)
    
    avgsleepstarthour_wdays = round(weekday['sleepstarthour'].mean(),1)
    
    avgsleepstarthour_wends = round(weekend['sleepstarthour'].mean(),1)


    
    if avgsleepstarthour >= 24:
        
        avgsleepstarthour = round(avgsleepstarthour -24)

    
    
    if avgsleepstarthour_wdays >= 24:
        
       avgsleepstarthour_wdays = round(avgsleepstarthour_wdays -24)

    if avgsleepstarthour_wends >= 24:
        
       avgsleepstarthour_wends = round(avgsleepstarthour_wends -24)
        
    
   
    avgwakeuptime = round(sleep_data['wakeuphour'].mean())
    avgwakeuptime_wdays = round(weekday['wakeuphour'].mean())
    avgwakeuptime_wends = round(weekend['wakeuphour'].mean())

 
    avgminutessleep = round(sleep_data['MinutesInBed'].mean()/60,2)



    sleep_cycles = sleep_data[~sleep_data['deepSleepSeconds'].isnull()]


    avgDeepSleep = round(sleep_cycles['deepSleepSeconds'].mean(),2)
    avgDeepSleep_inmins = avgDeepSleep/60
    avgDeepSleep_inhours = avgDeepSleep_inmins/60

 

    percentage_of_deep_sleep = round((avgDeepSleep_inhours*100)/avgminutessleep,2)

    avgLightSleep = round(sleep_cycles['lightSleepSeconds'].mean(),2)
    avgLightSleep_inmins = avgLightSleep/60
    avgLightSleep_inhours = avgLightSleep_inmins/60

 

    percentage_of_light_sleep = round((avgLightSleep_inhours*100)/avgminutessleep,2)



    avgAwakeSleep = round(sleep_cycles['awakeSleepSeconds'].mean(),2)
    avgAwakeSleep_inmins = avgAwakeSleep/60
    avgAwakeSleep_inhours = avgAwakeSleep_inmins/60

 

    percentage_of_awake_sleep = round((avgAwakeSleep_inhours*100)/avgminutessleep,2)






    weekday1 = weekday.filter(['dateonly','MinutesInBed','deepSleepSeconds','lightSleepSeconds', 'awakeSleepSeconds'], axis=1)

    weekday1.rename(columns={'dateonly': 'Date'}, inplace=True)


    weekday1.Date = pd.to_datetime(weekday1.Date)
    weekday1.set_index("Date", drop=True, inplace=True)
    weekday1.sort_index(inplace=True)

    
    weekend1 = weekend.filter(['dateonly','MinutesInBed','deepSleepSeconds','lightSleepSeconds', 'awakeSleepSeconds'], axis=1)
    weekend1.rename(columns={'dateonly': 'Date'}, inplace=True)

    weekend1.Date = pd.to_datetime(weekend1.Date)
    weekend1.set_index("Date", drop=True, inplace=True)
    weekend1.sort_index(inplace=True)

    
    weekend1.rename(columns={'MinutesInBed': 'Minutes sleeping'}, inplace=True)

    weekend1.rename(columns={'deepSleepSeconds': 'Minutes in Deep Sleep cycle' }, inplace=True)

    weekend1.rename(columns={'lightSleepSeconds': 'Minutes in Light Sleep cycle'}, inplace=True)
    weekend1.rename(columns={'awakeSleepSeconds': 'Minutes in REM Sleep cycle'}, inplace=True)




    weekday1.rename(columns={'MinutesInBed': 'Minutes sleeping'}, inplace=True)

    weekday1.rename(columns={'deepSleepSeconds': 'Minutes in Deep Sleep cycle' }, inplace=True)

    weekday1.rename(columns={'lightSleepSeconds': 'Minutes in Light Sleep cycle'}, inplace=True)
    weekday1.rename(columns={'awakeSleepSeconds': 'Minutes in REM Sleep cycle'}, inplace=True)


    fig, ax = plt.subplots(figsize=(5, 5))

    weekend1.plot(marker='o',   title= 'Weekend Sleep Data Facts', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)


    t2 = time.strftime("%Y%m%d-%H%M%S")

    n2 = 'garminweekendsleepfacts'

    fullnamegarminweekendsleepfacts = n2+t2+'.png'
    plt.savefig('static/'+fullnamegarminweekendsleepfacts)  
    plt.close()

    




    fig, ax = plt.subplots(figsize=(5, 5))
            
    weekday1.plot(marker='o',   title= 'Weekday Sleep Data Facts', ax=ax)
    plt.xticks(rotation=30)
    plt.xticks(fontsize=6)

    t3 = time.strftime("%Y%m%d-%H%M%S")

    n3 = 'garminweekdaysleepfacts'

    fullnamegarminweekdaysleepfacts = n3+t3+'.png'
    plt.savefig('static/'+fullnamegarminweekdaysleepfacts)  
    plt.close()
            

    count = sleep_data.groupby(sleep_data['sleepStartTime'].dt.date, sort=False, as_index=False).size()

    days_in_data_gs  = len(count)

    g3_stop = process_time()

    elapsed_time3_g =  g3_stop-g3_start


  

    return (avgSleepHours_wdays, avgSleepHours_wends, avgsleepstarthour_wdays, 
        avgsleepstarthour_wends, avgwakeuptime_wdays, avgwakeuptime_wends,
        percentage_of_deep_sleep, percentage_of_light_sleep, percentage_of_awake_sleep,
        fullnamegarminhoursofsleep, days_in_data_gs, elapsed_time3_g, rowcount3g,
        fullnamegarminweekdaysleepfacts, fullnamegarminweekendsleepfacts)