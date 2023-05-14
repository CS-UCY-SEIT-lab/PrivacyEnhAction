import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import priv_functions 
import pandas as pd

app = Flask(__name__)

print("Loading model") 
# HERE PUT THE CODE TO LOAD ANY PREDICTION MODEL NECESSARY FOR THE PROCESING
#global sess
#sess = tf.Session()
#set_session(sess)
#global model 
#model = load_model('my_cifar10_model.h5') 
#global graph
#graph = tf.get_default_graph()


@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        
        mode = request.form.getlist('mode')
        if len(mode) == 0:
            return render_template('index.html')  
        if mode[0] == 'watermeter':
            sel_mode = 1
            file = request.files['file_watermeter']
        if mode[0] == 'motionsensor':    
            sel_mode = 2
            file = request.files['file_motionsensor']
            
                  
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('processing',mode=sel_mode, filename=filename))
    return render_template('index.html')




@app.route('/processing/<mode>/<filename>') 
##mode = 1 -> Smart Water Meter
##mode = 2 ->Smart Motion Sensor
##mode = 3 ->FitBit
def processing(mode,filename):
    
    # Here You put your code Processing code
    
    X = process_file(mode,filename)
    print('Pre-Processing Finished! - X returned')
    print(X)
    results = "Processing Result"
    return render_template('results.html', results=results)




def process_file(mode,fileName):
    
    filename = os.path.join('uploads', fileName)
    print(filename)
    df = pd.read_csv(filename)
    df['result_time'] = pd.to_datetime(df['result_time'])   
    #df['result_time'] = pd.to_datetime(df['result_time'].dt.strftime('%dd/%mm/%yyyy %HH:%mm:%ss'))
    
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
    
    return X
    
 

app.run(host='142.11.210.23', port=8989)