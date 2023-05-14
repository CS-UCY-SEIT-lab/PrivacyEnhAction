
import os 

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template, session
import pickle


from flask_session import Session



from werkzeug.utils import secure_filename

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import priv_functions 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import requests
from bs4 import BeautifulSoup
import re
from sentence_splitter import SentenceSplitter
from nltk.stem.snowball import SnowballStemmer
import sys
import warnings


from sklearn.cluster import KMeans
import motion_sensor_source_code
from motion_sensor_source_code import process_motion_file
from motion_sensor_source_code import predict, pir_wr_inf1, pir_wr_inf2, pir_wr_inf3, pir_wr_inf4, pir_wr_inf5

from sklearn import metrics


import garmin_source_code
from garmin_source_code import process_garmin_location_inf, process_garmin_fitness_inf, process_garmin_sleep_inf

from fitbit_source_code import  process_fitbit_dailyActivity, process_fitbit_sleepdata, process_fitbit_heartrate

from mi4c_source_code import process_mi4c_activity_data,  process_mi4c_heart_rate_data, process_mi4c_sleep_data, process_mi4c_sleep_data2
from water_meter_source_code import process_water_file, process_water_sleep_inf

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

app = Flask(__name__)



# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['csv'])



model = pickle.load(open("model.pkl", "rb"))
model_inf = pickle.load(open("model_inf.pkl", "rb"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


@app.route('/') 
def main_page():
     return render_template('index.html') 


def remove_tags(html):
  
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
  
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
  
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


@app.route('/results_pp_inf_fitbit/')

def results_pp_inf_fitbit():

    return render_template('results_policy_inferences_fitbit.html') 

@app.route('/results_pp_inf_garmin/')

def results_pp_inf_garmin():

    return render_template('results_policy_inferences_garmin.html') 


@app.route('/results_pp_inf_mi4c/')

def results_pp_inf_mi4c():

    return render_template('results_policy_inferences_mi4c.html') 


@app.route('/predict_gdpr_rights/')
def predict_gdpr_rights():
    

    

    return render_template('results_policy_rights.html') 


    

@app.route('/predict_gdpr_rights2/')
def predict_gdpr_rights2():

   
      

    return render_template('results_policy_rights.html') 
    
    


@app.route('/devices_management')
def devices_management():
    return render_template('devices_management.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/about')
def about():
    return render_template('about.html')


def stemming(sentence):
    stemSentence = ""
    
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
        
    stemSentence = stemSentence.strip()
    
    return stemSentence

@app.route('/privacy_policy_analysis', methods=['GET', 'POST']) 
def privacy_policy_analysis():
    error = []
    results = {}

    if request.method == 'POST':
       
        mode = request.form.getlist('mode')
       
        if len(mode) == 0:
            return render_template('index.html') 
        
        
        if mode[0] == 'inferences':
            sel_mode = 2
            try:
               
                
                filename = os.path.join('uploads', 'all_inferences.csv')
              
                df = pd.read_csv(filename) 
                data = df
        

                if not sys.warnoptions:
                    warnings.simplefilter("ignore")

                data['policy'] = data['policy'].str.lower()
                data['policy'] = data['policy'].apply(cleanHtml)
                data['policy'] = data['policy'].apply(cleanPunc)
                data['policy'] = data['policy'].apply(keepAlpha)

               

                stemmer = SnowballStemmer("english")
               


                def stemming2(sentence):
                    stemSentence = ""
                    for word in sentence.split():
                        stem = stemmer.stem(word)
                        stemSentence += stem
                        stemSentence += " "
                    stemSentence = stemSentence.strip()
                    return stemSentence

                

                data['policy'] = data['policy'].apply(stemming2)
               
               
                
               
                X = data["policy"]
                y = np.asarray(data[data.columns[1:]])

              

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

               

                vetorizar = TfidfVectorizer(
                    stop_words='english',
                    sublinear_tf=True,
                    strip_accents='unicode',
                    analyzer='word',
                    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
                    ngram_range=(1, 1),
                    max_features=30000)

               


                vetorizar.fit(X_train)

                X_train_tfidf = vetorizar.transform(X_train)
                X_test_tfidf = vetorizar.transform(X_test)

                # initialize LabelPowerset multi-label classifier with a RandomForest
                classifier = BinaryRelevance(
                    classifier = SVC(),
                    require_dense = [False, True])

                # train
                classifier.fit(X_train_tfidf, y_train)
                predicted = classifier.predict(X_test_tfidf)

               

                url = request.form['policy_inferences']
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

                r = requests.get(url, headers=headers, verify=False)



                link_url_contents = r.text

               
                mytext= remove_tags(link_url_contents)
                mytext = mytext.replace("   ", " ")
               


                splitter = SentenceSplitter(language='en')

                mytext_split = splitter.split(text=mytext)


               
                test_df = pd.DataFrame(columns=['policy'])

                i =0

                for item in mytext_split:
                    test_df.at[i, 'policy'] = item
                    i = i +1



                df1 = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'Policy Text'])

                inf1 = pd.DataFrame(columns=['Policy Text'])
                inf2 = pd.DataFrame(columns=['Policy Text'])
                inf3 = pd.DataFrame(columns=['Policy Text'])
                inf4 = pd.DataFrame(columns=['Policy Text'])
                inf5 = pd.DataFrame(columns=['Policy Text'])
                inf6 = pd.DataFrame(columns=['Policy Text'])
                inf7 = pd.DataFrame(columns=['Policy Text'])
                
                col1_list = []
                col2_list = []
                col3_list = []
                col4_list = []
                col5_list = []
                col6_list = []
                col7_list = []
               
               
                policy_text_list = []

                inf1_policy_text_list = []
                inf2_policy_text_list = []
                inf3_policy_text_list = []
                inf4_policy_text_list = []
                inf5_policy_text_list = []
                inf6_policy_text_list = []
                inf7_policy_text_list = []
                

                for index, row in test_df.iterrows():

                    
                    new_sentences = row
                    
                    new_sentence_tfidf = vetorizar.transform(new_sentences)
                    predicted_sentences_BR = classifier.predict(new_sentence_tfidf)
                    
                    prediction = predicted_sentences_BR.toarray()

                    col1 = prediction[0][0] 
                   
                    col2 = prediction[0][1] 
                   

                    col3 = prediction[0][2] 

                   
                    
                    col4 = prediction[0][3] 
                    
                    col5 = prediction[0][4] 
                    
                    col6 = prediction[0][5] 
                    
                    col7 = prediction[0][6] 
                    

                    col1_list.append(col1)
                    col2_list.append(col2)

                   
                    col3_list.append(col3)
                  
                    col4_list.append(col4)
                  
                    col5_list.append(col5)
                    col6_list.append(col6)
                    
                    col7_list.append(col7)
                    
                    

                    policy_text_list.append(row['policy'])

                    br_prediction = predicted_sentences_BR.toarray().astype(bool)


                
                
                df1['c1'] = col1_list
                
                df1['c2'] = col2_list

              

                df1['c3'] = col3_list


                df1['c4'] = col4_list
             
                df1['c5'] = col5_list
             
                df1['c6'] = col6_list
                
                df1['c7'] = col7_list

                

                df1['Policy Text'] = policy_text_list
             


                for index1, row1 in df1.iterrows():
                    if row1['c1'] == 0:
                        pass
                        
                    else:
                        inf1_policy_text_list.append(row1['Policy Text'])             
                inf1['Policy Text'] = inf1_policy_text_list   

                if len(inf1) == 0:
                    flag1 = 0
                    inf1.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    
                    inf1['Policy Text'] = inf1_policy_text_list 
                    flag1 = 1 

                           



                for index2, row2 in df1.iterrows():
                    if row2['c2'] == 0:
                        
                        pass
                        
                    else:
                        inf2_policy_text_list.append(row2['Policy Text'])
                inf2['Policy Text'] = inf2_policy_text_list 

                if len(inf2) == 0:
                    flag2 = 0
                    inf2.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    
                    inf2['Policy Text'] = inf2_policy_text_list 
                    flag2 = 1




                for index3, row3 in df1.iterrows():
                    if row3['c3'] == 0:
                        
                        pass
                    else:
                        inf3_policy_text_list.append(row3['Policy Text'])
                inf3['Policy Text'] = inf3_policy_text_list 

                if len(inf3) == 0:
                    flag3 = 0
                    inf3.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    
                    inf3['Policy Text'] = inf3_policy_text_list
                    flag3 = 1

                for index4, row4 in df1.iterrows():
                    if row4['c4'] == 0:
                        
                        pass
                    else:
                        inf4_policy_text_list.append(row4['Policy Text'])
                inf4['Policy Text'] = inf4_policy_text_list 

                if len(inf4) == 0:
                    flag4 = 0
                    inf4.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    
                    inf4['Policy Text'] = inf4_policy_text_list 
                    flag4 = 1
  


                for index5, row5 in df1.iterrows():
                    if row5['c5'] == 0:
                        
                        pass
                    else:
                        inf5_policy_text_list.append(row5['Policy Text'])
                inf5['Policy Text'] = inf5_policy_text_list 

                if len(inf5) == 0:
                    flag5 = 0
                    inf5.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    
                    inf5['Policy Text'] = inf5_policy_text_list 
                    flag5 = 1




                for index6, row6 in df1.iterrows():
                    if row6['c6'] == 0:
                       
                        pass
                    else:
                        inf6_policy_text_list.append(row6['Policy Text'])
                inf6['Policy Text'] = inf6_policy_text_list 

                if len(inf6) == 0:
                    flag6 = 0
                    inf6.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    flag6 = 1
                    inf6['Policy Text'] = inf6_policy_text_list 


                for index7, row7 in df1.iterrows():
                  if row7['c7'] == 0:
                    
                    pass
                else:
                    inf7_policy_text_list.append(row7['Policy Text'])
                inf7['Policy Text'] = inf7_policy_text_list 

                if len(inf7) == 0:
                    flag7 = 0
                    inf7.at[0,'Policy Text'] = 'No inferences found.'
                else:
                    flag7 = 1
                    inf7['Policy Text'] = inf7_policy_text_list 

                count_flags = flag1+flag2+flag3+flag4+flag5+flag6+flag7


    
               

            except Exception as x:
                print(type(x),x)
                pass
            


            return render_template('results_policy_inferences.html', passedflag1 = flag1, 
                passedflag2 = flag2, passedflag3= flag3, passedflag4 = flag4, passedflag5=flag5, 
                passedflag6 = flag6, passedflag7=flag7, passedscore=count_flags)







        if mode[0] == 'rights':

            sel_mode = 1
            try:
               
                
                filename = os.path.join('uploads', 'all_devices.csv')
                df = pd.read_csv(filename) 

                



                data = df

                

                if not sys.warnoptions:
                    warnings.simplefilter("ignore")

                data['policy'] = data['policy'].str.lower()
                data['policy'] = data['policy'].apply(cleanHtml)
                data['policy'] = data['policy'].apply(cleanPunc)
                data['policy'] = data['policy'].apply(keepAlpha)

               

                stemmer = SnowballStemmer("english")
                


                def stemming2(sentence):
                    stemSentence = ""
                    for word in sentence.split():
                        stem = stemmer.stem(word)
                        stemSentence += stem
                        stemSentence += " "
                    stemSentence = stemSentence.strip()
                    return stemSentence

                

                data['policy'] = data['policy'].apply(stemming2)
               
               
               
                X = data["policy"]
                y = np.asarray(data[data.columns[1:]])

              

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

               

                vetorizar = TfidfVectorizer(
                    stop_words='english',
                    sublinear_tf=True,
                    strip_accents='unicode',
                    analyzer='word',
                    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
                    ngram_range=(1, 1),
                    max_features=30000)

               


                vetorizar.fit(X_train)

                X_train_tfidf = vetorizar.transform(X_train)
                X_test_tfidf = vetorizar.transform(X_test)

                # initialize LabelPowerset multi-label classifier with a RandomForest
                classifier = BinaryRelevance(
                    classifier = SVC(),
                    require_dense = [False, True])

                # train
                classifier.fit(X_train_tfidf, y_train)
                predicted = classifier.predict(X_test_tfidf)



                url = request.form['policy_rights']
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

                r = requests.get(url, headers=headers, verify=False)



                
                link_url_contents = r.text

                
                mytext= remove_tags(link_url_contents)
                mytext = mytext.replace("   ", " ")
               


                splitter = SentenceSplitter(language='en')

                mytext_split = splitter.split(text=mytext)


               
                test_df = pd.DataFrame(columns=['policy'])

                i =0

                for item in mytext_split:
                    test_df.at[i, 'policy'] = item
                    i = i +1



                df1 = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'Policy Text'])

                gdpr1 = pd.DataFrame(columns=['Policy Text'])
                gdpr2 = pd.DataFrame(columns=['Policy Text'])
                gdpr3 = pd.DataFrame(columns=['Policy Text'])
                gdpr4 = pd.DataFrame(columns=['Policy Text'])
                gdpr5 = pd.DataFrame(columns=['Policy Text'])
                gdpr6 = pd.DataFrame(columns=['Policy Text'])
                gdpr7 = pd.DataFrame(columns=['Policy Text'])
                gdpr8 = pd.DataFrame(columns=['Policy Text'])


                col1_list = []
                col2_list = []
                col3_list = []
                col4_list = []
                col5_list = []
                col6_list = []
                col7_list = []
                col8_list = []
                
               
                policy_text_list = []

                gdpr1_policy_text_list = []
                gdpr2_policy_text_list = []
                gdpr3_policy_text_list = []
                gdpr4_policy_text_list = []
                gdpr5_policy_text_list = []
                gdpr6_policy_text_list = []
                gdpr7_policy_text_list = []
                gdpr8_policy_text_list = []
               

                for index, row in test_df.iterrows():

                    
                    new_sentences = row
                   
                    new_sentence_tfidf = vetorizar.transform(new_sentences)

                   

                    predicted_sentences_BR = classifier.predict(new_sentence_tfidf)
                    
                    prediction = predicted_sentences_BR.toarray()

                   

                    col1 = prediction[0][0] 
                   
                    col2 = prediction[0][1] 
                   

                    col3 = prediction[0][2] 

                   
                    
                    col4 = prediction[0][3] 
                 
                    col5 = prediction[0][4] 
                  
                    col6 = prediction[0][5] 
                  
                    col7 = prediction[0][6] 
                    col8 = prediction[0][7] 

                    col1_list.append(col1)
                    col2_list.append(col2)

                   
                    col3_list.append(col3)
                  
                    col4_list.append(col4)
                  
                    col5_list.append(col5)
                    col6_list.append(col6)
                    
                    col7_list.append(col7)
                  
                    col8_list.append(col8)

                  
                
                    policy_text_list.append(row['policy'])

                 
                 
                    br_prediction = predicted_sentences_BR.toarray().astype(bool)

                              
                

                df1['c1'] = col1_list               

                df1['c2'] = col2_list

              

                df1['c3'] = col3_list

             

                df1['c4'] = col4_list
             
             
                df1['c5'] = col5_list
             


                df1['c6'] = col6_list
                

                df1['c7'] = col7_list

                df1['c8'] = col8_list

                df1['Policy Text'] = policy_text_list
             


                for index1, row1 in df1.iterrows():
                    if row1['c1'] == 0:
                        pass
                        
                    else:
                        gdpr1_policy_text_list.append(row1['Policy Text'])             
                gdpr1['Policy Text'] = gdpr1_policy_text_list   

                if len(gdpr1) == 0:
                    flag1 = 0
                    gdpr1.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy.'
                else:
                    
                    gdpr1['Policy Text'] = gdpr1_policy_text_list 
                    flag1 = 1 

                



                for index2, row2 in df1.iterrows():
                    if row2['c2'] == 0:
                       
                        pass
                        
                    else:
                        gdpr2_policy_text_list.append(row2['Policy Text'])
                gdpr2['Policy Text'] = gdpr2_policy_text_list 

                if len(gdpr2) == 0:
                    flag2 = 0
                    gdpr2.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    
                    gdpr2['Policy Text'] = gdpr2_policy_text_list 
                    flag2 = 1




                for index3, row3 in df1.iterrows():
                    if row3['c3'] == 0:
                        
                        pass
                    else:
                        gdpr3_policy_text_list.append(row3['Policy Text'])
                gdpr3['Policy Text'] = gdpr3_policy_text_list 

                if len(gdpr3) == 0:
                    flag3 = 0
                    gdpr3.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    
                    gdpr3['Policy Text'] = gdpr3_policy_text_list
                    flag3 = 1

                for index4, row4 in df1.iterrows():
                    if row4['c4'] == 0:
                       
                        pass
                    else:
                        gdpr4_policy_text_list.append(row4['Policy Text'])
                gdpr4['Policy Text'] = gdpr4_policy_text_list 

                if len(gdpr4) == 0:
                    flag4 = 0
                    gdpr4.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    
                    gdpr4['Policy Text'] = gdpr4_policy_text_list 
                    flag4 = 1
  


                for index5, row5 in df1.iterrows():
                    if row5['c5'] == 0:
                        
                        pass
                    else:
                        gdpr5_policy_text_list.append(row5['Policy Text'])
                gdpr5['Policy Text'] = gdpr5_policy_text_list 

                if len(gdpr5) == 0:
                    flag5 = 0
                    gdpr5.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    
                    gdpr5['Policy Text'] = gdpr5_policy_text_list 
                    flag5 = 1




                for index6, row6 in df1.iterrows():
                    if row6['c6'] == 0:
                      
                        pass
                    else:
                        gdpr6_policy_text_list.append(row6['Policy Text'])
                gdpr6['Policy Text'] = gdpr6_policy_text_list 

                if len(gdpr6) == 0:
                    flag6 = 0
                    gdpr6.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    flag6 = 1
                    gdpr6['Policy Text'] = gdpr6_policy_text_list 


                for index7, row7 in df1.iterrows():
                  if row7['c7'] == 0:
                   
                    pass
                else:
                    gdpr7_policy_text_list.append(row7['Policy Text'])
                gdpr7['Policy Text'] = gdpr7_policy_text_list 

                if len(gdpr7) == 0:
                    flag7 = 0
                    gdpr7.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    flag7 = 1
                    gdpr7['Policy Text'] = gdpr7_policy_text_list 



                for index8, row8 in df1.iterrows():
                    if row8['c8'] == 0:
                       
                        pass
                        
                    else:
                        gdpr8_policy_text_list.append(row8['Policy Text'])


                        
                gdpr8['Policy Text'] = gdpr8_policy_text_list 
               
                if len(gdpr8) == 0:
                    flag8 = 0
                    gdpr8.at[0,'Policy Text'] = 'This right is not addressed in this privacy policy'
                else:
                    flag8 = 1
                    gdpr8['Policy Text'] = gdpr8_policy_text_list 


                count_flags = flag1+flag2+flag3+flag4+flag5+flag6+flag7+flag8
                        
                                 
              

            except Exception as x:
                print(type(x),x)
                pass
             
            





            return render_template('results_policy_rights2.html', 
                tables= [gdpr1.to_html(classes='mystyle'),
                gdpr2.to_html(classes='gdpr2',bold_rows = True, border =4,  index = False, justify = 'justify-all',  na_rep =' '), 
                gdpr3.to_html(classes='gdpr3',index=False), gdpr4.to_html(classes='gdpr4',index=False), 
                gdpr5.to_html(classes='gdpr5',index=False, justify = 'left'), 
                gdpr6.to_html(classes='gdpr6',index=False),
                gdpr7.to_html(classes='gdpr7',index=False, justify = 'justify-all'), 
                gdpr8.to_html(classes='gdpr8',index=False)],  titles=['na', 'The Right to Be Informed', 'The Right of Access', 'The Right to Rectification',
                'The Right to Erasure', 'The Right to Restrict Processing', 'The Right to Data Portability', 'The Right to Object', 
                'Rights Related to Automated Decision-Making and Profiling '], passedflag1 = flag1, 
                passedflag2 = flag2, passedflag3= flag3, passedflag4 = flag4, passedflag5=flag5, 
                passedflag6 = flag6, passedflag7=flag7, passedflag8 = flag8, passedscore=count_flags )
            

          


    return render_template('privacy_policy_analysis.html')


@app.route('/right_to_be_informed/') 

def right_to_be_informed():

     
     var_data1 = request.args.get('tables', None)
     var_data2 = request.args.get('titles', None)
     
               
     return render_template('results_right_to_be_informed.html', passedgdpr1 = var_data1,passedgdpr2 = var_data2 )     
    




@app.route('/inf_page/', methods=['GET', 'POST']) 
def inf_page():
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
        if mode[0] == 'fitbit':    
            sel_mode = 3
            
            files = request.files.getlist("files_fitbit[]")
            file_list =[]
            for file in files:
                filename = secure_filename(file.filename)

                
                file.save(os.path.join('uploads',filename))
                file_list.append(filename)


           
            return redirect(url_for('processing_fitbit', file_list=file_list))
                  
        if mode[0] == 'garmin': 
            
            sel_mode = 4
            
            files = request.files.getlist("files_garmin[]")
            file_list =[]
            
            for file in files:
                filename = secure_filename(file.filename)

                
                file.save(os.path.join('uploads',filename))
                file_list.append(filename)


           
            return redirect(url_for('processing_garmin', file_list=file_list))
            
         
        if mode[0] == 'mifit':    
            sel_mode = 5
            
            files = request.files.getlist("files_mi4c[]")
            file_list =[]
            
            for file in files:
                filename = secure_filename(file.filename)

                
                file.save(os.path.join('uploads',filename))
                file_list.append(filename)


           
            return redirect(url_for('processing_mi4c', file_list=file_list))

            
        if (mode[0] != 'fitbit' or mode[0] != 'mifit' or mode[0] != 'garmin' ):
            filename = secure_filename(file.filename)
            
            file.save(os.path.join('uploads', filename))
            return redirect(url_for('processing',mode=sel_mode, filename=filename))
    



 
    return render_template('inference_detection.html')




@app.route('/processing/<mode>/<filename>') 
##mode = 1 -> Smart Water Meter
##mode = 2 ->Smart Motion Sensor

def processing(mode,filename):
    
    
    if mode == "1":
        
        occupancy = process_water_file(mode,filename) 

        wakeuptime_weekday, wakeuptime_weekend, sleeptime_weekday, sleeptime_weekend , no_of_wakeup_times = process_water_sleep_inf(mode,filename)

        return render_template('results_water_meter.html', infered_occupancy = occupancy, infered_wakeuptime_weekday = wakeuptime_weekday, 
            infered_wakeuptime_weekend = wakeuptime_weekend, infered_sleeptime_weekday = sleeptime_weekday, infered_sleeptime_weekend = sleeptime_weekend , 
            infered_no_of_wakeup_times = no_of_wakeup_times) 


    elif mode == "2":
        
        #call motion processing file
        
        X = process_motion_file(mode,filename)
        print('Pre-Processing Finished! - X returned')
      
        gr, yel, bl,re,pi  = predict(X);
    
        infcluster,wakeup_pir_wd  = pir_wr_inf1(re)

        #Inference 2: Sleep time
        inf2cluster,ave_sleep_time  = pir_wr_inf2(bl)

        inf3cluster,ave_leave_time = pir_wr_inf3(pi)
        inf4cluster, ave_return_time = pir_wr_inf4(gr)
        no_of_nw = pir_wr_inf5(pi)
  
   
        result1 ="you usually wake up time around "+str(wakeup_pir_wd)+" a.m."
        result2 = "you usually go to sleep around "+str(ave_sleep_time)+" p.m."

        result3 = "you usually leave for work/school at "+str(ave_leave_time)+" a.m."

        result4 = "you usually return home at "+str(ave_return_time)+" p.m."
   
        result5 = "you get out of bed around "+str(no_of_nw)+" times a night."

        
        
        

        return render_template('results_motion_sensor.html', infered_wakeuptime=result1, infered_sleeptime = result2,
            infered_leavehome = result3, infered_returnhome = result4, infered_nightwander = result5) 
    
    
    elif mode == "3":
        print("3")
        
    
   
        
   
    
    
    
    else: 
        
        results = "mode is: "+mode
        return render_template('results.html', results=results)
        
        
def Convert(string):
    li = list(string.split(' '))
    return li

 

@app.route('/processing_policy_rights/<policy_rights>') 

def processing_policy_rights(policy_rights):

    return render_template('results_policy_rights.html' ) 

   

@app.route('/processing_fitbit/<file_list>') 

def processing_fitbit(file_list):
   
   
    no_of_files = len(file_list)

    file_list = file_list.replace("[", "")
    file_list = file_list.replace("]", "")
    file_list = file_list.replace("'", "")
    file_list = file_list.replace(",", "")

    list_of_files = Convert(file_list)
   

    for file in list_of_files:

        file1 = list_of_files[0]
        file2 = list_of_files[1]
        file3 = list_of_files[2]
       

        
        count1, time1, rel, fullnamemon, fullnamewed, fullnamesat = process_fitbit_dailyActivity(file1)
        
        count3, time3, days_in_data2, avgSleepHours,avgSleepHours_wdays, avgSleepHours_wends, avgsleepstarthour,avgsleepstarthour_wdays, avgsleepstarthour_wends, avgwakeuptime, avgwakeuptime_wdays, avgwakeuptime_wends, percentage_of_deep_sleep, percentage_of_light_sleep,percentage_of_rem_sleep, fullnamehoursofsleep , fullnameweekdaysleepfacts,fullnameweekendsleepfacts  =  process_fitbit_sleepdata (file3)
    
           
        count2,time2, sum_max_hr, days_in_data,fullnamehhr,rowcount_highhr, sum_low_hr,rowcount_lowhr, fullnamelowhr = process_fitbit_heartrate(file2,avgsleepstarthour,avgwakeuptime)
        
            

      
        total = count1 + count2 + count3

        total_time = time1 + time2 + time3 +10

        process_time = round(total_time,2)


    return render_template('results_fitbit.html', nooffiles = no_of_files, total_count = total,  total_process_time = process_time,
        infered_days=days_in_data , 
        infered_sum_max_hr = sum_max_hr, infered_fullnamehhr = fullnamehhr,
        infered_rowcount_highhr = rowcount_highhr, infered_sum_low_hr = sum_low_hr,
        infered_rowcount_lowhr = rowcount_lowhr, infered_fullnamelowhr = fullnamelowhr,
        infered_rel = rel, infered_sleephours =avgSleepHours, infered_sleephours_wdays =avgSleepHours_wdays, infered_sleephours_wend =avgSleepHours_wends,
        infered_sleeptime = avgsleepstarthour, infered_sleeptime_wdays = avgsleepstarthour_wdays,infered_sleeptime_wends = avgsleepstarthour_wends,
        infered_waketime = avgwakeuptime , infered_waketime_wdays = avgwakeuptime_wdays ,infered_waketime_wends = avgwakeuptime_wends ,
        infered_lightsleep = percentage_of_light_sleep, infered_deepsleep = percentage_of_deep_sleep, infered_remsleep = percentage_of_rem_sleep, 
        infered_count1 = count1, infered_count2 = count2, infered_count3 = count3,
        infered_fullnamemon = fullnamemon, 
        infered_fullnamewed = fullnamewed, infered_fullnamesat = fullnamesat, infered_days2=days_in_data2,
        infered_fullnamehoursofsleep = fullnamehoursofsleep , infered_fullnameweekdaysleepfacts = fullnameweekdaysleepfacts,
        infered_fullnameweekendsleepfacts = fullnameweekendsleepfacts ) 
        



@app.route('/processing_mi4c/<file_list>') 

def processing_mi4c(file_list):
   
   
    no_of_files = len(file_list)

    file_list = file_list.replace("[", "")
    file_list = file_list.replace("]", "")
    file_list = file_list.replace("'", "")
    file_list = file_list.replace(",", "")

    list_of_files = Convert(file_list)
   

    for file in list_of_files:

        file1 = list_of_files[0]
        file2 = list_of_files[1]
        file3 = list_of_files[2]
        file4 = list_of_files[3]
       
        
       
        
        pr1_rowcount, pr1_time, days_in_data_pers,jew_rel, activity_level, mean_steps, pregnancy_pos, fullnamemi, fullnamemiactivityindex, fullnameprpos = process_mi4c_activity_data(file1, file2, file3, file4)
        
           
       
            
        pr3_rowcount, pr3_time, sum_max_hr3, sum_low_hr3, days_in_data3, fullnamemihighhr, fullnamemilowhr  =  process_mi4c_heart_rate_data (file2)
        


        pr2_rowcount, pr2_time, days_in_data, wdays_hours_of_sleep, wends_hours_of_sleep, wdays_sleeptime, wend_sleeptime, wdays_wakeuptime, wend_wakeuptime, deepsleepperc, shallowsleepperc, waketimeperc, fullnamewdhoursofsleep ,fullnamesleepstarthour, fullnamesleepfactsmi = process_mi4c_sleep_data2(file3)

        totalmi = pr1_rowcount + pr2_rowcount + pr3_rowcount

        total_timemi = pr1_time + pr2_time + pr3_time +10

        process_timemi = round(total_timemi,2)
        
        

        return render_template('results_mi4c.html', total_count_mi = totalmi,  total_process_time_mi = process_timemi,infered_days = days_in_data, 
             infered_rel = jew_rel, infered_activity_level = activity_level, infered_mean_steps = mean_steps, 
             infered_pregnancy_pos = pregnancy_pos,
             infered_sum_max_hr = sum_max_hr3,    
             infered_sum_low_hr = sum_low_hr3, infered_days_in_data = days_in_data3 , infered_wdays_hours_of_sleep = wdays_hours_of_sleep, 
             infered_wend_hours_of_sleep = wends_hours_of_sleep, infered_wdays_gotosleep =  wdays_sleeptime, infered_wend_gotosleep =wend_sleeptime,
             infered_wdays_wakeup = wdays_wakeuptime, infered_wend_wakeup = wend_wakeuptime, infered_deep = deepsleepperc,
             infered_shallow = shallowsleepperc, infered_wtime = waketimeperc, infered_fullnamemi = fullnamemi,
             infered_fullnamemiactivityindex = fullnamemiactivityindex, infered_fullnameprpos = fullnameprpos,
             infered_fullnamemihighhr = fullnamemihighhr, infered_fullnamemilowhr = fullnamemilowhr, 
             infered_fullnamewdhoursofsleep = fullnamewdhoursofsleep, 
             infered_fullnamesleepstarthour = fullnamesleepstarthour, infered_fullnamesleepfactsmi = fullnamesleepfactsmi,
             infered_pr1_rowcount = pr1_rowcount,
             infered_pr2_rowcount = pr2_rowcount,
             infered_pr3_rowcount = pr3_rowcount, infered_days_in_data_pers = days_in_data_pers, infered_days_in_data3 = days_in_data3)
             




@app.route('/processing_garmin/<file_list>') 

def processing_garmin(file_list):
   
   
    no_of_files = len(file_list)

    file_list = file_list.replace("[", "")
    file_list = file_list.replace("]", "")
    file_list = file_list.replace("'", "")
    file_list = file_list.replace(",", "")

    list_of_files = Convert(file_list)
   

    for file in list_of_files:

        file1 = list_of_files[0]
        file2 = list_of_files[1]
    
       
        most_common_loc, most_common_activity, fullnamegarminact, days_in_data_g, elapsed_time1_g, rowcount1g, fullnamegarminloc = process_garmin_location_inf(file1)

        fitness, vo2, days, dur, common_day, common_hour, fullnamevO2MaxValue,days_in_data_gf, elapsed_time2_g, rowcount2g, fullnamegarminactday= process_garmin_fitness_inf(file1)

        infered_avgSleepHours_wdays, infered_avgSleepHours_wends, infered_avgsleepstarthour_wdays, infered_avgsleepstarthour_wends, infered_avgwakeuptime_wdays, infered_avgwakeuptime_wends, infered_percentage_of_deep_sleep, infered_percentage_of_light_sleep, infered_percentage_of_awake_sleep , fullnamegarminhoursofsleep, days_in_data_gs, elapsed_time3_g, rowcount3g,fullnamegarminweekdaysleepfacts, fullnamegarminweekendsleepfacts= process_garmin_sleep_inf(file2)

        
        totalg = rowcount1g + rowcount2g + rowcount3g

        total_timeg = elapsed_time1_g + elapsed_time2_g + elapsed_time3_g +10

        process_timeg = round(total_timeg,2)
        
        return render_template('results_garmin.html', infered_location=most_common_loc, infered_activity =most_common_activity, 
            infered_fullnamegarminact = fullnamegarminact, total_process_time = process_timeg, total_count = totalg, 
            infered_days_in_data_g = days_in_data_g,
            infered_fitness = fitness,infered_vo2 = vo2, infered_days = days, 
            infered_dur = dur, infered_common_day = common_day, infered_common_hour = common_hour , 
            infered_avgSleepHours_wdays = infered_avgSleepHours_wdays, infered_avgSleepHours_wends = infered_avgSleepHours_wends, 
            infered_avgsleepstarthour_wdays = infered_avgsleepstarthour_wdays, infered_avgsleepstarthour_wends = infered_avgsleepstarthour_wends,
            infered_avgwakeuptime_wdays = infered_avgwakeuptime_wdays, infered_avgwakeuptime_wends = infered_avgwakeuptime_wends, 
            infered_percentage_of_deep_sleep = infered_percentage_of_deep_sleep, infered_percentage_of_light_sleep = infered_percentage_of_light_sleep,
            infered_percentage_of_awake_sleep = infered_percentage_of_awake_sleep, infered_rowcount1g = rowcount1g,
            infered_fullnamegarminloc = fullnamegarminloc, infered_fullnamevO2MaxValue = fullnamevO2MaxValue,
            infered_days_in_data_gf = days_in_data_gf, infered_elapsed_time2_g = elapsed_time2_g, infered_rowcount2g = rowcount2g,
            infered_fullnamegarminactday = fullnamegarminactday,
            infered_fullnamegarminhoursofsleep = fullnamegarminhoursofsleep, 
            infered_days_in_data_gs = days_in_data_gs, infered_elapsed_time3_g = elapsed_time3_g, 
            infered_rowcount3g = rowcount3g,
            infered_fullnamegarminweekdaysleepfacts = fullnamegarminweekdaysleepfacts, 
            infered_fullnamegarminweekendsleepfacts = fullnamegarminweekendsleepfacts) 
        
      
   

    


@app.route('/home/')

def home():

    return render_template('index.html') 






@app.route('/garmin_location_inf/') 

def garmin_location_inf():
     
     
     loc = request.args.get('pass_loc', None)
     act = request.args.get('pass_act', None)
     var_fullnamegarminact = request.args.get('pass_fullnamegarminact', None)
     var_process_timeg = request.args.get('pass_process_timeg', None)
     var_totalg = request.args.get('pass_totalg', None)
     var_days_in_data_g = request.args.get('pass_days_in_data_g', None)
     var_rowcount1g = request.args.get('pass_rowcount1g', None)
     var_fullnamegarminloc = request.args.get('pass_fullnamegarminloc', None)

     

  

     
     return render_template('results_garmin_location.html', passedloc =loc, passedact = act, 
        passedfullnamegarminact = var_fullnamegarminact, passedprocess_timeg = var_process_timeg, 
        passedtotalg = var_totalg ,passeddays_in_data_g = var_days_in_data_g, 
        passedrowcount1g = var_rowcount1g , passedfullnamegarminloc = var_fullnamegarminloc)



@app.route('/garmin_fitness_inf/') 

def garmin_fitness_inf():
    
     fit_var = request.args.get('pass_fit', None)
     vo2_var = request.args.get('pass_vo2', None)
     days_var = request.args.get('pass_days', None)
     dur_var = request.args.get('pass_dur', None)
     common_day_var = request.args.get('pass_common_day', None)
     common_hour_var = request.args.get('pass_common_hour', None)

     var_fullnamevO2MaxValue = request.args.get('pass_fullnamevO2MaxValue', None)
     var_days_in_data_gf = request.args.get('pass_days_in_data_gf', None)
     var_elapsed_time2_g = request.args.get('pass_elapsed_time2_g', None)
     var_rowcount2g = request.args.get('pass_rowcount2g', None)
     var_fullnamegarminactday = request.args.get('pass_fullnamegarminactday', None)

     
     
     return render_template('results_garmin_fitness.html', passedfit=fit_var,passedvo2=vo2_var,passeddays=days_var, passeddur=dur_var, 
        passedcommon_day=common_day_var,passedcommon_hour=common_hour_var, passedfullnamevO2MaxValue  = var_fullnamevO2MaxValue,
        passeddays_in_data_gf = var_days_in_data_gf, passedelapsed_time2_g = var_elapsed_time2_g, passedrowcount2g = var_rowcount2g,
        passedfullnamegarminactday=var_fullnamegarminactday)

@app.route('/garmin_sleep_inf/') 

def garmin_sleep_inf():
    
     vr1 = request.args.get('pass_infered_avgSleepHours_wdays', None)
     vr2 = request.args.get('pass_infered_avgSleepHours_wends', None)
     vr3 = request.args.get('pass_infered_avgsleepstarthour_wdays', None)
     vr4 = request.args.get('pass_infered_avgsleepstarthour_wends', None)
     vr5 = request.args.get('pass_infered_avgwakeuptime_wdays', None)
     vr6 = request.args.get('pass_infered_avgwakeuptime_wends', None)
     vr7 = request.args.get('pass_infered_percentage_of_deep_sleep', None)
     vr8 = request.args.get('pass_infered_percentage_of_light_sleep', None)
     vr9 = request.args.get('pass_infered_percentage_of_awake_sleep', None)

     var_fullnamegarminhoursofsleep = request.args.get('pass_fullnamegarminhoursofsleep', None)
     var_days_in_data_gs = request.args.get('pass_days_in_data_gs', None)
     var_elapsed_time3_g = request.args.get('pass_elapsed_time3_g', None)
     var_rowcount3g = request.args.get('pass_rowcount3g', None)
     var_fullnamegarminweekdaysleepfacts = request.args.get('pass_fullnamegarminweekdaysleepfacts', None)
     var_fullnamegarminweekendsleepfacts = request.args.get('pass_fullnamegarminweekendsleepfacts', None)


     
     return render_template('results_garmin_sleep.html', passedinfered_avgSleepHours_wdays=vr1,
        passedinfered_avgSleepHours_wends=vr2, passedinfered_avgsleepstarthour_wdays = vr3, passedinfered_avgsleepstarthour_wends = vr4,
        passedinfered_avgwakeuptime_wdays = vr5, passedinfered_avgwakeuptime_wends = vr6,passedinfered_percentage_of_deep_sleep = vr7 ,
        passedinfered_percentage_of_light_sleep = vr8, passedinfered_percentage_of_awake_sleep = vr9,
        passedfullnamegarminhoursofsleep = var_fullnamegarminhoursofsleep,
        passeddays_in_data_gs = var_days_in_data_gs,
        passedelapsed_time3_g = var_elapsed_time3_g,
        passedrowcount3g = var_rowcount3g,
        passedfullnamegarminweekdaysleepfacts = var_fullnamegarminweekdaysleepfacts,
        passedfullnamegarminweekendsleepfacts = var_fullnamegarminweekendsleepfacts)







@app.route('/fitbit_heart_inf/') 

def fitbit_heart_inf():

     var_count2 = request.args.get('pass_count2', None)
     var_time1 = request.args.get('pass_time1', None)
     
     
     var_days = request.args.get('pass_days', None)

     
     


     var_sum_max_hr = request.args.get('pass_sum_max_hr', None)
     var_rowcount_highhr = request.args.get('pass_rowcount_highhr', None)
     var_fullnamehhr = request.args.get('pass_fullnamehhr', None)
     var_sum_low_hr = request.args.get('pass_sum_low_hr', None)
     var_rowcount_lowhr = request.args.get('pass_rowcount_lowhr', None)
     var_fullnamelowhr = request.args.get('pass_fullnamelowhr', None)


          
     return render_template('results_fitbit_heartrate.html', passedcount2 = var_count2, passedtime1 = var_time1, 
        passeddays=var_days, passedsum_max_hr = var_sum_max_hr, passedrowcount_highhr = var_rowcount_highhr,
        passedfullnamehhr = var_fullnamehhr, passedsum_low_hr =var_sum_low_hr, 
        passedrowcount_lowhr = var_rowcount_lowhr, passedfullnamelowhr = var_fullnamelowhr )  




@app.route('/fitbit_pers_inf/') 

def fitbit_pers_inf():

     
     var_time2 = request.args.get('pass_time2', None)
     var_rel = request.args.get('pass_rel', None)
     var_count1 = request.args.get('pass_count1', None)
     var_pass_days = request.args.get('pass_days', None)

     var_pass_fullnamemon = request.args.get('pass_fullnamemon', None)
     var_pass_fullnamewed = request.args.get('pass_fullnamewed', None)
     var_pass_fullnamesat = request.args.get('pass_fullnamesat', None)

    
     return render_template('results_fitbit_personal_inferences.html',  
        passeddays = var_pass_days, passedtime2 = var_time2,passedrel=var_rel,
        passedcount1 = var_count1, passedfullnamemon = var_pass_fullnamemon, 
        passedfullnamewed = var_pass_fullnamewed, passedfullnamesat = var_pass_fullnamesat) 

@app.route('/fitbit_sleep_inf/') 

def fitbit_sleep_inf():

    var_count3 = request.args.get('pass_count3', None)
    var_time3 = request.args.get('pass_time3', None)
    var_avgsleephours = request.args.get('pass_sleephours', None)
    var_avgsleephours_wdays = request.args.get('pass_sleephours_wdays', None)
    var_avgsleephours_wends = request.args.get('pass_sleephours_wends', None)
    
    var_avgsleepstarthour = request.args.get('pass_sleepstarthour', None)
    var_avgsleepstarthour_wdays = request.args.get('pass_sleepstarthour_wdays', None)
    var_avgsleepstarthour_wends = request.args.get('pass_sleepstarthour_wends', None)

    var_avgwakeuptime = request.args.get('pass_wakeuptime', None)
    var_avgwakeuptime_wdays = request.args.get('pass_wakeuptime_wdays', None)
    var_avgwakeuptime_wends = request.args.get('pass_wakeuptime_wends', None)


    var_lightsleep = request.args.get('pass_lightsleep', None)
    var_deepsleep = request.args.get('pass_deepsleep', None)
    var_remsleep = request.args.get('pass_remsleep', None)
    var_pass_days2 = request.args.get('pass_days2', None)
    var_fullnamehoursofsleep = request.args.get('pass_fullnamehoursofsleep', None)
    var_fullnameweekendsleepfacts = request.args.get('pass_fullnameweekendsleepfacts', None)
    var_fullnameweekdaysleepfacts = request.args.get('pass_fullnameweekdaysleepfacts', None)

    
    
    return render_template('results_fitbit_sleep_inferences.html', passedcount3 = var_count3, passedtime3 = var_time3,  passedsleephour=var_avgsleephours,
        passedsleephour_wdays=var_avgsleephours_wdays,
        passedsleephour_wends=var_avgsleephours_wends, passedsleepstarthour=var_avgsleepstarthour, passedsleepstarthour_wdays=var_avgsleepstarthour_wdays, 
        passedsleepstarthour_wends=var_avgsleepstarthour_wends,passedwakeuptime=var_avgwakeuptime, passedwakeuptime_wdays=var_avgwakeuptime_wdays,
        passedwakeuptime_wends=var_avgwakeuptime_wends, passeddays = var_pass_days2,passedlightsleep= var_lightsleep,passeddeepsleep= var_deepsleep, 
        passedremsleep=var_remsleep, passeddays2 = var_pass_days2, passedfullnamehoursofsleep = var_fullnamehoursofsleep , 
        passedfullnameweekendsleepfacts =var_fullnameweekendsleepfacts, passedfullnameweekdaysleepfacts = var_fullnameweekdaysleepfacts)



@app.route('/mi4c_personal/') 

def mi4c_personal():


    var_count1 = request.args.get('pass_pr1_rowcount', None)
    var_time1 = request.args.get('pass_time1', None)
    var_days_in_data_pers = request.args.get('pass_days_in_data_pers', None)
    var_jew_rel = request.args.get('pass_jew_rel', None)
    var_activity_level = request.args.get('pass_activity_level', None)
    
    var_mean_steps = request.args.get('pass_mean_steps', None)
    var_pregnancy_pos= request.args.get('pass_pregnancy_pos', None)

    var_fullnamemi= request.args.get('pass_fullnamemi', None)
    var_fullnamemiactivityindex= request.args.get('pass_fullnamemiactivityindex', None)
    var_fullnameprpos= request.args.get('pass_fullnameprpos', None)


    
   

    return render_template('results_mic4_personal_inferences.html', passedpr1_rowcount = var_count1, 
        passedtime1 = var_time1, passedjew_rel=var_jew_rel, passedactivity_level=var_activity_level,
        passedmean_steps=var_mean_steps, passedpregnancy_pos=var_pregnancy_pos ,
        passedfullnamemi=var_fullnamemi, passedfullnamemiactivityindex=var_fullnamemiactivityindex, 
        passedfullnameprpos=var_fullnameprpos, passeddays_in_data_pers = var_days_in_data_pers) 







@app.route('/mic4_heartrate/') 

def mic4_heartrate():


    var_count3 = request.args.get('pass_count3', None)
    var_time3 = request.args.get('pass_time3', None)
    var_sum_max_hr = request.args.get('pass_sum_max_hr', None)
    var_sum_low_hr = request.args.get('pass_sum_low_hr', None)
    var_pr3_rowcount = request.args.get('pass_pr3_rowcount', None)
    var_days_in_data = request.args.get('pass_days_in_data', None)

    var_fullnamemihighhr = request.args.get('pass_fullnamemihighhr', None)
    var_fullnamemilowhr = request.args.get('pass_fullnamemilowhr', None)


    


    return render_template('results_mic4_heartrate_inferences.html', passedcount3 = var_count3, 
        passedtime3 = var_time3,passedsum_max_hr=var_sum_max_hr, passedsum_low_hr=var_sum_low_hr,
        passeddays_in_data=var_days_in_data, passedfullnamemihighhr=var_fullnamemihighhr,
        passedfullnamemilowhr=var_fullnamemilowhr, passedpr3_rowcount = var_pr3_rowcount) 



@app.route('/mic4c_sleepdata/') 

def mic4c_sleepdata():


    var_count2 = request.args.get('pass_count2', None)
    var_time2 = request.args.get('pass_time2', None)
    var_days = request.args.get('pass_days', None)
    var_pr2_rowcount = request.args.get('pass_pr2_rowcount', None)
    var_wdays_hours_of_sleep = request.args.get('pass_wdays_hours_of_sleep', None)
    var_wend_hours_of_sleep = request.args.get('pass_wend_hours_of_sleep', None)


    var_wdays_gotosleep = request.args.get('pass_wdays_gotosleep', None)
    var_wend_gotosleep = request.args.get('pass_wend_gotosleep', None)
    var_wdays_wakeup = request.args.get('pass_wdays_wakeup', None)
    var_wend_wakeup = request.args.get('pass_wend_wakeup', None)
    var_deep = request.args.get('pass_deep', None)
    var_shallow = request.args.get('pass_shallow', None)
    var_wtime = request.args.get('pass_wtime', None)

    var_fullnamewdhoursofsleep = request.args.get('pass_fullnamewdhoursofsleep', None)
    var_fullnamesleepstarthour = request.args.get('pass_fullnamesleepstarthour', None)
    var_fullnamesleepfactsmi = request.args.get('pass_fullnamesleepfactsmi', None)

  

    return render_template('results_mic4_sleepdata_inferences.html', passedcount2 = var_count2, 
        passedtime2 = var_time2, passeddays = var_days,   passedwdays_hours_of_sleep = var_wdays_hours_of_sleep,
        passedwend_hours_of_sleep = var_wend_hours_of_sleep, passedwdays_gotosleep = var_wdays_gotosleep, passedwend_gotosleep=var_wend_gotosleep ,
        passedwdays_wakeup=var_wdays_wakeup, passedwend_wakeup=var_wend_wakeup, passeddeep=var_deep,
        passedshallow = var_shallow, passedwtime =var_wtime, passedfullnamewdhoursofsleep = var_fullnamewdhoursofsleep, 
        passedfullnamesleepstarthour = var_fullnamesleepstarthour, passedfullnamesleepfactsmi = var_fullnamesleepfactsmi, 
        passedpr2_rowcount = var_pr2_rowcount )  




app.run(host='xxxx', port=xxx)