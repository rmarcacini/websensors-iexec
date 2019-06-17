#!/usr/bin/env python
# coding: utf-8

#libraries
import sys
import hashlib
import os 
import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
import datetime
import nltk
import sklearn
from sklearn.preprocessing import MinMaxScaler
from numbers import Number
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import sklearn.metrics
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import distance
import traceback
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
nltk.download('stopwords')
nltk.download('punkt')


# In[21]:


# reading events
debug = True

def getEventsFromURL(url_data):
    
    r = requests.get(url_data)
    events = r.json()
        
    # checking events format.
    log("Checking events format.")
    status, msg = checkEventData(events)
    log(status)
    log(msg)
    
    return events, status
    

def getTitleTokens(title):
    # stopwords removal
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(title)
    tokens = [i for i in tokens if not i in stop_words]
    
    return tokens

def getNamedEntities(title):
    nlp = spacy.load('en')
    doc = nlp(title)
    ner = []
    for ne in doc.ents:
        ner.append([ne.text, ne.label_])
    
    return ner
    
def getTimeFeatures(str_date):
    
    date_time_obj = datetime.datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')
    month =  date_time_obj.strftime('%B')
    day_week = date_time_obj.strftime('%A')
    year = date_time_obj.strftime("%Y")
    date = date_time_obj.strftime("%Y-%m-%d")
    
    time_features = [month, day_week, year, date]
    
    return time_features
    
def log(msg):
    if(debug): print(msg)
    

def getNegativeWords():
    negative_words = []
    filepath = 'resources/negative.words'  
    log('Reading negative words...')
    with open(filepath) as fp:  
       line = fp.readline()
       while line:
           negative_words.append(line.strip())
           line = fp.readline()
    
    log('Number of negative words: '+str(len(negative_words)))
    return negative_words
    
    
def removeEventsWithNegativeWords(events, negative_words):
    events_to_remove = []
    
    for event in events:
        for word in negative_words:
            title = event['title'].lower()
            if word in title:
                events_to_remove.append(event['id'])
                continue
        
    for event_id in events_to_remove:
        for event in events:
            if event['id']==event_id:
                log('Removing event '+event_id+' (contains negative words)')
                events.remove(event)
                continue
    
    return events

def removeEventsUngeoreferenced(events):
    events_to_remove = []
    
    for event in events:
        if 'lat' not in event:
            events_to_remove.append(event['id'])
            continue
        if 'lng' not in event:
            events_to_remove.append(event['id'])
            continue
        
    for event_id in events_to_remove:
        for event in events:
            if event['id']==event_id:
                log('Removing event '+event_id+' (absence of coordinates)')
                events.remove(event)
                continue
    
    return events
    
def checkEventData(events):
    
    total = len(events)
    if(total==0): return False, 'The set of positive events is empty.'
    if(total> 100): return False, 'The maximum number (100) of positive events has been reached.'
    
    for event in events:
        if 'id' not in event: return False, "Event ID does not exist"
        if len(str(event['id'])) == 0: return False, "Event ID is empty"
        if len(str(event['id'])) > 50: return False, "Event ID is too long"
        
        if 'title' not in event: return False, "Event Title does not exist"
        if len(str(event['title'])) == 0: return False, "Event Title is empty"
        if len(str(event['title'])) > 5000: return False, "Event Title is too long"
        
        if 'date' not in event: return False, "Event Date does not exist"
        if len(str(event['date'])) > 50: return False, "Event Date is too long"
        if len(str(event['date'])) == 0: return False, "Event Date is empty"
        try:
            date_time_obj = datetime.datetime.strptime(str(event['date']), '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            s = str(e)
            return False, "Event Date format is invalid. "+s
        
        if 'url' not in event: return False, "Event URL does not exist"
        if len(str(event['url'])) == 0: return False, "Event URL is empty"
        if len(str(event['url'])) > 1500: return False, "Event URL is too long"
        
        if 'lat' not in event: return False, "Event Lat does not exist"
        if len(str(event['lat'])) == 0: return False, "Event Lat is empty"
        if len(str(event['lat'])) > 50: return False, "Event Lat is too long"
        if isinstance(float(event['lat']), Number)==False: return False, "Event Lat is not numeric"
       
        if 'lng' not in event: return False, "Event Lng does not exist"
        if len(str(event['lng'])) == 0: return False, "Event Lng is empty"
        if len(str(event['lng'])) > 50: return False, "Event Lng is too long"
        if isinstance(float(event['lng']), Number)==False: return False, "Event Lng is not numeric"
        
        if 'local' not in event: return False, "Event Local does not exist"
        if len(str(event['local'])) == 0: return False, "Event Local is empty"
        if len(str(event['local'])) > 500: return False, "Event Local is too long"        
        
        return True, 'OK'

def loadGloveModel(gloveFile):
    log("Loading Glove Model")
    f = open(gloveFile,'r')
    words = {}
    embedding = []
    counter = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        vector = np.array([float(val) for val in splitLine[1:]])
        words[word.lower()]=counter
        embedding.append(vector)
        counter+=1

    log("Glove Model: "+str(len(words))+" words.")
    return words, embedding

def getWordVector(word,words,embedding):
    if(word in word):
        return embedding[words[word.lower()]]
    else:
        return None
    
    
def getEventContentWordVector(event,words,embedding):
    tokens = getTitleTokens(event['title'].lower())
    
    event_content = []
    
    for token in tokens:
        if token in words:
            v = getWordVector(token,words,embedding)
            event_content.append(v)
    
    if(len(event_content)==0):
        print(tokens)
        log("Warning! Event "+event['id']+" without word vectors.")
    
    return np.average(event_content,axis=0)
    
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
def iexec_callback(sensor_id, status):
    result = "1"+str(sensor_id)+status
    consensus = hashlib.md5(str.encode(result))

    file = open("/iexec_out/callback.iexec","w") 
    file.write(result) 
    file.close()

    file = open("/iexec_out/determinism.iexec","w") 
    file.write(consensus.hexdigest()) 
    file.close() 


def get_parameter(parameter,parameters_vec):
    for item in parameters_vec:
        if 'name' in item and 'value' in item and item['name']==parameter:
            log('Parameter '+item['name']+' = '+str(item['value']))
            return float(item['value'])

    return None


def main():
    
    if(len(sys.argv)!=2):
        log('Error. Number of parameters is invalid. Enter the (web) sensor id on the iExec platform.')
        iexec_callback('10000000','09')
        exit(0)
        
    sensor_id = sys.argv[1]
    try:
        test_int = int(sensor_id)
    except Exception as e:
        log('Error. Invalid sensor id: '+sensor_id)
        iexec_callback('10000000','09')
        exit(0)
        
    if(len(sensor_id)!=8):
        log('Error. Invalid sensor id: '+sensor_id)
        iexec_callback('10000000','09')
        exit(0)
        
    
    ##### Getting sensor data #####
    r = requests.get('https://websensors.net.br/sensors/data/?id='+sensor_id)
    sensor_data = r.json()
    
    if('id' not in sensor_data or 'train' not in sensor_data or 'predict' not in sensor_data):
        log('Error. Invalid sensor data at '+'https://websensors.net.br/sensors/data/?id='+sensor_id)
        iexec_callback(sensor_id,'08')
        exit(0)

    ##### Loading event train and predict #####

    log("Loading training event set...")
    eventsTrain, statusTrain = getEventsFromURL(sensor_data['train'])

    log("Loading event set to predict...")
    eventsPredict, statusPredict = getEventsFromURL(sensor_data['predict'])

    if(len(eventsTrain) < 3):
        log('Error. The training set is too small (eventsTrain < 3)')
        iexec_callback(sensor_id,'07')
        exit(0)
        
    if(len(eventsPredict) == 0):
        log('Error. The set of events to predict is empty.')
        iexec_callback(sensor_id,'07')
        exit(0)
    
    if(statusTrain==False or statusPredict==False):
        log('Error. Invalid format in event set.')
        iexec_callback(sensor_id,'07')
        exit(0)
        
    # preprocessing events
    words, embedding = loadGloveModel('resources/glove.6B.50d.txt')
    
    event_labels = {}
    event_set = {} # removing duplicates
    for event in eventsTrain:
        event_labels[event['id']]=1
        event_set[event['id']] = event
    for event in eventsPredict:
        event_labels[event['id']]=0
        event_set[event['id']] = event

    events = [] # final event dataset
    for event_id in event_set:
        events.append(event_set[event_id])

    negative_words = getNegativeWords()

    events = removeEventsWithNegativeWords(events,negative_words)
    events = removeEventsUngeoreferenced(events)
        
    log('Total of events: '+str(len(events)))
        
    if(len(events) < 3):
        log('Error. The event set is too small (events < 3)')
        iexec_callback(sensor_id,'07')
        exit(0)
        
    
    # algorithm parameters (default)
    min_events = 50;
    min_confidence = 0.7
    max_distance_radius_km = 500
    tsne_perplexity = 5
    tsne_epsilon = 10
    osvm_nu = 0.1
    osvm_gamma = 0.1
    textual_distance_weight = 1
    geographic_distance_weight = 0.5
    smooth = 2
        
    try:
        p1 = get_parameter('Min. Events',sensor_data['parameters'])
        if p1 != None: min_events=p1

        p2 = get_parameter('Min. Confidence',sensor_data['parameters'])
        if p2 != None: min_confidence=p2

        p3 = get_parameter('Radius',sensor_data['parameters'])
        if p3 != None: max_distance_radius_km=p3

        p4 = get_parameter('t-SNE Perplexity',sensor_data['parameters'])
        if p4 != None: tsne_perplexity=p4

        p5 = get_parameter('t-SNE Epsilon',sensor_data['parameters'])
        if p5 != None: tsne_epsilon=p5

        p6 = get_parameter('OSVM Nu',sensor_data['parameters'])
        if p6 != None: osvm_nu=p6

        p7 = get_parameter('OSVM Gamma',sensor_data['parameters'])
        if p7 != None: osvm_gamma=p7

        p8 = get_parameter('Textual Distance Weight',sensor_data['parameters'])
        if p8 != None: textual_distance_weight=p8

        p9 = get_parameter('Geographic Distance Weight',sensor_data['parameters'])
        if p9 != None: geographic_distance_weight=p9

        p10 = get_parameter('Smooth',sensor_data['parameters'])
        if p10 != None: smooth=p10
            
    except Exception as e:
        log('Error. Invalid sensor parameters: '+sensor_id)
        iexec_callback(sensor_id,'06')
        exit(0)
    
    ##### Learning Classifier Model ####
    try:
        
        event_vectors = {}

        # event index
        event_index = {}
        counter = 0
        for event in events:
            event_index[event['id']] = counter
            counter+=1

        mdist_events = np.zeros((len(events),len(events)))

        for event in events:
            v = getEventContentWordVector(event,words,embedding)
            event_vectors[event['id']]=v


        for event1 in events:
            for event2 in events:

                i = event_index[event1['id']]
                j = event_index[event2['id']]

                # textual data (what)
                v1 = event_vectors[event1['id']]
                v2 = event_vectors[event2['id']]
                dist = textual_distance_weight*distance.cosine(v1, v2)
                # smoothing reference events 
                if(event_labels[event1['id']]==1 and event_labels[event2['id']]==1):
                    dist /= smooth

                #print(event1['title'])
                #print(event2['title'])
                #print("Distance = "+str(dist))
                mdist_events[i,j]=dist


                # geographic data (where)
                lat1 = float(event1['lat'])
                lng1 = float(event1['lng'])
                lat2 = float(event2['lat'])
                lng2 = float(event2['lng'])

                dist_km = haversine(lat1,lng1,lat2,lng2)

                if(dist_km <=  max_distance_radius_km):
                    dist = 1 - (abs(dist_km - max_distance_radius_km) / max_distance_radius_km)
                else:
                    dist = dist_km / max_distance_radius_km

                dist *= geographic_distance_weight

                mdist_events[i,j] += dist

        # learning embedding model from distance matrix
        X_tsne = TSNE(metric="precomputed",perplexity=30,random_state=777).fit_transform(mdist_events)

        # learning event prediction model
        svm_event_train = []
        svm_event_prediction = []
        svm_event_prediction_id = []
        for event_id in event_index:
            if(event_labels[event_id]==1):
                x = X_tsne[event_index[event_id]]
                svm_event_train.append([x[0],x[1]])
            else:
                x = X_tsne[event_index[event_id]]
                svm_event_prediction.append([x[0],x[1]])
                svm_event_prediction_id.append(event_id)

        svm_event_train = np.array(svm_event_train)
        svm_event_prediction = np.array(svm_event_prediction)


        # fit the model
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(svm_event_train)

        y_train = clf.predict(svm_event_train)
        y_prediction = clf.predict(svm_event_prediction)

        # ploting the model
        # grid 
        xx, yy = np.meshgrid(np.linspace(np.min(X_tsne)*1.5, np.max(X_tsne)*1.5, 1000), np.linspace(np.min(X_tsne)*1.5, np.max(X_tsne)*1.5, 1000))

        # plot the line, the points, and the nearest vectors to the plane
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("Websensors-OCEV-GloVe50d-tSNE-OSVM-v1")
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        b1 = plt.scatter(svm_event_train[:, 0], svm_event_train[:, 1], c='white', s=s, edgecolors='k')
        b2 = plt.scatter(svm_event_prediction[:, 0], svm_event_prediction[:, 1], c='blueviolet', s=s,
                         edgecolors='k')

        n_error_train = y_train[y_train == -1].size

        plt.axis('tight')
        plt.xlim((np.min(X_tsne)*1.5, np.max(X_tsne)*1.5))
        plt.ylim((np.min(X_tsne)*1.5, np.max(X_tsne)*1.5))
        plt.legend([a.collections[0], b1, b2],
                   ["learned frontier", "training events",
                    "new events"],
                   loc='center left', bbox_to_anchor=(1, 0.5),
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "error train: %d/%d"
            % (n_error_train, len(y_train)))

        plt.savefig('classifier-model.png',bbox_inches='tight',dpi=100)
        
        
        # saving events
        log('Events predicted:')
        count = 0
        total_predicted = 0
        for event_id in svm_event_prediction_id:
            log(event_id+" = "+str(y_prediction[count]))
            if(y_prediction[count]==1):
                total_predicted += 1
            count += 1

            
        confidence = total_predicted / min_events
        if confidence > 1: confidence=1.0
        if(count < min_events):
            confidence = 0
            iexec_callback(sensor_id,'00')

        if(confidence >= min_confidence):
            log('A total of '+str(total_predicted)+' similar events were identified. Confidence = '+str(confidence))
            iexec_callback(sensor_id,'02')
        else:
            log('A total of '+str(total_predicted)+' similar events were identified. Confidence = '+str(confidence))
            iexec_callback(sensor_id,'01')
    
    except Exception as e:
        log('Error in event classifier. '+traceback.format_exception(*sys.exc_info()))
        iexec_callback(sensor_id,'05')
        exit(0)

if __name__ == "__main__":
    main()
    
    
