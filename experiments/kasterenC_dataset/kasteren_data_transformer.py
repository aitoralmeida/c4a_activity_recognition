# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:32:27 2015

@author: gazkune
"""

"""
This tool is to transform Kasteren datasets to various useful formats:
1- Our custom format (OUT_GORKA_FILE)
2- CASAS format but with annotated activities (to be used for evaluation)
3- CASAS format to be processed by the AD tool (activity label set to 'Other_Activity')
"""

import sys, getopt
import numpy as np
import time, datetime
import pandas as pd
from copy import deepcopy
from pandas.tseries.resample import TimeGrouper

ACTIVITY_LABELS = 'activity_labels.txt'
SENSOR_LABELS = 'sensor_labels.txt'
ACT_FILE = 'activities_reannotated.csv'
SENSE_FILE = 'sensors.csv'
OUT_GORKA_FILE = 'kasterenC.csv'
OUT_EVAL_FILE = 'kasterenC_groundtruth.csv'
OUT_CASAS_FILE = 'test_kasterenC.csv'


"""
Function to detect and remove overlapping activities
Input:
    act_df: pd.DataFrame with activities (start, end, activity)
    act_dict: dictionaty of activity ids and names {id:'Activity'}
Output:
    act_df: filtered act_df, where overlapping activities do not exist
"""
def detectOverlappingActivities(act_df, act_dict):
    # For testing purposes, remove index 53
    act_df = act_df.drop([53])
    for index in act_df.index:
        start = act_df.loc[index, 'start']
        end = act_df.loc[index, 'end']
        act_id = act_df.loc[index, 'activity']
        #print index, start < end
        
        
        #print 'Activity', act_dict[act_id], 'start:', start, 'end:', end
        overlapping = act_df[np.logical_and(act_df['start'] > start, act_df['end'] < end)]
        #activity_index = filtered_df[np.logical_or(filtered_df['a_start_end'] == 'start', filtered_df['a_start_end'] == 'end')].index
        if not overlapping.empty:
            #print '--------------------------'
            #print 'Activity', act_dict[act_id], start, end
            #print 'Overlapping activities:'
            #print overlapping.head()
            act_df = act_df.drop([index])
    
    #print 'Activities after removing overlapping'
    #print act_df.head(50)
    return act_df
        
"""
Use label files to build dictionaries for activities and sensors
"""
def sensActDicts():
    sensor_dict = {}
    act_dict = {}
    
    with open(SENSOR_LABELS, 'r') as sens_labels:
        for line in sens_labels:
            sensid = int(line.split(' ')[0])
            sensname = line.split(' ')[1].rstrip('\n')
            sensor_dict[sensid] = sensname
            
    with open(ACTIVITY_LABELS, 'r') as act_labels:
        for line in act_labels:
            actid = int(line.split(' ')[0])
            actname = line.split(' ')[1].rstrip('\n')
            act_dict[actid] = actname
            
    return sensor_dict, act_dict

"""
Dataset transformation function
"""

def transformDataset():    

    # open sense dataset file    
    #sense_df = pd.read_csv(SENSE_FILE, parse_dates=0, names=['start', 'end', 'sensor', 'value'])
    sense_df = pd.read_csv(SENSE_FILE, parse_dates=[0, 1], header=None, sep=',', names=['start', 'end', 'sensor', 'value'])
    print 'Sensor dataset:'
    print sense_df.head()
    
    # open activity dataset file    
    act_df = pd.read_csv(ACT_FILE, parse_dates=[0, 1], header=None, sep=',', names=['start', 'end', 'activity'])

    print 'Activity dataset'
    print act_df.head()
    
    # build sensor dictionary
    #sensor_dict = {1:'Microwave', 5:'HallToiletDoor', 6:'HallBathroomDoor', 7:'CupsCupboard', 8:'Fridge', 9:'PlatesCupboard', 12:'Frontdoor', 13:'Dishwasher', 14:'ToiletFlush', 17:'Freezer', 18:'PansCupboard', 20:'Washingmachine', 23:'GroceriesCupboard', 24:'HallBedroomDoor'}
    
    # build activity dict
    #act_dict = {1:'LeaveHouse', 4:'UseToilet', 5:'TakeShower', 10:'GoToBed', 13:'PrepareBreakfast', 15:'PrepareDinner', 17:'GetDrink'}
    [sensor_dict, act_dict] = sensActDicts()
    
    # List of activities which we want to store in the transformed dataset
    #target_acts = [13, 5, 15]    
    target_acts = act_dict.keys()
    
    #act_df = detectOverlappingActivities(act_df, act_dict)
    #print 'Activities after removing overlapping'
    #print act_df.head(50)
    
    # Initialize transformed dataset
    trans_df = pd.DataFrame(index = sense_df['start'].values)
    sensors = sense_df['sensor'].values    
    snames = []
    for i in xrange(len(sensors)):
        snames.append(sensor_dict[sensors[i]])
        
    trans_df['sensor'] = snames
    trans_df['activity'] = ['None']*len(trans_df)
    trans_df['start/end'] = ['']*len(trans_df)
    #print 'Trans df:'
    #print trans_df.head(50)
    
    # Label each sensor activation with an activity name, using act_df start/end times
    for index in act_df.index:        
        act_id = act_df.loc[index, 'activity']
        try:
            # Check whether the activity is in target (try/except block)
            target_acts.index(act_id)
            act_name = act_dict[act_id]
            index_list = trans_df[np.logical_and(trans_df.index >= act_df.loc[index, 'start'], trans_df.index <= act_df.loc[index, 'end'])].index
            #print 'Activity', act_name
            #print index_list
            trans_df.loc[index_list, 'activity'] = act_name
            aux = 0
            for i in index_list:
                if aux == 0:
                    trans_df.loc[i, 'start/end'] = 'start'
                elif aux == len(index_list) - 1:
                    trans_df.loc[i, 'start/end'] = 'end'
            
                aux = aux + 1
        except ValueError:
            pass
                
        
    # Remove all None actions to have a clear dataset (only activities)
    #trans_df = trans_df[trans_df['activity'] != 'None']
    # Repeat sensor column as location
    trans_df['location'] = trans_df['sensor']
    trans_df = trans_df[['sensor', 'location', 'activity', 'start/end']]
    print 'Trans df:'
    print trans_df.head(50)
    
    return trans_df
    

"""
Main function
"""

def main(argv):
        
    dataset_df = transformDataset()
    
    # Store the first file
    aux1 = dataset_df[['sensor', 'activity', 'start/end']]
    aux1.to_csv(OUT_GORKA_FILE)    
    
    # Store the second file: some trnasformations are required
    aux2 = dataset_df[['sensor', 'location', 'activity']]
    # In order to avoid 2 around dates, we have to convert the index in a normal column
    aux2.reset_index(level=0, inplace=True)
    print aux2.columns
    # Split 'index0 column into two
    #aux2['index'].apply(lambda x: x.strftime('%y-%m-%d %H:%M:%S'))
    aux2['index'] = aux2['index'].dt.strftime('%Y-%m-%d %H:%M:%S')
    aux2['day'], aux2['time'] = aux2['index'].str.split(' ', 1).str
    # Add a new column with 'ON' value
    aux2['state'] = 'ON'
    aux2 = aux2[['day', 'time', 'sensor', 'location', 'state', 'activity']]
    aux2.to_csv(OUT_EVAL_FILE, header=False, index=False, sep=' ')
    
    # Store the hird file
    # Use 'Other_Activity' for the 'activity' column for CASAS AD tool 
    aux2['activity'] = 'Other_Activity'    
    aux2 = aux2[['day', 'time', 'sensor', 'location', 'state', 'activity']]
    aux2.to_csv(OUT_CASAS_FILE, header=False, index=False, sep=' ')
    
if __name__ == "__main__":   
    main(sys.argv)