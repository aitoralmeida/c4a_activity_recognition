# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:32:27 2015

@author: gazkune
"""

"""
This tool is to transform Kasteren datasets to our csv format
"""

import sys, getopt
import numpy as np
import time, datetime
import pandas as pd
from copy import deepcopy
from pandas.tseries.resample import TimeGrouper

"""
Function to parse arguments from command line
Input:
    argv -> command line arguments
Output:
    sense_file -> csv file with sensor data (start, end, sensor-id, value=1)
    act_file -> csv file with activity data (start, end, activity-id)
    output -> csv file where timestamped sensor activation are listed where
        [timestamp, sensor, activity, start-end]
"""

def parseArgs(argv):
   sense_file = ''
   act_file = ''
   output = ''
   
   try:
      opts, args = getopt.getopt(argv,"hs:a:o:",["sense=","act=","out="])
   except getopt.GetoptError:
      print 'kasteren_data_transformer.py -sense <sense_dataset> -act <act_dataset> -out <output_dataset>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'kasteren_data_transformer.py -sense <sense_dataset> -act <act_dataset> -out <output_dataset>'
         sys.exit()
      elif opt in ("-s", "--sense"):
         sense_file = arg      
      elif opt in ("-a", "--act"):
         act_file = arg
      elif opt in ("-o", "--out"):
         output = arg
         
    
       
   return sense_file, act_file, output

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
Dataset transformation function
"""

def transformDataset(sense_file, act_file):
    # List of activities which we want to store in the transformed dataset
    target_acts = [13, 5, 15]    

    # open sense dataset file    
    sense_df = pd.read_csv(sense_file, parse_dates=0, names=['start', 'end', 'sensor', 'value'])
    print 'Sensor dataset:'
    print sense_df.head()
    
    # open activity dataset file    
    act_df = pd.read_csv(act_file, parse_dates=0, names=['start', 'end', 'activity'])
    print 'Activity dataset'
    print act_df.head()
    
    # build sensor dictionary
    sensor_dict = {1:'Microwave', 5:'HallToiletDoor', 6:'HallBathroomDoor', 7:'CupsCupboard', 8:'Fridge', 9:'PlatesCupboard', 12:'Frontdoor', 13:'Dishwasher', 14:'ToiletFlush', 17:'Freezer', 18:'PansCupboard', 20:'Washingmachine', 23:'GroceriesCupboard', 24:'HallBedroomDoor'}
    
    # build activity dict
    act_dict = {1:'LeaveHouse', 4:'UseToilet', 5:'TakeShower', 10:'GoToBed', 13:'PrepareBreakfast', 15:'PrepareDinner', 17:'GetDrink'}
    
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
    trans_df = trans_df[trans_df['activity'] != 'None']
    
    print 'Trans df:'
    print trans_df.head(50)
    
    return trans_df
    

"""
Main function
"""

def main(argv):
    # call the argument parser
    [sense_file, act_file, output] = parseArgs(argv[1:])
    print 'Sense:', sense_file
    print 'Act:', act_file
    print 'Output:', output
    
    dataset_df = transformDataset(sense_file, act_file)
    dataset_df.to_csv(output)
    
if __name__ == "__main__":   
    main(sys.argv)