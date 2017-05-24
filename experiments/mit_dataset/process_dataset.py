# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:03:11 2015

@author: gazkune
"""

"""
This tool is to annotate activities from a dataset
"""

import sys, getopt
import numpy as np
import time, datetime
import pandas as pd
import json
import csv

"""
Function to parse arguments from command line. If context_model is provided, there is no
need for seed_activity_models
Input:
    argv -> command line arguments
Output:
    dataset_file -> csv file where action properties are with time stamps and activity label
    output_file -> file to write TBD
"""

def parseArgs(argv):
   dataset_file = ''   
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hd:o:",["dataset=", "ofile="])      
   except getopt.GetoptError:
      print 'process_dataset.py -d <dataset> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'process_dataset.py -d <dataset> -o <outputfile>'
         sys.exit()
      elif opt in ("-d", "--dataset"):
         dataset_file = arg      
      elif opt in ("-o", "--ofile"):
         output_file = arg
   
   
   return dataset_file, output_file

"""
Function to read and format the csv file. The new format given 
is a list of dicts where:
activity: string
start: timestamp (datetime)
end: timestamp (datetime)
sens_ids: list of integers
sens_names: list of strings (fuse name + id to have unique names)
act_times: list of timestamps
dact_times: list of timestamps
Input:
    csvfilename: file name (string)
Output:
    formated_data: list of dictionaries for activities and sensors
"""
def formatCSVFile(csvfilename):
    
   csvfile = open(csvfilename)
   data = csv.reader(csvfile, delimiter=',')
   
   # Initialize block_size to 5, since the CSV file is repeated 
   # for 5 lines
   block_size = 5
   
   # Initialize the list to be filled and returned
   formated_data = []
   #iterare through the file, row by row   
   i = 0
   rows = []
   for row in data:
       #rows = [data[i], data[i+1], data[i+2], data[i+3], data[i+4]]
       rows.append(row)
       i = i + 1
       if i == block_size:           
           formated_data.append(csvRows2Dict(rows))
           i = 0
           rows = []
       
    
   return formated_data

       
"""
Function to read 5 row block of the CSV and return it in a dictionary. The format
of the dictionary is:
activity: string
start: timestamp (datetime)
end: timestamp (datetime)
sens_ids: list of integers
sens_names: list of strings (fuse name + id to have unique names)
act_times: list of timestamps
dact_times: list of timestamps
Input:
    rows: a list of 5 rows of the CSV file
Output:
    mydict: the dictionary with the formatted data
"""
def csvRows2Dict(rows):
    # Initialize the dict to be returned
    mydict = {}
    # Read five rows and create a dict
    
    # row 0 -> list of strings [Activity, Day, Start, End]
    row = rows[0]
    mydict['activity'] = row[0]
    # Obtain the start timestamp
    day = row[1]
    clock = row[2] # start time
    #print 'Row debugging:', row
    start = datetime.datetime.strptime(day + ' ' + clock, '%m/%d/%Y %H:%M:%S')
    mydict['start'] = start
    # Obtain the end timestamp
    clock = row[3] # end time
    end = datetime.datetime.strptime(day + ' ' + clock, '%m/%d/%Y %H:%M:%S')
    mydict['end'] = end
    
    # row 1 -> list of strings [sens_id1,..., sens_idN]
    row = rows[1]
    #print 'Row debugging:', row
    mydict['sens_ids'] = map(int, row)
    
    # row 2 -> list of strings [sens_name1,..., sens_nameN]
    mydict['sens_names'] = rows[2]
    for i in xrange(len(mydict['sens_names'])):
        mydict['sens_names'][i] = mydict['sens_names'][i] + '_' + str(mydict['sens_ids'][i])
    
    # row 3 -> list of strings [act_time1,..., act_timeN]
    row = rows[3]
    act_times = []
    for i in xrange(len(row)):
        act_times.append(datetime.datetime.strptime(day + ' ' + row[i], '%m/%d/%Y %H:%M:%S'))
        
    mydict['act_times'] = act_times
    
    # row 4 -> list of strings [dact_time1,..., dact_timeN]
    row = rows[4]
    dact_times = []
    for i in xrange(len(row)):
        dact_times.append(datetime.datetime.strptime(day + ' ' + row[i], '%m/%d/%Y %H:%M:%S'))
        
    mydict['dact_times'] = dact_times
    
    # Check whether sens_ids, sens_names, act_times and dact_times have the same length
    assert len(mydict['sens_ids']) == len(mydict['sens_names']) == len(mydict['act_times']) == len(mydict['dact_times']), 'Sizes of lists differ: sens_ids(' + len(mydict['sens_ids']) + '); sens_names(' + len(mydict['sens_names']) + '); act_times(' + len(mydict['act_times']) + '); dact_times(' + len(mydict['dact_times']) + ')'
    
    # Return de dictionary
    return mydict

"""
Function to visualize formated data from a list of dicts
Input:
    formated_data: list of dicts (formated CSV file)
    count: how many activities want to be visualized (if < 0 visualize all of them)
"""
def printFormatedData(formated_data, count):
    # Check the value of count
    if count < 0 or count > len(formated_data):
        count = len(formated_data)

    i = 0
    while i < count:
        print i, ' ', formated_data[i]['activity']
        print 'start:', formated_data[i]['start'].strftime('%m/%d/%Y %H:%M:%S')
        print 'end:', formated_data[i]['end'].strftime('%m/%d/%Y %H:%M:%S')
        print formated_data[i]['sens_ids']
        print formated_data[i]['sens_names']
        act_times = [dt.strftime('%H:%M:%S') for dt in formated_data[i]['act_times']]
        print act_times
        dact_times = [dt.strftime('%H:%M:%S') for dt in formated_data[i]['dact_times']]
        print dact_times
        print '------------------------------------'
        i = i + 1

"""
Function to select the activities which fulfil given criteria about
activity names and time constraints
Input:
    formated_data: list of dicts (converted CSV dataset)
    activities: list of strings (if empty, consider any activity)
    time_constraints: dict of datetime.datetime or datetime.time {'start: , 'end': }
    (selected activity's start time has to be inside the [start, end] lapse)
Output:
    selected_data: list of dicts (same format as input)
"""
def selectActivities(formated_data, activities, time_constraints):
    check_activities = True
    if len(activities) == 0:
        check_activities = False
        
    # Initialize selected_data
    selected_data = []
    
    # Iterate through formated_data and select those activities which 
    # fulfil the given criteria
    for i in xrange(len(formated_data)):
        auxdict = formated_data[i]
        if type(time_constraints['start']) == datetime.datetime:
            # Take the day into account
            if auxdict['start'] > time_constraints['start'] and auxdict['start'] < time_constraints['end']:
                # Use check_activities before appending
                if check_activities == False:
                    selected_data.append(auxdict)
                else:
                    if auxdict['activity'] in activities:
                        selected_data.append(auxdict)
                        
        elif type(time_constraints['start']) == datetime.time:
            # Check time constraints for every day in the dataset
            start_time = datetime.time(auxdict['start'].hour, auxdict['start'].minute, auxdict['start'].second)
            if start_time > time_constraints['start'] and start_time < time_constraints['end']:
                # Use check_activities before appending
                if check_activities == False:
                    selected_data.append(auxdict)
                else:
                    if auxdict['activity'] in activities:
                        selected_data.append(auxdict)
        else:
            # Error!
            print 'The type of the elements of parameter time_constraints are not correct!'
            print 'Expected datetime.datetime or datetime.time; Found:', type(time_constraints['start'])
            raise ValueError
    
    return selected_data

"""
Function to print the information from activity data in a readable way.
Only the activities and time slots performed in a day are printed (no sensors)
Input:
    activity_data: list of dicts (formated CSV file)
"""
def printDailyActivities(activity_data):
    # Generate a list of week days for printing
    #weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # We assume activity_data is sorted by starting time
    currentday = activity_data[0]['start']
    print currentday.strftime('%m/%d/%Y %A')
    
    for i in xrange(len(activity_data)):
        auxdict = activity_data[i]
        if currentday.day == auxdict['start'].day and currentday.month == auxdict['start'].month:
            print '  ', auxdict['activity'], '[', auxdict['start'].strftime('%H:%M:%S'), ',', auxdict['end'].strftime('%H:%M:%S'), ']'
        else:
            currentday = activity_data[i]['start']
            print currentday.strftime('%m/%d/%Y %A')
            print '  ', auxdict['activity'], '[', auxdict['start'].strftime('%H:%M:%S'), ',', auxdict['end'].strftime('%H:%M:%S'), ']'

"""
Function to calculate the statistics associated to an activity.
For now, only mean duration and standard deviation are calculated.
Input:
    formated_data: list of dicts (formated CSV file)
Output: 
    act_statistics: a list of dictionaries 
        {activity: name, instances: value, mean_duration: value, std_duration: value}
"""
def activityStatistics(formated_data):
    # Initialize the dictionary 
    std_dict = {}
    # Initialize the resulting list
    act_statistics = []
        
    # Iterate through formated_data and calculate statistics
    for i in xrange(len(formated_data)):
        # Extract the important activity info
        activity = formated_data[i]["activity"]
        duration = formated_data[i]["end"] - formated_data[i]["start"]
        if not activity in std_dict:
            std_dict[activity] = [duration.seconds]
        else:
            std_dict[activity].append(duration.seconds)
    
    # std_dict has a list of integers with registered duration per activity
    # convert this format to act_statistics lsit of dicts    
    for key in std_dict:
        auxdict = {}
        auxdict["activity"] = key
        auxdict["instances"] = len(std_dict[key])
        auxdict["mean_duration"] = np.mean(std_dict[key])
        auxdict["std_duration"] = np.std(std_dict[key])
        act_statistics.append(auxdict)
        
    
    return act_statistics

"""
Function to print in a structured way the information from act_statistics
"""
def printActStatistics(act_statistics):
    
    for i in xrange(len(act_statistics)):
       print act_statistics[i]["activity"]
       print '   instances:', act_statistics[i]["instances"]
       print '   mean:', act_statistics[i]["mean_duration"]
       print '   std:', act_statistics[i]["std_duration"]        
        
"""
Function to transform the list of dict format of data to pd.DataFrame.
Input:
    data: list of dicts as returned by formatCSVFile
    {activity: string
    start: timestamp (datetime)
    end: timestamp (datetime)
    sens_ids: list of integers
    sens_names: list of strings
    act_times: list of timestamps
    dact_times: list of timestamps}
Output:
    df: a pd.DataFrame, with the following columns:
    [timestamp, sensor_id, sensor_name, sens_event, activity, act_event]
"""
def transform2DF(data):
    # For each row of the DataFrame, build a dict where key = column_name
    # Append the dict to a list and build the DataFrame from that list
    rows = []
    
    for i in xrange(len(data)):
        dictrow = {}
        # First of all, store the activity start event in a row
        dictrow['timestamp'] = data[i]['start']
        dictrow['activity'] = data[i]['activity']
        dictrow['act_event'] = 'Start'
        rows.append(dictrow)
        
        # Store the activity end event in another row
        dictrow = {}
        dictrow['timestamp'] = data[i]['end']
        dictrow['activity'] = data[i]['activity']
        dictrow['act_event'] = 'End'
        rows.append(dictrow)
        
        
        # Go through sensors and store them in different rows
        sens_names = data[i]['sens_names']
        sens_ids = data[i]['sens_ids']
        act_times = data[i]['act_times']
        dact_times = data[i]['dact_times']
        for j in xrange(len(sens_names)):
            # A row for a sensor activation
            dictrow = {}
            dictrow['timestamp'] = act_times[j]
            dictrow['sensor_id'] = sens_ids[j]
            dictrow['sensor_name'] = sens_names[j]
            dictrow['sens_event'] = 'ON'
            rows.append(dictrow)
            # Another row for sensor deactivation
            dictrow = {}
            dictrow['timestamp'] = dact_times[j]
            dictrow['sensor_id'] = sens_ids[j]
            dictrow['sensor_name'] = sens_names[j]
            dictrow['sens_event'] = 'OFF'
            rows.append(dictrow)
            
    # We should have all the rows of our DataFrame in 'rows'
    # Build the DataFrame
    df = pd.DataFrame(rows)
    print df.head(50)
    df = df.set_index('timestamp')
    df.index.name = None
    
    # Now reorder the timestamp index
    df = df.sort_index()
    # Reorder the columns
    cols = ['sensor_id', 'sensor_name', 'sens_event', 'activity', 'act_event']
    df = df[cols]
    
    return df

"""
Function to count the sensors that appear for each activity. Average and 
standard deviation for sensor activation will also be shown
Input: 
    formated_data: a list of dicts, where
    activity: string
    start: timestamp (datetime)
    end: timestamp (datetime)
    sens_ids: list of integers
    sens_names: list of strings
    act_times: list of timestamps
    dact_times: list of timestamps
Output:
    TBD
"""
def sensorsPerActivity(formated_data):
    # Initialize the output variable
    activations = []
    # Initialize the auxiliar list
    auxlist = []
    # Iterate through formated_data and grab info
    for i in xrange(len(formated_data)):
        if not formated_data[i]['activity'] in auxlist:
            auxlist.append(formated_data[i]['activity'])
            auxdict = {}
            auxdict['activity'] = formated_data[i]['activity']
            auxdict['count'] = 1.0
            auxdict['sensors'] = []
            auxdict['freq'] = []
            # 'duration' will be a list of lists where all durations of a sensor
            # for this activity are stored to calculate the mean and std duration
            auxdict['durations'] = []
            auxdict['mean_duration'] = []
            auxdict['std_duration'] = []
            activations.append(auxdict)
            index = len(auxlist) - 1
        else:
            index = auxlist.index(formated_data[i]['activity'])
            # Increase the 'count' variable
            activations[index]['count'] = activations[index]['count'] + 1
            
        # Now we have the index of the activity
        # Iterate though the 'sens_names' array and count
        sens_names = formated_data[i]['sens_names']
        act_times = formated_data[i]['act_times']
        dact_times = formated_data[i]['dact_times']
        for j in xrange(len(sens_names)):
            # check whether sens_names is in activations[index]
            if not sens_names[j] in activations[index]['sensors']:
                activations[index]['sensors'].append(sens_names[j])
                activations[index]['freq'].append(1.0)
                duration = dact_times[j] - act_times[j]
                dur = duration.total_seconds()         
                activations[index]['durations'].append([dur])
            else:
                # Obtain the index for the 'sensors' list
                sindex = activations[index]['sensors'].index(sens_names[j])
                activations[index]['freq'][sindex] = activations[index]['freq'][sindex] + 1.0
                duration = dact_times[j] - act_times[j]
                dur = duration.total_seconds()                
                activations[index]['durations'][sindex].append(dur)
                
    # At this point, 'activations' has all the required info
    # Now calculate the mean and std for each sensor in each activity
    # Calculate also the frequency of each sensor
    for i in xrange(len(activations)):
        for j in xrange(len(activations[i]['sensors'])):
            activations[i]['freq'][j] = activations[i]['freq'][j] / activations[i]['count']
            #print activations[i]['durations'][j]
            avg = np.mean(activations[i]['durations'][j])
            std = np.std(activations[i]['durations'][j])
            activations[i]['mean_duration'].append(avg)
            activations[i]['std_duration'].append(std)
            
    # 'activations' is now ready to be returned
    return activations

"""
Function to print in a structured way the information in 'activations'
"""
def printActivations(activations):
    
    for i in xrange(len(activations)):
        print activations[i]['activity']
        print '   count:', activations[i]['count']
        print '   sensors:'
        for j in xrange(len(activations[i]['sensors'])):
            print '     ', activations[i]['sensors'][j], ', freq:', activations[i]['freq'][j], ', duration: (', activations[i]['mean_duration'][j], activations[i]['std_duration'][j], ')'

"""
Function to extract the action sequences from all the activities.
Input:
    formated_data: a list of dicts, where each dict is like:
        activity: string
        start: timestamp (datetime)
        end: timestamp (datetime)
        sens_ids: list of integers
        sens_names: list of strings
        act_times: list of timestamps
        dact_times: list of timestamps
Output:
    actions_seq: a list of dicts, where each dict is like:
        activity: string
        id_seq: list of lists ([][]) where each row represents a sensor sequence
            by ids (int)
        name_seq: list of lists ([][]) where each row represents a sensor sequence
            by names (string)
        freq: a list of frequencies (0..1) for each sequence in previous lists
"""
def getActionsSequences(formated_data):
    # TODO
    # Init the output list
    actions_seq = []
    # Init the dict for the output list elements
    #aux_dict = {}
    
    # Initialize the auxiliar list to keep the treaten activities in each iteration
    auxlist = []
    
    # Iterate 'formated_data'
    for i in xrange(len(formated_data)):
        if not formated_data[i]['activity'] in auxlist:
            auxlist.append(formated_data[i]['activity'])
            auxdict = {}
            auxdict['activity'] = formated_data[i]['activity']
            auxdict['id_seq'] = []
            auxdict['id_seq'].append(formated_data[i]['sens_ids'])
            auxdict['name_seq'] = []
            auxdict['name_seq'].append(formated_data[i]['sens_names'])
            # To calculate frequencies, we will store just the count of
            # each sequence; 0..1 values will be calculated in the end
            auxdict['freq'] = [1.0]
            # Apend 'auxdict' to 'actions_seq' output variable
            actions_seq.append(auxdict)
            
        else:
            index = auxlist.index(formated_data[i]['activity'])
            # The activity has already appeared
            # Now check whether the sensor sequences are the same
            seq_n = formated_data[i]['sens_names']
            seq_id = formated_data[i]['sens_ids']
            try:
                j = actions_seq[index]['id_seq'].index(seq_id)
                # The sequence already exists
                # Update only the corresponding 'freq' value
                actions_seq[index]['freq'][j] += 1.0
            except ValueError:
                # The sequence is not in the activity
                actions_seq[index]['id_seq'].append(seq_id)
                # Append also the sensor names sequence
                actions_seq[index]['name_seq'].append(seq_n)
                # Update 'freq' list appending a new value of 1
                actions_seq[index]['freq'].append(1.0)

    # At this point, 'actions_seq' has all the information
    # Now calculate the values for 'freq' (0..1)
    for i in xrange(len(actions_seq)):
        total = sum(actions_seq[i]['freq'])
        actions_seq[i]['freq'][:] = [x / total for x in actions_seq[i]['freq']]
        
    return actions_seq

"""
Function to print in a structured way the information in 'actions_seq'
"""
def printActionsSeq(actions_seq):
    
    for i in xrange(len(actions_seq)):
        print '------------------------'
        print actions_seq[i]['activity']
        for j in xrange(len(actions_seq[i]['freq'])):
            print '   freq:', actions_seq[i]['freq'][j]
            print '   ', actions_seq[i]['name_seq'][j]      


"""
Main function
"""

def main(argv):
   # call the argument parser 
   [dataset_file, output_file] = parseArgs(argv[1:])
   #print 'Dataset:', dataset_file   
   #print 'Output file:', output_file   
   
   
   formated_data = formatCSVFile(dataset_file)
   
   # Select only some activities from the dataset
   time_constraints = {}
   time_constraints['start'] = datetime.time(0, 0, 0)
   time_constraints['end'] = datetime.time(23, 59, 59)
   #target_activities = ['Toileting', 'Grooming', 'Preparing lunch']
   target_activities = []
   selected_data = selectActivities(formated_data, target_activities, time_constraints)
   # Sort selected activities using the start time of the activity
   selected_data = sorted(selected_data, key=lambda activity: activity['start'])
   
   #act_statistics = activityStatistics(selected_data)
   #printActStatistics(act_statistics)
   activations = sensorsPerActivity(selected_data)
   printActivations(activations)
   
   # Extract sensor sequences for activities
   #actions_seq = getActionsSequences(formated_data)
   #printActionsSeq(actions_seq)
   """      
   print ''    
   printDailyActivities(selected_data)
   """
   # Transform the dataset to pd.DataFrame
   #activities_df = transform2DF(selected_data)
   #activities_df.to_csv(output_file)
   #print activities_df.head(60)
   
   #printFormatedData(selected_data, -1)



if __name__ == "__main__":   
    main(sys.argv)