# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:34:59 2017

@author: gazkune
"""

import sys

import pandas as pd
import csv



MIT_DATASET = 'log_s1.csv'
TRANSFORMED_DATASET = 'mit_s1.csv'
SENSOR_NAMES = 'sensors_s1.txt'


# Main function
def main(argv):
    print "Let's do this!"

    # Read the MIT dataset    
    df = pd.read_csv(MIT_DATASET, parse_dates=[0], index_col=0)    
    
    # Get rid off unuseful columns for this experiment
    del df['sensor_id']
    #del df['sens_event']
    
    # Add a new column and initialise with 'None'    
    df['label'] = 'None'
    
    # Update 'label' column with the activity name
    # Replace blank spaces in sensor names with underscores
    for i in xrange(len(df.index)):
        if df.ix[i, 'act_event'] == 'Start':
            activity = str(df.ix[i, 'activity']).replace(" ", "_")
            
        if df.ix[i, 'act_event'] == 'End':
            activity = 'None'
            
        df.ix[i, 'label'] = activity
        
        old_name = str(df.ix[i, 'sensor_name'])        
        df.ix[i, 'sensor_name'] = old_name.replace(" ", "_")
    
    # Remove rows where no sensor activation exists
    df = df.drop(df.index[df['act_event'] == 'Start'])
    df = df.drop(df.index[df['act_event'] == 'End'])
    df = df.drop(df.index[df['sens_event'] == 'OFF'])
    
    # Get rid off unuseful columns for this experiment
    del df['activity']
    del df['act_event']
           
    # Duplicate 'sensor_name' column
    df['location'] = df['sensor_name']
    
    # Reorder the columns
    df = df[['sensor_name', 'location', 'sens_event', 'label']]
    
    print df.head(10)
    
    # Store the df in a csv file
    # In order to store in the desired format and to avoid blank space problems
    # and quoting, convert the index to a standard column first
    df.reset_index(level=df.index.names, inplace=True)
    df['index'] = df['index'].astype(str)
    df['day'], df['hour'] = df['index'].str.split(' ', 1).str
    del df['index']
    
    # Reorder the columns
    df = df[['day', 'hour', 'sensor_name', 'location', 'sens_event', 'label']]
    
    # Write to csv
    #df.to_csv(TRANSFORMED_DATASET, header=False, sep=' ', index=False)
    
    # Now change the 'label' column to 'Other_Activity' to be processed by CASAS
    df['label'] = 'Other_Activity'
    
    # Write to a new csv file with 'test' prefix
    #df.to_csv('test_'+TRANSFORMED_DATASET, header=False, sep=' ', index=False)
    
    # Write also all the names of sensors
    names = df['sensor_name'].unique()
    
    with open(SENSOR_NAMES, "w") as text_file:
        for i in xrange(len(names)):
            text_file.write(names[i] + " ")

    
    
    
if __name__ == "__main__":
   main(sys.argv)
