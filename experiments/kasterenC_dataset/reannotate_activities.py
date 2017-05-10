# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:52:10 2017

@author: gazkune
"""

import pandas as pd

ACT_FILE = 'activities.csv'
OUTPUT = 'activities_reannotated.csv'

# open activity dataset file
act_df = pd.read_csv(ACT_FILE, parse_dates=[0, 1], header=None, sep=',', names=['start', 'end', 'activity'])
columns = act_df.columns
resdf = pd.DataFrame(columns=columns)

i = 0
while i < len(act_df):
    print i
    if i < len(act_df)-1:
        end_i = act_df.ix[i, 'end']
        next_start = act_df.ix[i+1, 'start']
        
        if end_i > next_start:
            # There is an overlapping between activity i and i+1
            # Find all overlapping activities
            overlapping = [i+1]
            j = i+2
            finish = False
            while not finish:
                if end_i > act_df.ix[j, 'start']:
                    overlapping.append(j)
                    j = j+1
                else:
                    finish = True
            # Now overlapping has the indices of overlapping activities in act_df
            # Add new rows to resdf in consequence
            start_i = act_df.ix[i, 'start']
            activity_i = int(act_df.ix[i, 'activity'])
            s = pd.Series([start_i, next_start - pd.DateOffset(seconds=1), activity_i], index=columns)
            resdf = resdf.append(s, ignore_index=True)
            for row in overlapping:
                if row -1 > i:
                    # Add new activity_i with start and end times between row-1 and row
                    start = act_df.ix[row-1, 'end'] + pd.DateOffset(seconds=1)
                    end = act_df.ix[row, 'start'] - pd.DateOffset(seconds=1)
                    s = pd.Series([start, end, activity_i], index=columns)
                    resdf = resdf.append(s, ignore_index=True)

                # Add activity in row
                resdf = resdf.append(act_df.ix[row], ignore_index=True)
                # Check whether we have to treat the activity before
            # Now add the last piece of activity_i
            last_ix = overlapping[len(overlapping)-1]
            start = act_df.ix[last_ix, 'end'] + pd.DateOffset(seconds=1)
            s = pd.Series([start, end_i, activity_i], index=columns)
            resdf = resdf.append(s, ignore_index=True)
            
            # Update i
            i = last_ix + 1
        else:
            # No overlapping; just append the row to resdf
            start = act_df.ix[i, 'start']
            end = act_df.ix[i, 'end']
            activity = int(act_df.ix[i, 'activity'])
            s = pd.Series([start, end, activity], index=columns)
            resdf = resdf.append(s, ignore_index=True)
            # update i
            i = i + 1
    else:
        i = i + 1
            
# Write resdf to OUTPUT
resdf['activity'] = resdf['activity'].astype(int)
resdf.to_csv(OUTPUT, header=False, index=False)