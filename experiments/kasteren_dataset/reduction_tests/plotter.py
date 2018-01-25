#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:50:58 2018

@author: gazkune
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd


RESULTS_FILE = 'results.csv'

results = pd.read_csv(RESULTS_FILE, header=0)

prec = 100*results['precision'].values
rec = 100*results['recall'].values
f1 = 100*results['f1'].values
days = results['days'].values

plt.plot(days, f1, '-o')
plt.plot(days, prec, '-x')
plt.plot(days, rec, '-*')
#plt.title('Performance for day number')
plt.ylabel('value (max = 100)')
plt.xlabel('number of days')
lgd = plt.legend(['F-Measure', 'Precision', 'Recall'], bbox_to_anchor=(1.04,1), loc="upper left")

save = True
if save == True:
    plt.savefig('reduction_tests.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.gcf().clear()
else:
    plt.show()