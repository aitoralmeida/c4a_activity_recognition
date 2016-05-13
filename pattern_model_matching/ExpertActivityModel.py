# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:49:28 2016

@author: gazkune
"""
from datetime import datetime

class ExpertActivityModel:
    
    def __init__(self, name=None, infodict=None):       
        
        if name is not None and isinstance(infodict, dict):
            self.name = name
            self.locations = infodict["locations"]
            self.actions = infodict["actions"]
            self.duration = infodict["duration"]
            # convert strings to datetime types
            #self.start = infodict["start"]
            self.start = []
            self.convertStartList(infodict["start"])
        else:
            self.name = ""
            self.locations = []
            self.actions = []
            self.duration = -1
            self.start = []
        

    def printEAM(self):
        print self.name
        print '   locations:', self.locations
        print '   actions:', self.actions
        print '   duration:', self.duration
        print '   start:'
        for timerange in self.start:
            print '    ', timerange[0].strftime("%H:%M"), '-', timerange[1].strftime("%H:%M")
        
    def convertStartList(self, rangelist):
        for timerange in rangelist:
            time1 = datetime.strptime(timerange[0], "%H:%M")
            time2 = datetime.strptime(timerange[1], "%H:%M")
            self.start.append([time1, time2])