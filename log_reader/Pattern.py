# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:24:45 2016

@author: gazkune
"""

"""
Class Pattern to represent a pattern found by AD
"""

class Pattern:
    
    def __init__(self):
        # Pattern value calculated by AD
        self.value = 0.0
        # Actions or events inside the pattern
        self.actions = []
        # The pattern number assigned by AD
        self.number = -1
        # The number of instances of this pattern found
        # by AD in the dataset
        self.instances = 0
        
    def setValue(self, value):
        self.value = value
        
    def appendAction(self, action):
        self.actions.append(action)
        
    def setNumber(self, number):
        self.number = number
        
    def setInstances(self, inst_number):
        self.instances = inst_number
        
    def reset(self):
        self.value = 0.0
        self.actions = []
        self.number = -1
        self.instances = 0
        
    def printPattern(self):
        print 'Pattern', self.number
        print '  value:', self.value
        print '  instances:', self.instances
        print '  actions:', self.actions