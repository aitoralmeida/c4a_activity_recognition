# -*- coding: utf-8 -
"""
Created on Wed Apr 27 15:07:22 2016

@author: gazkune
"""
from Pattern import Pattern

class Cluster:
    def __init__(self):
        self.patterns = []
        self.number = -1
        
    def addPattern(self, pattern):
        self.patterns.append(pattern)
        
    def setNumber(self, number):
        self.number = number
        
    def reset(self):
        self.patterns = []
        self.number = -1
        
    def printCluster(self):
        print 'Cluster', self.number
        for pattern in self.patterns:
            pattern.printPattern()