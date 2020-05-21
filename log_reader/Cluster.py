# -*- coding: utf-8 -
"""
Created on Wed Apr 27 15:07:22 2016

@author: gazkune
"""
from log_reader.Pattern import Pattern

class Cluster:
    def __init__(self):
        """ Constructor
        
        Usage example:
            cluster = Cluster()
            
        Parameters
        ----------
        None
            
        Returns
        ----------
        Instance of the class
        
        """
        self.patterns = []
        self.number = -1
        
    def add_pattern(self, pattern):
        """ Method to add a pattern to the instance
        
        Usage example:
            add_pattern(pattern)
            
        Parameters
        ----------
        pattern : Pattern
            a instance of the Pattern class 
            
        Returns
        ----------
        None
        
        """
        self.patterns.append(pattern)
        
    def set_number(self, number):
        """ Setter for attribute number
        
        Usage example:
            set_number(5)
            
        Parameters
        ----------
        value : integer
            an integer with the number of the pattern
            
        Returns
        ----------
        None
        
        """
        self.number = number
        
    def reset(self):
        """ Method to reset the instance
                    
        Usage example:
            reset()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        self.patterns = []
        self.number = -1
        
    def print_cluster(self):
        """ Method to print the instance
                    
        Usage example:
            print_cluster()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        print('Cluster', self.number)
        for pattern in self.patterns:
            pattern.print_pattern()