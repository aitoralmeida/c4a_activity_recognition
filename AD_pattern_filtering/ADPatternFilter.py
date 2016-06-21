# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:12:28 2016

@author: gazkune
"""

"""
A class to filter the patterns detected by AD 
"""

import sys, getopt
from copy import deepcopy

import numpy as np

from log_reader.Pattern import Pattern
from log_reader.Cluster import Cluster
from log_reader.LogReader import LogReader



class ADPatternFilter:
    
    # Constructor
    def __init__(self, logfile):
        """ Constructor
        
        Usage example:
            patternfilter = ADPatternFilter(logfile)
            
        Parameters
        ----------        
        logfile : string
            the name of a text file generated as a log of the AD tool
                    
        Returns
        ----------
        Instance of the class
        
        """    
        # Create an instance of LogReader
        self.log = LogReader(logfile)
        self.removedPatterns = []
        self.defPatternlist = []
        
    def filter_patterns(self):
        """ Method to filter spurious patterns.
            
        Usage example:
            filter_patterns()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        # Parse the logfile and obtain the list of patterns and clusters
        self.log.parse_log()
        self.remove_patterns1()
        #self.remove_patterns2()
        

    def remove_patterns1(self):
        """ This method implements the strategy 1 to remove patterns
        Take into account only the number of instances for removing purposes
            
        Usage example:
            remove_patterns1()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        perc = 0.10
        threshold = self.log.minInstances + (self.log.maxInstances - self.log.minInstances)*perc
        print 'Used threshold value for instances:', threshold
        for pattern in self.log.patternlist:
            if pattern.instances < threshold:
                self.removedPatterns.append(pattern)
            else:
                self.defPatternlist.append(pattern)
                
    
    def remove_patterns2(self):
        """ This method implements the strategy 2 to remove patterns
        Take into account only the pattern value for removing purposes
            
        Usage example:
            remove_patterns2()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        perc = 0.10
        threshold = self.log.minPatternValue + (self.log.maxPatternValue - self.log.minPatternValue)*perc
        print 'Used threshold value for pattern values:', threshold
        for pattern in self.log.patternlist:
            if pattern.value < threshold:
                self.removedPatterns.append(pattern)
            else:
                self.defPatternlist.append(pattern)
                
    
    def remove_patterns3(self):
        """ This method implements the strategy 3 to remove patterns
        Take into account only the pattern instances for removing purposes
        but calculate the threshold using the IQR (Interquartile range)
            
        Usage example:
            remove_patterns3()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        instances = []
        for pattern in self.log.patternlist:
            instances.append(pattern.instances)
            
        iqr = np.subtract(*np.percentile(instances, [75, 25]))
        threshold = self.log.minInstances + iqr
        print 'Used threshold value for instances:', threshold, 'iqr:', iqr
        for pattern in self.log.patternlist:
            if pattern.instances < threshold:
                self.removedPatterns.append(pattern)
            else:
                self.defPatternlist.append(pattern)
                
    
    def remove_patterns4(self):
        """ This method implements the strategy 4 to remove patterns
        Take into account only the pattern values for removing purposes
        but calculate the threshold using the IQR (Interquartile range)
            
        Usage example:
            remove_patterns4()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        values = []
        for pattern in self.log.patternlist:
            values.append(pattern.value)
            
        iqr = np.subtract(*np.percentile(values, [75, 25]))
        threshold = self.log.minPatternValue + iqr
        print 'Used threshold value for pattern values:', threshold, 'iqr:', iqr
        for pattern in self.log.patternlist:
            if pattern.value < threshold:
                self.removedPatterns.append(pattern)
            else:
                self.defPatternlist.append(pattern)
    
    #Method to reset the internal lists    
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
        self.removedPatterns = []
        self.defPatternlist = []
            
        
######################################################################################
        
"""
Function to parse arguments from command line
Input:
    argv -> command line arguments
Output:
    inputfile -> log file generated by AD (custom format)

"""

def parse_args(argv):
    """ Function to parse arguments from command line
                    
    Usage example:            
        [eamsfile, annotatedfile, logfile, contextmodel, outputfile] = parse_args(argv[1:])
                
    Parameters
    ----------
    argv : list
        the arguments to be parsed as passed to the function
                        
    Returns
    -------
    logfile : string
        the file name for the log generated by AD (CASAS tool)
        
    """
    inputfile = ''
   
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print 'ADPatternFilter.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'ADPatternFilter.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
         
    return inputfile
  
"""
Main function
"""
def main(argv):
    """ Main
            
    Usage example:
        main(argv)
            
    Parameters
    ----------
    argv : list
        the arguments to be parsed as passed to the function
                
    Returns
    -------
    None
        
    """
    # call the argument parser 
    inputfile_name = parse_args(argv[1:])
    print 'Provided arguments:'       
    print inputfile_name
   
    patternfilter = ADPatternFilter(inputfile_name)
   
    # parse the file for patterns and clusters
    patternfilter.filter_patterns()
   
    """
    for pattern in patternfilter.removedPatterns:
       pattern.printPattern()
    """
    """
    for pattern in patternfilter.defPatternlist:
       pattern.printPattern()
    """
    patternfilter.remove_patterns1()
    print len(patternfilter.removedPatterns)
   
    patternfilter.reset()
   
    patternfilter.remove_patterns2()
    print len(patternfilter.removedPatterns)
   
    patternfilter.reset()
   
    patternfilter.remove_patterns3()
    print len(patternfilter.removedPatterns)
   
    patternfilter.reset()
   
    patternfilter.remove_patterns4()
    print len(patternfilter.removedPatterns)
   
if __name__ == "__main__":
   main(sys.argv)