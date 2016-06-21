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
        """ Constructor
        
        Usage example:
            pattern = Pattern()
            
        Parameters
        ----------
        None
            
        Returns
        ----------
        Instance of the class
        
        """
        # Pattern value calculated by AD
        self.value = 0.0
        # Actions or events inside the pattern
        self.actions = []
        # The pattern number assigned by AD
        self.number = -1
        # The number of instances of this pattern found
        # by AD in the dataset
        self.instances = 0
        
    def set_value(self, value):
        """ Setter for attribute value
        
        Usage example:
            set_value(5.3)
            
        Parameters
        ----------
        value : float
            a float with the value of the pattern
            
        Returns
        ----------
        None
        
        """
        self.value = value
        
    def append_action(self, action):
        """ Method to append an action to the instance
        
        Usage example:
            append_action("hasCoffee")
            
        Parameters
        ----------
        action : string
            a string with the action name
            
        Returns
        ----------
        None
        
        """
        self.actions.append(action)
        
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
        
    def set_instances(self, inst_number):
        """ Setter for attribute instances
        
        Usage example:
            set_instances(5)
            
        Parameters
        ----------
        value : integer
            an integer with the number of instances of the pattern
            
        Returns
        ----------
        None
        
        """
        self.instances = inst_number
        
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
        self.value = 0.0
        self.actions = []
        self.number = -1
        self.instances = 0
        
    def print_pattern(self):
        """ Method to print the instance
                    
        Usage example:
            print_pattern()
                
        Parameters
        ----------
        None
                
        Returns
        -------
        None
        
        """
        print 'Pattern', self.number
        print '  value:', self.value
        print '  instances:', self.instances
        print '  actions:', self.actions