# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:34:08 2017

@author: gazkune
"""

"""
This class evaluates the results of the activity recognition process. It
calculates both the confusion matrix and the precision, recall and F1 metrics.

In the case of the metrics it calculates the three types of averages: micro, 
macro and weighted. As this is a multiclass classification problem use either 
macro or weighted values, dependign if the dataset is balanced or unbalanced.
"""

import ast
import getopt
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab
import datetime


class NewEvaluator:
        
    def __init__(self, groundtruth, evaluable, labels):
        """
        Constructor
        """
        self.groundtruth = pd.read_csv(groundtruth, parse_dates=[0, 1], header=None, sep=',', names=['start', 'end', 'activity'])
                
        self.evaluable = pd.read_csv(evaluable, parse_dates=[0], index_col= 0, converters={'detected_activities': ast.literal_eval})

        # List of activities in the groundtruth dataset
        self.act_dict = {}
        with open(labels, 'r') as act_labels:
            for line in act_labels:
                actid = int(line.split(' ')[0])
                actname = line.split(' ')[1].rstrip('\n')
                self.act_dict[actid] = actname
                
              
        # Let's transform numeric activities in self.groundtruth to activity names
        for i in self.groundtruth.index:
            self.groundtruth.ix[i, 'activity'] = self.act_dict[self.groundtruth.ix[i, 'activity']]
        
        self.activities = self.groundtruth.activity.unique()                
        self.activities = np.insert(self.activities, 0, 'None')
        # Use two lists: one for groundtruth activity and another one for predicted activity
        # These list will be used to build de confusion matrix and calculate different metrics
        self.y_groundtruth = []
        self.y_predicted = []
        
    def evaluate(self):
        """ Function to evaluate the AR performance using the activity annotation in groundtruth
        """
        evstart = self.evaluable.index[0]
        gtstart = self.groundtruth.ix[0, 'start']
        abs_start = gtstart
        # Treat the first case outside the loop
        if evstart < gtstart:
            abs_start = evstart
            eval_act = self.obtain_unique_activities(abs_start, gtstart)
            for j in xrange(len(eval_act)):
                self.y_groundtruth.append('None')
                self.y_predicted.append(eval_act[j])
            
        
        for i in self.groundtruth.index:
            # Treat the real activity
            start = self.groundtruth.ix[i, 'start']
            end = self.groundtruth.ix[i, 'end']
            gtact = self.groundtruth.ix[i, 'activity']            
            
            eval_act = self.obtain_unique_activities(start, end)
            
            if i < 10:
                print '(', start, ',', end, ')', gtact
                print '   ', eval_act
            
            for j in xrange(len(eval_act)):
                self.y_groundtruth.append(gtact)
                self.y_predicted.append(eval_act[j])
                
            # Now treat the 'None' activity between this and the following activity
            if i < len(self.groundtruth) - 1:
                start = end + pd.DateOffset(seconds=1)
                end = self.groundtruth.ix[i+1, 'start'] - pd.DateOffset(seconds=1)
                eval_act = self.obtain_unique_activities(start, end)
                for j in xrange(len(eval_act)):
                    self.y_groundtruth.append('None')
                    self.y_predicted.append(eval_act[j])
                    
        # Treat the last activity outside the loop
        evlast = len(self.evaluable) - 1
        gtlast = len(self.groundtruth) - 1
        evend = self.evaluable.index[evlast]
        gtend = self.groundtruth.ix[gtlast, 'end']
        if evend > gtend:
            start = gtend + pd.DateOffset(seconds=1)
            eval_act = self.obtain_unique_activities(start, evend)
            for j in xrange(len(eval_act)):
                self.y_groundtruth.append('None')
                self.y_predicted.append(eval_act[j])
            
            
    def obtain_unique_activities(self, start, end):
        # aux is a np array, where each element is a list
        aux = np.unique(self.evaluable.ix[start:end, 'detected_activities'].values)
        activities = []
        for i in xrange(len(aux)):
            # acts should be a list of strings            
            acts = aux[i]
            for j in xrange(len(acts)):
                if not acts[j] in activities:
                    activities.append(acts[j])
        
        return activities
        
    def calculate_evaluation_metrics(self):
        """Calculates the evaluation metrics (precision, recall and F1) for the
        predicted examples. It calculates the micro, macro and weighted values
        of each metric.
        
        Usage example:
            y_ground_truth = ['make_coffe', 'brush_teeth', 'wash_hands']
            y_predicted = ['make_coffe', 'wash_hands', 'wash_hands']
            metrics = calculate_evaluation_metrics (y_ground_truth, y_predicted)
    
        Parameters
        ----------
        y_ground_truth : array, shape = [n_samples]
            Classes that appear in the ground truth.
    
        y_predicted: array, shape = [n_samples]
            Predicted classes. Take into account that the must follow the same
            order as in y_ground_truth
       
        Returns
        -------
        metric_results : dict
            Dictionary with the values for the metrics (precision, recall and 
            f1)    
        """
        
        metric_types =  ['micro', 'macro', 'weighted']
        metric_results = {
            'precision' : {},
            'recall' : {},
            'f1' : {}        
        }
        
        for t in metric_types:
            metric_results['precision'][t] = metrics.precision_score(self.y_groundtruth, self.y_predicted, average = t)
            metric_results['recall'][t] = metrics.recall_score(self.y_groundtruth, self.y_predicted, average = t)
            metric_results['f1'][t] = metrics.f1_score(self.y_groundtruth, self.y_predicted, average = t)
            
        return metric_results
        
    def create_confusion_matrix(self):
        """Creates the confusión matrix of the predicted values.
        
        Usage example:
            y_ground_truth = ['make_coffe', 'brush_teeth', 'wash_hands']
            y_predicted = ['make_coffe', 'wash_hands', 'wash_hands']
            labels = ['make_coffe', 'brush_teeth', 'wash_hands']
            conf_matrix = calculate_evaluation_metrics (y_ground_truth, y_predicted, labels)
    
        Parameters
        ----------
        y_ground_truth : array, shape = [n_samples]
            Classes that appear in the ground truth.
    
        y_predicted: array, shape = [n_samples]
            Predicted classes. Take into account that the must follow the same
            order as in y_ground_truth
            
        labels: array, shape = [n_classes]
            Expected order for the labels in the confusion matrix.
       
        Returns
        -------
        conf_matrix : array, shape = [n_classes, n_classes]
            Confusion matrix  
        """
        conf_matrix = confusion_matrix(self.y_groundtruth, self.y_predicted, labels=self.activities)
        return conf_matrix

                
            
        
########################################################################################################################          


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
    groundtruth = "activities_reannotated.csv"
    evaluable = "pm_output.csv"
    labels = "activity_labels.txt"
    
    evaluator = NewEvaluator(groundtruth, evaluable, labels)
    
    print "Groundtruth:"
    print evaluator.groundtruth.head(10)
    
    print "-----------------------------------------"
    
    print "Evaluable:"
    print evaluator.evaluable.head(10)
    
    print "-----------------------------------------"
    
    evaluator.evaluate()
    
    print "Activities:"
    print evaluator.activities
    
    cm = evaluator.create_confusion_matrix()
    print evaluator.activities
    print cm
    
    print "Activities:"
    print evaluator.activities    
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    np.set_printoptions(precision=3, linewidth=1000)
    print(cm_normalized)
    
    #Dictionary with the values for the metrics (precision, recall and f1)    
    metrics = evaluator.calculate_evaluation_metrics()
    print 'precision:', metrics['precision']
    print 'recall:', metrics['recall']
    print 'f1:', metrics['f1']
    
    
if __name__ == "__main__":
   main(sys.argv)
    