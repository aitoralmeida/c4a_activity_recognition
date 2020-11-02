# -*- coding: utf-8 -*-
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

class AREvaluator:
    
    def __init__(self, groundtruth, evaluable):  
        self.groundtruth = pd.read_csv(groundtruth, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
        self.groundtruth.columns = ['sensor', 'action', 'event', 'activity']
        self.groundtruth.index.names = ["timestamp"]
        
        self.evaluable = pd.read_csv(evaluable, parse_dates=True, index_col=0, converters={'detected_activities': ast.literal_eval})

        # List of activities in the groundtruth dataset        
        self.activities = self.groundtruth.activity.unique()
               
        # Use two lists: one for groundtruth activity and another one for predicted activity
        # These list will be used to build de confusion matrix and calculate different metrics
        self.y_groundtruth = []
        self.y_predicted = []
        
            
    def pattern_based_evaluation(self):
        """First evaluation method. 
        The idea is to take the patterns in evaluable, take the start and end time of the pattern
        and look at the same time section of the groundtruth file. Extract the activities of the groundtruth
        for that section and compare with the activities of evaluable. The objetive is to create a confusion
        matrix.    
        
        Usage example:
            pattern_based_evaluation()
    
        Parameters
        ----------
        None
       
        Returns
        -------
        None
        """
        # Initialize start and prev_pat
        start = self.evaluable.index[0]
        prev_pat = self.evaluable.loc[start, 'pattern']
        print('First pattern:', prev_pat)
        # iterate through evaluable        
        for i in range(1, len(self.evaluable)):
            ts = self.evaluable.index[i]
            pat = self.evaluable.loc[ts, 'pattern']
            print(pat)
            print(prev_pat)
            if pat != prev_pat:
                end = self.evaluable.index[i-1]
                # At this point, start and end have the right values for a discovered pattern                
                gt_activities = self.extract_groundtruth_activities(start, end)
                ev_activities = self.evaluable.loc[end, 'detected_activities']
                print( "Pattern:", prev_pat, "(", start, ",", end, ")")
                print( "   GT:", gt_activities)
                print( "   EV:", ev_activities )               
                # New approach, using the scikit-learn confusion matrix
                self.lenient_metric(gt_activities, ev_activities)
                # Now update prev_pat and start
                prev_pat = pat
                start = ts
            
        
    
    def extract_groundtruth_activities(self, start, end):
        """ Method to extract the activities that appear in the groundtruth between
        the timestamps start and end
        
        Usage example:
            start = pd.Timestamp('2016-01-01 00:00:00')
            end = pd.Timestamp('2016-01-01 00:00:12')
            action_list = extract_groundtruth_activities(start, end)
            
        Parameters
        ----------
        start : Pandas.Timestamp
            start time of the action sequence to extract
        end : Pandas.Timestamp
            end time of the action sequence to extract
            
        Returns
        -------
        activities: list
            a list of unique activities that appear between star and end
        
        """
        #print( '   extractGTActivities: start', start, ', end', end        
        return list(self.groundtruth.loc[start:end, 'activity'].unique())    
                

    def lenient_metric(self, gt_activities, ev_activities):
        """ Method to update the confusion matrix given the groundtruth activities
        and detected activities of a given section of time
        
        Usage example:
            gt_activities = ['MakeCoffee', 'WashHands']
            ev_activities = ['MakeCoffee', 'None']
            
        Parameters
        ----------
        gt_activities : list
            activities in the groundtruth
        ev_activities : list
            predicted activities for the same section as groundtruth
            
        Returns
        -------
        None
        
        """
        if len(gt_activities) > len(ev_activities):
            if 'None' in gt_activities:
                i = gt_activities.index('None')
                gt_activities.pop(i)            
            for i in range(len(gt_activities) - len(ev_activities)):
                ev_activities.append('None')
                
        elif len(gt_activities) < len(ev_activities):
            for i in range(len(ev_activities) - len(gt_activities)):
                gt_activities.append('None')
                
        # At this point, both lists should be the same length
        # Now we can compare both lists
        # First of all compute hits (activities that appear in both lists)
        hits = np.intersect1d(np.array(gt_activities), np.array(ev_activities))
        hits = hits.tolist()
        for i in range(len(hits)):
            #self.cm.update(hits[i], hits[i])            
            self.y_predicted.append(hits[i])
            self.y_groundtruth.append(hits[i])
        
        # Now compute divergences
        a = np.setdiff1d(np.array(gt_activities), np.array(ev_activities), True)
        b = np.setdiff1d(np.array(ev_activities), np.array(gt_activities), True)
        if len(a) == len(b):
            for i in range(len(a)):
                self.y_predicted.append(b[i])
                self.y_groundtruth.append(a[i])
        else:
            raise ValueError('Arrays a and b have different lengths!')
        
                
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
    
    def plot(self, matrix):
        """Plots the confusion matrix.
        
        Usage example:
            matrix = self.calculate_evaluation_metrics            
            self.plot(matrix)
    
        Parameters
        ----------
        matrix : array, shape = [n_classes, n_classes]
            Confusion matrix        
       
        Returns
        -------
        None
        """
        # Use percentages for the confusion matrix
        matrix = matrix*100
        pylab.figure()
        pylab.imshow(matrix, interpolation='nearest', cmap=pylab.cm.jet)
        pylab.title("Confusion Matrix")

        for i, vi in enumerate(matrix):
            for j, vj in enumerate(vi):
                #pylab.text(j, i+.1, "%.2f" % vj, fontsize=12)
                pylab.text(j-.3, i+.1, "%.2f" % vj, fontsize=6)

        pylab.colorbar()

        classes = np.arange(len(self.activities))
        pylab.tick_params(axis='both', which='major', labelsize=6)
        pylab.tick_params(axis='both', which='minor', labelsize=6)
        #pylab.tick_params(axis='x', labelrotation=90)
        pylab.xticks(classes, self.activities, rotation='vertical')
        pylab.yticks(classes, self.activities)

        pylab.ylabel('Expected label')
        pylab.xlabel('Predicted label')
        #pylab.show()
        pylab.savefig('cm.png')
        
    def calculate_manual_metrics(self, cm):
        """Plots the confusion matrix.
        Usage example:
            metrics = self.calculate_manual_metrics            
    
        Parameters
        ----------
        cm : array, shape = [n_classes, n_classes]
            Confusion matrix        
       
        Returns
        -------
        metrics : dict 
        """
        class_precision = []
        class_recall = []
        class_f1 = []
                 
        prf_dict = {}
        # calculate total numbers for fp, fn and tp for micro metrics
        total_fp = cm.sum(axis=0) - cm.diagonal()
        total_fp = total_fp.sum()
        total_fn = cm.sum(axis=1) - cm.diagonal()
        total_fn = total_fn.sum()
        total_tp = cm.diagonal().sum()
        # Calculate macro and weighted metrics
        for i in range(len(cm)):            
            tp = cm[i][i]
        
            col = cm[:, i]    
            col = np.delete(col, i)    
            fp = sum(col)
        
            row = cm[i, :]    
            row = np.delete(row, i)
        
            fn = sum(row)
            if tp == 0.0 and fp == 0.0:
                precision = 0
            else:
                precision = tp/(tp+fp)
        
            recall = tp/(tp+fn)
            class_precision.append(precision)
            class_recall.append(recall)
    
            if precision == 0.0 and recall == 0.0:
                class_f1.append(0.0)
            else:
                class_f1.append(2*precision*recall / (precision+recall))
    
        prf_dict["precision"] = {}
        prf_dict["precision"]["macro"] = np.mean(class_precision)
        prf_dict["precision"]["weighted"] = np.average(class_precision, weights=cm.sum(axis=1))
        prf_dict["precision"]["micro"] = total_tp / (total_tp + total_fp)
        
        prf_dict["recall"] = {}
        prf_dict["recall"]["macro"] = np.mean(class_recall)
        prf_dict["recall"]["weighted"] = np.average(class_recall, weights=cm.sum(axis=1))
        prf_dict["recall"]["micro"] = total_tp / (total_tp + total_fn)
        
        prf_dict["f1"] = {}
        prf_dict["f1"]["macro"] = np.mean(class_f1)
        prf_dict["f1"]["weighted"] = np.average(class_f1, weights=cm.sum(axis=1))
        prf_dict["f1"]["micro"] = 2*prf_dict["precision"]["micro"]*prf_dict["recall"]["micro"] / (prf_dict["precision"]["micro"]+prf_dict["recall"]["micro"])
                
        return prf_dict
                

########################################################################################################################          
 
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
        [groundtruth, evaluable] = parse_args(argv[1:])        
            
    Parameters
    ----------
    argv : list
        the arguments to be parsed as passed to the function
                
    Returns
    -------
    groundtruth: string
        name of the file which contains the groundtruth
    evaluable: string
        name of the file which contains the evaluable file
        
    """
    groundtruth = ''
    evaluable = ''
      
    try:
      opts, args = getopt.getopt(argv,"hg:e:",["gfile=", "efile="])
    except getopt.GetoptError:
      print( 'AREvaluator.py -g <groundtruth> -e <evaluable>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print( 'AREvaluator.py -g <groundtruth> -e <evaluable>')
         sys.exit()
      elif opt in ("-g", "--gfile"):
         groundtruth = arg
      elif opt in ("-e", "--efile"):
         evaluable = arg
      
         
    return groundtruth, evaluable
  

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
    [groundtruth, evaluable] = parse_args(argv[1:])
    print( 'Provided arguments:')
    print( groundtruth, evaluable)
    evaluator = AREvaluator(groundtruth, evaluable)    
    
    print( evaluator.groundtruth.head(10)  ) 
    print( '-------------------------------------------'   )
    print( evaluator.evaluable.head(10)   )
    evaluator.pattern_based_evaluation()
    
    # Test new approach with scikit-learn
    cm = evaluator.create_confusion_matrix()
    print( cm)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    
    #Dictionary with the values for the metrics (precision, recall and f1)    
    metrics = evaluator.calculate_evaluation_metrics()
    print( "Scikit metrics")
    print( 'precision:', metrics['precision'])
    print( 'recall:', metrics['recall'])
    print( 'f1:', metrics['f1'])
    
    # Calculate and print( manual metrics
    man_metrics = evaluator.calculate_manual_metrics(cm_normalized)
    print( "Manual metrics")
    print( 'precision:', man_metrics['precision'])
    print( 'recall:', man_metrics['recall'])
    print( 'f1:', man_metrics['f1'])
    
    #evaluator.plot(cm)
    # Just a trick -> I don't use absolute routes!!
    cmfile = '/experiments/kasterenC-cm.npy'
    np.save(cmfile, cm_normalized)
    evaluator.plot(cm_normalized)
   
if __name__ == "__main__":
   main(sys.argv)

   