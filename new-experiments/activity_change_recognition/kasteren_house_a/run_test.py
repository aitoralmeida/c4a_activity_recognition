# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:32:42 2014

@author: gazkune
"""

import sys, time
sys.path.append('../..')


from pattern_model_matching.PatternModelMatching import main as pm_main
from ar_evaluator.AREvaluator import main as ar_main




def main(argv):
    # Pattern Model Matching arguments
    eamsfile = "eams.json"
    #patternsfile = "test_kasteren_removed.csv.annotated"
    patternsfile = "test_kasteren_cosine_distance.csv.annotated"
    #adlogfile = "kasteren_removed_log.txt"
    adlogfile = "kasteren_removed_log.txt"
    contextmodelfile = "context_model.json"
    #pmoutputfile = "pm_output.csv"
    pmoutputfile = "pm_output_prueba.csv"
    
    # AR Evaluator arguments
    #groundtruth = "base_kasteren_reduced.csv"
    groundtruth = "base_kasteren_reduced.csv"
    #evaluable = "pm_output.csv"
    evaluable = "pm_output_prueba.csv"
    
    
    # Call PatternModelMatching    
    arguments = ['PatternModelMatching.py', '-e', eamsfile, '-a', patternsfile, '-l', adlogfile, '-c', contextmodelfile, '-o', pmoutputfile]
    pm_main(arguments)
    time.sleep(1)       
    
    # Call AR Evaluator
    arguments = ['AREvaluator.py', '-g', groundtruth, '-e', evaluable]
    ar_main(arguments)
       
if __name__ == "__main__":    
    main(sys.argv)
