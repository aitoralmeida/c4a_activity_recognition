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
    patternsfile = "reduction_tests/test_kasteren_20d.csv.annotated"
    #adlogfile = "kasteren_removed_log.txt"
    adlogfile = "reduction_tests/test_kasterenA_20d_log.txt"
    contextmodelfile = "context_model.json"
    #pmoutputfile = "pm_output.csv"
    pmoutputfile = "reduction_tests/pm_output_20d.csv"
    
    # AR Evaluator arguments
    #groundtruth = "base_kasteren_reduced.csv"
    groundtruth = "reduction_tests/base_kasteren_20d.csv"
    #evaluable = "pm_output.csv"
    evaluable = "reduction_tests/pm_output_20d.csv"
    
    
    # Call PatternModelMatching    
    arguments = ['PatternModelMatching.py', '-e', eamsfile, '-a', patternsfile, '-l', adlogfile, '-c', contextmodelfile, '-o', pmoutputfile]
    pm_main(arguments)
    time.sleep(1)       
    
    # Call AR Evaluator
    arguments = ['AREvaluator.py', '-g', groundtruth, '-e', evaluable]
    ar_main(arguments)
       
if __name__ == "__main__":    
    main(sys.argv)
