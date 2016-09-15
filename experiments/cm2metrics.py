# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:24:27 2016

@author: gazkune
"""
import numpy as np

kasteren_cm = np.array([[61.5, 6.4, 0.8, 9.8, 4.6, 0.6, 15.4, 0.8], [0.9, 98.3, 0.2, 0.4, 0.0, 0.1, 0.2, 0.0], [12.1, 4.5, 72.6, 5.3, 3.9, 0.5, 0.8, 0.3], [9.8, 0.8, 5.3, 84.2, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.3, 0.2, 99.4, 0.0, 0.0, 0.0], [25.7, 0.0, 0.9, 0.0, 0.9, 56.0, 11.9, 4.6], [19.8, 1.4, 0.3, 0.0, 0.0, 9.2, 67.8, 1.4], [15.3, 5.1, 3.4, 1.7, 0.0, 6.8, 10.2, 57.6]])
kasteren_cm_hmm = np.array([[60.9, 6.4, 0.7, 9.8, 4.6, 0.4, 16.3, 0.9], [0.8, 98.4, 0.2, 0.3, 0.0, 0.1, 0.2, 0.0], [6.3, 2.9, 83.9, 3.2, 2.6, 0.3, 0.3, 0.5], [7.2, 0.0, 3.8, 89.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.4, 0.1, 99.4, 0.0, 0.0, 0.0], [22.0, 0.0, 0.9, 0.0, 0.9, 52.3, 16.5, 7.3], [12.4, 0.3, 0.3, 0.0, 0.0, 6.0, 78.4, 2.6], [11.9, 1.7, 1.7, 0.0, 0.0, 3.4, 13.6, 67.8]])
kasteren_cm_crf = np.array([[80.4, 12.3, 0.4, 0.4, 4.3, 0.0, 2.0, 0.2], [1.1, 98.4, 0.2, 0.1, 0.1, 0.0, 0.1, 0.0], [7.6, 12.4, 69.5, 1.8, 8.7, 0.0, 0.0, 0.0], [23.0, 8.7, 2.3, 66.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.2, 0.0, 99.7, 0.0, 0.0, 0.0], [32.1, 0.0, 0.9, 0.0, 0.9, 55.0, 9.2, 1.8], [33.0, 7.2, 0.6, 0.0, 0.6, 4.3, 53.4, 0.9], [32.2, 5.1, 3.4, 0.0, 0.0, 5.1, 10.2, 44.1]])

our_cm = np.array([[42.0, 8.0, 9.0, 9.0, 3.0, 2.0, 5.0, 20.0], [0.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 129.0, 0.0, 1.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 37.0, 0.0, 0.0, 3.0, 1.0], [1.0, 0.0, 2.0, 0.0, 21.0, 0.0, 0.0, 0.0], [6.0, 0.0, 1.0, 0.0, 0.0, 36.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 2.0], [13.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 37.0]])

com_cm1 = np.array([[91.6, 0.0, 0.0, 7.2, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], [0.0, 88.3, 0.0, 10.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 97.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.8, 2.2, 95.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 55.6, 0.5, 43.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.3, 98.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 98.3, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 99.3, 0.0, 0.0], [26.4, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 72.9, 0.0], [76.6, 0.0, 0.0, 19.8, 0.0, 0.0, 0.0, 2.9, 0.4, 0.0]])
com_cm2 = np.array([[83.9, 0.0, 3.4, 11.8, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0], [0.0, 93.2, 6.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.6, 94.8, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.6, 0.2, 98.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.4, 95.7, 2.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.3, 0.0, 4.6, 93.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.8, 0.8, 93.1, 5.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [39.4, 0.0, 42.5, 17.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [23.5, 0.1, 0.7, 2.3, 0.0, 0.0, 0.0, 0.0, 73.2, 0.0], [83.0, 0.0, 3.2, 13.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]])

#cm.astype(float)

#print cm

cm_dict = {}
cm_dict["kasteren_hmm"] = kasteren_cm_hmm
cm_dict["kasteren_crf"] = kasteren_cm_crf
cm_dict["our"] = our_cm
#cm_dict["com_one"] = com_cm1
#cm_dict["com_two"] = com_cm2

metrics_dict = {}
for key in cm_dict:
    
    class_precision = []
    class_recall = []
    class_f1 = []
    cm = cm_dict[key]
    
    prf_dict = {}
    # calculate total numbers for fp, fn and tp for micro metrics
    total_fp = cm.sum(axis=0) - cm.diagonal()
    total_fp = total_fp.sum()
    total_fn = cm.sum(axis=1) - cm.diagonal()
    total_fn = total_fn.sum()
    total_tp = cm.diagonal().sum()
    # Calculate macro and weighted metrics
    for i in xrange(len(cm)):    
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
    metrics_dict[key] = prf_dict
 
#print "Precision per class:", class_precision
#print "Recall per class:", class_recall
for key in metrics_dict:
    print key
    print "   Precision:", metrics_dict[key]["precision"]#["macro"]
    print "   Recall:", metrics_dict[key]["recall"]#["macro"]
    print "   F1:", metrics_dict[key]["f1"]#["macro"]
    print "-------------------------------------------"