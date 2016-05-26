# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:32:04 2016

@author: gazkune
"""

import pylab
import collections
import numpy as np
import sys


class ConfusionMatrix:
    def __init__(self, labels):
        self.labels = labels
        self.confusion_dictionary = self.build_confusion_dictionary(labels)

    def update(self, predicted_label, expected_label):
        print 'ConfusionMatrix: predicted', predicted_label, 'expected', expected_label
        self.confusion_dictionary[expected_label][predicted_label] += 1

    def build_confusion_dictionary(self, label_set):
        expected_labels = collections.OrderedDict()

        for expected_label in label_set:
            expected_labels[expected_label] = collections.OrderedDict()

            for predicted_label in label_set:
                expected_labels[expected_label][predicted_label] = 0.0

        return expected_labels

    def convert_to_matrix(self, dictionary):
        length = len(dictionary)
        confusion_dictionary = np.zeros((length, length))

        i = 0
        for row in dictionary:
            j = 0
            for column in dictionary:
                confusion_dictionary[i][j] = dictionary[row][column]
                j += 1
            i += 1

        return confusion_dictionary

    def get_confusion_matrix(self):
        matrix = self.convert_to_matrix(self.confusion_dictionary)
        #return self.normalize(matrix)
        return matrix

    def normalize(self, matrix):
        amin = np.amin(matrix)
        amax = np.amax(matrix)

        return [[(((y - amin) * (1 - 0)) / (amax - amin)) for y in x] for x in matrix]

    def plot(self):
        matrix = self.get_confusion_matrix()

        pylab.figure()
        pylab.imshow(matrix, interpolation='nearest', cmap=pylab.cm.jet)
        pylab.title("Confusion Matrix")

        for i, vi in enumerate(matrix):
            for j, vj in enumerate(vi):
                pylab.text(j, i+.1, "%.1f" % vj, fontsize=12)

        pylab.colorbar()

        classes = np.arange(len(self.labels))
        pylab.xticks(classes, self.labels)
        pylab.yticks(classes, self.labels)

        pylab.ylabel('Expected label')
        pylab.xlabel('Predicted label')
        pylab.show()
        
########################################################################################################################          
 
  
"""
Main function
"""
def main(argv):
   labels = ["cat", "dog", "velociraptor", "kraken", "pony"]
   confusionMatrix = ConfusionMatrix(labels)

   confusionMatrix.update("cat", "cat")
   confusionMatrix.update("cat", "dog")

   confusionMatrix.update("kraken", "velociraptor")
   confusionMatrix.update("velociraptor", "velociraptor")

   confusionMatrix.plot() 
   
   
   
if __name__ == "__main__":
   main(sys.argv)