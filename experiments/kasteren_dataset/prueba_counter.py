# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:15:52 2016

@author: gazkune
"""
from collections import Counter
import json

#
#l = [['GroceriesCupboard', 'Fridge', 'Fridge', 'Fridge', 'Fridge', 'Fridge', 'GroceriesCupboard', 'Fridge', 'PlatesCupboard', 'PansCupboard', 'PansCupboard', 'CupsCupboard', 'CupsCupboard', 'Freezer', 'Fridge', 'Fridge', 'Fridge', 'PlatesCupboard', 'Microwave', 'Microwave', 'Fridge', 'Fridge', 'PlatesCupboard', 'Fridge'],
#['GroceriesCupboard', 'Fridge', 'PansCupboard', 'Fridge', 'Freezer', 'Fridge', 'Fridge', 'Fridge', 'PlatesCupboard', 'PlatesCupboard'],
#['Fridge', 'CupsCupboard', 'PlatesCupboard', 'Fridge', 'GroceriesCupboard', 'Fridge', 'PansCupboard', 'Fridge', 'PansCupboard', 'PansCupboard', 'Fridge', 'Fridge', 'Fridge', 'PansCupboard', 'Fridge', 'CupsCupboard', 'PlatesCupboard', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'Fridge', 'Freezer', 'Fridge', 'PansCupboard', 'Fridge'],
#['Freezer', 'PlatesCupboard', 'PlatesCupboard', 'Freezer', 'Freezer'],
#['Freezer', 'GroceriesCupboard', 'PlatesCupboard', 'HallBathroomDoor', 'ToiletFlush', 'HallBathroomDoor', 'PansCupboard', 'PansCupboard', 'PlatesCupboard', 'Fridge', 'Fridge', 'Fridge', 'PlatesCupboard'],
#['GroceriesCupboard', 'Freezer', 'CupsCupboard', 'PansCupboard', 'PansCupboard', 'GroceriesCupboard', 'Microwave', 'PansCupboard', 'GroceriesCupboard', 'PlatesCupboard', 'Fridge', 'PansCupboard', 'PansCupboard', 'PlatesCupboard'],
#['Fridge', 'Fridge', 'GroceriesCupboard', 'Freezer', 'Freezer', 'GroceriesCupboard', 'Fridge', 'Freezer', 'Freezer', 'PlatesCupboard', 'PlatesCupboard'],
#['Freezer', 'PlatesCupboard', 'Freezer', 'Freezer'],
#['GroceriesCupboard', 'GroceriesCupboard', 'Freezer', 'Microwave', 'PlatesCupboard', 'Microwave', 'Microwave', 'Microwave'],
#['Dishwasher', 'PansCupboard', 'PansCupboard', 'PlatesCupboard', 'PlatesCupboard', 'CupsCupboard', 'GroceriesCupboard', 'GroceriesCupboard', 'HallBathroomDoor', 'ToiletFlush', 'HallBathroomDoor', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'PansCupboard', 'PlatesCupboard']]
#
#
#l2 = []
#
#for lista in l:
#    lista = set(lista)    
#    
#    for elemento in lista:
#        l2.append(elemento)
#        
#c = Counter(l2)
#print c

activities = {}
previous_activity = ""
actions_set = []
fichero = "/home/gazkune/repositories/gorka.azkune/src/dataset_transformer/kasterenDataset-m.csv"
with open(fichero, 'r') as f:    
    for l in f:
        
        tokens = l.strip().split(',')
        activity = tokens[2]
        action = tokens[1]
                
        if activity != previous_activity:
            if previous_activity in activities:
                activities[previous_activity].append(actions_set)
            else:
                activities[previous_activity] = [actions_set]
            actions_set = []
            
            
        actions_set.append(action)
        previous_activity = activity
        

sensors = set()
for activity in activities:
    actions = activities[activity]
    l2 = []
    for lista in actions:
        lista = set(lista)
        for elemento in lista:
            l2.append(elemento)
            sensors.add(elemento)
    c = Counter(l2)
    print activity
    print c
    
print sensors     