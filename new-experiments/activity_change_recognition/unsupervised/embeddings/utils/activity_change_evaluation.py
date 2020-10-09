import numpy as np
import sys

def get_detection_delay(scores, y, timestamps, threshold, lower=False):
    detection_delay = 0

    counter = 0
    for i in range(0, len(scores)):
        if lower:
            if scores[i] < threshold:
                counter += 1
                detection_delay += search_nearest_change_point(y, timestamps, i)
        else: 
            if scores[i] > threshold:
                counter += 1
                detection_delay += search_nearest_change_point(y, timestamps, i)

    if counter == 0:
        return detection_delay
    else:
        return detection_delay / counter

def get_detection_delay_desc(scores, y, timestamps, threshold, lower=False):
    detection_delay = 0

    counter = 0
    for i in range(0, len(scores)-1):
        if scores[i+1] - scores[i] > threshold:
            counter += 1
            detection_delay += search_nearest_change_point(y, timestamps, i)

    if counter == 0:
        return detection_delay
    else:
        return detection_delay / counter

def get_conf_matrix_with_offset_strategy(scores, y, timestamps, threshold, offset, lower=False):
    cf_matrix = np.zeros((2,2))

    for i in range(0, len(scores)):
        if lower:
            if scores[i] < threshold:
                correctly_detected = check_detected_change_point_with_offset(timestamps, y, i, offset)
                if correctly_detected:
                    cf_matrix[1][1] += 1
                else:
                    cf_matrix[0][1] += 1
            else:
                if y[i] == 0:
                    cf_matrix[0][0] += 1
                else:
                    cf_matrix[1][0] += 1
        else:
            if scores[i] > threshold:
                correctly_detected = check_detected_change_point_with_offset(timestamps, y, i, offset)
                if correctly_detected:
                    cf_matrix[1][1] += 1
                else:
                    cf_matrix[0][1] += 1
            else:
                if y[i] == 0:
                    cf_matrix[0][0] += 1
                else:
                    cf_matrix[1][0] += 1
    
    return cf_matrix

def get_conf_matrix_with_offset_strategy_desc(scores, y, timestamps, threshold, offset):
    cf_matrix = np.zeros((2,2))

    if y[0] == 0:
        cf_matrix[0][0] += 1
    else:
        cf_matrix[1][0] += 1
    
    for i in range(0, len(scores)-1):
        if scores[i+1] - scores[i] > threshold:
            correctly_detected = check_detected_change_point_with_offset(timestamps, y, i+1, offset)
            if correctly_detected:
                cf_matrix[1][1] += 1
            else:
                cf_matrix[0][1] += 1
        else:
            if y[i+1] == 0:
                cf_matrix[0][0] += 1
            else:
                cf_matrix[1][0] += 1
    
    return cf_matrix

def check_detected_change_point_with_offset(timestamps, y, position, offset):
    timestamp_ref = timestamps[position]

    # right
    for i in range(position, len(y)):
        difference = timestamps[i] - timestamp_ref
        if offset < difference:
            break
        else:
            if y[i] == 1:
                return True
    # left
    i = position - 1
    while i >= 0:
        difference = timestamp_ref - timestamps[i]
        if offset < difference:
            break
        else:
            if y[i] == 1:
                return True
        i -= 1

    return False

def search_nearest_change_point(y, timestamps, position):
    timestamp_ref = timestamps[position]

    # right
    right_nearest_cp_difference = sys.maxsize
    for i in range(position, len(y)):
        if y[i] == 1:
            right_nearest_cp_difference = timestamps[i] - timestamp_ref
            break
    
    # left
    left_nearest_cp_difference = sys.maxsize
    i = position - 1
    while i >= 0:
        if y[i] == 1:
            left_nearest_cp_difference = timestamp_ref - timestamps[i]
            break
        i -= 1

    return min(right_nearest_cp_difference, left_nearest_cp_difference)