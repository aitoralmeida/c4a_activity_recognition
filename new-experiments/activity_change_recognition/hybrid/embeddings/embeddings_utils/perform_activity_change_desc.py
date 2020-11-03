def get_change_points_desc(scores, threshold):
    change_points = []
    change_points.append(0)
    
    for i in range(0, len(scores)-1):
        if scores[i+1] - scores[i] > threshold:
            change_points.append(1)
        else:
            change_points.append(0)
    
    return change_points