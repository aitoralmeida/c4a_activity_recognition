import sys
import numpy as np

sys.path.append('../densratio')
from core import densratio

##################################################################################################################
# R_ULSIF based CPD algorithm translation from http://allmodelsarewrong.net/software.html
# Adapted for Kasteren dataset feature extraction based on Aminikhanghahi et al. paper
# START
##################################################################################################################  
            
def change_detection(feature_vectors, n, alpha):
    scores = []

    windows = np.zeros((feature_vectors.shape[1], feature_vectors.shape[0]))
    for i in range(0, feature_vectors.shape[0]):
        windows[:,i] = feature_vectors[i,].reshape(1,-1)

    num_samples = windows.shape[1]
    print("Num window samples in change detection: " + str(num_samples))
    t = n

    while((t+n) <= num_samples):
        y = windows[:,(t-n):(n+t)]
        y_ref = y[:,0:n]
        y_test = y[:,n:]

        densratio_obj = densratio(y_test, y_ref, alpha=alpha)

        scores.append(densratio_obj.alpha_PE)

        if t % 20 == 0:
            print(t)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores

##################################################################################################################
# R_ULSIF based CPD algorithm translation from MatLab code http://allmodelsarewrong.net/software.html
# Adapted for Kasteren dataset feature extraction based on Aminikhanghahi et al. paper
# END
################################################################################################################## 