from Classifer.CLF import *
from time import time


feature = '/home/atticus/PycharmProjects/Audio/Features/ship_26_11.csv'
metadata = '/home/atticus/PycharmProjects/Audio/Features/meta.csv'

Classifer(feature,metadata,model_name='SVM',leave_one_out=True,type=12)

#Classifer(feature,metadata,model_name='NN',leave_one_out=True,type=12)