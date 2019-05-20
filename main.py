from Classifer.DenseNet import *
from Classifer.KNN import *
from Classifer.SVM import *
from Classifer.ELM.ELMClassifer import *
from Classifer.GMM import *
from time import time


feature = '/home/atticus/PycharmProjects/Audio/Features/ship_26_5.csv'
metadata = '/home/atticus/PycharmProjects/Audio/Features/meta.csv'

DenseNetClassifer(feature,metadata,epoch=100,leave_one_out=True,batch_size=512,type=5)
#SVMClasifer(feature,metadata,leave_one_out=True,type=5)
#KNNClassifer(feature,metadata,leave_one_out=True,type=5)
#ELMClassifer(feature, metadata, type=5)
#GMMClassifer(feature, metadata,type=5)
