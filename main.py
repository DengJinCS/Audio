from Classifer.DenseNet import *
from Classifer.KNN import *
from Classifer.SVM import *
from Classifer.ELM.ELMClassifer import *
from time import time


feature = '/home/atticus/PycharmProjects/Audio/Features/ship_incp.csv'
metadata = '/home/atticus/PycharmProjects/Audio/Features/meta.csv'
t= time()
print(t)
DenseNetClassifer(feature,metadata,epoch=10,batch_size=512,type=5)
SVMClasifer(feature,metadata,type=5)
KNNClassifer(feature,metadata,type=5)
ELMClassifer(feature, metadata, type=5)
print(time(),time()-t)