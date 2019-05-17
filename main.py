from Classifer.DenseNet import *
from Classifer.KNN import *
from Classifer.SVM import *
from Classifer.ELM.ELMClassifer import *


feature = '/home/atticus/PycharmProjects/Audio/Features/ship.csv'
metadata = '/home/atticus/PycharmProjects/Audio/Features/meta.csv'

DenseNetClassifer(feature,metadata,epoch=25,batch_size=512,type=5)
SVMClasifer(feature,metadata,type=5)
KNNClassifer(feature,metadata,type=5)
ELMClassifer(feature, metadata, type=5)