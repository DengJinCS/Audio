from Classifer.DenseNet import read_meta,read_features
from sklearn.mixture import GaussianMixture as GMM
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#产生实验数据
from sklearn.datasets.samples_generator import make_blobs

def GMMClassifer(feature,metadata,leave_one_out=False,type=5):
    ##############################################################
    #           leave one-out for each file in dataset           #
    ##############################################################
    info = read_meta(metadata)
    print(info)
    acc = np.zeros(len(info))
    matrix = np.zeros((type, type), int)
    # 混淆矩阵

    for i in range(len(info)):
        print("Train GMM Classifier.....")
        data, _, _, _ = read_features(feature, type=type)
        ID = info[i][0]
        if read_features(feature, leave_out_ID=ID, type=type) == None:
            continue
        if leave_one_out:
            X_train, X_test, y_train, y_test = read_features(feature, leave_out_ID=ID, type=type)
        else:
            X_train, X_test, y_train, y_test = read_features(feature, type=type)
        model = GMM(n_components=type).fit(X_train, y_train)


        print(f'*********************Training Classifer When Leave {ID} Out************************')
        print(f'Train:{len(X_train)}   Test:{len(X_test)}    Test Label:{len(y_test)}')
        print(f'*********************Training Classifer When Leave {ID} Out************************')

        predictions = model.predict(X_test)
        right = 0
        for j in range(len(predictions)):
            pre = predictions[j]  # 获得预测值
            label = y_test[j]  # 获得真实值
            if pre == label:
                right += 1
            matrix[pre][label] += 1

        acc[i] = right / len(predictions)
        print("**************************************************************************")
        print(f'Leave ID:{ID},ANS:{info[i][2]} out Test ACC:{acc[i]}')
        print(f'                         matrix[pre][label]')
        print(matrix)
        print("**************************************************************************")
        print()
    print("ACC MEAN=", acc.mean())
    print(matrix)
