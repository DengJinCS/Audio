import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas import DataFrame

import warnings
warnings.filterwarnings('ignore')
from keras import models
from keras import layers


def read_features(features,leave_out_ID=None,type=5):
    ####################################################################
    #                 Analysing the features in Pandas                 #
    ####################################################################
    if leave_out_ID:
        data = pd.read_csv(features)
        print("Origin:", data.shape)
        encoder = LabelEncoder()
        data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
        # Encoding the Labels

        #Leave-One-Out
        data = DataFrame(data)
        test = data[data['filename']==leave_out_ID]
        if len(test)==0:
            return None
        train = data[data['filename']!=leave_out_ID]

        # 均衡数据，跟最少的类别数量匹配
        mintrain= min([len(train[train['label'] == i]) for i in range(type)])
        print("MINTRAIN:",mintrain)

        train = train.drop(['filename'], axis=1)
        Train_tmp = train
        for t in range(type):
            #选取某个类别区间内的mintrain个随机样例
            #均衡数据集，使得每个类别数据样本一样多
            tmp = train[train['label'] == t]
            frec = mintrain/len(tmp)
            TrainA = tmp.sample(frac=frec)
            print("For category",t,"There have ",len(TrainA),"Samples")
            if(t == 0):
                Train_tmp = TrainA
            else:
                Train_tmp = pd.concat((Train_tmp,TrainA))
        train = Train_tmp
        #均衡数据，使每个类别都有一样多的训练
        #选取第i类的mintrain个随机样本样本

        test = test.drop(['filename'], axis=1)
        scaler = StandardScaler()

        feature_train = scaler.fit_transform(np.array(train.iloc[:, :-1], dtype=float))
        feature_test = scaler.fit_transform(np.array(test.iloc[:, :-1], dtype=float))
        print("Train Data:", feature_train.shape)
        print("Test Data:", feature_test.shape)

        label_train = np.array(train.iloc[:, -1])
        label_test = np.array(test.iloc[:, -1])

        return feature_train,feature_test,label_train,label_test
    else:
        data = pd.read_csv(features)
        # Dropping unneccesary columns
        data = data.drop(['filename'], axis=1)

        # Encoding the Labels
        genre_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        label = encoder.fit_transform(genre_list)
        # 编码为数字数组向量
        # Scaling the Feature columns
        scaler = StandardScaler()
        feature = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
        X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.3)
        return X_train, X_test, y_train, y_test

def read_meta(metadata):
    ####################################################################
    #                                get metadata                      #
    ####################################################################
    meta = pd.read_csv(metadata)
    ID = meta.iloc[:, 0]
    ID = np.array(ID)
    print("ID:", ID.shape)

    Type = meta.iloc[:, 1]
    encoder = LabelEncoder()
    Type = encoder.fit_transform(Type)
    # 编码为数字数组向量
    print("Type:", Type.shape)

    Class = meta.iloc[:, -1]
    encoder = LabelEncoder()
    Class = encoder.fit_transform(Class)
    # 编码为数字数组向量
    print("Class:", Class.shape)
    # 拼接为列向量保持原始格式
    info = np.hstack((ID.reshape(len(ID), 1),
                      Type.reshape(len(Type), 1),
                      Class.reshape(len(Class), 1)))
    print("Info:", info.shape)
    return info

def DenseNetClassifer(feature,metadata,epoch,batch_size,type=5):
    ##############################################################
    #           leave one-out for each file in dataset           #
    ##############################################################
    info = read_meta(metadata)
    print(info)
    acc = np.zeros(len(info))
    matrix = np.zeros((5,5),int)
    #混淆矩阵
    for i in range(len(info)):
        data,_,_,_ = read_features(feature,type=type)

        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_shape=(data.shape[1],)))

        model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(64, activation='relu'))

        model.add(layers.Dense(5, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        ID = info[i][0]
        if read_features(feature,leave_out_ID=ID,type=type) == None:
            continue
        X_train, X_test, y_train, y_test = read_features(feature,leave_out_ID=ID,type=type)
        print(f'*********************Training Classifer When Leave {ID} Out************************')
        print(f'Train:{len(X_train)}   Test:{len(X_test)}    Test Label:{len(y_test)}')
        print(f'*********************Training Classifer When Leave {ID} Out************************')

        history = model.fit(X_train,
                            y_train,
                            epochs=epoch,
                            batch_size=batch_size)
        predictions = model.predict(X_test)
        for j in range(len(predictions)):
            ans = predictions[j]
            max = ans[0]
            pre = 0

            for k in range(len(ans)):
                if ans[k] > max:
                    max = ans[k]
                    pre = k  # 获得预测值
            label = y_test[j]#获得真实值
            matrix[pre][label] += 1

        test_loss, acc[i] = model.evaluate(X_test, y_test)
        print("**************************************************************************")
        print(f'Leave ID:{ID},ANS:{info[i][2]} out Test ACC:{acc[i]}')
        print(f'                         matrix[pre][label]')
        print(matrix)
        print("**************************************************************************")
        print()
    print("ACC MEAN=",acc.mean())
    print(matrix)



"""
#Validating our approach
x_val = X_train[:1000]
partial_x_train = X_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=100,
          batch_size=256,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)
print("Loss & ACC:",results)


#Predictions on Test Data
predictions = model.predict(X_test)

"""

