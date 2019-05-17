from Classifer.ELM.ELM import *
from Classifer.ELM.random_layer import *
from Classifer.DenseNet import *
from sklearn.metrics import accuracy_score

hyperparameters = [100, 250, 500, 1200, 1800, 2500] # remove 100 and 250


def ELMClassifer(feature,metadata,type=5):
    ##############################################################
    #           leave one-out for each file in dataset           #
    ##############################################################
    info = read_meta(metadata)
    print(info)
    acc = np.zeros(len(info))
    matrix = np.zeros((5, 5), int)
    # 混淆矩阵

    for i in range(len(info)):
        print("Train ELM Classifier.....")
        data, _, _, _ = read_features(feature, type=type)
        ID = info[i][0]
        if read_features(feature, leave_out_ID=ID, type=type) == None:
            continue
        #X_train, X_test, y_train, y_test = read_features(feature, leave_out_ID=ID, type=type)
        X_train, X_test, y_train, y_test = read_features(feature, type=type)

        print("USING ELMClassifer...")
        #选择最佳超参数
        score_max = 0
        h_max = -1
        for h in hyperparameters:  # select best params in validation set
            print('Now in: ' + str(h))
            rl = RandomLayer(n_hidden=h, activation_func='reclinear', alpha=1)
            model = GenELMClassifier(hidden_layer=rl)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            print('- Score: ' + str(score))
            if score > score_max:
                score_max = score
                h_max = h
            print('Accuracy val set: ' + str(score_max))
            print('Best hyperparameter: ' + str(h_max))


        rl = RandomLayer(n_hidden=h_max, activation_func='reclinear', alpha=1)
        model = GenELMClassifier(hidden_layer=rl)
        model.fit(X_train, y_train)

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
