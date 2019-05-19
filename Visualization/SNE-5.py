from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from time import time
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import csv
import os

#绘制2D散点图
def plot_embdding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(111)
    print(label)
    color = ['red','green','yellow','blue','orange']
    labels = ['A','B','C','D','E']
    for k in range(5):
        plt.scatter([data[i, 0] for i in range(int(label[k]), int(label[k+1]))],  # X
                    [data[i, 1] for i in range(int(label[k]), int(label[k+1]))],  # Y
                    color=color[k],
                    label=labels[k],
                    marker='o')
    plt.legend(loc='best')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig

#绘制3D散点图
def plot_embdding_3D(data, label,title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(111)
    ax = Axes3D(fig)
    print(label)
    color = ['red', 'green', 'yellow', 'blue','orange']
    labels = ['A', 'B', 'C', 'D','E']
    for j in range(5):
        ax.scatter([data[i, 0] for i in range(int(label[j]), int(label[j+1]))],  # X
                   [data[i, 1] for i in range(int(label[j]), int(label[j+1]))],  # Y
                   [data[i, 2] for i in range(int(label[j]), int(label[j+1]))],  # Z
                   color = color[j],
                   label=labels[j],
                   marker='o'
        )

    ax.legend(loc='best')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.title(title)
    #ax.title("TEST")
    plt.title(title)
    plt.show()
    return fig

#读取特征，返回特征矩阵和计数
def read_feture(csvfile):
    data = pd.read_csv(csvfile)
    print("Origin:", data.shape)

    # Dropping unneccesary columns
    data = data.drop(['filename'], axis=1)
    print("Without name:", data.shape)

    feature = np.array(data.iloc[:, :-1], dtype=float)
    print("feature:", feature.shape)

    genres = 'A B C D E'.split()
    label = np.zeros(6)
    for i in range(1, 6):
        label[i] = len(data[data.label == genres[i - 1]]) + label[i - 1]
    print("count:", label)

    return feature, label

#可视化并结果写入csv
def v2csv2D(features,labels,csvfile):
    print(labels)
    print('Computing t-sne embedding')

    tsne = TSNE(n_components=2, perplexity=300, verbose=2, learning_rate=30, init='pca', random_state=0,
                early_exaggeration=300, n_iter=500)
    t0 = time()
    result = tsne.fit_transform(features)
    print(result)
    ##############################
    # 将可视化结果写到csv文件
    ##############################
    header = 'x y label'
    header = header.split()
    file = open(csvfile, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    file = open(csvfile, 'a', newline='')
    with file:
        for i in range(len(result)):
            if i < labels[1]:
                l = 'A'
            elif i < labels[2]:
                l = 'B'
            elif i < labels[3]:
                l = 'C'
            elif i < labels[4]:
                l = 'D'
            else:
                l = 'E'
            ################get label
            to_append = f'{result[i][0]} {result[i][1]} {l}'

            ##################write to csv
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            print(f'{i}:writing feature...')

#可视化并结果写入csv
def v2csv3D(features,labels,csvfile):
    print(labels)
    print('Computing t-sne embedding')

    tsne = TSNE(n_components=3, perplexity=300, verbose=2, learning_rate=30, init='pca', random_state=0,
                early_exaggeration=300, n_iter=500)
    result = tsne.fit_transform(features)
    print(result)
    ##############################
    # 将可视化结果写到csv文件
    ##############################
    header = 'x y z label'
    header = header.split()
    file = open(csvfile, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    file = open(csvfile, 'a', newline='')
    with file:
        for i in range(len(result)):
            if i < labels[1]:
                l = 'A'
            elif i < labels[2]:
                l = 'B'
            elif i < labels[3]:
                l = 'C'
            elif i < labels[4]:
                l = 'D'
            else:
                l = 'E'
            ################get label
            to_append = f'{result[i][0]} {result[i][1]} {result[i][2]} {l}'
            ##################write to csv
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            print(f'{i}:writing feature...')
#读取可视化结果
def read_result(csvfile):
    data = pd.read_csv(csvfile)
    feature = np.array(data.iloc[:, :-1], dtype=float)
    print("Visaul data shape:", feature.shape)
    return feature

def plot2D(feature_csv,result_csv):
    t0 = time()
    feature,label = read_feture(feature_csv)
    if not os.path.exists(result_csv):
        v2csv2D(feature,label,result_csv)
    result = read_result(result_csv)
    plot_embdding(result, label,'2D t-SNE embedding of the Common feature_vectors (time %.2fs)' % (time() - t0))
    print('t-SNE embedding of the feature_vectors (time %.2fs)' % (time() - t0))

def plot3D(feature_csv,result_csv):
    t0 = time()
    feature, label = read_feture(feature_csv)
    if not os.path.exists(result_csv):
        v2csv3D(feature, label, result_csv)
    result = read_result(result_csv)
    plot_embdding_3D(result, label, '3D t-SNE embedding of the Common feature_vectors (time %.2fs)' % (time() - t0))
    print('t-SNE embedding of the feature_vectors (time %.2fs)' % (time() - t0))


"""
feature_csv = '../Features/ship.csv'
result_csv_2D = './result_csv/ship_2D.csv'
result_csv_3D = './result_csv/ship_3D.csv'
plot2D(feature_csv,result_csv_2D)
plot3D(feature_csv,result_csv_3D)

feature_csv = '../Features/ship_vgg.csv'
result_csv_2D = './result_csv/ship_vgg_2D.csv'
result_csv_3D = './result_csv/ship_vgg_3D.csv'
plot2D(feature_csv,result_csv_2D)
plot3D(feature_csv,result_csv_3D)
"""

feature_csv = '../Features/ship_incp.csv'
result_csv_2D = './result_csv/ship_incp_2D.csv'
result_csv_3D = './result_csv/ship_incp_3D.csv'
plot2D(feature_csv,result_csv_2D)
plot3D(feature_csv,result_csv_3D)