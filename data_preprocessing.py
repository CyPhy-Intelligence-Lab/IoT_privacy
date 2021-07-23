import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import pickle

def read_csv():
    path = '/Users/tongwu/Downloads/data-2021-07-02_Box13.csv'
    df = pd.read_csv(path, usecols=['channel_name', 'time', 'value', 'label'])

    df_sorted = df.sort_values(['time', 'channel_name'], ascending=[True, True])
    sensors = np.array(df_sorted.iloc[0:18, 0])
    values = np.array(df_sorted['value'])
    labels = np.array(df_sorted.iloc[::18, 3])

    values = values.reshape(-1, 18)

    time = np.array(df_sorted.iloc[::18, 1])

    df_new = pd.DataFrame(values, columns=sensors)
    df_new['time'] = time
    df_new['label'] = labels
    df_new = df_new[['time'] + [col for col in df_new.columns if col != 'time']]
    df_new.to_csv('data/mastro_sample_data_cleaned.csv')
    return df_new


def occ_label_organize():
    df = pd.read_csv('data/data-2021-07-02-2021-07-09-Box13.csv')
    list = []
    for label in df['label']:
        if label == 'NN' or label == 'NF':
            list.append(0)
        else:
            list.append(1)
    df['label'] = np.array(list)
    return df.iloc[:, 2:-1]


def ident_label_organize():
    df = pd.read_csv('data/data-2021-07-02-2021-07-09-Box13.csv')
    list = []
    for label in df['label']:
        if label[0] == '1':
            list.append(1)
        elif label[0] == '2':
            list.append(2)
        elif label[0] == '3':
            list.append(3)
        elif label[0] == 'N':
            list.append(0)

    df['label'] = np.array(list)
    return df.iloc[:, 2:-1]


def occ_svm_classification():
    df = occ_label_organize()
    data = df.values
    X_train, X_test, y_train, y_test = train_test_split\
        (data[:, :-1], data[:, -1], test_size=0.3, random_state=109)

    #X_train = norm(X_train)
    #X_test = norm(X_test)

    X_train = X_train[:]
    y_train = y_train[:]
    X_test = X_test[:]
    y_test = y_test[:]
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    with open('occ_svm_13box.pkl', 'wb') as f:
        pickle.dump(clf, f)


def norm(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled = scaler.fit_transform(data)
    return scaled


def ident_svm_classification():
    df = ident_label_organize()
    #onehot = pd.get_dummies(df['label'], columns=['l1', 'l2', 'l3'])
    #df = df.drop('label', axis=1)
    data = df.values
    #X_train, X_test, y_train, y_test = train_test_split \
    #    (data[:, :-1], onehot.values, test_size=0.3, random_state=109)
    X_train, X_test, y_train, y_test = train_test_split \
        (data[:, :-1], data[:, -1], test_size=0.3, random_state=109)

    X_train = norm(X_train)
    X_test = norm(X_test)

    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:3000]
    y_test = y_test[:3000]
    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo', verbose=True)  # Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)


    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_test, y_pred, average=None))


def two_layer_cnn(X_train, y_train, X_test, y_test, hyper_parameters):
    c1, c2, l2_value = hyper_parameters
    input = Input(shape=(60, 11, 1))
    bn1 = BatchNormalization()(input)
    conv1 = Conv1D(c1, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn1)
    pool1 = MaxPooling1D(pool_size=(2, 2))(conv1)
    bn2 = BatchNormalization()(pool1)
    conv2 = Conv1D(c2, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn2)
    pool2 = MaxPooling1D(pool_size=(2, 2))(conv2)
    flat = Flatten()(pool2)

occ_svm_classification()