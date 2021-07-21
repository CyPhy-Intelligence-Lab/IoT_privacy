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
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy


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

read_csv()