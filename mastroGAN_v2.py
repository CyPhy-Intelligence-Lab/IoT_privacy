import keras
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights_v2
import numpy as np
import time
import random
from sklearn.preprocessing import MinMaxScaler
from visualization_metrics import visualization
import pickle
from sklearn import svm
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

LEARNING_RATE = 0.0001
A = 1
B = -1

def norm(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled = scaler.fit_transform(data)
    return scaled


def get_real_data(data_dim, batch_size):
    for i in range(300):
        a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
        base = np.linspace(-1, 1, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
        yield a * np.power(base, 2) + (a-1)


def get_mastro_data(batch_size):
    # load mastro data
    df = pd.read_csv('./data/data-2021-07-02-2021-07-09-Box13.csv')
    df = df.values
    np.random.shuffle(df)
    mastro_data = df[:, 2:-2]
    mastro_label = df[:, -2]
    occ_list = []
    ident_list = []
    for label in mastro_label:
        if label == 'NN' or label == 'NF':
            occ_list.append(0)
            ident_list.append(0)
        elif label[0] == '1':
            occ_list.append(1)
            ident_list.append(1)
        elif label[0] == '2':
            occ_list.append(1)
            ident_list.append(2)
        elif label[0] == '3':
            occ_list.append(1)
            ident_list.append(3)

    occ_label = np.array(occ_list)
    ident_label = np.array(ident_list)

    mastro_data = norm(mastro_data)

    for i in range(int(len(mastro_data)/batch_size)):
        yield (mastro_data[i*batch_size:(i+1)*batch_size],
               occ_label[i*batch_size:(i+1)*batch_size], ident_label[i*batch_size:(i+1)*batch_size])


def get_mastro_data_by_size(size):
    # load mastro data
    df = pd.read_csv('./data/mastro_sample_data_cleaned.csv')
    df = df.values
    np.random.shuffle(df)
    mastro_data = df[:, 2:-1]
    mastro_label = df[:, -1]

    mastro_data = norm(mastro_data)
    return mastro_data[:size]


class GAN(keras.Model):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, n, training=None, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_generator(self):
        model = keras.Sequential([
            keras.Input([None, self.latent_dim]),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(self.data_dim),
        ], name="generator")
        model.summary()
        return model

    def _get_discriminator(self):
        model = keras.Sequential([
            keras.Input([None, self.data_dim]),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(1)
        ], name="discriminator")
        model.summary()
        return model

    def train_d(self, data, label, occ_label, ident_label):
        with tf.GradientTape() as tape:
            pred = self.d.call(data, training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def train_g(self, d_label, occ_label, ident_label):
        with tf.GradientTape() as tape:
            g_data = self.call(len(d_label), training=True)
            pred = self.d.call(g_data, training=False)
            loss = self.loss_func(d_label, pred)
        loss = loss + A * svm_occ_classifier(g_data, occ_label) + B * svm_ident_classifier(g_data, ident_label)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_data, binary_accuracy(d_label, pred)

    def step(self, data, occ_label, ident_label):
        # train g
        d_label = tf.ones((len(data) * 2, 1), tf.float32)  # let d think generated are real
        g_loss, g_data, g_acc = self.train_g(d_label, occ_label, ident_label)

        # train d
        d_label = tf.concat((tf.ones((len(data), 1), tf.float32), tf.zeros((len(g_data) // 2, 1), tf.float32)), axis=0)
        data = tf.concat((data, g_data[:len(g_data) // 2]), axis=0)
        d_loss, d_acc = self.train_d(data, d_label, occ_label, ident_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (data, occ_label, ident_label) in enumerate(get_mastro_data(BATCH_SIZE)):
        #for t, data in enumerate(get_real_data(DATA_DIM, BATCH_SIZE)):
            d_loss, d_acc, g_loss, g_acc = gan.step(data, occ_label, ident_label)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        #save_gan(gan, ep)
    save_weights_v2(gan, 'gan_c1_c2')
    #cvt_gif(gan, shrink=2)


def svm_occ_classifier(fake_data, labels):
    with open('occ_svm_13box.pkl', 'rb') as f:
        clf = pickle.load(f)
    predicted_label = clf.predict(fake_data)
    loss_func = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = loss_func(labels, predicted_label)
    return loss

def svm_ident_classifier(fake_data, labels):
    with open('ident_svm_13box.pkl', 'rb') as f:
        clf = pickle.load(f)
    predicted_label = clf.predict(fake_data)
    loss_func = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = loss_func(labels, predicted_label)
    return loss


if __name__ == "__main__":
    LATENT_DIM = 10
    DATA_DIM = 18
    BATCH_SIZE = 32
    EPOCH = 200


    set_soft_gpu(True)
    m = GAN(LATENT_DIM, DATA_DIM)

    train(m, EPOCH)

    fake_data = m.call(1000).numpy()
    visualization(get_mastro_data_by_size(1000), fake_data, 'pca')
    visualization(get_mastro_data_by_size(1000), fake_data, 'tsne')

    print()
