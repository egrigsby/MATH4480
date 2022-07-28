import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

def load_breastcancer():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    file_path = tf.keras.utils.get_file("wdbc_data", DATA_URL)
    df = pd.read_csv(file_path, header=None)
    df = df.drop([0], axis=1)
    df = df.replace({'B':0, 'M':1})
    df =df[df.columns[list(range(1,31))+[0]]]
    df.columns = range(df.columns.size)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x , y

def load_pokerhands():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"
    file_path = tf.keras.utils.get_file("poker-hand-testing_data", DATA_URL)
    df = pd.read_csv(file_path, header=None)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_spambase():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    file_path = tf.keras.utils.get_file("spambase_data", DATA_URL)
    df = pd.read_csv(file_path, header=None)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_iris():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    file_path = tf.keras.utils.get_file("iris_data", DATA_URL)
    df = pd.read_csv(file_path, header=None)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_parkinsons():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    file_path = tf.keras.utils.get_file("parkinsons_data", DATA_URL)
    df = pd.read_csv(file_path, header=None)
    df = df.drop(index=df.index[0], axis=0)
    df = df.drop([0], axis=1)
    df.columns = range(df.columns.size)
    df[list(range(0,23))] = df[list(range(0,23))].apply(pd.to_numeric)
    df =df[df.columns[list(range(16))+list(range(17,23))+[16]]]
    df.columns = range(df.columns.size)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_white_wine():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    file_path = tf.keras.utils.get_file("winequality-white_csv", DATA_URL)
    df = pd.read_csv(file_path, header=None, sep=";")
    df = df.drop(index=df.index[0], axis=0)
    df[list(range(12))] = df[list(range(12))].apply(pd.to_numeric)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_auto_mpg():
    DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    file_path = tf.keras.utils.get_file("auto-mpg_data", DATA_URL)
    df = pd.read_csv(file_path, na_values = "?", comment='\t', sep=" ", skipinitialspace=True, header=None)
    df.replace(to_replace='?', value=np.nan)
    df =df.dropna()
    df =df[df.columns[list(range(1,8))+[0]]]
    df.columns = range(df.columns.size)
    x = df.drop([df.shape[1]-1], axis = 1).values
    y = df[df.shape[1]-1].values.reshape(-1,1)
    return df, x, y

def load_fashion_mnist():
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test

def load_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

def load_horses_or_humans():
    ds = tfds.load('horses_or_humans', as_supervised=True)
    ds_np = tfds.as_numpy(ds)
    x_train = np.array([image for (image, label) in ds_np['train']])
    x_test = np.array([image for (image, label) in ds_np['test']])
    y_train = np.array([label for (image, label) in ds_np['train']])
    y_test = np.array([label for (image, label) in ds_np['test']])
    return x_train, y_train, x_test, y_test

def load_rock_paper_scissors():
    ds = tfds.load('rock_paper_scissors', as_supervised=True)
    ds_np = tfds.as_numpy(ds)
    x_train = np.array([image for (image, label) in ds_np['train']])
    x_test = np.array([image for (image, label) in ds_np['test']])
    y_train = np.array([label for (image, label) in ds_np['train']])
    y_test = np.array([label for (image, label) in ds_np['test']])
    return x_train, y_train, x_test, y_test

def plot_some_examples():
    for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
